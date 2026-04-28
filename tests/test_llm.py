import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


sys.path.append(str(Path(__file__).parent.parent))

from kbprojection.llm import LLMGenerationError, _load_dotenv_if_present, call_llm, extract_kb_from_output
from kbprojection.models import (
    ExperimentStatus,
    LLMKBInjection,
    LLMKBResponse,
    LangProResult,
    NLIProblem,
    NLILabel,
    ProblemConfig,
)
from kbprojection.orchestration import process_single_problem


class TestLLMHandling(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.problem = NLIProblem(
            id="manual-1",
            premises=["A dog runs."],
            hypothesis="An animal moves.",
            gold_label=NLILabel.ENTAILMENT,
            dataset="manual",
            split="test",
        )

    def test_load_dotenv_sets_missing_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dotenv_path = Path(tmp_dir) / ".env"
            dotenv_path.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True):
                _load_dotenv_if_present(dotenv_path)
                self.assertEqual("test-key", __import__("os").environ["OPENROUTER_API_KEY"])

    def test_extract_kb_from_output_ignores_missing_closing_paren(self):
        self.assertEqual([], extract_kb_from_output("isa_wn(man, person"))
        self.assertEqual(["isa_wn(man, person)"], extract_kb_from_output("isa_wn(man, person)"))

    @patch("kbprojection.llm.asyncio.sleep", new_callable=AsyncMock)
    @patch("kbprojection.llm.AsyncGenericAIClient.generate", new_callable=AsyncMock)
    @patch("kbprojection.llm.AsyncGenericAIClient._setup_client", return_value=None)
    async def test_call_llm_retries_legacy_malformed_relation(
        self,
        _mock_setup,
        mock_generate,
        _mock_sleep,
    ):
        mock_generate.side_effect = [
            LLMKBResponse(output=[LLMKBInjection(KB_injection="isa_wn(man, person")]),
            LLMKBResponse(output=[LLMKBInjection(KB_injection="isa_wn(man, person)")]),
        ]

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            result = await call_llm("openai", "gpt-5-mini", "legacy_icl", self.problem, max_retries=1)

        self.assertEqual(["isa_wn(man, person)"], result)
        self.assertEqual(2, mock_generate.await_count)

    @patch("kbprojection.llm.AsyncGenericAIClient.generate", new_callable=AsyncMock)
    @patch("kbprojection.llm.AsyncGenericAIClient._setup_client", return_value=None)
    async def test_call_llm_raises_provider_error_after_retries(self, _mock_setup, mock_generate):
        mock_generate.side_effect = RuntimeError("boom")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with self.assertRaises(LLMGenerationError) as ctx:
                await call_llm("openai", "gpt-5-mini", "icl", self.problem, max_retries=0)

        self.assertIn("openai", str(ctx.exception))
        self.assertIn("gpt-5-mini", str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))

    @patch("kbprojection.orchestration.call_llm", new_callable=AsyncMock)
    @patch("kbprojection.orchestration.langpro_api_call", new_callable=AsyncMock)
    async def test_process_single_problem_records_llm_error(self, mock_langpro, mock_call_llm):
        mock_call_llm.side_effect = RuntimeError("provider exploded")
        mock_langpro.return_value = LangProResult(label=NLILabel.NEUTRAL)

        result = await process_single_problem(
            self.problem,
            config=ProblemConfig(verbose=False),
        )

        self.assertEqual(result.final_status, ExperimentStatus.KB_GENERATION_FAILED)
        self.assertEqual(result.llm_error, "provider exploded")

    @patch("kbprojection.orchestration.call_llm", new_callable=AsyncMock)
    @patch("kbprojection.orchestration.langpro_api_call", new_callable=AsyncMock)
    async def test_process_single_problem_records_kb_generation_empty(self, mock_langpro, mock_call_llm):
        mock_call_llm.return_value = []
        mock_langpro.return_value = LangProResult(label=NLILabel.NEUTRAL)

        result = await process_single_problem(
            self.problem,
            config=ProblemConfig(verbose=False),
        )

        self.assertEqual(result.final_status, ExperimentStatus.KB_GENERATION_EMPTY)
        self.assertEqual(result.kb_raw, [])
        self.assertIsNone(result.llm_error)


if __name__ == "__main__":
    unittest.main()
