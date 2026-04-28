import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from kbprojection.models import (
    ExperimentResult,
    ExperimentStatus,
    LangProResult,
    NLIProblem,
    NLILabel,
    ProblemConfig,
)
from kbprojection.runners import (
    arun_problem,
    arun_problems,
    infer_provider,
    run_problem,
    run_problems,
    serialize_result_payload,
)


def make_problem(problem_id: str = "p1") -> NLIProblem:
    return NLIProblem(
        id=problem_id,
        premises=["A dog runs."],
        hypothesis="An animal moves.",
        gold_label=NLILabel.ENTAILMENT,
        dataset="manual",
        split="test",
    )


def make_result(problem: NLIProblem) -> ExperimentResult:
    return ExperimentResult(
        problem=problem,
        pred_no_kb=NLILabel.NEUTRAL,
        final_status=ExperimentStatus.KB_NOT_SOLVED,
        prover_calls=[LangProResult(label=NLILabel.NEUTRAL)],
    )


class TestRunnerUtilities(unittest.TestCase):
    def test_infer_provider_explicit_provider_wins(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(infer_provider("anything", "openrouter"), "openrouter")

    def test_infer_provider_prefers_openrouter_for_slash_model(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}, clear=True):
            self.assertEqual(infer_provider("openai/gpt-5.4-mini"), "openrouter")

    def test_infer_provider_prefers_openai_for_plain_model(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}, clear=True):
            self.assertEqual(infer_provider("gpt-5-mini"), "openai")

    def test_infer_provider_raises_without_keys(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                infer_provider("gpt-5-mini")

    def test_serialize_result_payload_adds_metadata_and_discards_prover_calls(self):
        payload = serialize_result_payload(
            make_result(make_problem()),
            model="gpt-5-mini",
            provider="openai",
            prompt_style="icl",
            discard_prover_calls=True,
        )

        self.assertEqual(payload["model"], "gpt-5-mini")
        self.assertEqual(payload["provider"], "openai")
        self.assertEqual(payload["prompt_style"], "icl")
        self.assertIsNone(payload["prover_calls"])

    def test_serialize_result_payload_keeps_prover_calls_by_default(self):
        payload = serialize_result_payload(make_result(make_problem()))
        self.assertIsInstance(payload["prover_calls"], list)


class TestPublicRunners(unittest.IsolatedAsyncioTestCase):
    @patch("kbprojection.runners.process_single_problem", new_callable=AsyncMock)
    async def test_arun_problem_builds_config_and_context(self, mock_process):
        problem = make_problem()
        mock_process.return_value = make_result(problem)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}, clear=True):
            result = await arun_problem(problem, model="gpt-5-mini", verbose=False)

        self.assertIs(result, mock_process.return_value)
        _, kwargs = mock_process.call_args
        self.assertEqual(kwargs["config"].llm_provider, "openai")
        self.assertEqual(kwargs["config"].model, "gpt-5-mini")
        self.assertFalse(kwargs["config"].verbose)
        self.assertIsNotNone(kwargs["context"])

    @patch("kbprojection.runners.process_single_problem", new_callable=AsyncMock)
    async def test_arun_problem_uses_supplied_config(self, mock_process):
        problem = make_problem()
        config = ProblemConfig(llm_provider="openrouter", model="openai/gpt-5.4-mini", verbose=False)
        mock_process.return_value = make_result(problem)

        await arun_problem(problem, config=config)

        _, kwargs = mock_process.call_args
        self.assertIs(kwargs["config"], config)

    @patch("kbprojection.runners.process_single_problem", new_callable=AsyncMock)
    async def test_arun_problems_preserves_input_order(self, mock_process):
        problems = [make_problem("p1"), make_problem("p2")]

        async def fake_process(problem, **_kwargs):
            if problem.id == "p1":
                await asyncio.sleep(0.02)
            return make_result(problem)

        mock_process.side_effect = fake_process

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}, clear=True):
            payloads = await arun_problems(
                problems,
                model="gpt-5-mini",
                concurrency=2,
                show_progress=False,
            )

        self.assertEqual([payload["problem"]["id"] for payload in payloads], ["p1", "p2"])

    async def test_arun_problems_empty_list(self):
        self.assertEqual(await arun_problems([], provider="openai", show_progress=False), [])

    async def test_arun_problems_validates_concurrency(self):
        with self.assertRaises(ValueError):
            await arun_problems([make_problem()], provider="openai", concurrency=0, show_progress=False)

    @patch("kbprojection.runners.process_single_problem", new_callable=AsyncMock)
    async def test_arun_problems_discards_prover_calls(self, mock_process):
        problem = make_problem()
        mock_process.return_value = make_result(problem)

        payloads = await arun_problems(
            [problem],
            provider="openai",
            discard_prover_calls=True,
            show_progress=False,
        )

        self.assertIsNone(payloads[0]["prover_calls"])


class TestSyncRunners(unittest.TestCase):
    @patch("kbprojection.runners.process_single_problem", new_callable=AsyncMock)
    def test_run_problem(self, mock_process):
        problem = make_problem()
        mock_process.return_value = make_result(problem)

        result = run_problem(problem, provider="openai")

        self.assertIs(result, mock_process.return_value)

    @patch("kbprojection.runners.process_single_problem", new_callable=AsyncMock)
    def test_run_problems(self, mock_process):
        problem = make_problem()
        mock_process.return_value = make_result(problem)

        payloads = run_problems([problem], provider="openai", show_progress=False)

        self.assertEqual([payload["problem"]["id"] for payload in payloads], ["p1"])


if __name__ == "__main__":
    unittest.main()
