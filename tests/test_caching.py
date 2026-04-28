import sys
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


sys.path.append(str(Path(__file__).parent.parent))

from kbprojection.langpro import clear_langpro_cache, langpro_api_call, set_langpro_cache_backend
from kbprojection.langpro_cache import InMemoryLangProCache, SQLiteLangProCache


MOCK_RESPONSE_TEXT = (
    '{"prob": [], "proofs": {}}'
)


class TestLangProCaching(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.memory_backend = InMemoryLangProCache()
        set_langpro_cache_backend(self.memory_backend)
        clear_langpro_cache()

    def tearDown(self):
        set_langpro_cache_backend(InMemoryLangProCache())

    @patch("kbprojection.langpro._execute_local_langpro_request", new_callable=AsyncMock)
    async def test_in_memory_cache_reuses_same_kb_for_local_langpro(self, mock_local_request):
        mock_local_request.return_value = (MOCK_RESPONSE_TEXT, None)
        premises = ["A implies B", "A"]
        hypothesis = "B"
        kb_list = ["relation(a,b)", "relation(c,d)"]
        kb_list_reversed = ["relation(c,d)", "relation(a,b)"]

        res_a = await langpro_api_call(premises, hypothesis, endpoint="local://auto", kb=kb_list)
        res_b = await langpro_api_call(
            premises,
            hypothesis,
            endpoint="local://auto",
            kb=kb_list_reversed,
        )

        self.assertEqual(mock_local_request.call_count, 1)
        self.assertEqual(res_a.label, res_b.label)

    @patch("kbprojection.langpro._execute_local_langpro_request", new_callable=AsyncMock)
    async def test_sqlite_cache_reuses_same_kb_across_local_calls(self, mock_local_request):
        mock_local_request.return_value = (MOCK_RESPONSE_TEXT, None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = SQLiteLangProCache(Path(tmp_dir) / "langpro_cache.sqlite3")
            set_langpro_cache_backend(backend)
            clear_langpro_cache()

            premises = ["A implies B", "A"]
            hypothesis = "B"
            kb_list = ["relation(a,b)", "relation(c,d)"]
            kb_list_reversed = ["relation(c,d)", "relation(a,b)"]

            res_a = await langpro_api_call(premises, hypothesis, endpoint="local://auto", kb=kb_list)
            res_b = await langpro_api_call(
                premises,
                hypothesis,
                endpoint="local://auto",
                kb=kb_list_reversed,
            )

            self.assertEqual(mock_local_request.call_count, 1)
            self.assertEqual(res_a.label, res_b.label)

            set_langpro_cache_backend(InMemoryLangProCache())

    @patch("kbprojection.langpro._execute_local_langpro_request", new_callable=AsyncMock)
    async def test_inflight_cache_deduplicates_concurrent_calls(self, mock_local_request):
        async def delayed_response(*_args, **_kwargs):
            await asyncio.sleep(0.01)
            return MOCK_RESPONSE_TEXT, None

        mock_local_request.side_effect = delayed_response
        premises = ["A implies B", "A"]
        hypothesis = "B"
        kb_list = ["relation(a,b)", "relation(c,d)"]

        res_a, res_b = await asyncio.gather(
            langpro_api_call(premises, hypothesis, endpoint="local://auto", kb=kb_list),
            langpro_api_call(premises, hypothesis, endpoint="local://auto", kb=list(reversed(kb_list))),
        )

        self.assertEqual(mock_local_request.call_count, 1)
        self.assertEqual(res_a.label, res_b.label)


if __name__ == "__main__":
    unittest.main()
