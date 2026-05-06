import sys
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


sys.path.append(str(Path(__file__).parent.parent))

from kbprojection.async_runtime import AsyncRunLimits, create_async_run_context
from kbprojection.langpro import (
    _HYBRID_BACKEND_HEALTH,
    _eligible_hybrid_backends,
    _mark_hybrid_backend_failure,
    _mark_hybrid_backend_success,
    _make_legacy_endpoint_langpro_cache_key,
    _reset_hybrid_backend_health,
    clear_langpro_cache,
    langpro_api_call,
    set_langpro_cache_backend,
)
from kbprojection.langpro_cache import InMemoryLangProCache, SQLiteLangProCache
from kbprojection.models import NLILabel
from kbprojection.settings import DEFAULT_LANGPRO_ENDPOINT


MOCK_RESPONSE_TEXT = (
    '{"prob": [], "proofs": {}}'
)


class TestLangProCaching(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.memory_backend = InMemoryLangProCache()
        set_langpro_cache_backend(self.memory_backend)
        clear_langpro_cache()
        _reset_hybrid_backend_health()

    def tearDown(self):
        set_langpro_cache_backend(InMemoryLangProCache())
        _reset_hybrid_backend_health()

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

    @patch("kbprojection.langpro._execute_remote_langpro_request_without_limit", new_callable=AsyncMock)
    @patch("kbprojection.langpro._execute_local_langpro_request", new_callable=AsyncMock)
    async def test_cache_ignores_endpoint_for_local_and_remote_calls(
        self,
        mock_local_request,
        mock_remote_request,
    ):
        mock_local_request.return_value = (MOCK_RESPONSE_TEXT, None)
        mock_remote_request.return_value = (MOCK_RESPONSE_TEXT, None)
        premises = ["A implies B", "A"]
        hypothesis = "B"
        kb_list = ["relation(a,b)"]

        res_a = await langpro_api_call(premises, hypothesis, endpoint="local://auto", kb=kb_list)
        res_b = await langpro_api_call(
            premises,
            hypothesis,
            endpoint=DEFAULT_LANGPRO_ENDPOINT,
            kb=kb_list,
        )

        self.assertEqual(mock_local_request.call_count, 1)
        self.assertEqual(mock_remote_request.call_count, 0)
        self.assertEqual(res_a.label, res_b.label)

    @patch("kbprojection.langpro._execute_langpro_request", new_callable=AsyncMock)
    async def test_legacy_endpoint_cache_entry_is_migrated_without_deletion(self, mock_execute):
        premises = ["A implies B", "A"]
        hypothesis = "B"
        kb_list = ["relation(a,b)"]
        legacy_key = _make_legacy_endpoint_langpro_cache_key(
            premises,
            hypothesis,
            DEFAULT_LANGPRO_ENDPOINT,
            "easyccg",
            200,
            kb_list,
            "all",
            True,
            True,
        )
        self.memory_backend.set(legacy_key, MOCK_RESPONSE_TEXT)

        result = await langpro_api_call(
            premises,
            hypothesis,
            endpoint=DEFAULT_LANGPRO_ENDPOINT,
            kb=kb_list,
        )

        self.assertEqual(result.label, NLILabel.NEUTRAL)
        self.assertEqual(mock_execute.call_count, 0)
        self.assertEqual(self.memory_backend.get(legacy_key), MOCK_RESPONSE_TEXT)

    @patch("kbprojection.langpro._execute_langpro_request", new_callable=AsyncMock)
    async def test_inflight_cache_deduplicates_across_endpoints(self, mock_execute):
        async def delayed_response(*_args, **_kwargs):
            await asyncio.sleep(0.01)
            return MOCK_RESPONSE_TEXT, None

        mock_execute.side_effect = delayed_response
        premises = ["A implies B", "A"]
        hypothesis = "B"
        kb_list = ["relation(a,b)"]

        res_a, res_b = await asyncio.gather(
            langpro_api_call(premises, hypothesis, endpoint="local://auto", kb=kb_list),
            langpro_api_call(
                premises,
                hypothesis,
                endpoint=DEFAULT_LANGPRO_ENDPOINT,
                kb=kb_list,
            ),
        )

        self.assertEqual(mock_execute.call_count, 1)
        self.assertEqual(res_a.label, res_b.label)

    @patch("kbprojection.langpro._execute_hybrid_backend_with_token", new_callable=AsyncMock)
    async def test_hybrid_routes_to_local_when_remote_saturated(self, mock_backend):
        mock_backend.return_value = (MOCK_RESPONSE_TEXT, None)
        context = create_async_run_context(
            AsyncRunLimits(
                llm_concurrency=1,
                langpro_concurrency=1,
                local_langpro_concurrency=1,
            )
        )
        await context.langpro_semaphore.acquire()
        try:
            await langpro_api_call(
                ["A implies B", "A"],
                "B",
                endpoint="hybrid://auto",
                kb=["relation(a,b)"],
                context=context,
            )
        finally:
            context.langpro_semaphore.release()

        self.assertEqual(mock_backend.call_args.args[0], "local")

    @patch("kbprojection.langpro._execute_hybrid_backend_with_token", new_callable=AsyncMock)
    async def test_hybrid_routes_to_remote_when_local_saturated(self, mock_backend):
        mock_backend.return_value = (MOCK_RESPONSE_TEXT, None)
        context = create_async_run_context(
            AsyncRunLimits(
                llm_concurrency=1,
                langpro_concurrency=1,
                local_langpro_concurrency=1,
            )
        )
        await context.local_langpro_semaphore.acquire()
        try:
            await langpro_api_call(
                ["A implies B", "A"],
                "B",
                endpoint="hybrid://auto",
                kb=["relation(a,b)"],
                context=context,
            )
        finally:
            context.local_langpro_semaphore.release()

        self.assertEqual(mock_backend.call_args.args[0], "remote")

    @patch("kbprojection.langpro._acquire_hybrid_backend_token", new_callable=AsyncMock)
    @patch("kbprojection.langpro._execute_hybrid_backend_with_token", new_callable=AsyncMock)
    async def test_hybrid_remote_failure_falls_back_to_local(self, mock_backend, mock_acquire):
        mock_acquire.return_value = "remote"

        async def backend_response(backend, *_args, **_kwargs):
            if backend == "remote":
                return None, "remote down"
            return MOCK_RESPONSE_TEXT, None

        mock_backend.side_effect = backend_response

        result = await langpro_api_call(
            ["A implies B", "A"],
            "B",
            endpoint="hybrid://auto",
            kb=["relation(a,b)"],
        )

        self.assertEqual(result.label, NLILabel.NEUTRAL)
        self.assertEqual([call.args[0] for call in mock_backend.call_args_list], ["remote", "local"])
        self.assertEqual(_HYBRID_BACKEND_HEALTH["remote"].consecutive_failures, 1)

    @patch("kbprojection.langpro._acquire_hybrid_backend_token", new_callable=AsyncMock)
    @patch("kbprojection.langpro._execute_hybrid_backend_with_token", new_callable=AsyncMock)
    async def test_hybrid_local_failure_falls_back_to_remote(self, mock_backend, mock_acquire):
        mock_acquire.return_value = "local"

        async def backend_response(backend, *_args, **_kwargs):
            if backend == "local":
                return None, "local missing"
            return MOCK_RESPONSE_TEXT, None

        mock_backend.side_effect = backend_response

        result = await langpro_api_call(
            ["A implies B", "A"],
            "B",
            endpoint="hybrid://auto",
            kb=["relation(a,b)"],
        )

        self.assertEqual(result.label, NLILabel.NEUTRAL)
        self.assertEqual([call.args[0] for call in mock_backend.call_args_list], ["local", "remote"])
        self.assertEqual(_HYBRID_BACKEND_HEALTH["local"].consecutive_failures, 1)

    @patch("kbprojection.langpro._acquire_hybrid_backend_token", new_callable=AsyncMock)
    @patch("kbprojection.langpro._execute_hybrid_backend_with_token", new_callable=AsyncMock)
    async def test_hybrid_reports_both_backend_errors_when_both_fail(self, mock_backend, mock_acquire):
        mock_acquire.return_value = "remote"

        async def backend_response(backend, *_args, **_kwargs):
            return None, f"{backend} failed"

        mock_backend.side_effect = backend_response

        result = await langpro_api_call(
            ["A implies B", "A"],
            "B",
            endpoint="hybrid://auto",
            kb=["relation(a,b)"],
        )

        self.assertEqual(result.label, NLILabel.UNKNOWN)
        self.assertIn("remote: remote failed", result.error or "")
        self.assertIn("local: local failed", result.error or "")

    @patch("kbprojection.langpro._execute_hybrid_backend_with_token", new_callable=AsyncMock)
    async def test_hybrid_cache_hit_skips_backends(self, mock_backend):
        mock_backend.return_value = (MOCK_RESPONSE_TEXT, None)
        premises = ["A implies B", "A"]
        hypothesis = "B"
        kb_list = ["relation(a,b)"]

        await langpro_api_call(premises, hypothesis, endpoint="hybrid://auto", kb=kb_list)
        await langpro_api_call(premises, hypothesis, endpoint="hybrid://auto", kb=kb_list)

        self.assertEqual(mock_backend.call_count, 1)

    def test_hybrid_backend_degradation_and_probe_backoff(self):
        _mark_hybrid_backend_failure("remote", 100.0)
        _mark_hybrid_backend_failure("remote", 101.0)
        self.assertIn("remote", _eligible_hybrid_backends(102.0))

        _mark_hybrid_backend_failure("remote", 102.0)
        self.assertNotIn("remote", _eligible_hybrid_backends(103.0))
        self.assertEqual(_HYBRID_BACKEND_HEALTH["remote"].next_probe_at, 117.0)

        self.assertIn("remote", _eligible_hybrid_backends(117.0))
        _mark_hybrid_backend_failure("remote", 117.0)
        self.assertEqual(_HYBRID_BACKEND_HEALTH["remote"].next_probe_at, 147.0)

        _mark_hybrid_backend_success("remote")
        self.assertIn("remote", _eligible_hybrid_backends(118.0))
        self.assertEqual(_HYBRID_BACKEND_HEALTH["remote"].consecutive_failures, 0)

    def test_hybrid_tries_both_when_both_backends_degraded(self):
        for backend in ("remote", "local"):
            _mark_hybrid_backend_failure(backend, 100.0)
            _mark_hybrid_backend_failure(backend, 101.0)
            _mark_hybrid_backend_failure(backend, 102.0)

        self.assertEqual(_eligible_hybrid_backends(103.0), ["remote", "local"])

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
