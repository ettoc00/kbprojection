import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.async_runtime import AsyncRunLimits, create_async_run_context
from kbprojection.langpro import langpro_api_call
from kbprojection.settings import get_default_results_dir

DEFAULT_PATTERNS = [
    "2x2_50_compare__openrouter__*.json",
    "results_replay_openrouter_*__*.json",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _iter_target_files(results_dir: Path, pattern: str) -> list[Path]:
    return sorted(results_dir.glob(pattern))


def _serialize_langpro_result(result: Any) -> dict[str, Any]:
    payload = json.loads(result.model_dump_json(fallback=str))
    proofs = getattr(result, "proofs", {}) or {}
    payload["proofs_rendered"] = {
        label: _render_tree(tree)
        for label, tree in proofs.items()
    }
    return payload


def _render_tree(tree: Any) -> str:
    if tree is None:
        return ""
    if hasattr(tree, "pformat"):
        try:
            return tree.pformat()
        except Exception:
            pass
    return str(tree)


def _normalize_items(payload: Any) -> tuple[list[dict[str, Any]], str]:
    if isinstance(payload, dict) and "runs" in payload:
        tagged: list[dict[str, Any]] = []
        for prompt_style, run in (payload.get("runs") or {}).items():
            for problem_id, item in (run.get("items_by_id") or {}).items():
                tagged.append(
                    {
                        "_container": "compare",
                        "_prompt_style": prompt_style,
                        "_problem_id": problem_id,
                        "_item": item,
                    }
                )
        return tagged, "compare"

    if isinstance(payload, list):
        return [{"_container": "replay", "_item": item} for item in payload], "replay"

    raise ValueError("Unsupported result payload format")


async def _call_langpro_for_item(item: dict[str, Any], context: Any) -> tuple[dict[str, Any], dict[str, str]]:
    problem = item["problem"]
    premises = list(problem["premises"])
    hypothesis = problem["hypothesis"]
    mismatches: dict[str, str] = {}

    baseline = await langpro_api_call(
        premises,
        hypothesis,
        report=False,
        context=context,
    )
    item["prover_call_no_kb"] = _serialize_langpro_result(baseline)
    if item.get("pred_no_kb") is not None and str(item.get("pred_no_kb")) != str(baseline.label.value):
        mismatches["pred_no_kb"] = f"{item.get('pred_no_kb')} -> {baseline.label.value}"

    raw_kb = list(item.get("kb_raw") or [])
    if item.get("status_with_raw_kb") != "pending" and raw_kb:
        raw = await langpro_api_call(
            premises,
            hypothesis,
            kb=raw_kb,
            report=False,
            context=context,
        )
        item["prover_call_raw_kb"] = _serialize_langpro_result(raw)
        if item.get("pred_with_raw_kb") is not None and str(item.get("pred_with_raw_kb")) != str(raw.label.value):
            mismatches["pred_with_raw_kb"] = f"{item.get('pred_with_raw_kb')} -> {raw.label.value}"

    norm_kb = list(item.get("kb_filtered") or [])
    if item.get("status_with_kb") != "pending" and norm_kb:
        norm = await langpro_api_call(
            premises,
            hypothesis,
            kb=norm_kb,
            report=False,
            context=context,
        )
        item["prover_call_kb"] = _serialize_langpro_result(norm)
        if item.get("pred_with_kb") is not None and str(item.get("pred_with_kb")) != str(norm.label.value):
            mismatches["pred_with_kb"] = f"{item.get('pred_with_kb')} -> {norm.label.value}"

    return item, mismatches


def _make_request_key(problem: dict[str, Any], kb: list[str]) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    return (
        tuple(problem["premises"]),
        problem["hypothesis"],
        tuple(sorted(kb)),
    )


async def _call_langpro_for_request(
    problem: dict[str, Any],
    kb: list[str],
    context: Any,
) -> dict[str, Any]:
    result = await langpro_api_call(
        list(problem["premises"]),
        problem["hypothesis"],
        kb=kb or None,
        report=False,
        context=context,
    )
    return _serialize_langpro_result(result)


async def _backfill_file(path: Path, context: Any) -> tuple[int, int]:
    payload = _load_json(path)
    tagged_items, kind = _normalize_items(payload)
    updated = 0
    mismatched = 0

    requests: dict[tuple[str, tuple[str, ...], str, tuple[str, ...]], dict[str, Any]] = {}
    consumers: dict[tuple[str, tuple[str, ...], str, tuple[str, ...]], list[tuple[dict[str, Any], str, str]]] = defaultdict(list)

    for tagged in tagged_items:
        item = tagged["_item"]
        problem = item["problem"]

        no_kb = []
        no_kb_request_key = ("no_kb",) + _make_request_key(problem, no_kb)
        requests.setdefault(no_kb_request_key, {"problem": problem, "kb": no_kb})
        consumers[no_kb_request_key].append((item, "prover_call_no_kb", "pred_no_kb"))

        raw_kb = list(item.get("kb_raw") or [])
        if item.get("status_with_raw_kb") != "pending" and raw_kb:
            raw_request_key = ("raw_kb",) + _make_request_key(problem, raw_kb)
            requests.setdefault(raw_request_key, {"problem": problem, "kb": raw_kb})
            consumers[raw_request_key].append((item, "prover_call_raw_kb", "pred_with_raw_kb"))

        norm_kb = list(item.get("kb_filtered") or [])
        if item.get("status_with_kb") != "pending" and norm_kb:
            norm_request_key = ("norm_kb",) + _make_request_key(problem, norm_kb)
            requests.setdefault(norm_request_key, {"problem": problem, "kb": norm_kb})
            consumers[norm_request_key].append((item, "prover_call_kb", "pred_with_kb"))

    request_keys = list(requests)
    total = len(request_keys)
    for index, request_key in enumerate(request_keys, start=1):
        request = requests[request_key]
        serialized = await _call_langpro_for_request(request["problem"], request["kb"], context)
        predicted = serialized.get("label")
        for item, field_name, pred_field in consumers[request_key]:
            item[field_name] = serialized
            previous = item.get(pred_field)
            if previous is not None and str(previous) != str(predicted):
                item.setdefault("prover_backfill_mismatches", {})[pred_field] = f"{previous} -> {predicted}"
        if index % 25 == 0 or index == total:
            print(f"{path.name}: unique_requests {index}/{total}")

    for tagged in tagged_items:
        item = tagged["_item"]
        if item.get("prover_backfill_mismatches"):
            mismatched += 1
        updated += 1

    if kind == "compare":
        for tagged in tagged_items:
            payload["runs"][tagged["_prompt_style"]]["items_by_id"][tagged["_problem_id"]] = tagged["_item"]
    else:
        payload = [tagged["_item"] for tagged in tagged_items]

    _save_json(path, payload)
    return updated, mismatched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill prover calls into saved benchmark result JSON files without rerunning LLM generation."
    )
    parser.add_argument(
        "--results-dir",
        default=str(get_default_results_dir()),
        help="Directory containing result JSON files.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern(s) of result files to update.",
    )
    parser.add_argument(
        "--skip-backups",
        action="store_true",
        help="Do not create .bak copies before editing files in place.",
    )
    parser.add_argument(
        "--langpro-concurrency",
        type=int,
        default=4,
        help="Concurrent remote LangPro calls.",
    )
    parser.add_argument(
        "--local-langpro-concurrency",
        type=int,
        default=2,
        help="Concurrent local LangPro calls.",
    )
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    files: list[Path] = []
    patterns = args.pattern or DEFAULT_PATTERNS
    for pattern in patterns:
        files.extend(_iter_target_files(results_dir, pattern))
    files = sorted({path for path in files if "__backup_" not in path.name})
    if not files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    if not args.skip_backups:
        for path in files:
            backup = path.with_suffix(path.suffix + ".bak")
            if not backup.exists():
                backup.write_bytes(path.read_bytes())

    context = create_async_run_context(
        AsyncRunLimits(
            llm_concurrency=1,
            langpro_concurrency=args.langpro_concurrency,
            local_langpro_concurrency=args.local_langpro_concurrency,
        )
    )

    total_updated = 0
    total_mismatched = 0
    for path in files:
        updated, mismatched = await _backfill_file(path, context)
        total_updated += updated
        total_mismatched += mismatched
        print(f"Updated {path.name}: rows={updated} mismatches={mismatched}")

    print(f"Finished. rows={total_updated} mismatches={total_mismatched}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
