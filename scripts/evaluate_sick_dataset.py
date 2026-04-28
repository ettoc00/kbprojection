import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


DEFAULT_SPLITS = ("train", "dev", "test")


def load_dotenv_if_present(dotenv_path: Optional[Path] = None) -> None:
    if dotenv_path is None:
        dotenv_path = PROJECT_ROOT / ".env"

    if not dotenv_path.exists():
        return

    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def infer_provider(model: str, explicit_provider: Optional[str]) -> str:
    if explicit_provider:
        return explicit_provider

    if "/" in model and os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if "/" not in model and os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"

    raise ValueError(
        "Could not infer provider from the environment. "
        "Set a provider API key or pass --provider explicitly."
    )


def problem_key(problem: Any) -> str:
    return f"{problem.split}:{problem.id}"


def load_existing_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_run_log(path: Optional[Path], event: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def serializable_result_payload(result: Any, discard_prover_calls: bool = False) -> Dict[str, Any]:
    exclude = {"prover_calls"} if discard_prover_calls else None
    payload = json.loads(result.model_dump_json(exclude=exclude, fallback=str))
    if discard_prover_calls:
        payload["prover_calls"] = None
    return payload


def build_default_results_path(results_dir: Path, provider: str, model: str, prompt_style: str) -> Path:
    safe_provider = sanitize_filename_part(provider)
    safe_model = sanitize_filename_part(model)
    safe_prompt = sanitize_filename_part(prompt_style)
    return results_dir / f"sick_full__{safe_prompt}__{safe_provider}__{safe_model}.json"


def normalize_label_filters(raw_labels: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not raw_labels:
        return None
    return {label.strip().lower() for label in raw_labels if label.strip()}


def load_sick_problems(splits: Sequence[str], label_filter: Optional[set[str]]) -> List[Any]:
    from kbprojection.loaders.sick import SICKLoader

    loader = SICKLoader()
    loader.load(splits=list(splits))

    problems: List[Any] = []
    for split in splits:
        problems.extend(loader.iter_problems(split=split, label_filter=label_filter))
    return problems


def build_initial_state(
    results_path: Path,
    model: str,
    provider: str,
    prompt_style: str,
    splits: Sequence[str],
    problem_keys: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "meta": {
            "results_file": str(results_path),
            "dataset": "sick",
            "splits": list(splits),
            "problem_keys": problem_keys,
            "model": model,
            "provider": provider,
            "prompt_style": prompt_style,
            "test_mode": args.test_mode,
            "run_ablation": args.run_ablation,
            "post_process": not args.no_post_process,
            "label_filter": args.labels,
        },
        "items_by_key": {},
        "summary": {},
    }


def ensure_compatible_state(
    state: Dict[str, Any],
    results_path: Path,
    model: str,
    provider: str,
    prompt_style: str,
    splits: Sequence[str],
    problem_keys: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not state:
        return build_initial_state(
            results_path,
            model,
            provider,
            prompt_style,
            splits,
            problem_keys,
            args,
        )

    meta = state.get("meta", {})
    expected = {
        "dataset": "sick",
        "model": model,
        "provider": provider,
        "prompt_style": prompt_style,
        "test_mode": args.test_mode,
        "run_ablation": args.run_ablation,
        "post_process": not args.no_post_process,
    }
    mismatches = [
        f"{key}={meta.get(key)!r} expected {value!r}"
        for key, value in expected.items()
        if meta.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Existing output file is incompatible: {results_path}. "
            + "; ".join(mismatches)
        )

    state.setdefault("items_by_key", {})
    state.setdefault("summary", {})
    state["meta"]["results_file"] = str(results_path)
    state["meta"]["splits"] = list(splits)
    state["meta"]["problem_keys"] = problem_keys
    state["meta"]["label_filter"] = args.labels
    return state


def build_summary(state: Dict[str, Any], all_problem_keys: List[str]) -> Dict[str, Any]:
    items = state.get("items_by_key", {})
    completed_items = [items[key] for key in all_problem_keys if key in items]

    by_status: Dict[str, int] = {}
    by_split: Dict[str, Dict[str, int]] = {}
    correct_counts = {
        "no_kb": 0,
        "raw_kb": 0,
        "normalised_kb": 0,
    }
    successful_calls = {
        "no_kb": 0,
        "raw_kb": 0,
        "normalised_kb": 0,
    }

    for item in completed_items:
        status = str(item.get("final_status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1

        problem = item.get("problem", {})
        split = str(problem.get("split") or "unknown")
        split_counts = by_split.setdefault(split, {"total": 0})
        split_counts["total"] += 1
        split_counts[status] = split_counts.get(status, 0) + 1

        gold_label = problem.get("gold_label")
        if item.get("status_no_kb") == "success":
            successful_calls["no_kb"] += 1
            if item.get("pred_no_kb") == gold_label:
                correct_counts["no_kb"] += 1
        if item.get("status_with_raw_kb") == "success":
            successful_calls["raw_kb"] += 1
            if item.get("pred_with_raw_kb") == gold_label:
                correct_counts["raw_kb"] += 1
        if item.get("status_with_kb") == "success":
            successful_calls["normalised_kb"] += 1
            if item.get("pred_with_kb") == gold_label:
                correct_counts["normalised_kb"] += 1

    return {
        "total": len(all_problem_keys),
        "completed": len(completed_items),
        "remaining": len(all_problem_keys) - len(completed_items),
        "by_status": dict(sorted(by_status.items())),
        "by_split": by_split,
        "correct_counts": correct_counts,
        "successful_calls": successful_calls,
        "accuracy_on_successful_calls": {
            key: (
                correct_counts[key] / successful_calls[key]
                if successful_calls[key]
                else 0.0
            )
            for key in correct_counts
        },
    }


async def run_async(args: argparse.Namespace) -> int:
    from kbprojection.async_runtime import AsyncRunLimits, create_async_run_context
    from kbprojection.models import ProblemConfig, TestMode
    from kbprojection.orchestration import process_single_problem
    from kbprojection.settings import get_default_results_dir

    load_dotenv_if_present()

    provider = infer_provider(args.model, args.provider)
    splits = tuple(args.splits)
    label_filter = normalize_label_filters(args.labels)
    problems = load_sick_problems(splits, label_filter)
    if args.limit is not None:
        problems = problems[: args.limit]

    all_problem_keys = [problem_key(problem) for problem in problems]
    results_path = (
        Path(args.results_file).resolve()
        if args.results_file
        else build_default_results_path(
            get_default_results_dir(),
            provider,
            args.model,
            args.prompt_style,
        ).resolve()
    )
    run_log_path = (
        Path(args.run_log_file).resolve()
        if args.run_log_file
        else results_path.with_suffix(".jsonl")
    )

    state = ensure_compatible_state(
        load_existing_state(results_path),
        results_path,
        args.model,
        provider,
        args.prompt_style,
        splits,
        all_problem_keys,
        args,
    )
    save_state(results_path, state)

    context = create_async_run_context(
        AsyncRunLimits(
            llm_concurrency=args.llm_concurrency,
            langpro_concurrency=args.langpro_concurrency,
            local_langpro_concurrency=args.local_langpro_concurrency,
        )
    )
    config = ProblemConfig(
        llm_provider=provider,
        model=args.model,
        prompt_style=args.prompt_style,
        test_mode=TestMode(args.test_mode),
        run_ablation=args.run_ablation,
        post_process=not args.no_post_process,
        verbose=args.verbose,
    )

    completed_keys = set(state["items_by_key"])
    jobs = [problem for problem in problems if problem_key(problem) not in completed_keys]
    job_semaphore = asyncio.Semaphore(args.job_concurrency)

    async def process_job(problem: Any) -> Tuple[str, Dict[str, Any]]:
        key = problem_key(problem)
        append_run_log(
            run_log_path,
            {
                "event": "start_problem",
                "key": key,
                "split": problem.split,
                "id": problem.id,
                "gold_label": problem.gold_label.value,
            },
        )
        try:
            async with job_semaphore:
                if args.problem_timeout_seconds and args.problem_timeout_seconds > 0:
                    result = await asyncio.wait_for(
                        process_single_problem(problem, config=config, context=context),
                        timeout=args.problem_timeout_seconds,
                    )
                else:
                    result = await process_single_problem(problem, config=config, context=context)

            payload = serializable_result_payload(
                result,
                discard_prover_calls=args.discard_prover_calls,
            )
            append_run_log(
                run_log_path,
                {
                    "event": "finish_problem",
                    "key": key,
                    "split": problem.split,
                    "id": problem.id,
                    "final_status": str(payload.get("final_status")),
                },
            )
            payload["model"] = args.model
            payload["provider"] = provider
            payload["prompt_style"] = args.prompt_style
            return key, payload
        except Exception as exc:
            log_problem_exception(key, exc)
            raise

    def log_problem_exception(key: str, exc: BaseException) -> None:
        error_payload = {
            "event": "problem_exception",
            "key": key,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        append_run_log(run_log_path, error_payload)
        print(
            f"\n[error] Aborting on {key}: {type(exc).__name__}: {exc}\n"
            f"[error] Wrote run log to {run_log_path}",
            file=sys.stderr,
        )

    async def cancel_pending_tasks() -> None:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    print(
        f"\n=== Running SICK {','.join(splits)}: {len(jobs)} remaining / "
        f"{len(problems)} total for model={args.model} provider={provider} "
        f"prompt={args.prompt_style} ==="
    )
    print(f"Run log: {run_log_path}")
    if completed_keys:
        print(f"Resuming from {results_path}: {len(completed_keys)} items already completed.")

    completed_since_save = 0
    tasks = [asyncio.create_task(process_job(problem)) for problem in jobs]
    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="sick", unit="problem")
    try:
        for completed in progress:
            try:
                key, payload = await completed
            except Exception as exc:
                await cancel_pending_tasks()
                state["summary"] = build_summary(state, all_problem_keys)
                save_state(results_path, state)
                raise

            progress.set_postfix_str(key)
            state["items_by_key"][key] = payload
            completed_since_save += 1
            if completed_since_save >= args.save_every:
                state["summary"] = build_summary(state, all_problem_keys)
                save_state(results_path, state)
                completed_since_save = 0
    finally:
        progress.close()

    state["summary"] = build_summary(state, all_problem_keys)
    save_state(results_path, state)

    print()
    print(json.dumps(state["summary"], indent=2))
    print(f"\nWrote {results_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one model/prompt over the whole SICK dataset with resumable JSON output."
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model identifier, e.g. gpt-5-mini, claude-sonnet-4-5, or openai/gpt-5-mini.",
    )
    parser.add_argument(
        "-r",
        "--results-file",
        default=None,
        help="Output JSON path. Defaults to <results-dir>/sick_full__<prompt>__<provider>__<model>.json.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "openrouter", "claude", "gemini"],
        help="Optional provider override. By default the script infers the provider from the model and environment.",
    )
    parser.add_argument(
        "--prompt-style",
        default="icl",
        help="Prompt style to use, e.g. icl, cot, legacy_icl.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
        help="SICK splits to evaluate. Defaults to train dev test.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        choices=["entailment", "contradiction", "neutral"],
        help="Optional gold-label filter. Defaults to all labels.",
    )
    parser.add_argument(
        "--test-mode",
        default="both",
        choices=["no_kb", "raw_kb", "normalised", "filtered", "both", "full"],
        help="Pipeline mode. Defaults to both raw and normalised KB evaluation. 'filtered' is a legacy alias.",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run ablations for fixed examples. This can be expensive on the full dataset.",
    )
    parser.add_argument(
        "--no-post-process",
        action="store_true",
        help="Disable KB post-processing before filtered-KB evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional first-N limit for smoke tests before running the whole dataset.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--llm-concurrency", type=int, default=2)
    parser.add_argument("--langpro-concurrency", type=int, default=4)
    parser.add_argument("--local-langpro-concurrency", type=int, default=2)
    parser.add_argument("--job-concurrency", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--problem-timeout-seconds",
        type=float,
        default=900.0,
        help=(
            "Abort the run if a single problem takes longer than this many seconds. "
            "Use 0 to disable the per-problem timeout."
        ),
    )
    parser.add_argument(
        "--run-log-file",
        default=None,
        help=(
            "JSONL run log path. Defaults to the results path with .jsonl suffix. "
            "Logs problem starts, finishes, and aborting exceptions."
        ),
    )
    parser.add_argument(
        "--discard-prover-calls",
        action="store_true",
        help="Discard prover call details from result JSON to keep files smaller.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    return asyncio.run(run_async(args))


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
