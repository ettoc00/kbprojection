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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


DEFAULT_SPLITS = ("dev",)


def load_dotenv_if_present(dotenv_path: Optional[Path] = None) -> None:
    dotenv_path = dotenv_path or PROJECT_ROOT / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
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
    raise ValueError("Could not infer provider. Set a provider API key or pass --provider.")


def problem_key(problem: Any) -> str:
    return f"{problem.split}:{problem.id}"


def load_existing_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_run_log(path: Optional[Path], event: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def serializable_result_payload(result: Any, discard_prover_calls: bool = False) -> Dict[str, Any]:
    exclude = {"prover_calls"} if discard_prover_calls else None
    payload = json.loads(result.model_dump_json(exclude=exclude, fallback=str))
    if discard_prover_calls:
        payload["prover_calls"] = None
    return payload


def build_default_results_path(results_dir: Path, provider: str, model: str, prompt_style: str) -> Path:
    return results_dir / (
        f"snli__{sanitize_filename_part(prompt_style)}__{sanitize_filename_part(provider)}__"
        f"{sanitize_filename_part(model)}.json"
    )


def normalize_label_filters(raw_labels: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not raw_labels:
        return None
    return {label.strip().lower() for label in raw_labels if label.strip()}


def load_snli_problems(splits: Sequence[str], label_filter: Optional[set[str]]) -> List[Any]:
    from kbprojection.loaders.snli import SNLILoader

    loader = SNLILoader()
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
            "dataset": "snli",
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
        return build_initial_state(results_path, model, provider, prompt_style, splits, problem_keys, args)
    meta = state.get("meta", {})
    expected = {
        "dataset": "snli",
        "model": model,
        "provider": provider,
        "prompt_style": prompt_style,
        "test_mode": args.test_mode,
        "run_ablation": args.run_ablation,
        "post_process": not args.no_post_process,
    }
    mismatches = [f"{key}={meta.get(key)!r} expected {value!r}" for key, value in expected.items() if meta.get(key) != value]
    if mismatches:
        raise ValueError(f"Existing output file is incompatible: {results_path}. " + "; ".join(mismatches))
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
    correct_counts = {"no_kb": 0, "raw_kb": 0, "normalised_kb": 0}
    successful_calls = {"no_kb": 0, "raw_kb": 0, "normalised_kb": 0}
    for item in completed_items:
        status = str(item.get("final_status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        gold = (item.get("problem") or {}).get("gold_label")
        if item.get("status_no_kb") == "success":
            successful_calls["no_kb"] += 1
            correct_counts["no_kb"] += int(item.get("pred_no_kb") == gold)
        if item.get("status_with_raw_kb") == "success":
            successful_calls["raw_kb"] += 1
            correct_counts["raw_kb"] += int(item.get("pred_with_raw_kb") == gold)
        if item.get("status_with_kb") == "success":
            successful_calls["normalised_kb"] += 1
            correct_counts["normalised_kb"] += int(item.get("pred_with_kb") == gold)
    return {
        "total": len(all_problem_keys),
        "completed": len(completed_items),
        "remaining": len(all_problem_keys) - len(completed_items),
        "by_status": dict(sorted(by_status.items())),
        "correct_counts": correct_counts,
        "successful_calls": successful_calls,
        "accuracy_on_successful_calls": {
            key: (correct_counts[key] / successful_calls[key] if successful_calls[key] else 0.0)
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
    problems = load_snli_problems(splits, normalize_label_filters(args.labels))
    if args.limit is not None:
        problems = problems[: args.limit]
    all_problem_keys = [problem_key(problem) for problem in problems]
    results_path = (
        Path(args.results_file).resolve()
        if args.results_file
        else build_default_results_path(get_default_results_dir(), provider, args.model, args.prompt_style).resolve()
    )
    run_log_path = Path(args.run_log_file).resolve() if args.run_log_file else results_path.with_suffix(".jsonl")
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
    semaphore = asyncio.Semaphore(args.job_concurrency)

    async def process_job(problem: Any) -> Tuple[str, Dict[str, Any]]:
        key = problem_key(problem)
        append_run_log(run_log_path, {"event": "start_problem", "key": key})
        try:
            async with semaphore:
                if args.problem_timeout_seconds and args.problem_timeout_seconds > 0:
                    result = await asyncio.wait_for(
                        process_single_problem(problem, config=config, context=context),
                        timeout=args.problem_timeout_seconds,
                    )
                else:
                    result = await process_single_problem(problem, config=config, context=context)
            payload = serializable_result_payload(result, discard_prover_calls=args.discard_prover_calls)
            payload["model"] = args.model
            payload["provider"] = provider
            payload["prompt_style"] = args.prompt_style
            append_run_log(run_log_path, {"event": "finish_problem", "key": key, "final_status": payload.get("final_status")})
            return key, payload
        except Exception as exc:
            append_run_log(
                run_log_path,
                {
                    "event": "problem_exception",
                    "key": key,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    print(
        f"Running SNLI {','.join(splits)}: {len(jobs)} remaining / {len(problems)} total "
        f"model={args.model} provider={provider} prompt={args.prompt_style}"
    )
    print(f"Output: {results_path}")
    tasks = [asyncio.create_task(process_job(problem)) for problem in jobs]
    completed_since_save = 0
    try:
        for task in asyncio.as_completed(tasks):
            key, payload = await task
            state["items_by_key"][key] = payload
            completed_since_save += 1
            print(f"{len(state['items_by_key'])}/{len(all_problem_keys)} {key} {payload.get('final_status')}")
            if completed_since_save >= args.save_every:
                state["summary"] = build_summary(state, all_problem_keys)
                save_state(results_path, state)
                completed_since_save = 0
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        state["summary"] = build_summary(state, all_problem_keys)
        save_state(results_path, state)
        raise
    state["summary"] = build_summary(state, all_problem_keys)
    save_state(results_path, state)
    print(json.dumps(state["summary"], indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one model/prompt over SNLI with resumable JSON output.")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-r", "--results-file", default=None)
    parser.add_argument("--provider", default=None, choices=["openai", "openrouter", "claude", "gemini"])
    parser.add_argument("--prompt-style", default="icl")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), choices=["train", "dev", "test"])
    parser.add_argument("--labels", nargs="+", default=None, choices=["entailment", "contradiction", "neutral"])
    parser.add_argument("--test-mode", default="both", choices=["no_kb", "raw_kb", "normalised", "filtered", "both", "full"])
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--no-post-process", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--llm-concurrency", type=int, default=2)
    parser.add_argument("--langpro-concurrency", type=int, default=4)
    parser.add_argument("--local-langpro-concurrency", type=int, default=2)
    parser.add_argument("--job-concurrency", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--problem-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--run-log-file", default=None)
    parser.add_argument("--discard-prover-calls", action="store_true")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run_async(parse_args()))


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
