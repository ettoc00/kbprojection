import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


DEFAULT_EXTRACTION_FILE = PROJECT_ROOT / "scripts" / "analysis" / "random_entailment_verb_prover_scope_ids_20260505.json"
DEFAULT_MODELS = (
    "google/gemini-3-flash-preview",
    "qwen/qwen3.6-max-preview",
    "anthropic/claude-4.7-opus",
    "openai/gpt-5.4",
    "google/gemma-4-31b-it",
)


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


def load_existing_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def append_run_log(path: Optional[Path], event: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def item_key(problem: Any) -> str:
    return f"{problem.dataset}:{problem.split}:{problem.id}"


def build_results_path(results_dir: Path, provider: str, model: str) -> Path:
    return results_dir / f"entailment_scope_compare__{provider}__{sanitize_filename_part(model)}.json"


def load_extraction_problems(path: Path, datasets: Sequence[str]) -> List[Any]:
    from kbprojection.loaders.sick import SICKLoader
    from kbprojection.loaders.snli import SNLILoader

    payload = json.loads(path.read_text(encoding="utf-8"))
    loaders = {
        "snli": SNLILoader(),
        "sick": SICKLoader(),
    }
    splits_by_dataset: Dict[str, set[str]] = {dataset: set() for dataset in datasets}
    for dataset in datasets:
        for item in payload.get(dataset, []):
            splits_by_dataset[dataset].add(item["split"])
    for dataset, splits in splits_by_dataset.items():
        if splits:
            loaders[dataset].load(splits=sorted(splits))

    problems = []
    for dataset in datasets:
        for item in payload.get(dataset, []):
            problems.append(loaders[dataset].get_problem(item["id"], split=item["split"]))
    return problems


def build_initial_state(
    results_path: Path,
    extraction_file: Path,
    model: str,
    provider: str,
    prompt_style: str,
    problem_keys: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "meta": {
            "results_file": str(results_path),
            "benchmark_kind": "entailment_scope",
            "extraction_file": str(extraction_file),
            "model": model,
            "provider": provider,
            "prompt_styles": [prompt_style],
            "problem_ids": problem_keys,
            "test_mode": args.test_mode,
            "post_process": not args.no_post_process,
            "run_ablation": args.run_ablation,
        },
        "runs": {
            prompt_style: {
                "items_by_id": {},
            }
        },
        "summary": {},
    }


def ensure_state(
    state: Dict[str, Any],
    results_path: Path,
    extraction_file: Path,
    model: str,
    provider: str,
    prompt_style: str,
    problem_keys: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not state:
        return build_initial_state(results_path, extraction_file, model, provider, prompt_style, problem_keys, args)

    meta = state.get("meta", {})
    expected = {
        "benchmark_kind": "entailment_scope",
        "model": model,
        "provider": provider,
        "post_process": not args.no_post_process,
        "run_ablation": args.run_ablation,
    }
    mismatches = [
        f"{key}={meta.get(key)!r} expected {value!r}"
        for key, value in expected.items()
        if meta.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"Existing output file is incompatible: {results_path}. " + "; ".join(mismatches))

    state.setdefault("runs", {}).setdefault(prompt_style, {}).setdefault("items_by_id", {})
    state.setdefault("summary", {})
    state["meta"]["results_file"] = str(results_path)
    state["meta"]["extraction_file"] = str(extraction_file)
    state["meta"]["prompt_styles"] = [prompt_style]
    state["meta"]["problem_ids"] = problem_keys
    test_mode_history = state["meta"].setdefault("test_mode_history", [])
    previous_test_mode = meta.get("test_mode")
    if previous_test_mode and previous_test_mode not in test_mode_history:
        test_mode_history.append(previous_test_mode)
    if args.test_mode not in test_mode_history:
        test_mode_history.append(args.test_mode)
    state["meta"]["test_mode"] = args.test_mode
    return state


def build_summary(state: Dict[str, Any], prompt_style: str, problem_keys: Sequence[str]) -> Dict[str, Any]:
    items = state["runs"][prompt_style]["items_by_id"]
    completed = [items[key] for key in problem_keys if key in items]
    by_status: Dict[str, int] = {}
    for item in completed:
        status = str(item.get("final_status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "total": len(problem_keys),
        "completed": len(completed),
        "remaining": len(problem_keys) - len(completed),
        "by_status": dict(sorted(by_status.items())),
    }


def serializable_result_payload(result: Any, discard_prover_calls: bool = False) -> Dict[str, Any]:
    exclude = {"prover_calls"} if discard_prover_calls else None
    payload = json.loads(result.model_dump_json(exclude=exclude, fallback=str))
    if discard_prover_calls:
        payload["prover_calls"] = None
    return payload


async def run_model(
    args: argparse.Namespace,
    model: str,
    problems: Sequence[Any],
    context: Any,
    problem_locks: Dict[str, asyncio.Lock],
) -> Path:
    from kbprojection.models import ProblemConfig, TestMode
    from kbprojection.orchestration import process_single_problem
    from kbprojection.settings import get_default_results_dir

    provider = args.provider
    results_dir = Path(args.results_dir).resolve() if args.results_dir else get_default_results_dir()
    results_path = build_results_path(results_dir, provider, model).resolve()
    run_log_path = results_path.with_suffix(".jsonl") if args.run_logs else None
    prompt_style = args.prompt_style
    problem_keys = [item_key(problem) for problem in problems]
    state = ensure_state(
        load_existing_state(results_path),
        results_path,
        Path(args.extraction_file).resolve(),
        model,
        provider,
        prompt_style,
        problem_keys,
        args,
    )
    save_state(results_path, state)

    config = ProblemConfig(
        llm_provider=provider,
        model=model,
        prompt_style=prompt_style,
        test_mode=TestMode(args.test_mode),
        run_ablation=args.run_ablation,
        post_process=not args.no_post_process,
        verbose=args.verbose,
    )

    completed_keys = set(state["runs"][prompt_style]["items_by_id"])
    jobs = [problem for problem in problems if item_key(problem) not in completed_keys]
    pending_jobs = deque(jobs)
    pending_jobs_lock = asyncio.Lock()
    timeout_counts: Counter[str] = Counter()
    completed_since_save = 0

    async def claim_job() -> Optional[Any]:
        async with pending_jobs_lock:
            if not pending_jobs:
                return None
            rotations = len(pending_jobs)
            for _ in range(rotations):
                problem = pending_jobs[0]
                if not problem_locks[item_key(problem)].locked():
                    return pending_jobs.popleft()
                pending_jobs.rotate(-1)
            return None

    async def process_job(problem: Any) -> Tuple[str, Dict[str, Any]]:
        key = item_key(problem)
        append_run_log(run_log_path, {"event": "start_problem", "model": model, "key": key})
        try:
            async with problem_locks[key]:
                if args.problem_timeout_seconds and args.problem_timeout_seconds > 0:
                    result = await asyncio.wait_for(
                        process_single_problem(problem, config=config, context=context),
                        timeout=args.problem_timeout_seconds,
                    )
                else:
                    result = await process_single_problem(problem, config=config, context=context)
            payload = serializable_result_payload(result, discard_prover_calls=args.discard_prover_calls)
            payload["model"] = model
            payload["provider"] = provider
            payload["prompt_style"] = prompt_style
            append_run_log(run_log_path, {"event": "finish_problem", "model": model, "key": key, "final_status": payload.get("final_status")})
            return key, payload
        except asyncio.TimeoutError:
            from kbprojection.langpro import penalize_hybrid_local_backend_for_timeout

            timeout_counts[key] += 1
            penalize_hybrid_local_backend_for_timeout()
            append_run_log(
                run_log_path,
                {
                    "event": "problem_timeout_requeued",
                    "model": model,
                    "key": key,
                    "attempt": timeout_counts[key],
                    "timeout_seconds": args.problem_timeout_seconds,
                },
            )
            raise
        except Exception as exc:
            append_run_log(
                run_log_path,
                {
                    "event": "problem_exception",
                    "model": model,
                    "key": key,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    async def worker() -> None:
        nonlocal completed_since_save
        while True:
            problem = await claim_job()
            if problem is None:
                async with pending_jobs_lock:
                    if not pending_jobs:
                        return
                await asyncio.sleep(0.25)
                continue
            try:
                key, payload = await process_job(problem)
            except asyncio.TimeoutError:
                async with pending_jobs_lock:
                    pending_jobs.append(problem)
                await asyncio.sleep(0.25)
                continue
            state["runs"][prompt_style]["items_by_id"][key] = payload
            completed_since_save += 1
            progress.update(1)
            progress.set_postfix_str(key[:40])
            if completed_since_save >= args.save_every:
                state["summary"] = build_summary(state, prompt_style, problem_keys)
                save_state(results_path, state)
                completed_since_save = 0

    progress = tqdm(total=len(jobs), desc=sanitize_filename_part(model), unit="problem", leave=False)
    tasks = [asyncio.create_task(worker()) for _ in range(args.job_concurrency)]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        state["summary"] = build_summary(state, prompt_style, problem_keys)
        save_state(results_path, state)
        raise
    finally:
        progress.close()

    state["summary"] = build_summary(state, prompt_style, problem_keys)
    save_state(results_path, state)
    return results_path


async def run_async(args: argparse.Namespace) -> int:
    from kbprojection.async_runtime import AsyncRunLimits, create_async_run_context

    load_dotenv_if_present()
    extraction_file = Path(args.extraction_file).resolve()
    problems = load_extraction_problems(extraction_file, args.datasets)
    if args.limit is not None:
        problems = problems[: args.limit]

    context = create_async_run_context(
        AsyncRunLimits(
            llm_concurrency=args.llm_concurrency,
            langpro_concurrency=args.langpro_concurrency,
            local_langpro_concurrency=args.local_langpro_concurrency,
        )
    )
    problem_locks = {item_key(problem): asyncio.Lock() for problem in problems}
    model_semaphore = asyncio.Semaphore(args.model_concurrency)

    async def run_model_limited(model: str) -> Path:
        async with model_semaphore:
            return await run_model(args, model, problems, context, problem_locks)

    tasks = [asyncio.create_task(run_model_limited(model)) for model in args.models]
    written = []
    for completed in asyncio.as_completed(tasks):
        written.append(await completed)

    print("Wrote result files:")
    for path in written:
        print(f"- {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenRouter models on the entailment-only prover-scope extraction.")
    parser.add_argument("--extraction-file", default=str(DEFAULT_EXTRACTION_FILE))
    parser.add_argument("--datasets", nargs="+", default=["snli", "sick"], choices=["snli", "sick"])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--provider", default="openrouter", choices=["openrouter"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--prompt-style", default="icl")
    parser.add_argument("--test-mode", default="both", choices=["no_kb", "raw_kb", "normalised", "filtered", "both", "full"])
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--no-post-process", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model-concurrency", type=int, default=2)
    parser.add_argument("--llm-concurrency", type=int, default=4)
    parser.add_argument("--job-concurrency", type=int, default=8)
    parser.add_argument("--langpro-concurrency", type=int, default=4)
    parser.add_argument("--local-langpro-concurrency", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--problem-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--discard-prover-calls", action="store_true")
    parser.add_argument("--run-logs", action="store_true")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
