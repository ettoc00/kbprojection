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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.async_runtime import AsyncRunLimits, create_async_run_context
from kbprojection.loaders.sick import SICKLoader
from kbprojection.models import ProblemConfig, TestMode
from kbprojection.orchestration import process_single_problem
from kbprojection.settings import get_default_results_dir


DEFAULT_FOCUSED_SET = get_default_results_dir() / "icl_focused_validation_set.json"


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


def problem_key(problem: Any) -> str:
    return f"{problem.split}:{problem.id}"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_log(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **payload}, default=str) + "\n")


def serializable_result_payload(result: Any, discard_prover_calls: bool) -> Dict[str, Any]:
    exclude = {"prover_calls"} if discard_prover_calls else None
    payload = json.loads(result.model_dump_json(exclude=exclude, fallback=str))
    if discard_prover_calls:
        payload["prover_calls"] = None
    return payload


def build_summary(state: Dict[str, Any], all_problem_keys: Sequence[str]) -> Dict[str, Any]:
    items = state.get("items_by_key", {})
    completed = [items[key] for key in all_problem_keys if key in items]
    by_status: Dict[str, int] = {}
    for item in completed:
        status = str(item.get("final_status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "total": len(all_problem_keys),
        "completed": len(completed),
        "remaining": len(all_problem_keys) - len(completed),
        "by_status": dict(sorted(by_status.items())),
    }


def default_output_path(provider: str, model: str) -> Path:
    return get_default_results_dir() / (
        f"icl_focused_validation__openrouter_safe__{sanitize_filename_part(provider)}__"
        f"{sanitize_filename_part(model)}.json"
    )


def load_focused_problems(focused_set_path: Path) -> Tuple[List[Any], Dict[str, Any]]:
    focused = load_json(focused_set_path)
    keys = sorted(focused["sick_problems"], key=lambda value: (value.split(":", 1)[0], int(value.split(":", 1)[1])))
    loader = SICKLoader()
    splits = sorted({key.split(":", 1)[0] for key in keys})
    loader.load(splits=splits)
    problems = []
    for key in keys:
        split, problem_id = key.split(":", 1)
        problems.append(loader.get_problem(problem_id, split=split))
    return problems, focused


def initial_state(
    output_path: Path,
    focused_set_path: Path,
    provider: str,
    model: str,
    prompt_style: str,
    problem_keys: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "meta": {
            "results_file": str(output_path),
            "focused_set_file": str(focused_set_path),
            "dataset": "sick",
            "problem_keys": list(problem_keys),
            "provider": provider,
            "model": model,
            "prompt_style": prompt_style,
            "test_mode": args.test_mode,
            "post_process": not args.no_post_process,
            "run_ablation": args.run_ablation,
        },
        "items_by_key": {},
        "summary": {},
    }


def ensure_state(
    output_path: Path,
    focused_set_path: Path,
    provider: str,
    model: str,
    prompt_style: str,
    problem_keys: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not output_path.exists():
        return initial_state(output_path, focused_set_path, provider, model, prompt_style, problem_keys, args)
    state = load_json(output_path)
    meta = state.get("meta", {})
    expected = {
        "provider": provider,
        "model": model,
        "prompt_style": prompt_style,
        "test_mode": args.test_mode,
        "post_process": not args.no_post_process,
        "run_ablation": args.run_ablation,
    }
    mismatches = [f"{key}={meta.get(key)!r} expected {value!r}" for key, value in expected.items() if meta.get(key) != value]
    if mismatches:
        raise ValueError(f"Existing output file is incompatible: {output_path}. " + "; ".join(mismatches))
    state.setdefault("items_by_key", {})
    state.setdefault("summary", {})
    state["meta"]["problem_keys"] = list(problem_keys)
    state["meta"]["focused_set_file"] = str(focused_set_path)
    return state


async def run_async(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    focused_set_path = args.focused_set.resolve()
    problems, _focused = load_focused_problems(focused_set_path)
    if args.limit:
        problems = problems[: args.limit]
    problem_keys = [problem_key(problem) for problem in problems]

    output_path = (args.output.resolve() if args.output else default_output_path(args.provider, args.model).resolve())
    log_path = output_path.with_suffix(".jsonl")
    state = ensure_state(output_path, focused_set_path, args.provider, args.model, args.prompt_style, problem_keys, args)
    save_json(output_path, state)

    config = ProblemConfig(
        llm_provider=args.provider,
        model=args.model,
        prompt_style=args.prompt_style,
        test_mode=TestMode(args.test_mode),
        post_process=not args.no_post_process,
        run_ablation=args.run_ablation,
        verbose=args.verbose,
    )
    context = create_async_run_context(
        AsyncRunLimits(
            llm_concurrency=args.llm_concurrency,
            langpro_concurrency=args.langpro_concurrency,
            local_langpro_concurrency=args.local_langpro_concurrency,
        )
    )
    completed = set(state["items_by_key"])
    jobs = [problem for problem in problems if problem_key(problem) not in completed]
    semaphore = asyncio.Semaphore(args.job_concurrency)

    async def run_one(problem: Any) -> Tuple[str, Dict[str, Any]]:
        key = problem_key(problem)
        append_log(log_path, {"event": "start_problem", "key": key})
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
            payload["provider"] = args.provider
            payload["prompt_style"] = args.prompt_style
            append_log(log_path, {"event": "finish_problem", "key": key, "final_status": payload.get("final_status")})
            return key, payload
        except Exception as exc:
            append_log(
                log_path,
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
        f"Running focused validation: {len(jobs)} remaining / {len(problems)} total "
        f"model={args.model} provider={args.provider} prompt={args.prompt_style}"
    )
    print(f"Output: {output_path}")
    tasks = [asyncio.create_task(run_one(problem)) for problem in jobs]
    completed_since_save = 0
    try:
        for task in asyncio.as_completed(tasks):
            key, payload = await task
            state["items_by_key"][key] = payload
            completed_since_save += 1
            print(f"{len(state['items_by_key'])}/{len(problem_keys)} {key} {payload.get('final_status')}")
            if completed_since_save >= args.save_every:
                state["summary"] = build_summary(state, problem_keys)
                save_json(output_path, state)
                completed_since_save = 0
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        state["summary"] = build_summary(state, problem_keys)
        save_json(output_path, state)
        raise

    state["summary"] = build_summary(state, problem_keys)
    save_json(output_path, state)
    print(json.dumps(state["summary"], indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revised ICL prompt on the deterministic focused SICK validation set.")
    parser.add_argument("--focused-set", type=Path, default=DEFAULT_FOCUSED_SET)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="openrouter", choices=["openai", "openrouter", "claude", "gemini"])
    parser.add_argument("--prompt-style", default="icl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--test-mode", default="both", choices=["no_kb", "raw_kb", "normalised", "filtered", "both", "full"])
    parser.add_argument("--no-post-process", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--llm-concurrency", type=int, default=2)
    parser.add_argument("--langpro-concurrency", type=int, default=4)
    parser.add_argument("--local-langpro-concurrency", type=int, default=2)
    parser.add_argument("--job-concurrency", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--problem-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--discard-prover-calls", action="store_true")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run_async(parse_args()))


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
