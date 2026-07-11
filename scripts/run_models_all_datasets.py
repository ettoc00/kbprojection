import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_MODELS = [
    "google/gemini-3-flash-preview",
    "qwen/qwen3.6-plus",
    "google/gemma-4-31b-it",
]
DEFAULT_DATASETS = ["sick", "snli"]
DEFAULT_SPLITS = {
    "sick": ("train", "dev", "test"),
    "snli": ("train", "dev", "test"),
}


def load_dotenv_if_present(dotenv_path: Optional[Path] = None) -> None:
    if dotenv_path is None:
        dotenv_path = PROJECT_ROOT / ".env"

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
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
    raise ValueError("Could not infer provider. Set API keys or pass --provider explicitly.")


def problem_key(problem: Any) -> str:
    return f"{problem.split}:{problem.id}"


def build_results_path(
    results_dir: Path,
    dataset: str,
    provider: str,
    model: str,
    prompt_style: str,
    test_mode: str,
) -> Path:
    safe_dataset = sanitize_filename_part(dataset)
    safe_provider = sanitize_filename_part(provider)
    safe_model = sanitize_filename_part(model)
    safe_prompt = sanitize_filename_part(prompt_style)
    safe_mode = sanitize_filename_part(test_mode)
    return results_dir / f"{safe_dataset}_full__{safe_prompt}__{safe_mode}__{safe_provider}__{safe_model}.json"


def load_existing_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_state(
    state: Dict[str, Any],
    *,
    dataset: str,
    model: str,
    provider: str,
    prompt_style: str,
    test_mode: str,
    splits: Sequence[str],
    all_keys: List[str],
    results_path: Path,
) -> Dict[str, Any]:
    expected = {
        "dataset": dataset,
        "model": model,
        "provider": provider,
        "prompt_style": prompt_style,
        "test_mode": test_mode,
    }
    if not state:
        return {
            "meta": {
                "results_file": str(results_path),
                **expected,
                "splits": list(splits),
                "problem_keys": all_keys,
            },
            "items_by_key": {},
            "summary": {},
        }

    meta = state.get("meta", {})
    mismatches = [
        f"{k}={meta.get(k)!r} expected {v!r}"
        for k, v in expected.items()
        if meta.get(k) != v
    ]
    if mismatches:
        raise ValueError(
            f"Existing output file is incompatible: {results_path}. " + "; ".join(mismatches)
        )

    state.setdefault("items_by_key", {})
    state.setdefault("summary", {})
    state["meta"]["results_file"] = str(results_path)
    state["meta"]["splits"] = list(splits)
    state["meta"]["problem_keys"] = all_keys
    return state


def build_summary(state: Dict[str, Any], all_keys: List[str]) -> Dict[str, Any]:
    items = state.get("items_by_key", {})
    completed = [items[k] for k in all_keys if k in items]
    by_status: Dict[str, int] = {}
    for item in completed:
        status = str(item.get("final_status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "total": len(all_keys),
        "completed": len(completed),
        "remaining": len(all_keys) - len(completed),
        "by_status": by_status,
    }


def load_problems(dataset: str, splits: Sequence[str]) -> List[Any]:
    if dataset == "sick":
        from kbprojection.loaders.sick import SICKLoader

        loader = SICKLoader()
    elif dataset == "snli":
        from kbprojection.loaders.snli import SNLILoader

        loader = SNLILoader()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    loader.load(splits=list(splits))
    problems: List[Any] = []
    for split in splits:
        problems.extend(loader.iter_problems(split=split))
    return problems


async def run_once(
    *,
    dataset: str,
    splits: Sequence[str],
    model: str,
    provider: str,
    prompt_style: str,
    test_mode: str,
    results_path: Path,
    limit: Optional[int],
    save_every: int,
    llm_concurrency: int,
    langpro_concurrency: int,
    local_langpro_concurrency: int,
    job_concurrency: int,
    discard_prover_calls: bool,
) -> None:
    from kbprojection.async_runtime import AsyncRunLimits, create_async_run_context
    from kbprojection.models import ProblemConfig, TestMode
    from kbprojection.orchestration import process_single_problem

    all_problems = load_problems(dataset, splits)
    if limit is not None:
        all_problems = all_problems[:limit]
    all_keys = [problem_key(p) for p in all_problems]

    state = ensure_state(
        load_existing_state(results_path),
        dataset=dataset,
        model=model,
        provider=provider,
        prompt_style=prompt_style,
        test_mode=test_mode,
        splits=splits,
        all_keys=all_keys,
        results_path=results_path,
    )

    completed_keys = set(state.get("items_by_key", {}).keys())
    jobs = [p for p in all_problems if problem_key(p) not in completed_keys]
    if not jobs:
        state["summary"] = build_summary(state, all_keys)
        save_state(results_path, state)
        print(f"\n[{dataset}] {model}: already complete at {results_path}")
        return

    config = ProblemConfig(
        llm_provider=provider,
        model=model,
        prompt_style=prompt_style,
        test_mode=TestMode(test_mode),
        post_process=True,
        run_ablation=False,
        verbose=False,
    )
    context = create_async_run_context(
        AsyncRunLimits(
            llm_concurrency=llm_concurrency,
            langpro_concurrency=langpro_concurrency,
            local_langpro_concurrency=local_langpro_concurrency,
        )
    )
    sem = asyncio.Semaphore(job_concurrency)

    async def process_job(problem: Any) -> tuple[str, Dict[str, Any]]:
        key = problem_key(problem)
        async with sem:
            result = await process_single_problem(problem, config=config, context=context)
        payload = json.loads(
            result.model_dump_json(
                exclude={"prover_calls"} if discard_prover_calls else None,
                fallback=str,
            )
        )
        payload["model"] = model
        payload["provider"] = provider
        payload["prompt_style"] = prompt_style
        return key, payload

    print(
        f"\n=== {dataset.upper()} | model={model} | provider={provider} | "
        f"mode={test_mode} | remaining={len(jobs)}/{len(all_problems)} ==="
    )
    print(f"Output: {results_path}")

    tasks = [asyncio.create_task(process_job(problem)) for problem in jobs]
    progress = tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"{dataset}:{sanitize_filename_part(model)}",
        unit="problem",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    completed_since_save = 0
    try:
        for fut in progress:
            key, payload = await fut
            progress.set_postfix_str(key)
            state["items_by_key"][key] = payload
            completed_since_save += 1
            if completed_since_save >= save_every:
                state["summary"] = build_summary(state, all_keys)
                save_state(results_path, state)
                completed_since_save = 0
    finally:
        progress.close()

    state["summary"] = build_summary(state, all_keys)
    save_state(results_path, state)
    print(json.dumps(state["summary"], indent=2))


async def run_all(args: argparse.Namespace) -> None:
    load_dotenv_if_present()
    from kbprojection.settings import get_default_results_dir

    if not os.environ.get("KBPROJECTION_CACHE_DIR"):
        default_cache_dir = (PROJECT_ROOT / ".cache").resolve()
        default_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["KBPROJECTION_CACHE_DIR"] = str(default_cache_dir)

    results_dir = Path(args.results_dir).resolve() if args.results_dir else get_default_results_dir()
    models = args.models or DEFAULT_MODELS
    datasets = args.datasets or DEFAULT_DATASETS

    for model in models:
        provider = infer_provider(model, args.provider)
        for dataset in datasets:
            splits = DEFAULT_SPLITS[dataset]
            results_path = build_results_path(
                results_dir,
                dataset,
                provider,
                model,
                args.prompt_style,
                args.test_mode,
            ).resolve()
            await run_once(
                dataset=dataset,
                splits=splits,
                model=model,
                provider=provider,
                prompt_style=args.prompt_style,
                test_mode=args.test_mode,
                results_path=results_path,
                limit=args.limit,
                save_every=args.save_every,
                llm_concurrency=args.llm_concurrency,
                langpro_concurrency=args.langpro_concurrency,
                local_langpro_concurrency=args.local_langpro_concurrency,
                job_concurrency=args.job_concurrency,
                discard_prover_calls=args.discard_prover_calls,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple models across SICK/SNLI in one resumable command."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model identifiers to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=DEFAULT_DATASETS,
        help="Datasets to run in sequence.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "openrouter", "claude", "gemini"],
        default=None,
        help="Optional provider override for all models.",
    )
    parser.add_argument("--prompt-style", default="icl")
    parser.add_argument(
        "--test-mode",
        default="raw_kb",
        choices=["no_kb", "raw_kb", "normalised", "filtered", "both", "full"],
        help="Defaults to raw_kb as requested.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset first-N limit.")
    parser.add_argument("--results-dir", default=None, help="Optional output directory.")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--llm-concurrency", type=int, default=2)
    parser.add_argument("--langpro-concurrency", type=int, default=4)
    parser.add_argument("--local-langpro-concurrency", type=int, default=2)
    parser.add_argument("--job-concurrency", type=int, default=8)
    parser.add_argument("--discard-prover-calls", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_all(parse_args()))
