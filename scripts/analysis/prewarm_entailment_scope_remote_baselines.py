import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


DEFAULT_EXTRACTION_FILE = PROJECT_ROOT / "scripts" / "analysis" / "random_entailment_verb_prover_scope_ids_20260505.json"


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


def item_key(problem: Any) -> str:
    return f"{problem.dataset}:{problem.split}:{problem.id}"


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


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") in {"done", "cached"}:
                completed.add(record["key"])
    return completed


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


async def run_async(args: argparse.Namespace) -> int:
    from kbprojection.async_runtime import AsyncRunLimits, create_async_run_context
    from kbprojection.langpro import (
        _make_langpro_cache_key,
        get_langpro_cache_backend,
        langpro_api_call,
    )
    from kbprojection.settings import DEFAULT_LANGPRO_ENDPOINT

    load_dotenv_if_present()
    checkpoint_path = Path(args.checkpoint_file).resolve()
    problems = load_extraction_problems(Path(args.extraction_file).resolve(), args.datasets)
    if args.limit is not None:
        problems = problems[: args.limit]

    completed = load_completed(checkpoint_path)
    cache_backend = get_langpro_cache_backend()

    def has_cached_baseline(problem: Any) -> bool:
        cache_key = _make_langpro_cache_key(
            list(problem.premises or []),
            problem.hypothesis,
            "easyccg",
            200,
            [],
            "all",
            True,
            True,
        )
        return cache_backend.get(cache_key) is not None

    jobs = []
    skipped_cached = 0
    for problem in problems:
        key = item_key(problem)
        if key in completed:
            continue
        if has_cached_baseline(problem):
            append_jsonl(checkpoint_path, {"status": "cached", "key": key})
            skipped_cached += 1
            continue
        jobs.append(problem)
    print(f"Skipping {skipped_cached} already cached baselines; prewarming {len(jobs)} misses.")
    context = create_async_run_context(
        AsyncRunLimits(
            llm_concurrency=1,
            langpro_concurrency=args.concurrency,
            local_langpro_concurrency=1,
        )
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(problem: Any) -> None:
        key = item_key(problem)
        async with semaphore:
            result = await langpro_api_call(
                problem.premises,
                problem.hypothesis,
                endpoint=DEFAULT_LANGPRO_ENDPOINT,
                kb=None,
                report=False,
                timeout_seconds=args.timeout_seconds,
                context=context,
            )
        append_jsonl(
            checkpoint_path,
            {
                "status": "done",
                "key": key,
                "label": result.label.value,
                "error": result.error,
            },
        )

    progress = tqdm(total=len(jobs), desc="remote-baseline-prewarm", unit="problem")
    tasks = [asyncio.create_task(worker(problem)) for problem in jobs]
    try:
        for completed_task in asyncio.as_completed(tasks):
            await completed_task
            progress.update(1)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        progress.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prewarm no-KB LangPro cache using the remote endpoint.")
    parser.add_argument("--extraction-file", default=str(DEFAULT_EXTRACTION_FILE))
    parser.add_argument("--datasets", nargs="+", default=["snli", "sick"], choices=["snli", "sick"])
    parser.add_argument("--checkpoint-file", default=str(PROJECT_ROOT / "scripts" / "analysis" / "remote_baseline_prewarm_entailment_scope.jsonl"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
