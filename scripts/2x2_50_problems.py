import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


PROMPT_STYLES = ("legacy_icl", "icl")
BENCHMARK_IDS = {
    "sick": {
        "train": [
            "1792", "2281", "2809", "3853", "4181", "4766", "4974", "5149",
            "5198", "5435", "5462", "5554", "5555", "6584", "8624",
        ],
        "dev": ["3586"],
        "test": ["1262", "1497", "3641", "3974", "4297", "4494", "4589", "4972", "5230"],
    },
    "snli": {
        "train": [
            "3629664676.jpg#4r1e", "vg len26r4e", "vg len66r1e", "3159569570.jpg#4r1e",
            "4294390957.jpg#3r1e", "1184967930.jpg#4r4e", "2521878609.jpg#4r1e",
            "303607405.jpg#4r3e", "436393371.jpg#3r1e", "vg len84r1e", "326456451.jpg#4r1e",
            "vg len120r5e", "2543017787.jpg#4r1e", "2647049174.jpg#4r1e", "2618866067.jpg#4r1e",
            "vg len47r3e", "vg len47r4e", "vg len46r5e", "3228793611.jpg#4r1e",
            "3134092148.jpg#3r2e", "2217728745.jpg#4r1e", "3479245321.jpg#4r1e",
            "2741990005.jpg#4r1e", "3614595423.jpg#4r1e", "207584893.jpg#3r1e",
        ],
    },
}


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


def load_existing_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def serializable_result_payload(result: Any, discard_prover_calls: bool = False) -> Dict[str, Any]:
    exclude = {"prover_calls"} if discard_prover_calls else None
    payload = json.loads(result.model_dump_json(exclude=exclude, fallback=str))
    if discard_prover_calls:
        payload["prover_calls"] = None
    return payload


def prompt_hash(prompt_style: str) -> str:
    from kbprojection.models import LLMKBResponse
    from kbprojection.prompts import get_prompt

    effective_prompt = {
        "prompt_style": prompt_style,
        "template": get_prompt(prompt_style),
    }
    if prompt_style.startswith("legacy_"):
        effective_prompt["legacy_response_schema"] = LLMKBResponse.model_json_schema()

    serialized = json.dumps(effective_prompt, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


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


def success_count(items_by_id: Dict[str, Dict[str, Any]], field: str, status_field: str, problem_ids: List[str]) -> int:
    count = 0
    for problem_id in problem_ids:
        item = items_by_id.get(problem_id)
        if not item:
            continue
        if item.get(status_field) == "success" and item.get(field) == item["problem"]["gold_label"]:
            count += 1
    return count


def already_correct_count(items_by_id: Dict[str, Dict[str, Any]], problem_ids: List[str]) -> int:
    count = 0
    for problem_id in problem_ids:
        item = items_by_id.get(problem_id)
        if not item:
            continue
        if item.get("final_status") in {"baseline_solved", "already_correct"}:
            count += 1
    return count


def build_summary(state: Dict[str, Any], problem_ids: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": len(problem_ids),
        "table": {},
    }

    for prompt_style in PROMPT_STYLES:
        items_by_id = state["runs"][prompt_style]["items_by_id"]
        already_correct = already_correct_count(items_by_id, problem_ids)
        raw = success_count(items_by_id, "pred_with_raw_kb", "status_with_raw_kb", problem_ids)
        normalised = success_count(items_by_id, "pred_with_kb", "status_with_kb", problem_ids)
        summary["table"][prompt_style] = {
            "already_correct": already_correct,
            "raw_success": raw,
            "raw_total_solved": already_correct + raw,
            "normalised_success": normalised,
            "normalised_total_solved": already_correct + normalised,
            "raw_rate": raw / len(problem_ids) if problem_ids else 0.0,
            "normalised_rate": normalised / len(problem_ids) if problem_ids else 0.0,
        }

    summary["markdown"] = (
        "| Prompt | Raw | Normalised |\n"
        "|---|---:|---:|\n"
        f"| legacy_icl | {summary['table']['legacy_icl']['raw_success']}/"
        f"{summary['table']['legacy_icl']['raw_total_solved']}/{len(problem_ids)} | "
        f"{summary['table']['legacy_icl']['normalised_success']}/"
        f"{summary['table']['legacy_icl']['normalised_total_solved']}/{len(problem_ids)} |\n"
        f"| icl | {summary['table']['icl']['raw_success']}/"
        f"{summary['table']['icl']['raw_total_solved']}/{len(problem_ids)} | "
        f"{summary['table']['icl']['normalised_success']}/"
        f"{summary['table']['icl']['normalised_total_solved']}/{len(problem_ids)} |"
    )
    return summary


def build_initial_state(
    results_path: Path,
    model: str,
    provider: str,
    problem_ids: List[str],
    prompt_hashes: Dict[str, str],
    prompt_cache_files: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "meta": {
            "results_file": str(results_path),
            "model": model,
            "provider": provider,
            "prompt_styles": list(PROMPT_STYLES),
            "prompt_hashes": prompt_hashes,
            "prompt_cache_files": prompt_cache_files,
            "problem_ids": problem_ids,
            "benchmark": BENCHMARK_IDS,
        },
        "runs": {
            prompt_style: {
                "items_by_id": {},
            }
            for prompt_style in PROMPT_STYLES
        },
    }


def ensure_compatible_state(
    state: Dict[str, Any],
    results_path: Path,
    model: str,
    provider: str,
    problem_ids: List[str],
    prompt_hashes: Dict[str, str],
    prompt_cache_files: Dict[str, str],
) -> Dict[str, Any]:
    if not state:
        return build_initial_state(
            results_path,
            model,
            provider,
            problem_ids,
            prompt_hashes,
            prompt_cache_files,
        )

    meta = state.get("meta", {})
    if meta.get("model") != model or meta.get("provider") != provider:
        raise ValueError(
            f"Existing output file is for model={meta.get('model')} provider={meta.get('provider')}, "
            f"but this run requested model={model} provider={provider}."
        )

    for prompt_style in PROMPT_STYLES:
        state.setdefault("runs", {}).setdefault(prompt_style, {}).setdefault("items_by_id", {})

    existing_prompt_hashes = meta.get("prompt_hashes") or {}
    for prompt_style, current_hash in prompt_hashes.items():
        existing_hash = existing_prompt_hashes.get(prompt_style)
        if existing_hash and existing_hash != current_hash:
            state["runs"][prompt_style]["items_by_id"] = {}

    state["meta"]["results_file"] = str(results_path)
    state["meta"]["prompt_hashes"] = prompt_hashes
    state["meta"]["prompt_cache_files"] = prompt_cache_files
    state["meta"]["problem_ids"] = problem_ids
    state["meta"]["benchmark"] = BENCHMARK_IDS
    return state


def build_prompt_cache_path(
    cache_dir: Path,
    provider: str,
    model: str,
    prompt_style: str,
    prompt_fingerprint: str,
) -> Path:
    safe_provider = sanitize_filename_part(provider)
    safe_model = sanitize_filename_part(model)
    safe_prompt = sanitize_filename_part(prompt_style)
    return cache_dir / f"2x2_50__{safe_prompt}__{safe_provider}__{safe_model}__{prompt_fingerprint}.json"


def build_default_results_path(results_dir: Path, provider: str, model: str) -> Path:
    safe_provider = sanitize_filename_part(provider)
    safe_model = sanitize_filename_part(model)
    return results_dir / f"2x2_50_compare__{safe_provider}__{safe_model}.json"


def build_initial_prompt_cache_state(
    cache_path: Path,
    model: str,
    provider: str,
    prompt_style: str,
    prompt_fingerprint: str,
    problem_ids: List[str],
) -> Dict[str, Any]:
    return {
        "meta": {
            "cache_file": str(cache_path),
            "cache_kind": "2x2_50_prompt_results",
            "model": model,
            "provider": provider,
            "prompt_style": prompt_style,
            "prompt_hash": prompt_fingerprint,
            "problem_ids": problem_ids,
            "benchmark": BENCHMARK_IDS,
        },
        "items_by_id": {},
    }


def ensure_compatible_prompt_cache_state(
    state: Dict[str, Any],
    cache_path: Path,
    model: str,
    provider: str,
    prompt_style: str,
    prompt_fingerprint: str,
    problem_ids: List[str],
) -> Dict[str, Any]:
    if not state:
        return build_initial_prompt_cache_state(
            cache_path,
            model,
            provider,
            prompt_style,
            prompt_fingerprint,
            problem_ids,
        )

    meta = state.get("meta", {})
    expected = {
        "model": model,
        "provider": provider,
        "prompt_style": prompt_style,
        "prompt_hash": prompt_fingerprint,
    }
    mismatches = [
        f"{key}={meta.get(key)!r} expected {value!r}"
        for key, value in expected.items()
        if meta.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Existing prompt cache file is incompatible: {cache_path}. "
            + "; ".join(mismatches)
        )

    state.setdefault("items_by_id", {})
    state["meta"]["cache_file"] = str(cache_path)
    state["meta"]["cache_kind"] = "2x2_50_prompt_results"
    state["meta"]["problem_ids"] = problem_ids
    state["meta"]["benchmark"] = BENCHMARK_IDS
    return state


def sync_prompt_cache_with_comparison_state(
    state: Dict[str, Any],
    prompt_cache_states: Dict[str, Dict[str, Any]],
    problem_ids: List[str],
) -> None:
    for prompt_style in PROMPT_STYLES:
        comparison_items = state["runs"][prompt_style]["items_by_id"]
        cache_items = prompt_cache_states[prompt_style]["items_by_id"]

        for problem_id in problem_ids:
            if problem_id in comparison_items and problem_id not in cache_items:
                cache_items[problem_id] = comparison_items[problem_id]

        for problem_id in problem_ids:
            if problem_id in cache_items and problem_id not in comparison_items:
                comparison_items[problem_id] = cache_items[problem_id]


def load_benchmark_problems() -> List[Any]:
    from kbprojection.loaders.sick import SICKLoader
    from kbprojection.loaders.snli import SNLILoader

    sick_loader = SICKLoader()
    snli_loader = SNLILoader()
    sick_loader.load(splits=["train", "dev", "test"])
    snli_loader.load(splits=["train"])

    problems: List[Any] = []

    for split, ids in BENCHMARK_IDS["sick"].items():
        for problem_id in ids:
            problems.append(sick_loader.get_problem(problem_id, split=split))

    for split, ids in BENCHMARK_IDS["snli"].items():
        for problem_id in ids:
            problems.append(snli_loader.get_problem(problem_id, split=split))

    return problems


async def _precompute_baselines(problems: List[Any], context: Any) -> Dict[str, Any]:
    from kbprojection.langpro import langpro_api_call

    async def compute(problem: Any) -> Tuple[str, Any]:
        result = await langpro_api_call(problem.premises, problem.hypothesis, report=False, context=context)
        return problem.id, result

    baselines: Dict[str, Any] = {}
    tasks = [asyncio.create_task(compute(problem)) for problem in problems]
    if not tasks:
        return baselines

    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="no_kb", unit="problem")
    for completed in progress:
        problem_id, result = await completed
        baselines[problem_id] = result
    return baselines


async def run_async(args: argparse.Namespace) -> int:
    from kbprojection.async_runtime import AsyncRunLimits, create_async_run_context
    from kbprojection.models import NLIProblem, ProblemConfig, TestMode
    from kbprojection.orchestration import process_single_problem
    from kbprojection.settings import get_default_results_dir

    load_dotenv_if_present()

    provider = infer_provider(args.model, args.provider)
    default_results_dir = get_default_results_dir()
    results_file_arg = args.results_file or args.output_file
    results_path = (
        Path(results_file_arg).resolve()
        if results_file_arg
        else build_default_results_path(default_results_dir, provider, args.model).resolve()
    )
    cache_dir = Path(args.prompt_cache_dir).resolve() if args.prompt_cache_dir else default_results_dir

    problems = [NLIProblem.model_validate(problem.model_dump()) for problem in load_benchmark_problems()]
    problem_ids = [problem.id for problem in problems]
    problems_by_id = {problem.id: problem for problem in problems}
    prompt_hashes = {prompt_style: prompt_hash(prompt_style) for prompt_style in PROMPT_STYLES}
    prompt_cache_paths = {
        prompt_style: build_prompt_cache_path(
            cache_dir,
            provider,
            args.model,
            prompt_style,
            prompt_hashes[prompt_style],
        )
        for prompt_style in PROMPT_STYLES
    }
    prompt_cache_files = {
        prompt_style: str(path)
        for prompt_style, path in prompt_cache_paths.items()
    }

    state = ensure_compatible_state(
        load_existing_state(results_path),
        results_path,
        args.model,
        provider,
        problem_ids,
        prompt_hashes,
        prompt_cache_files,
    )
    prompt_cache_states = {
        prompt_style: ensure_compatible_prompt_cache_state(
            load_existing_state(prompt_cache_paths[prompt_style]),
            prompt_cache_paths[prompt_style],
            args.model,
            provider,
            prompt_style,
            prompt_hashes[prompt_style],
            problem_ids,
        )
        for prompt_style in PROMPT_STYLES
    }
    sync_prompt_cache_with_comparison_state(state, prompt_cache_states, problem_ids)
    for prompt_style, cache_path in prompt_cache_paths.items():
        save_state(cache_path, prompt_cache_states[prompt_style])
    save_state(results_path, state)

    context = create_async_run_context(
        AsyncRunLimits(
            llm_concurrency=args.llm_concurrency,
            langpro_concurrency=args.langpro_concurrency,
            local_langpro_concurrency=args.local_langpro_concurrency,
        )
    )

    jobs: List[Tuple[str, Any]] = []
    for prompt_style in PROMPT_STYLES:
        completed_problem_ids = set(state["runs"][prompt_style]["items_by_id"])
        jobs.extend(
            (prompt_style, problem)
            for problem in problems
            if problem.id not in completed_problem_ids
        )

    baseline_by_problem_id: Dict[str, Any] = {}
    if args.precompute_baselines:
        baseline_problem_ids = sorted({problem.id for _, problem in jobs})
        baseline_problems = [problems_by_id[problem_id] for problem_id in baseline_problem_ids]
        baseline_by_problem_id = await _precompute_baselines(baseline_problems, context)

    job_semaphore = asyncio.Semaphore(args.job_concurrency)

    async def process_job(prompt_style: str, problem: Any) -> Tuple[str, str, Dict[str, Any]]:
        config = ProblemConfig(
            llm_provider=provider,
            model=args.model,
            prompt_style=prompt_style,
            test_mode=TestMode.BOTH,
            run_ablation=False,
            verbose=args.verbose,
        )
        async with job_semaphore:
            result = await process_single_problem(
                problem,
                config=config,
                context=context,
                baseline_no_kb=baseline_by_problem_id.get(problem.id),
            )
            payload = serializable_result_payload(
                result,
                discard_prover_calls=args.discard_prover_calls,
            )
            payload["model"] = args.model
            payload["provider"] = provider
            payload["prompt_style"] = prompt_style
            return prompt_style, problem.id, payload

    print(f"\n=== Running {len(jobs)} jobs for model={args.model} provider={provider} ===")
    for prompt_style in PROMPT_STYLES:
        completed_count = len(state["runs"][prompt_style]["items_by_id"])
        if completed_count:
            print(f"Resuming {prompt_style}: {completed_count}/{len(problems)} already completed.")

    completed_since_save = 0
    tasks = [asyncio.create_task(process_job(prompt_style, problem)) for prompt_style, problem in jobs]
    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="2x2", unit="job")
    for completed in progress:
        prompt_style, problem_id, payload = await completed
        progress.set_postfix_str(f"{prompt_style}/{problem_id}")
        state["runs"][prompt_style]["items_by_id"][problem_id] = payload
        prompt_cache_states[prompt_style]["items_by_id"][problem_id] = payload
        completed_since_save += 1
        if completed_since_save >= args.save_every:
            save_state(results_path, state)
            save_state(prompt_cache_paths[prompt_style], prompt_cache_states[prompt_style])
            completed_since_save = 0

    summary = build_summary(state, problem_ids)
    for prompt_style, cache_path in prompt_cache_paths.items():
        save_state(cache_path, prompt_cache_states[prompt_style])
    save_state(results_path, state)

    print()
    print(summary["markdown"])
    print(f"\nWrote {results_path}")
    print("Prompt result caches:")
    for prompt_style in PROMPT_STYLES:
        print(f"  {prompt_style}: {prompt_cache_paths[prompt_style]}")
    return 0


def run(args: argparse.Namespace) -> int:
    return asyncio.run(run_async(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the fixed 50-problem benchmark and produce a legacy_icl vs icl 2x2 table."
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model identifier to replay, e.g. openai/gpt-5.4-mini.",
    )
    parser.add_argument(
        "-r",
        "--results-file",
        default=None,
        help=(
            "JSON comparison results path. Defaults to "
            "<results-dir>/2x2_50_compare__<provider>__<model>.json."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default=None,
        help="Deprecated alias for --results-file.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "openrouter", "claude", "gemini"],
        help="Optional provider override. By default the script infers the provider from the model and environment.",
    )
    parser.add_argument(
        "--prompt-cache-dir",
        default=None,
        help=(
            "Directory for reusable per-prompt result caches. "
            "Defaults to the kbprojection results directory."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-problem logs from the pipeline.",
    )
    parser.add_argument("--llm-concurrency", type=int, default=2)
    parser.add_argument("--langpro-concurrency", type=int, default=4)
    parser.add_argument("--local-langpro-concurrency", type=int, default=2)
    parser.add_argument("--job-concurrency", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--precompute-baselines",
        action="store_true",
        help=(
            "Precompute all no-KB LangPro baselines before prompt jobs. "
            "By default, baselines are computed inside jobs and reused via the LangPro cache."
        ),
    )
    parser.add_argument(
        "--discard-prover-calls",
        action="store_true",
        help="Discard prover call trees/proofs from result JSON to keep files smaller.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
