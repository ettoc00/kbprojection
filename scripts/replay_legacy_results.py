import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.settings import get_default_results_dir

DEFAULT_BASELINE = get_default_results_dir() / "results_6.4_openai_gpt-5-mini__legacy_icl.json"
SOLVED_STATUSES = {"normalised_kb_solved", "raw_kb_solved", "fixed", "fixed_raw_kb"}


def sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_results(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_existing_output(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return load_results(path)


def save_results(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2)


def accuracy_for_field(items: List[Dict[str, Any]], field: str) -> float:
    if not items:
        return 0.0

    correct = 0
    for item in items:
        if item.get(field) == item["problem"]["gold_label"]:
            correct += 1
    return correct / len(items)


def solved_rate(items: List[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    solved = sum(1 for item in items if item.get("final_status") in SOLVED_STATUSES)
    return solved / len(items)


def solved_ids(items: List[Dict[str, Any]]) -> set[str]:
    return {
        item["problem"]["id"]
        for item in items
        if item.get("final_status") in SOLVED_STATUSES
    }


def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total": len(items),
        "baseline_acc": accuracy_for_field(items, "pred_no_kb"),
        "raw_acc": accuracy_for_field(items, "pred_with_raw_kb"),
        "normalised_acc": accuracy_for_field(items, "pred_with_kb"),
        "solved_rate": solved_rate(items),
        "solved_ids": solved_ids(items),
    }


def serializable_result_payload(result: Any) -> Dict[str, Any]:
    payload = result.model_dump(
        mode="json",
        exclude={"prover_calls"},
    )
    payload["prover_calls"] = None
    return payload


def compare_runs(baseline: List[Dict[str, Any]], candidate: List[Dict[str, Any]]) -> str:
    base = summarize(baseline)
    cand = summarize(candidate)

    gained = sorted(cand["solved_ids"] - base["solved_ids"])
    lost = sorted(base["solved_ids"] - cand["solved_ids"])

    lines = [
        "Comparison against legacy gpt-5-mini baseline",
        f"  Total problems: {cand['total']}",
        (
            f"  Baseline acc: {cand['baseline_acc']:.1%} "
            f"(baseline {base['baseline_acc']:.1%}, delta {cand['baseline_acc'] - base['baseline_acc']:+.1%})"
        ),
        (
            f"  Raw KB acc: {cand['raw_acc']:.1%} "
            f"(baseline {base['raw_acc']:.1%}, delta {cand['raw_acc'] - base['raw_acc']:+.1%})"
        ),
        (
            f"  Normalised KB acc: {cand['normalised_acc']:.1%} "
            f"(baseline {base['normalised_acc']:.1%}, delta {cand['normalised_acc'] - base['normalised_acc']:+.1%})"
        ),
        (
            f"  Solved rate: {cand['solved_rate']:.1%} "
            f"(baseline {base['solved_rate']:.1%}, delta {cand['solved_rate'] - base['solved_rate']:+.1%})"
        ),
        f"  Solved IDs gained: {', '.join(gained[:10]) if gained else 'none'}",
        f"  Solved IDs lost: {', '.join(lost[:10]) if lost else 'none'}",
    ]
    return "\n".join(lines)


def build_output_path(output: Path | None, provider: str, model: str, prompt_style: str) -> Path:
    if output is not None:
        return output

    safe_provider = sanitize_filename_part(provider)
    safe_model = sanitize_filename_part(model)
    safe_prompt = sanitize_filename_part(prompt_style)
    return get_default_results_dir() / f"results_replay_{safe_provider}_{safe_model}__{safe_prompt}.json"


async def run_async(args: argparse.Namespace) -> int:
    if args.provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 2

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    from kbprojection.models import NLIProblem, ProblemConfig, TestMode
    from kbprojection.orchestration import process_single_problem

    baseline_path = Path(args.baseline).resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline results file not found: {baseline_path}. Pass --baseline to an existing JSON file."
        )
    baseline_items = load_results(baseline_path)
    output_path = build_output_path(
        Path(args.output).resolve() if args.output else None,
        args.provider,
        args.model,
        args.prompt_style,
    )
    replay_items = load_existing_output(output_path)
    completed_problem_ids = {item["problem"]["id"] for item in replay_items}

    config = ProblemConfig(
        llm_provider=args.provider,
        model=args.model,
        prompt_style=args.prompt_style,
        test_mode=TestMode.BOTH,
        run_ablation=False,
        verbose=args.verbose,
    )

    total = len(baseline_items)
    for index, baseline_item in enumerate(baseline_items, start=1):
        problem = NLIProblem.model_validate(baseline_item["problem"])
        if problem.id in completed_problem_ids:
            print(f"[{index}/{total}] Skipping completed {problem.dataset}/{problem.split}/{problem.id}")
            continue
        print(f"[{index}/{total}] Replaying {problem.dataset}/{problem.split}/{problem.id}")
        result = await process_single_problem(problem, config=config)

        payload = serializable_result_payload(result)
        payload["model"] = args.model
        payload["provider"] = args.provider
        payload["prompt_style"] = args.prompt_style
        payload["baseline_reference"] = str(baseline_path)
        replay_items.append(payload)
        completed_problem_ids.add(problem.id)
        save_results(output_path, replay_items)

    print()
    print(f"Wrote replay results to {output_path}")
    print(compare_runs(baseline_items, replay_items))
    return 0


def run(args: argparse.Namespace) -> int:
    return asyncio.run(run_async(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the exact legacy benchmark problem set against a new model/provider."
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Path to the legacy baseline JSON file.",
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=["openai", "openrouter", "claude", "gemini"],
        help="LLM provider to use for the replay.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.4-mini",
        help="Model identifier to replay.",
    )
    parser.add_argument(
        "--prompt-style",
        default="legacy_icl",
        help="Prompt style to use for the replay.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for the replay JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-problem logs from the pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
