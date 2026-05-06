import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.prompts import get_prompt
from kbprojection.settings import get_default_results_dir


SOLVED_STATUSES = {"raw_kb_solved", "normalised_kb_solved"}
EXAMPLE_RE = re.compile(
    r"### Example (?P<number>\d+): (?P<title>[^\n]+)\n"
    r"Premise:\s*(?P<premise>.*?)\n"
    r"Hypothesis:\s*(?P<hypothesis>.*?)\n"
    r"\[KB_START\]\n"
    r"(?P<kb>.*?)\n"
    r"\[KB_END\]",
    re.DOTALL,
)


def default_result_path(filename: str) -> Path:
    return get_default_results_dir() / filename


def load_items(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["items_by_key"]


def is_solved(item: Dict[str, Any]) -> bool:
    return item.get("final_status") in SOLVED_STATUSES


def sort_key(key: str) -> tuple[str, int | str]:
    split, _, number = key.partition(":")
    try:
        return split, int(number)
    except ValueError:
        return split, number


def stable_sample(keys: Iterable[str], size: int, seed: int) -> List[str]:
    keys = sorted(set(keys), key=sort_key)
    if len(keys) <= size:
        return keys
    rng = random.Random(seed)
    rng.shuffle(keys)
    return sorted(keys[:size], key=sort_key)


def raw_vs_normalised_harm(items: Dict[str, Dict[str, Any]]) -> set[str]:
    return {
        key
        for key, item in items.items()
        if item.get("pred_with_raw_kb") == (item.get("problem") or {}).get("gold_label")
        and item.get("pred_with_kb") != (item.get("problem") or {}).get("gold_label")
    }


def extract_synthetic_examples() -> List[Dict[str, Any]]:
    examples = []
    for match in EXAMPLE_RE.finditer(get_prompt("icl")):
        examples.append(
            {
                "id": f"synthetic_icl_example_{match.group('number')}",
                "title": match.group("title").strip(),
                "premises": [re.sub(r"\s+", " ", match.group("premise").strip())],
                "hypothesis": re.sub(r"\s+", " ", match.group("hypothesis").strip()),
                "kb": [line.strip() for line in match.group("kb").splitlines() if line.strip()],
            }
        )
    return examples


def problem_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    problem = item.get("problem") or {}
    return {
        "split": problem.get("split"),
        "id": problem.get("id"),
        "gold_label": problem.get("gold_label"),
        "premises": problem.get("premises") or [],
        "hypothesis": problem.get("hypothesis"),
    }


def parse_args() -> argparse.Namespace:
    results_dir = get_default_results_dir()
    parser = argparse.ArgumentParser(description="Build the deterministic focused validation set for ICL prompt revisions.")
    parser.add_argument(
        "--gpt54-results",
        type=Path,
        default=default_result_path("sick_full__icl__openrouter__openai_gpt-5.4.json"),
    )
    parser.add_argument(
        "--opus47-results",
        type=Path,
        default=default_result_path("sick_full__icl__openrouter__anthropic_claude-opus-4.7.json"),
    )
    parser.add_argument("--output", type=Path, default=results_dir / "icl_focused_validation_set.json")
    parser.add_argument("--shared-failure-sample", type=int, default=100)
    parser.add_argument("--success-regression-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260503)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpt = load_items(args.gpt54_results)
    opus = load_items(args.opus47_results)
    common_keys = set(gpt) & set(opus)

    gpt_only = {key for key in common_keys if is_solved(gpt[key]) and not is_solved(opus[key])}
    opus_only = {key for key in common_keys if is_solved(opus[key]) and not is_solved(gpt[key])}
    shared_generated_failures = {
        key
        for key in common_keys
        if gpt[key].get("final_status") == "kb_not_solved"
        and opus[key].get("final_status") == "kb_not_solved"
        and ((gpt[key].get("kb_raw") or []) or (opus[key].get("kb_raw") or []))
    }
    harm = raw_vs_normalised_harm(gpt) | raw_vs_normalised_harm(opus)
    current_successes = {key for key in common_keys if is_solved(gpt[key]) or is_solved(opus[key])}

    buckets = {
        "gpt_only_wins": sorted(gpt_only, key=sort_key),
        "opus_only_wins": sorted(opus_only, key=sort_key),
        "shared_generated_failures_sample": stable_sample(shared_generated_failures, args.shared_failure_sample, args.seed),
        "raw_vs_normalised_harm": sorted(harm, key=sort_key),
        "current_success_regression_sample": stable_sample(current_successes, args.success_regression_sample, args.seed + 1),
    }
    all_keys = sorted(set().union(*(set(values) for values in buckets.values())), key=sort_key)
    payload = {
        "inputs": {
            "gpt54_results": str(args.gpt54_results),
            "opus47_results": str(args.opus47_results),
        },
        "seed": args.seed,
        "counts": {name: len(values) for name, values in buckets.items()} | {"unique_sick_keys": len(all_keys)},
        "buckets": buckets,
        "sick_problems": {key: problem_brief(gpt.get(key) or opus[key]) for key in all_keys},
        "synthetic_examples": extract_synthetic_examples(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps(payload["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
