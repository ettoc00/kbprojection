import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


DEFAULT_RESULTS_DIR = Path.home() / "AppData" / "Local" / "kbprojection" / "results"
DEFAULT_FILES = {
    "gpt54": "sick_full__icl__openrouter__openai_gpt-5.4.json",
    "mini": "sick_full__icl__openrouter__openai_gpt-5.4-mini.json",
    "opus": "sick_full__icl__openrouter__anthropic_claude-opus-4.7.json",
}

SUBSET_STATUSES = (
    "raw_kb_solved",
    "normalised_kb_solved",
    "kb_not_solved",
    "kb_generation_empty",
    "kb_normalisation_empty",
)
SOLVED_KB_STATUSES = {"raw_kb_solved", "normalised_kb_solved"}


def load_dotenv_if_present(dotenv_path: Optional[Path] = None) -> None:
    import os

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


def load_items(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["items_by_key"]


def status_sets(items: Dict[str, Dict[str, Any]]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for key, item in items.items():
        out.setdefault(str(item.get("final_status") or "unknown"), set()).add(key)
    return out


def solved_kb_set(items: Dict[str, Dict[str, Any]]) -> set[str]:
    return {
        key
        for key, item in items.items()
        if item.get("final_status") in SOLVED_KB_STATUSES
    }


def count_by_status(items: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(item.get("final_status") or "unknown") for item in items.values()).items()))


def sample_keys(keys: Iterable[str], n: int = 5) -> List[str]:
    return sorted(keys, key=lambda value: (value.split(":", 1)[0], int(value.split(":", 1)[1])))[:n]


def item_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    problem = item["problem"]
    return {
        "gold": problem.get("gold_label"),
        "pred_no_kb": item.get("pred_no_kb"),
        "pred_raw": item.get("pred_with_raw_kb"),
        "pred_norm": item.get("pred_with_kb"),
        "raw_kb": item.get("kb_raw") or [],
        "normalised_kb": item.get("kb_filtered") or [],
        "premises": problem.get("premises"),
        "hypothesis": problem.get("hypothesis"),
    }


def ordered_union(*lists: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for values in lists:
        for value in values or []:
            if value not in seen:
                seen.add(value)
                out.append(value)
    return out


def materially_different(a: Sequence[str], b: Sequence[str]) -> bool:
    return set(a or []) != set(b or [])


def status_zero_filled_counts(items: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    statuses = (
        "baseline_solved",
        "baseline_prover_failed",
        "kb_generation_failed",
        "kb_generation_empty",
        "kb_normalisation_empty",
        "raw_kb_solved",
        "raw_kb_not_solved",
        "normalised_kb_solved",
        "normalised_kb_not_solved",
        "normalised_kb_prover_failed",
        "kb_not_solved",
        "unknown",
    )
    counts = Counter(str(item.get("final_status") or "unknown") for item in items.values())
    return {status: counts.get(status, 0) for status in statuses}


def kb_not_solved_breakdown(items: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    rows = [item for item in items.values() if item.get("final_status") == "kb_not_solved"]
    return {
        "total": len(rows),
        "early_no_kb_wrong_non_neutral_no_generated_kb": sum(
            1
            for item in rows
            if item.get("status_no_kb") == "success"
            and item.get("pred_no_kb") not in (None, "neutral")
            and not item.get("kb_raw")
        ),
        "generated_kb_available": sum(1 for item in rows if bool(item.get("kb_raw"))),
        "normalised_kb_available": sum(1 for item in rows if bool(item.get("kb_filtered"))),
    }


def material_difference_breakdown(
    left: Dict[str, Dict[str, Any]],
    right: Dict[str, Dict[str, Any]],
    keys: Iterable[str],
) -> Dict[str, int]:
    counts = Counter()
    for key in keys:
        left_raw = set(left[key].get("kb_raw") or [])
        right_raw = set(right[key].get("kb_raw") or [])
        left_norm = set(left[key].get("kb_filtered") or [])
        right_norm = set(right[key].get("kb_filtered") or [])
        raw_diff = left_raw != right_raw
        norm_diff = left_norm != right_norm
        if raw_diff and norm_diff:
            counts["different_both"] += 1
        elif raw_diff:
            counts["different_raw_only"] += 1
        elif norm_diff:
            counts["different_normalised_only"] += 1
        else:
            counts["same_raw_and_normalised_sets"] += 1
        if raw_diff and not norm_diff:
            counts["same_after_normalisation"] += 1
        if bool(left_raw) != bool(right_raw):
            counts["one_raw_empty_one_nonempty"] += 1
        if bool(left_norm) != bool(right_norm):
            counts["one_normalised_empty_one_nonempty"] += 1
    return dict(sorted(counts.items()))


def subset_table(left: Dict[str, Dict[str, Any]], right: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    left_sets = status_sets(left)
    right_sets = status_sets(right)
    rows = []
    for status in SUBSET_STATUSES:
        left_keys = left_sets.get(status, set())
        right_keys = right_sets.get(status, set())
        rows.append(
            {
                "status": status,
                "mini_count": len(left_keys),
                "full_count": len(right_keys),
                "overlap": len(left_keys & right_keys),
                "only_mini": len(left_keys - right_keys),
                "only_full": len(right_keys - left_keys),
                "only_mini_examples": sample_keys(left_keys - right_keys),
                "only_full_examples": sample_keys(right_keys - left_keys),
            }
        )
    return rows


def pairwise_outcomes(models: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    sets = {name: status_sets(items) for name, items in models.items()}
    kb_solved = {name: solved_kb_set(items) for name, items in models.items()}
    rows = {}
    names = list(models)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            rows[f"{left}_vs_{right}"] = {
                "kb_solved_overlap": len(kb_solved[left] & kb_solved[right]),
                f"kb_solved_only_{left}": len(kb_solved[left] - kb_solved[right]),
                f"kb_solved_only_{right}": len(kb_solved[right] - kb_solved[left]),
                "raw_overlap": len(sets[left].get("raw_kb_solved", set()) & sets[right].get("raw_kb_solved", set())),
                f"raw_only_{left}": len(sets[left].get("raw_kb_solved", set()) - sets[right].get("raw_kb_solved", set())),
                f"raw_only_{right}": len(sets[right].get("raw_kb_solved", set()) - sets[left].get("raw_kb_solved", set())),
                "normalised_overlap": len(
                    sets[left].get("normalised_kb_solved", set())
                    & sets[right].get("normalised_kb_solved", set())
                ),
                f"normalised_only_{left}": len(
                    sets[left].get("normalised_kb_solved", set())
                    - sets[right].get("normalised_kb_solved", set())
                ),
                f"normalised_only_{right}": len(
                    sets[right].get("normalised_kb_solved", set())
                    - sets[left].get("normalised_kb_solved", set())
                ),
            }
    return rows


async def run_combined_sample(
    gpt54: Dict[str, Dict[str, Any]],
    opus: Dict[str, Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    from kbprojection.langpro import langpro_api_call

    candidates = []
    for key in sorted(set(gpt54) & set(opus), key=lambda value: (value.split(":", 1)[0], int(value.split(":", 1)[1]))):
        g = gpt54[key]
        o = opus[key]
        if g.get("final_status") != "kb_not_solved" or o.get("final_status") != "kb_not_solved":
            continue
        g_raw = g.get("kb_raw") or []
        o_raw = o.get("kb_raw") or []
        g_norm = g.get("kb_filtered") or []
        o_norm = o.get("kb_filtered") or []
        if not (g_raw and o_raw):
            continue
        if materially_different(g_raw, o_raw) or materially_different(g_norm, o_norm):
            candidates.append(key)
        if len(candidates) >= limit:
            break

    rows = []
    for key in candidates:
        item = gpt54[key]
        problem = item["problem"]
        gold = problem["gold_label"]
        g = gpt54[key]
        o = opus[key]
        combined_raw = ordered_union(g.get("kb_raw") or [], o.get("kb_raw") or [])
        combined_norm = ordered_union(g.get("kb_filtered") or [], o.get("kb_filtered") or [])
        raw_result = await langpro_api_call(problem["premises"], problem["hypothesis"], kb=combined_raw)
        norm_result = None
        if combined_norm:
            norm_result = await langpro_api_call(problem["premises"], problem["hypothesis"], kb=combined_norm)
        rows.append(
            {
                "key": key,
                "gold": gold,
                "combined_raw_pred": raw_result.label.value,
                "combined_raw_error": raw_result.error,
                "combined_raw_solved": raw_result.label.value == gold and not raw_result.error,
                "combined_normalised_pred": norm_result.label.value if norm_result else None,
                "combined_normalised_error": norm_result.error if norm_result else None,
                "combined_normalised_solved": (
                    norm_result.label.value == gold and not norm_result.error if norm_result else False
                ),
                "gpt54_raw": g.get("kb_raw") or [],
                "opus_raw": o.get("kb_raw") or [],
                "gpt54_normalised": g.get("kb_filtered") or [],
                "opus_normalised": o.get("kb_filtered") or [],
            }
        )
    return rows


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("sick_result_comparison.json"))
    parser.add_argument("--combined-sample", type=int, default=0)
    parser.add_argument("--combined-random-seed", type=int, default=None)
    args = parser.parse_args()

    load_dotenv_if_present()
    paths = {name: args.results_dir / filename for name, filename in DEFAULT_FILES.items()}
    models = {name: load_items(path) for name, path in paths.items()}

    report = {
        "files": {name: str(path) for name, path in paths.items()},
        "status_counts": {name: count_by_status(items) for name, items in models.items()},
        "status_counts_zero_filled": {name: status_zero_filled_counts(items) for name, items in models.items()},
        "kb_not_solved_breakdown": {name: kb_not_solved_breakdown(items) for name, items in models.items()},
        "mini_vs_gpt54_subset_table": subset_table(models["mini"], models["gpt54"]),
        "pairwise_model_outcomes": pairwise_outcomes(models),
        "baseline_sets_equal": {
            "gpt54_vs_mini": status_sets(models["gpt54"]).get("baseline_solved", set())
            == status_sets(models["mini"]).get("baseline_solved", set()),
            "gpt54_vs_opus": status_sets(models["gpt54"]).get("baseline_solved", set())
            == status_sets(models["opus"]).get("baseline_solved", set()),
        },
    }

    kb_not_solved_overlap = (
        status_sets(models["gpt54"]).get("kb_not_solved", set())
        & status_sets(models["opus"]).get("kb_not_solved", set())
    )
    different_kb = [
        key
        for key in kb_not_solved_overlap
        if materially_different(models["gpt54"][key].get("kb_raw") or [], models["opus"][key].get("kb_raw") or [])
        or materially_different(models["gpt54"][key].get("kb_filtered") or [], models["opus"][key].get("kb_filtered") or [])
    ]
    report["gpt54_opus_kb_not_solved"] = {
        "overlap": len(kb_not_solved_overlap),
        "different_kb_count": len(different_kb),
        "difference_breakdown": material_difference_breakdown(models["gpt54"], models["opus"], kb_not_solved_overlap),
        "different_kb_examples": sample_keys(different_kb, 10),
    }

    all_keys = set.intersection(*(set(items) for items in models.values()))
    kb_sets = {name: solved_kb_set(items) for name, items in models.items()}
    raw_sets = {name: status_sets(items).get("raw_kb_solved", set()) for name, items in models.items()}
    norm_sets = {name: status_sets(items).get("normalised_kb_solved", set()) for name, items in models.items()}
    baseline = status_sets(models["gpt54"]).get("baseline_solved", set())
    report["oracle_upper_bounds"] = {
        "all_files_unique_completed_keys": {name: len(items) for name, items in models.items()},
        "common_keys": len(all_keys),
        "baseline_solved": len(baseline),
        "any_model_raw_kb_solved": len(set().union(*raw_sets.values())),
        "any_model_normalised_kb_solved": len(set().union(*norm_sets.values())),
        "any_model_any_kb_solved": len(set().union(*kb_sets.values())),
        "baseline_plus_any_model_any_kb_solved": len(baseline | set().union(*kb_sets.values())),
        "unique_any_kb_by_model": {
            name: len(kb_sets[name] - set().union(*(kb_sets[other] for other in models if other != name)))
            for name in models
        },
    }

    report["examples"] = {
        "mini_normalised_only_vs_gpt54": {
            key: item_brief(models["mini"][key])
            for key in sample_keys(
                status_sets(models["mini"]).get("normalised_kb_solved", set())
                - status_sets(models["gpt54"]).get("normalised_kb_solved", set()),
                3,
            )
        },
        "gpt54_normalised_only_vs_mini": {
            key: item_brief(models["gpt54"][key])
            for key in sample_keys(
                status_sets(models["gpt54"]).get("normalised_kb_solved", set())
                - status_sets(models["mini"]).get("normalised_kb_solved", set()),
                3,
            )
        },
        "opus_raw_only_vs_gpt54": {
            key: item_brief(models["opus"][key])
            for key in sample_keys(
                status_sets(models["opus"]).get("raw_kb_solved", set())
                - status_sets(models["gpt54"]).get("raw_kb_solved", set()),
                3,
            )
        },
        "gpt54_normalised_only_vs_opus": {
            key: item_brief(models["gpt54"][key])
            for key in sample_keys(
                status_sets(models["gpt54"]).get("normalised_kb_solved", set())
                - status_sets(models["opus"]).get("normalised_kb_solved", set()),
                3,
            )
        },
    }

    if args.combined_sample:
        if args.combined_random_seed is not None:
            candidate_keys = sample_keys(different_kb, len(different_kb))
            random.Random(args.combined_random_seed).shuffle(candidate_keys)
            selected = set(candidate_keys[: args.combined_sample])
            original_gpt54 = models["gpt54"]
            original_opus = models["opus"]
            models["gpt54"] = {key: original_gpt54[key] for key in selected}
            models["opus"] = {key: original_opus[key] for key in selected}
            report["combined_kb_sampling"] = {
                "mode": "random",
                "seed": args.combined_random_seed,
                "requested": args.combined_sample,
                "available": len(different_kb),
            }
        else:
            report["combined_kb_sampling"] = {
                "mode": "sorted_first",
                "requested": args.combined_sample,
                "available": len(different_kb),
            }
        report["combined_kb_sample"] = await run_combined_sample(models["gpt54"], models["opus"], args.combined_sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
