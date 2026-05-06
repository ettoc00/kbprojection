import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.settings import get_default_results_dir


SOLVED_STATUSES = {"raw_kb_solved", "normalised_kb_solved"}
RELATION_RE = re.compile(r"^(isa_wn|disj)\((.*?),\s*(.*?)\)$")
PREPOSITIONS = {
    "about",
    "after",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "through",
    "to",
    "with",
}


def default_result_path(filename: str) -> Path:
    return get_default_results_dir() / filename


def load_items(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["items_by_key"]


def count_by_status(items: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(item.get("final_status") or "unknown") for item in items.values()).items()))


def pairwise_status_matrix(
    left: Dict[str, Dict[str, Any]],
    right: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    counts = Counter()
    for key in set(left) & set(right):
        if left[key].get("final_status") == "baseline_solved":
            continue
        pair = f"{left[key].get('final_status')} / {right[key].get('final_status')}"
        counts[pair] += 1
    return dict(counts.most_common())


def is_solved(item: Dict[str, Any]) -> bool:
    return item.get("final_status") in SOLVED_STATUSES


def relation_parts(relation: str) -> Optional[tuple[str, str, str]]:
    match = RELATION_RE.match(str(relation).strip())
    if not match:
        return None
    return match.group(1), match.group(2).strip(), match.group(3).strip()


def relation_args(kb: Sequence[str]) -> List[str]:
    args: List[str] = []
    for relation in kb or []:
        parts = relation_parts(relation)
        if parts:
            args.extend([parts[1], parts[2]])
    return args


def arg_flags(arg: str) -> Dict[str, bool]:
    tokens = [token for token in re.split(r"\s+", arg.strip().lower()) if token]
    return {
        "arg_gt_2_words": len(tokens) > 2,
        "arg_has_prep": any(token in PREPOSITIONS for token in tokens),
        "arg_gerund_like": any(token.endswith("ing") for token in tokens),
    }


def phrase_shape_counts(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for item in items:
        for arg in relation_args(item.get("kb_raw") or []):
            for flag, value in arg_flags(arg).items():
                if value:
                    counts[flag] += 1
    return dict(sorted(counts.items()))


def leaves(tree: Any) -> List[str]:
    if isinstance(tree, str):
        parts = tree.splitlines()
        if len(parts) >= 3:
            return [parts[2].strip().lower()]
        return []
    if isinstance(tree, list):
        out: List[str] = []
        for child in tree:
            out.extend(leaves(child))
        return out
    return []


def ccg_lemma_sequences(item: Dict[str, Any]) -> List[List[str]]:
    calls = item.get("prover_calls") or []
    if not calls:
        return []
    for call in reversed(calls):
        trees = call.get("ccg_trees") or []
        if trees:
            return [leaves(tree) for tree in trees]
    return []


def phrase_in_lemmas(phrase: str, lemmas: Sequence[str]) -> bool:
    tokens = [token for token in phrase.lower().split() if token]
    if not tokens:
        return False
    if len(tokens) > len(lemmas):
        return False
    return any(list(lemmas[index : index + len(tokens)]) == tokens for index in range(len(lemmas) - len(tokens) + 1))


def kb_arg_alignment(item: Dict[str, Any], kb: Sequence[str]) -> List[Dict[str, Any]]:
    lemma_sequences = ccg_lemma_sequences(item)
    premise_lemmas = lemma_sequences[0] if lemma_sequences else []
    hypothesis_lemmas = lemma_sequences[1] if len(lemma_sequences) > 1 else []
    rows = []
    for arg in relation_args(kb):
        in_premise = phrase_in_lemmas(arg, premise_lemmas)
        in_hypothesis = phrase_in_lemmas(arg, hypothesis_lemmas)
        flags = arg_flags(arg)
        rows.append(
            {
                "arg": arg,
                "in_premise_lemmas": in_premise,
                "in_hypothesis_lemmas": in_hypothesis,
                "appears_not_to_align_with_parsed_lemmas": bool(lemma_sequences) and not (in_premise or in_hypothesis),
                **flags,
            }
        )
    return rows


def normalized_direction_changes(raw_kb: Sequence[str], norm_kb: Sequence[str]) -> List[Dict[str, str]]:
    raw_pairs = []
    for relation in raw_kb or []:
        parts = relation_parts(relation)
        if parts:
            raw_pairs.append(parts)
    norm_set = set(norm_kb or [])
    changes = []
    for predicate, left, right in raw_pairs:
        reverse = f"{predicate}({right}, {left})"
        if reverse in norm_set and f"{predicate}({left}, {right})" not in norm_set:
            changes.append({"raw": f"{predicate}({left}, {right})", "normalised": reverse})
    return changes


def visible_tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "being",
        "by",
        "is",
        "no",
        "not",
        "of",
        "on",
        "some",
        "the",
        "there",
        "to",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z]+", text.lower())
        if len(token) > 2 and token not in stop
    }


def omitted_visible_mismatch(item: Dict[str, Any], kb: Sequence[str]) -> bool:
    problem = item.get("problem") or {}
    premise_text = " ".join(problem.get("premises") or [])
    hypothesis_text = problem.get("hypothesis") or ""
    mismatches = visible_tokens(premise_text) ^ visible_tokens(hypothesis_text)
    kb_tokens = set()
    for arg in relation_args(kb):
        kb_tokens.update(visible_tokens(arg))
    # This is a heuristic cue for inspection, not a correctness judgment.
    return len(mismatches - kb_tokens) >= 2 and bool(kb_tokens)


def item_brief(key: str, gpt: Dict[str, Any], opus: Dict[str, Any]) -> Dict[str, Any]:
    problem = gpt.get("problem") or opus.get("problem") or {}
    return {
        "key": key,
        "gold": problem.get("gold_label"),
        "baseline_prediction": gpt.get("pred_no_kb"),
        "premises": problem.get("premises") or [],
        "hypothesis": problem.get("hypothesis"),
        "gpt54": model_brief(gpt),
        "opus47": model_brief(opus),
    }


def model_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    raw_kb = item.get("kb_raw") or []
    norm_kb = item.get("kb_filtered") or []
    return {
        "final_status": item.get("final_status"),
        "raw_kb": raw_kb,
        "normalised_kb": norm_kb,
        "pred_with_raw_kb": item.get("pred_with_raw_kb"),
        "pred_with_kb": item.get("pred_with_kb"),
        "raw_arg_alignment": kb_arg_alignment(item, raw_kb),
        "normalised_arg_alignment": kb_arg_alignment(item, norm_kb),
        "normalisation_direction_changes": normalized_direction_changes(raw_kb, norm_kb),
        "generated_kb_omitted_visible_mismatch": omitted_visible_mismatch(item, raw_kb),
    }


def sort_key(key: str) -> tuple[str, int | str]:
    split, _, number = key.partition(":")
    try:
        return split, int(number)
    except ValueError:
        return split, number


def select(keys: Iterable[str], limit: int) -> List[str]:
    return sorted(keys, key=sort_key)[:limit]


def raw_vs_normalised_harm(items: Dict[str, Dict[str, Any]]) -> List[str]:
    return [
        key
        for key, item in items.items()
        if item.get("pred_with_raw_kb") == (item.get("problem") or {}).get("gold_label")
        and item.get("pred_with_kb") != (item.get("problem") or {}).get("gold_label")
    ]


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
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
    empty_generation_differences = {
        key
        for key in common_keys
        if bool(gpt[key].get("kb_raw") or []) != bool(opus[key].get("kb_raw") or [])
    }
    harm_keys = sorted(set(raw_vs_normalised_harm(gpt)) | set(raw_vs_normalised_harm(opus)), key=sort_key)

    inspected_keys = []
    inspected_keys.extend(select(gpt_only, args.per_bucket_limit))
    inspected_keys.extend(select(opus_only, args.per_bucket_limit))
    inspected_keys.extend(select(shared_generated_failures, args.per_bucket_limit))
    inspected_keys.extend(select(harm_keys, args.per_bucket_limit))
    inspected_keys = list(dict.fromkeys(inspected_keys))

    return {
        "inputs": {
            "gpt54_results": str(args.gpt54_results),
            "opus47_results": str(args.opus47_results),
        },
        "status_counts": {
            "gpt54": count_by_status(gpt),
            "opus47": count_by_status(opus),
        },
        "pairwise_status_matrix": pairwise_status_matrix(gpt, opus),
        "sets": {
            "gpt_only_kb_wins": len(gpt_only),
            "opus_only_kb_wins": len(opus_only),
            "shared_generated_failures": len(shared_generated_failures),
            "empty_generation_differences": len(empty_generation_differences),
            "raw_vs_normalised_harm": len(harm_keys),
        },
        "phrase_shape_counts": {
            "gpt_only_gpt_raw": phrase_shape_counts(gpt[key] for key in gpt_only),
            "gpt_only_opus_raw": phrase_shape_counts(opus[key] for key in gpt_only),
            "opus_only_gpt_raw": phrase_shape_counts(gpt[key] for key in opus_only),
            "opus_only_opus_raw": phrase_shape_counts(opus[key] for key in opus_only),
            "shared_failures_gpt_raw": phrase_shape_counts(gpt[key] for key in shared_generated_failures),
            "shared_failures_opus_raw": phrase_shape_counts(opus[key] for key in shared_generated_failures),
        },
        "example_keys": {
            "gpt_only_kb_wins": select(gpt_only, args.example_key_limit),
            "opus_only_kb_wins": select(opus_only, args.example_key_limit),
            "shared_generated_failures": select(shared_generated_failures, args.example_key_limit),
            "empty_generation_differences": select(empty_generation_differences, args.example_key_limit),
            "raw_vs_normalised_harm": select(harm_keys, args.example_key_limit),
        },
        "inspected_items": [item_brief(key, gpt[key], opus[key]) for key in inspected_keys],
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# ICL Prompt Failure Analysis",
        "",
        "## Inputs",
        "",
        f"- GPT-5.4: `{report['inputs']['gpt54_results']}`",
        f"- Opus 4.7: `{report['inputs']['opus47_results']}`",
        "",
        "## Status Counts",
        "",
        "```json",
        json.dumps(report["status_counts"], indent=2),
        "```",
        "",
        "## Set Sizes",
        "",
    ]
    for name, value in report["sets"].items():
        lines.append(f"- `{name}`: {value}")
    lines.extend(["", "## Phrase Shape Counts", "", "```json", json.dumps(report["phrase_shape_counts"], indent=2), "```"])
    lines.extend(["", "## Example Keys", "", "```json", json.dumps(report["example_keys"], indent=2), "```"])
    lines.extend(["", "## Inspected Items", ""])
    for item in report["inspected_items"]:
        lines.extend(
            [
                f"### {item['key']}",
                "",
                f"- Gold: `{item['gold']}`",
                f"- Baseline: `{item['baseline_prediction']}`",
                f"- Premise: {' '.join(item['premises'])}",
                f"- Hypothesis: {item['hypothesis']}",
                f"- GPT-5.4: `{item['gpt54']['final_status']}` raw={item['gpt54']['raw_kb']} norm={item['gpt54']['normalised_kb']}",
                f"- Opus 4.7: `{item['opus47']['final_status']}` raw={item['opus47']['raw_kb']} norm={item['opus47']['normalised_kb']}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    results_dir = get_default_results_dir()
    parser = argparse.ArgumentParser(description="Analyze GPT-5.4 vs Opus 4.7 SICK ICL prompt failures.")
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
    parser.add_argument("--output-json", type=Path, default=results_dir / "icl_prompt_failure_analysis.json")
    parser.add_argument("--output-md", type=Path, default=results_dir / "icl_prompt_failure_analysis.md")
    parser.add_argument("--per-bucket-limit", type=int, default=25)
    parser.add_argument("--example-key-limit", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.output_md)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps(report["sets"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
