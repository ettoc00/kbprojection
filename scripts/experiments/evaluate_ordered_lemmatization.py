#!/usr/bin/env python3
"""Replay frozen lemma variants under duplicate-controlled positional/set scoring.

No generation, tagging, downloads or prover calls are performed. Both metrics
remove duplicate directed argument pairs; positional scoring retains their first
occurrence. Predicate names are ignored, as in the existing repository scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import calculate_multi_reference_f1 as scorer

ARMS = ("baseline", "additive_verbs", "all_verbs", "context_pos", "additive_pos")
MODES = ("unique_position", "set")
REFERENCES = ("Alternative_KB", "Ettore_KB", "Jorryt_KB", "Lasha_KB", "Stefan_KB")
EXAMPLE_FIELDS = ("ID", "arm", "mode", "premise", "hypothesis", "KB", "reference", "item_f1", "recall")


@lru_cache(maxsize=None)
def representations(raw):
    sequence = scorer.parse_kb_sequence(raw)
    pairs = scorer.parse_kb_cell(raw)
    if frozenset(sequence) != pairs:
        raise ValueError(f"Inconsistent sequence/set parsing: {raw!r}")
    return {"set": pairs, "unique_position": tuple(dict.fromkeys(sequence))}


def counts(prediction, reference, mode):
    function = scorer.relation_counts if mode == "set" else scorer.position_sensitive_relation_counts
    return function(prediction[mode], reference[mode])


def best_reference(prediction, references, mode):
    """Use the existing scorer's F1/TP/-FP/-FN/column-name tie-breaking."""
    choices = []
    for column, raw, representation in references:
        value = counts(prediction, representation, mode)
        choices.append((scorer.item_selection_score(value), column, value, raw))
    return max(choices, key=lambda item: (item[0], item[2].tp, -item[2].fp, -item[2].fn, item[1]))


def evaluate(predictions, gold):
    """Return per-run metrics and the combined Gemma example.

    F1 independently selects the best reference for each variant and metric.
    Recall freezes the reference chosen by the original output with set scoring.
    Missing predictions are excluded; explicit NO_RELATION is evaluated.
    """
    groups = defaultdict(list)
    seen = set()
    for row in predictions:
        key = row["model"], row["repeat"], row["ID"]
        if key in seen:
            raise ValueError(f"Duplicate prediction slot: {key}")
        seen.add(key)
        if row["ID"] not in gold:
            raise ValueError(f"Missing reference item: {row['ID']}")
        groups[key[:2]].append(row)
    records, examples = [], []
    for (model, repeat), rows in sorted(groups.items()):
        totals = {(arm, mode): scorer.Counts() for arm in ARMS for mode in MODES}
        fixed = {(arm, mode): scorer.Counts() for arm in ARMS for mode in MODES}
        exact = defaultdict(int)
        evaluated = 0
        for row in rows:
            missing = [scorer.is_blank(row[arm + "_KB"]) for arm in ARMS]
            if any(missing) and not all(missing):
                raise ValueError(f"Variants have different missingness: {(model, repeat, row['ID'])}")
            if all(missing):
                continue
            references = [(column, gold[row["ID"]][column], representations(gold[row["ID"]][column]))
                          for column in REFERENCES if not scorer.is_blank(gold[row["ID"]].get(column, ""))]
            if not references:
                continue
            baseline = representations(row["baseline_KB"])
            fixed_raw = best_reference(baseline, references, "set")[3]
            fixed_representation = representations(fixed_raw)
            evaluated += 1
            for arm in ARMS:
                prediction = representations(row[arm + "_KB"])
                for mode in MODES:
                    best = best_reference(prediction, references, mode)
                    totals[arm, mode].add(best[2])
                    exact[arm, mode] += any(prediction[mode] == ref[mode] for _, _, ref in references)
                    fixed_counts = counts(prediction, fixed_representation, mode)
                    fixed[arm, mode].add(fixed_counts)
                    if (row["ID"], model, repeat) == ("2898", "google/gemma-3-4b-it", "1"):
                        examples.append({"ID": row["ID"], "arm": arm, "mode": mode,
                                         "premise": row["premise"], "hypothesis": row["hypothesis"],
                                         "KB": row[arm + "_KB"], "reference": fixed_raw,
                                         "item_f1": scorer.item_selection_score(fixed_counts),
                                         "recall": fixed_counts.recall})
        for arm in ARMS:
            for mode in MODES:
                value = fixed[arm, mode]
                records.append({"model": model, "repeat": repeat, "arm": arm, "scoring": mode,
                                "micro_f1": totals[arm, mode].f1,
                                "exact_match": exact[arm, mode] / evaluated if evaluated else float("nan"),
                                "fixed_reference_recall": value.recall,
                                "fixed_tp": value.tp, "fixed_fp": value.fp, "fixed_fn": value.fn,
                                "fixed_evaluated": evaluated})
    return records, examples


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["model"], row["arm"], row["scoring"]].append(row)
    result = []
    for (model, arm, mode), rows in sorted(groups.items()):
        item = {"model": model, "arm": arm, "scoring": mode}
        for metric in ("micro_f1", "fixed_reference_recall", "exact_match"):
            values = [100 * row[metric] for row in rows]
            item[metric + "_mean"] = statistics.mean(values)
            item[metric + "_sd"] = (statistics.stdev(values) if len(values) > 1 and
                                     all(math.isfinite(v) for v in values) else float("nan"))
        result.append(item)
    return result


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path):
    return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()


def report(summary, records, examples):
    runs = sorted({row["repeat"] for row in records})
    lines = ["ORDERED LEMMATIZATION: POSITIONAL / ORDER-AGNOSTIC SCORING", "",
             f"Offline comparison of five frozen variants across {len(runs)} generation runs.",
             "Each cell is positional / order-agnostic, in percent.",
             "Both rules remove duplicate pairs; positional scoring preserves first occurrence.",
             "Argument direction is retained. Predicate names are ignored.", "",
             "Micro-F1 aggregates selected-reference TP/FP/FN within each run.",
             "F1 selects the best reference separately for each variant and scoring rule.",
             "Exact match accepts any available reference. Recall uses the reference selected",
             "by the original prediction with set scoring, fixed across variants and rules.",
             "Missing outputs are excluded; explicit NO_RELATION predictions are evaluated."]
    index = {(r["model"], r["arm"], r["scoring"]): r for r in summary}
    for metric, title in (("micro_f1", "MICRO-F1"), ("fixed_reference_recall", "FIXED-REFERENCE MICRO-RECALL"),
                          ("exact_match", "EXACT MATCH")):
        lines += ["", f"{title} (%) - mean across {len(runs)} generation runs", "",
                  f"{'Model':32} {'Original':>15} {'+Verb':>15} {'Verb replace':>15} {'POS replace':>15} {'+POS':>15}"]
        for model in sorted({r["model"] for r in summary}):
            cells = [" / ".join(f"{index[model, arm, mode][metric + '_mean']:.2f}" for mode in MODES) for arm in ARMS]
            lines.append(f"{model:32} " + " ".join(f"{cell:>15}" for cell in cells))
    if examples:
        lines += ["", "COMBINED EXAMPLE: GEMMA, GENERATION RUN 1", "",
                  "P: " + examples[0]["premise"], "H: " + examples[0]["hypothesis"],
                  "Reference: " + examples[0]["reference"], ""]
        for row in examples:
            lines.append(f"{row['arm']:16} {row['mode']:16} F1={row['item_f1']:.4f} recall={row['recall']:.4f}  {row['KB']}")
        lines += ["", "An appended correct pair can be missed by positional scoring when it does",
                  "not occupy the reference position. Set scoring credits that pair while",
                  "F1 still penalizes unmatched originals retained by addition."]
    lines += ["", "Order-agnostic scoring was already the default for the lemmatization results.",
              "This comparison does not change LLM generations or test LangPro axiom order."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "experiment_results/lemmatization/paired_items.csv")
    parser.add_argument("--references", type=Path, default=ROOT / "data/all_usable_items_362.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "experiment_results/ordered_lemmatization")
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; use a new directory for replay")
    sources = [args.predictions, args.references, Path(__file__), ROOT / "calculate_multi_reference_f1.py",
               ROOT / "calculate_inter_annotator_agreement.py"]
    hashes = {relative_path(path): sha256(path) for path in sources}
    predictions, references = read_csv(args.predictions), read_csv(args.references)
    if not predictions or not references:
        raise ValueError("Predictions and references must contain data rows.")
    gold = {row["ID"]: row for row in references}
    if len(gold) != len(references):
        raise ValueError("Duplicate reference IDs.")
    records, examples = evaluate(predictions, gold)
    summary = summarize(records)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "metrics_by_run.csv", records)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "combined_example.csv", examples, EXAMPLE_FIELDS)
    (args.output / "REPORT.txt").write_text(report(summary, records, examples), encoding="utf-8")
    for path in sources:
        if sha256(path) != hashes[relative_path(path)]:
            raise RuntimeError(f"Input changed during evaluation: {relative_path(path)}")
    manifest = {"completed_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": hashes, "source_path_base": "repository root",
                "prediction_slots": len(predictions), "reference_items": len(gold),
                "evaluated_output_slots": sum(r["fixed_evaluated"] for r in records
                                              if r["arm"] == "baseline" and r["scoring"] == "set"),
                "conditions": list(ARMS), "scoring_rules": list(MODES),
                "model_run_condition_scoring_rows": len(records), "summary_rows": len(summary),
                "fixed_reference_selection": "baseline set F1, TP, -FP, -FN, reference column",
                "network_requests": 0,
                "artifact_sha256": {name: sha256(args.output / name) for name in
                                    ("metrics_by_run.csv", "summary.csv", "combined_example.csv", "REPORT.txt")}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} run rows and {len(summary)} summary rows to {args.output}")


if __name__ == "__main__":
    main()
