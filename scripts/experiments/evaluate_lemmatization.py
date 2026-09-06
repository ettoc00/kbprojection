"""Replay five lemma conditions on saved generations; no model/prover calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import nltk
from nltk.stem import WordNetLemmatizer

import calculate_multi_reference_f1 as scorer
from kbprojection.filtering import add_lemma_variants, lemmatize_kb

ARMS = ("baseline", "additive_verbs", "all_verbs", "context_pos", "additive_pos")
REFS = ("Alternative_KB", "Ettore_KB", "Jorryt_KB", "Lasha_KB", "Stefan_KB")
DEFAULT_INPUT = ROOT / "experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_no_filter_outputs.csv"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_record(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name,
            "sha256": sha(path)}


def select_reference(prediction, references):
    choices = []
    for column, raw in references:
        counts = scorer.relation_counts(prediction, scorer.parse_kb_cell(raw))
        choices.append((column, raw, counts))
    return max(choices, key=lambda item: (
        scorer.item_selection_score(item[2]), item[2].tp,
        -item[2].fp, -item[2].fn, item[0],
    ))


def preflight_resources(data_dir):
    if data_dir:
        nltk.data.path.insert(0, str(data_dir.resolve()))
    # Find installed resources, never use nltk.download during evaluation.
    resources = {}
    for name in ("corpora/wordnet.zip", "tokenizers/punkt_tab/english/",
                 "taggers/averaged_perceptron_tagger_eng/"):
        location = Path(str(nltk.data.find(name)))
        if location.is_file():
            resources[name] = {"sha256": sha(location)}
        else:
            resources[name] = {
                p.relative_to(location).as_posix(): sha(p)
                for p in sorted(location.rglob("*")) if p.is_file()
            }
    return resources


def evaluate(inputs, gold, lemmatizer):
    groups = defaultdict(list)
    seen = set()
    for source in inputs:
        key = (source["ID"], source["model"], source["repeat"])
        if key in seen:
            raise ValueError(f"Duplicate item/model/run (select one prompt): {key}")
        seen.add(key)
        if source["ID"] not in gold:
            raise ValueError(f"No source/reference item for {source['ID']}")
        context = gold[source["ID"]]
        for field in ("premise", "hypothesis"):
            if source.get(field) != context[field]:
                raise ValueError(f"Mismatching {field} for {key}")
        raw = source["KB"]
        verb = lemmatize_kb(raw, lemmatizer=lemmatizer)
        pos = lemmatize_kb(raw, context["premise"], context["hypothesis"],
                          mode="context_pos", lemmatizer=lemmatizer)
        row = {k: source[k] for k in ("ID", "model", "repeat", "premise", "hypothesis")}
        row.update(error=source.get("error", ""), baseline_KB=raw,
                   additive_verbs_KB=add_lemma_variants(raw, verb), all_verbs_KB=verb,
                   context_pos_KB=pos, additive_pos_KB=add_lemma_variants(raw, pos))
        row.update({column: context.get(column, "") for column in REFS})
        refs = [(column, row[column]) for column in REFS if not scorer.is_blank(row[column])]
        row["fixed_reference_column"] = row["fixed_reference_KB"] = ""
        if not scorer.is_blank(raw) and refs:
            column, reference, _ = select_reference(scorer.parse_kb_cell(raw), refs)
            row["fixed_reference_column"], row["fixed_reference_KB"] = column, reference
        groups[source["model"], source["repeat"]].append(row)

    metrics, fixed, paired, examples = [], [], [], []
    for (model, repeat), rows in sorted(groups.items()):
        paired.extend({k: v for k, v in row.items() if k not in REFS} for row in rows)
        for arm in ARMS:
            result = scorer.evaluate_prediction_column(
                rows, arm + "_KB", list(REFS), empty_prediction_is_no_relation=False,
                calculate_position_sensitive=True,
            )
            record = dict(model=model, repeat=repeat, arm=arm, total_items=len(rows),
                          evaluated_items=result.evaluated_items,
                          error_runs=sum(bool(row["error"].strip()) for row in rows),
                          skipped_missing_prediction=result.skipped_missing_prediction,
                          skipped_no_reference=result.skipped_no_reference,
                          exact_best_matches=result.exact_best_matches,
                          exact_best_match_rate=result.exact_best_matches / result.evaluated_items,
                          no_relation_best_matches=result.no_relation_best_matches)
            for prefix, counts in (("", result.selected_counts),
                                   ("position_sensitive_", result.position_sensitive_counts)):
                record.update({prefix + k: getattr(counts, k) for k in ("tp", "fp", "fn", "precision", "recall")})
                record[prefix + "micro_f1"] = counts.f1
            metrics.append(record)
            totals = scorer.Counts()
            evaluated = nonempty = 0
            for row in rows:
                if not row["fixed_reference_column"]:
                    continue
                reference = scorer.parse_kb_cell(row["fixed_reference_KB"])
                counts = scorer.relation_counts(scorer.parse_kb_cell(row[arm + "_KB"]), reference)
                totals.add(counts)
                evaluated += 1
                nonempty += bool(reference)
                if model == "google/gemma-3-4b-it" and repeat == "1" and row["ID"] == "2898":
                    examples.append(dict(ID=row["ID"], model=model, repeat=repeat, arm=arm,
                        premise=row["premise"], hypothesis=row["hypothesis"],
                        original_KB=row["baseline_KB"], KB=row[arm + "_KB"],
                        reference=row["fixed_reference_KB"], item_f1=scorer.item_selection_score(counts)))
            fixed.append(dict(model=model, repeat=repeat, arm=arm, tp=totals.tp, fp=totals.fp,
                              fn=totals.fn, evaluated=evaluated, nonempty_reference=nonempty,
                              reference_pair_count=totals.tp + totals.fn,
                              recall=totals.recall, precision=totals.precision, f1=totals.f1))
        print(f"Scored {model}, generation run {repeat}", flush=True)
    summary = []
    for model in sorted({r["model"] for r in metrics}):
        for arm in ARMS:
            row = dict(model=model, arm=arm)
            for prefix, source, fields in (
                ("", metrics, ("micro_f1", "precision", "recall", "exact_best_match_rate")),
                ("fixed_", fixed, ("recall", "precision", "f1")),
            ):
                subset = [r for r in source if r["model"] == model and r["arm"] == arm]
                for field in fields:
                    values = [r[field] for r in subset]
                    row[prefix + field + "_mean"] = statistics.mean(values)
                    row[prefix + field + "_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.append(row)
    return paired, metrics, fixed, summary, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--references", type=Path, default=ROOT / "data/all_usable_items_362.csv")
    parser.add_argument("--nltk-data", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "experiment_results/lemmatization")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; use a new directory for replay")
    resources = preflight_resources(args.nltk_data)
    inputs, references = read_csv(args.input), read_csv(args.references)
    gold = {row["ID"]: row for row in references}
    if len(gold) != len(references):
        raise ValueError("Duplicate reference IDs")
    paired, metrics, fixed, summary, examples = evaluate(inputs, gold, WordNetLemmatizer())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("paired_items.csv", paired), ("metrics_by_run.csv", metrics),
                       ("fixed_reference_by_run.csv", fixed), ("summary.csv", summary)):
        write_csv(args.output / name, rows)
    if examples:
        write_csv(args.output / "examples.csv", examples)
    lines = ["LEMMATIZATION: FIVE-CONDITION OFFLINE COMPARISON", "",
             f"{len(gold)} problems; {len(paired)} saved output slots.",
             "Micro-F1 is computed from accumulated TP/FP/FN within each run.",
             "Tables report percentages, averaged across generation runs.",
             "References are unchanged. Missing outputs are excluded; NO_RELATION is evaluated.",
             "Additive variants retain original relations. No filters, swaps or diff-only variants.",
             "POS alignment uses arg1 in premise and arg2 in hypothesis; ambiguous/absent spans are preserved.",
             "Fixed-reference recall uses the baseline-selected reference for every variant.",
             "This exploratory full-data comparison is not held-out validation.", "",
             f"{'Model':34} {'Condition':16} {'Micro-F1':>9} {'Fixed recall':>13}"]
    for row in summary:
        lines.append(f"{row['model']:34} {row['arm']:16} {100*row['micro_f1_mean']:9.2f} {100*row['fixed_recall_mean']:13.2f}")
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                    inputs=[input_record(args.input), input_record(args.references)],
                    code=[input_record(__file__), input_record(ROOT / "kbprojection/filtering.py"),
                          input_record(ROOT / "calculate_multi_reference_f1.py"),
                          input_record(ROOT / "calculate_inter_annotator_agreement.py")],
                    python_version=sys.version, nltk_version=nltk.__version__, resources=resources,
                    output_slots=len(paired), evaluated_output_slots=sum(bool(r["fixed_reference_column"]) for r in paired),
                    conditions=list(ARMS), model_run_condition_rows=len(metrics), network_requests=0,
                    artifact_sha256={p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
