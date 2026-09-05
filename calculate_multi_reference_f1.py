#!/usr/bin/env python3
"""Multi-reference KB evaluation for annotator agreement overview CSV files.

The main metric mirrors multi-reference machine-translation evaluation:
compare one prediction KB against every available reference KB for the same
item, keep the best matching reference for that item, then micro-average the
selected TP/FP/FN counts.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

from calculate_inter_annotator_agreement import normalize_kb


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else float("nan")

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else float("nan")

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else float("nan")


@dataclass
class EvaluationResult:
    prediction_column: str
    reference_columns: list[str]
    selected_counts: Counts
    evaluated_items: int = 0
    skipped_missing_prediction: int = 0
    skipped_no_reference: int = 0
    exact_best_matches: int = 0
    no_relation_best_matches: int = 0
    position_sensitive_counts: Counts | None = None


def is_blank(value: object) -> bool:
    return str(value or "").strip() == ""


def parse_kb_cell(value: object) -> frozenset[tuple[str, ...]]:
    text = str(value or "").strip()
    if text.upper() == "NO_RELATION":
        return frozenset()
    return normalize_kb(text)


def parse_kb_sequence(value: object) -> tuple[tuple[str, ...], ...]:
    """Normalize a KB cell while preserving the written relation order."""
    text = str(value or "").strip().lower().replace("?", "")
    if text.upper() == "NO_RELATION" or not normalize_kb(text):
        return ()

    relations: list[tuple[str, ...]] = []
    for match in re.finditer(r"\(([^()]*)\)", text):
        inner = match.group(1).strip()
        if not inner or inner in {"no, relation", "no relation", "none"}:
            continue
        relations.append(tuple(part.strip() for part in inner.split(",")))

    if relations:
        return tuple(relations)
    return ((text,),)


def relation_counts(
    prediction: frozenset[tuple[str, ...]],
    reference: frozenset[tuple[str, ...]],
) -> Counts:
    return Counts(
        tp=len(prediction & reference),
        fp=len(prediction - reference),
        fn=len(reference - prediction),
    )


def position_sensitive_relation_counts(
    prediction: tuple[tuple[str, ...], ...],
    reference: tuple[tuple[str, ...], ...],
) -> Counts:
    """Count exact relation matches at the same sequence position.

    A mismatch at a shared position contributes one FP and one FN. Relations
    extending beyond the shorter sequence contribute an FP (prediction) or FN
    (reference).
    """
    matching_positions = sum(
        predicted_relation == reference_relation
        for predicted_relation, reference_relation in zip(prediction, reference)
    )
    return Counts(
        tp=matching_positions,
        fp=len(prediction) - matching_positions,
        fn=len(reference) - matching_positions,
    )


def item_selection_score(counts: Counts) -> float:
    """F1 used only for choosing the best reference for one item.

    Empty prediction plus empty reference is a perfect item-level match, but it
    contributes no TP/FP/FN to relation-level micro-F1.
    """
    if counts.tp == 0 and counts.fp == 0 and counts.fn == 0:
        return 1.0
    return counts.f1


def format_score(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def sort_score(value: float) -> float:
    return -1.0 if math.isnan(value) else value


def kb_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name.endswith("_KB")]


def generated_llm_kb_columns(fieldnames: list[str]) -> list[str]:
    return [
        name
        for name in fieldnames
        if name.startswith("LLM__") and name.endswith("_KB")
    ]


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")
        return reader.fieldnames, list(reader)


def evaluate_prediction_column(
    rows: list[dict[str, str]],
    prediction_column: str,
    reference_columns: list[str],
    *,
    empty_prediction_is_no_relation: bool,
    calculate_position_sensitive: bool = False,
    details_path: Path | None = None,
) -> EvaluationResult:
    result = EvaluationResult(
        prediction_column=prediction_column,
        reference_columns=reference_columns,
        selected_counts=Counts(),
    )
    detail_rows: list[dict[str, object]] = []
    if calculate_position_sensitive:
        result.position_sensitive_counts = Counts()

    for row in rows:
        raw_prediction = row.get(prediction_column, "")
        if is_blank(raw_prediction) and not empty_prediction_is_no_relation:
            result.skipped_missing_prediction += 1
            continue

        references = [
            (column, parse_kb_cell(row.get(column, "")))
            for column in reference_columns
            if not is_blank(row.get(column, ""))
        ]
        if not references:
            result.skipped_no_reference += 1
            continue

        prediction = parse_kb_cell(raw_prediction)
        scored_references = []
        for column, reference in references:
            counts = relation_counts(prediction, reference)
            scored_references.append((item_selection_score(counts), column, reference, counts))

        best_score, best_column, best_reference, best_counts = max(
            scored_references,
            key=lambda entry: (
                entry[0],
                entry[3].tp,
                -entry[3].fp,
                -entry[3].fn,
                entry[1],
            ),
        )
        result.selected_counts.add(best_counts)

        position_best_column = ""
        position_best_score = float("nan")
        position_best_counts: Counts | None = None
        if calculate_position_sensitive:
            prediction_sequence = parse_kb_sequence(raw_prediction)
            position_scored_references = []
            for column, _ in references:
                reference_sequence = parse_kb_sequence(row.get(column, ""))
                counts = position_sensitive_relation_counts(
                    prediction_sequence, reference_sequence
                )
                position_scored_references.append(
                    (item_selection_score(counts), column, counts)
                )
            position_best_score, position_best_column, position_best_counts = max(
                position_scored_references,
                key=lambda entry: (
                    entry[0],
                    entry[2].tp,
                    -entry[2].fp,
                    -entry[2].fn,
                    entry[1],
                ),
            )
            assert result.position_sensitive_counts is not None
            result.position_sensitive_counts.add(position_best_counts)
        result.evaluated_items += 1
        if prediction == best_reference:
            result.exact_best_matches += 1
            if not prediction:
                result.no_relation_best_matches += 1

        detail_rows.append(
            {
                "ID": row.get("ID", ""),
                "prediction_column": prediction_column,
                "best_reference_column": best_column,
                "best_item_f1": best_score,
                "tp": best_counts.tp,
                "fp": best_counts.fp,
                "fn": best_counts.fn,
                "prediction_kb": raw_prediction,
                "best_reference_kb": row.get(best_column, ""),
                "position_sensitive_best_reference_column": position_best_column,
                "position_sensitive_best_item_f1": position_best_score,
                "position_sensitive_tp": position_best_counts.tp if position_best_counts else "",
                "position_sensitive_fp": position_best_counts.fp if position_best_counts else "",
                "position_sensitive_fn": position_best_counts.fn if position_best_counts else "",
            }
        )

    if details_path is not None:
        with details_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "ID",
                    "prediction_column",
                    "best_reference_column",
                    "best_item_f1",
                    "tp",
                    "fp",
                    "fn",
                    "prediction_kb",
                    "best_reference_kb",
                    "position_sensitive_best_reference_column",
                    "position_sensitive_best_item_f1",
                    "position_sensitive_tp",
                    "position_sensitive_fp",
                    "position_sensitive_fn",
                ],
            )
            writer.writeheader()
            writer.writerows(detail_rows)

    return result


def evaluate_global_best_reference(
    rows: list[dict[str, str]],
    prediction_column: str,
    reference_columns: list[str],
    *,
    empty_prediction_is_no_relation: bool,
) -> list[tuple[str, Counts, int]]:
    results: list[tuple[str, Counts, int]] = []
    for reference_column in reference_columns:
        counts = Counts()
        evaluated_items = 0
        for row in rows:
            raw_prediction = row.get(prediction_column, "")
            raw_reference = row.get(reference_column, "")
            if is_blank(raw_prediction) and not empty_prediction_is_no_relation:
                continue
            if is_blank(raw_reference):
                continue
            counts.add(relation_counts(parse_kb_cell(raw_prediction), parse_kb_cell(raw_reference)))
            evaluated_items += 1
        results.append((reference_column, counts, evaluated_items))
    return sorted(results, key=lambda entry: (-sort_score(entry[1].f1), entry[0]))


def print_result(result: EvaluationResult) -> None:
    counts = result.selected_counts
    print(f"\n{result.prediction_column} vs best of {', '.join(result.reference_columns)}")
    print(
        "  multi_reference_micro_f1 "
        f"P={format_score(counts.precision)} "
        f"R={format_score(counts.recall)} "
        f"F1={format_score(counts.f1)} "
        f"(TP={counts.tp}, FP={counts.fp}, FN={counts.fn})"
    )
    exact_rate = result.exact_best_matches / result.evaluated_items if result.evaluated_items else float("nan")
    print(
        "  items "
        f"evaluated={result.evaluated_items}, "
        f"exact_best_match={format_score(exact_rate)} "
        f"({result.exact_best_matches}/{result.evaluated_items}), "
        f"no_relation_best_matches={result.no_relation_best_matches}"
    )
    print(
        "  skipped "
        f"missing_prediction={result.skipped_missing_prediction}, "
        f"no_available_reference={result.skipped_no_reference}"
    )
    if result.position_sensitive_counts is not None:
        position_counts = result.position_sensitive_counts
        print(
            "  position_sensitive_relation_sequence_micro_f1 "
            f"P={format_score(position_counts.precision)} "
            f"R={format_score(position_counts.recall)} "
            f"F1={format_score(position_counts.f1)} "
            f"(TP={position_counts.tp}, FP={position_counts.fp}, "
            f"FN={position_counts.fn})"
        )


def print_global_reference_results(
    prediction_column: str,
    global_results: list[tuple[str, Counts, int]],
) -> None:
    print(f"  best_single_reference_for_{prediction_column}:")
    for reference_column, counts, evaluated_items in global_results:
        print(
            f"    {reference_column}: "
            f"F1={format_score(counts.f1)} "
            f"P={format_score(counts.precision)} "
            f"R={format_score(counts.recall)} "
            f"(items={evaluated_items}, TP={counts.tp}, FP={counts.fp}, FN={counts.fn})"
        )


def write_summary(path: Path, results: list[EvaluationResult]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prediction_column",
                "reference_columns",
                "evaluated_items",
                "precision",
                "recall",
                "micro_f1",
                "tp",
                "fp",
                "fn",
                "exact_best_matches",
                "exact_best_match_rate",
                "no_relation_best_matches",
                "skipped_missing_prediction",
                "skipped_no_reference",
                "position_sensitive_precision",
                "position_sensitive_recall",
                "position_sensitive_micro_f1",
                "position_sensitive_tp",
                "position_sensitive_fp",
                "position_sensitive_fn",
            ],
        )
        writer.writeheader()
        for result in results:
            counts = result.selected_counts
            position_counts = result.position_sensitive_counts
            exact_rate = result.exact_best_matches / result.evaluated_items if result.evaluated_items else float("nan")
            writer.writerow(
                {
                    "prediction_column": result.prediction_column,
                    "reference_columns": ";".join(result.reference_columns),
                    "evaluated_items": result.evaluated_items,
                    "precision": counts.precision,
                    "recall": counts.recall,
                    "micro_f1": counts.f1,
                    "tp": counts.tp,
                    "fp": counts.fp,
                    "fn": counts.fn,
                    "exact_best_matches": result.exact_best_matches,
                    "exact_best_match_rate": exact_rate,
                    "no_relation_best_matches": result.no_relation_best_matches,
                    "skipped_missing_prediction": result.skipped_missing_prediction,
                    "skipped_no_reference": result.skipped_no_reference,
                    "position_sensitive_precision": position_counts.precision if position_counts else "",
                    "position_sensitive_recall": position_counts.recall if position_counts else "",
                    "position_sensitive_micro_f1": position_counts.f1 if position_counts else "",
                    "position_sensitive_tp": position_counts.tp if position_counts else "",
                    "position_sensitive_fp": position_counts.fp if position_counts else "",
                    "position_sensitive_fn": position_counts.fn if position_counts else "",
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute multi-reference KB micro-F1 from an agreement overview CSV."
    )
    parser.add_argument(
        "--csv",
        default="annotator agreement - inter_annotator_agreement_overview.csv",
        help="Input agreement overview CSV.",
    )
    parser.add_argument(
        "--prediction-column",
        help=(
            "Column to evaluate as the system/LLM KB. If omitted with "
            "--reference-columns, all LLM__*_KB columns are evaluated. "
            "If omitted without --reference-columns, every *_KB column is "
            "evaluated against the others."
        ),
    )
    parser.add_argument(
        "--reference-columns",
        nargs="+",
        help="Reference KB columns. Defaults to all other *_KB columns.",
    )
    parser.add_argument(
        "--empty-prediction-is-no-relation",
        action="store_true",
        help="Treat blank prediction cells as explicit NO_RELATION instead of skipping them as missing.",
    )
    parser.add_argument(
        "--position-sensitive",
        action="store_true",
        help=(
            "Also calculate position-sensitive relation-sequence micro-F1. "
            "Relations only match when they occur at the same sequence position."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default="multi_reference_f1_summary.csv",
        help="Path to write the summary CSV. Use '' to disable.",
    )
    parser.add_argument(
        "--details-csv",
        help="Optional per-item details CSV. Only allowed when --prediction-column is set.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.csv)
    fieldnames, rows = load_rows(input_path)
    available_kb_columns = kb_columns(fieldnames)
    if not available_kb_columns:
        raise SystemExit("No *_KB columns found in the input CSV.")

    if args.prediction_column:
        prediction_columns = [args.prediction_column]
    elif args.reference_columns:
        prediction_columns = generated_llm_kb_columns(fieldnames)
        if not prediction_columns:
            raise SystemExit(
                "No generated LLM prediction columns found. Expected columns matching LLM__*_KB, "
                "or pass --prediction-column explicitly."
            )
    else:
        prediction_columns = available_kb_columns

    details_path = Path(args.details_csv) if args.details_csv else None
    if details_path is not None and len(prediction_columns) != 1:
        raise SystemExit("--details-csv can only be used with --prediction-column.")

    all_results: list[EvaluationResult] = []
    for prediction_column in prediction_columns:
        if prediction_column not in fieldnames:
            raise SystemExit(f"Prediction column not found: {prediction_column}")
        reference_columns = args.reference_columns or [
            column for column in available_kb_columns if column != prediction_column
        ]
        missing_references = [column for column in reference_columns if column not in fieldnames]
        if missing_references:
            raise SystemExit(f"Reference column(s) not found: {', '.join(missing_references)}")
        if not reference_columns:
            raise SystemExit(f"No reference columns available for {prediction_column}.")

        result = evaluate_prediction_column(
            rows,
            prediction_column,
            reference_columns,
            empty_prediction_is_no_relation=args.empty_prediction_is_no_relation,
            calculate_position_sensitive=args.position_sensitive,
            details_path=details_path,
        )
        all_results.append(result)
        print_result(result)
        print_global_reference_results(
            prediction_column,
            evaluate_global_best_reference(
                rows,
                prediction_column,
                reference_columns,
                empty_prediction_is_no_relation=args.empty_prediction_is_no_relation,
            ),
        )

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        write_summary(summary_path, all_results)
        print(f"\nWrote summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
