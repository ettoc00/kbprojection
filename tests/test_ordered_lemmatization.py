"""Check scoring semantics that affect the combined lemmatization ablation."""

import unittest

from scripts.experiments import evaluate_ordered_lemmatization as evaluation


class OrderedLemmatizationTests(unittest.TestCase):
    def test_relation_order_changes_only_positional_scoring(self):
        prediction = evaluation.representations("isa_wn(cat, animal); isa_wn(dog, animal)")
        reference = evaluation.representations("isa_wn(dog, animal); isa_wn(cat, animal)")
        self.assertEqual(evaluation.counts(prediction, reference, "set").f1, 1)
        self.assertEqual(evaluation.counts(prediction, reference, "unique_position").f1, 0)

    def test_duplicate_control_preserves_first_occurrence(self):
        raw = "isa_wn(cat, animal); isa_wn(cat, animal); isa_wn(dog, animal)"
        prediction = evaluation.representations(raw)
        reference = evaluation.representations("isa_wn(cat, animal); isa_wn(dog, animal)")
        self.assertEqual(prediction, reference)
        for mode in evaluation.MODES:
            self.assertEqual(evaluation.counts(prediction, reference, mode).f1, 1)

    def test_argument_reversal_is_not_order_agnostic(self):
        prediction = evaluation.representations("isa_wn(cat, animal)")
        reference = evaluation.representations("isa_wn(animal, cat)")
        for mode in evaluation.MODES:
            self.assertEqual(evaluation.counts(prediction, reference, mode).f1, 0)

    def test_reference_tie_breaking_matches_existing_scorer(self):
        prediction = evaluation.representations("isa_wn(cat, animal)")
        references = [(column, raw, evaluation.representations(raw)) for column, raw in
                      [("Alternative_KB", "isa_wn(cat, animal); isa_wn(dog, animal)"),
                       ("Stefan_KB", "isa_wn(cat, animal); isa_wn(bird, animal)")]]
        self.assertEqual(evaluation.best_reference(prediction, references, "set")[1], "Stefan_KB")

    def test_missing_is_excluded_but_no_relation_counts_as_exact(self):
        rows = []
        for item_id, kb in [("missing", ""), ("empty", "NO_RELATION"), ("pair", "isa_wn(cat, animal)")]:
            rows.append({"ID": item_id, "model": "test", "repeat": "1",
                         **{arm + "_KB": kb for arm in evaluation.ARMS}})
        gold = {"missing": {"Alternative_KB": "NO_RELATION"},
                "empty": {"Alternative_KB": "NO_RELATION"},
                "pair": {"Alternative_KB": "isa_wn(cat, animal)"}}
        records, _ = evaluation.evaluate(rows, gold)
        for record in records:
            self.assertEqual(record["fixed_evaluated"], 2)
            self.assertEqual(record["exact_match"], 1)
            self.assertEqual(record["micro_f1"], 1)
            self.assertEqual(record["fixed_tp"], 1)

    def test_recall_reference_is_frozen_before_transforming(self):
        row = {"ID": "x", "model": "test", "repeat": "1",
               **{arm + "_KB": "isa_wn(cat, animal)" for arm in evaluation.ARMS}}
        row["context_pos_KB"] = "isa_wn(dog, animal)"
        gold = {"x": {"Alternative_KB": "isa_wn(cat, animal)", "Stefan_KB": "isa_wn(dog, animal)"}}
        records, _ = evaluation.evaluate([row], gold)
        selected = next(r for r in records if r["arm"] == "context_pos" and r["scoring"] == "set")
        self.assertEqual(selected["micro_f1"], 1)
        self.assertEqual(selected["exact_match"], 1)
        self.assertEqual(selected["fixed_reference_recall"], 0)


if __name__ == "__main__":
    unittest.main()
