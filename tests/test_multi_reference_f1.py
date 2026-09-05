import unittest

from calculate_multi_reference_f1 import (
    argument_order_agnostic_relation_counts,
    evaluate_prediction_column,
    parse_kb_cell,
    relation_counts,
)


class ArgumentOrderAgnosticMetricTests(unittest.TestCase):
    def test_reversed_arguments_only_match_in_diagnostic_metric(self):
        prediction = parse_kb_cell("isa_wn(fruit, apple)")
        reference = parse_kb_cell("isa_wn(apple, fruit)")

        directed = relation_counts(prediction, reference)
        agnostic = argument_order_agnostic_relation_counts(prediction, reference)

        self.assertEqual((directed.tp, directed.fp, directed.fn), (0, 1, 1))
        self.assertEqual((agnostic.tp, agnostic.fp, agnostic.fn), (1, 0, 0))

    def test_unrelated_pairs_remain_distinct(self):
        prediction = parse_kb_cell("isa_wn(apple, vehicle)")
        reference = parse_kb_cell("isa_wn(apple, fruit)")

        agnostic = argument_order_agnostic_relation_counts(prediction, reference)

        self.assertEqual((agnostic.tp, agnostic.fp, agnostic.fn), (0, 1, 1))

    def test_best_reference_is_reselected_for_agnostic_metric(self):
        rows = [
            {
                "ID": "example",
                "prediction": "isa_wn(fruit, apple)",
                "reference_a": "isa_wn(apple, fruit)",
                "reference_b": "isa_wn(car, vehicle)",
            }
        ]

        result = evaluate_prediction_column(
            rows,
            "prediction",
            ["reference_a", "reference_b"],
            empty_prediction_is_no_relation=False,
            calculate_argument_order_agnostic=True,
        )

        self.assertEqual(
            (
                result.selected_counts.tp,
                result.selected_counts.fp,
                result.selected_counts.fn,
            ),
            (0, 1, 1),
        )
        self.assertIsNotNone(result.argument_order_agnostic_counts)
        self.assertEqual(
            (
                result.argument_order_agnostic_counts.tp,
                result.argument_order_agnostic_counts.fp,
                result.argument_order_agnostic_counts.fn,
            ),
            (1, 0, 0),
        )
        self.assertEqual(result.argument_order_agnostic_exact_best_matches, 1)

    def test_agnostic_exact_match_requires_the_complete_relation_set(self):
        rows = [
            {
                "ID": "example",
                "prediction": "isa_wn(fruit, apple)",
                "reference": "isa_wn(apple, fruit); isa_wn(cat, animal)",
            }
        ]

        result = evaluate_prediction_column(
            rows,
            "prediction",
            ["reference"],
            empty_prediction_is_no_relation=False,
            calculate_argument_order_agnostic=True,
        )

        self.assertEqual(result.argument_order_agnostic_exact_best_matches, 0)


if __name__ == "__main__":
    unittest.main()
