"""Linguistic and legacy-candidate regression checks (installed NLTK data)."""
import unittest
from unittest.mock import patch

from nltk.stem import WordNetLemmatizer
from kbprojection import filtering as f


class LemmatizationTests(unittest.TestCase):
    def setUp(self):
        self.lemmatizer = WordNetLemmatizer()

    def transform(self, kb, p="", h="", **kwargs):
        return f.lemmatize_kb(kb, p, h, lemmatizer=self.lemmatizer, **kwargs)

    def test_context_recovers_noun_pair(self):
        kb = "isa_wn(barbells, weights)"
        p, h = "The man is lifting barbells.", "The man is lifting weights."
        self.assertEqual(self.transform(kb), "isa_wn(barbells, weight)")
        self.assertEqual(self.transform(kb, p, h, mode="context_pos"), "isa_wn(barbell, weight)")

    def test_adjective_is_preserved_while_verb_rule_changes_it(self):
        kb = "isa_wn(cartoon, animated)"
        self.assertEqual(self.transform(kb), "isa_wn(cartoon, animate)")
        self.assertEqual(self.transform(kb, "A cartoon airplane lands.",
                         "An animated airplane lands.", mode="context_pos"), kb)

    def test_context_verb_and_absent_span(self):
        self.assertEqual(f.lemmatize_argument("performing", sentence="A band is performing.",
            mode="context_pos", lemmatizer=self.lemmatizer), "perform")
        self.assertEqual(f.lemmatize_argument("missing words", sentence="A band is performing.",
            mode="context_pos", lemmatizer=self.lemmatizer), "missing words")

    def test_conflicting_occurrence_tags_preserve_argument(self):
        with patch.object(f, "_tag_context", return_value=(("saw", "NN"), ("saw", "VBD"))):
            self.assertEqual(f.lemmatize_argument("saw", sentence="saw saw", mode="context_pos",
                lemmatizer=self.lemmatizer), "saw")

    def test_addition_preserves_order_duplicates_predicate_and_direction(self):
        original = "isa_wn(running, moving); isa_wn(running, moving); disj(red, blue)"
        result = self.transform(original, additive=True)
        self.assertEqual(result, original + "; isa_wn(run, move)")
        self.assertEqual(f.add_lemma_variants("isa_wn(dog, animal)",
            "disj(dog, animal); isa_wn(animal, dog)"),
            "isa_wn(dog, animal); disj(dog, animal); isa_wn(animal, dog)")

    def test_missing_explicit_empty_and_malformed_arity(self):
        for kb in ("", "NO_RELATION", "isa_wn(running, moving, walking)"):
            self.assertEqual(self.transform(kb), kb)
        with self.assertRaises(ValueError):
            self.transform("unparsed response")

    def test_existing_candidate_generation_retains_verb_and_swap_behavior(self):
        with patch.object(f, "get_lemmatizer", return_value=self.lemmatizer), \
             patch.object(f, "check_nltk"):
            candidates = f.generate_all_candidates("isa_wn", "running", "moving",
                "isa_wn(running, moving)", post_process=False)
        self.assertEqual([(c.relation, c.provenance) for c in candidates], [
            ("isa_wn(running, moving)", "llm"),
            ("isa_wn(moving, running)", "derived_swap"),
            ("isa_wn(run, move)", "derived_lemma"),
            ("isa_wn(move, run)", "derived_lemma_swap"),
        ])


if __name__ == "__main__":
    unittest.main()
