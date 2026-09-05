import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from kbprojection.llm import _extract_lasha_kb_from_output
from kbprojection.prompts import (
    ETTORE_BASE_PROMPT, LASHA_BASE_PROMPT, fill_prompt, get_prompt, list_prompts,
)


class TestPrompts(unittest.TestCase):
    def test_synthetic_icl_prompt_emphasizes_langpro_helpfulness(self):
        prompt = get_prompt("icl_synthetic")

        self.assertIn("genuinely KB-helpful for LangPro", prompt)
        self.assertIn("Prefer CCG/prover-friendly lemma heads", prompt)
        self.assertIn("Would each fact give LangPro a concrete new bridge", prompt)
        self.assertIn("Negation-aware direction", prompt)
        self.assertIn("isa_wn(jump, play)", prompt)
        self.assertIn("disj(sit, dance)", prompt)

    def test_fill_prompt_substitutes_multi_premise_input(self):
        prompt = fill_prompt("cot", ["Premise one.", "Premise two."], "Hypothesis.")

        self.assertIn("Premise: Premise one.\nPremise two.", prompt)
        self.assertIn("Hypothesis: Hypothesis.", prompt)
        self.assertNotIn("${premise}", prompt)
        self.assertNotIn("${hypothesis}", prompt)

    def test_fill_prompt_substitutes_lasha_uppercase_placeholders(self):
        prompt = fill_prompt("lasha", ["Premise one.", "Premise two."], "Hypothesis.")

        self.assertIn("premise: Premise one.\nPremise two.", prompt)
        self.assertIn("hypothesis: Hypothesis.", prompt)
        self.assertNotIn("${PREMISE}", prompt)
        self.assertNotIn("${HYPOTHESIS}", prompt)
        self.assertIn("entails(woman, person)", prompt)

    def test_fill_prompt_accepts_partial_predicate_dictionary(self):
        prompt = fill_prompt(
            "lasha",
            ["A woman is dancing."],
            "A person is moving.",
            variables={"predicates": {"entailment": "isa_wn"}},
        )

        self.assertIn("isa_wn(woman, person)", prompt)
        self.assertNotIn("${PREDICATE_ENTAILMENT}", prompt)
        self.assertNotIn("${PREDICATE_DISJUNCTION}", prompt)

    def test_extract_lasha_kb_from_entailment_output(self):
        output = (
            "answer: entailment\n"
            "relations: { entails(young lady, girl), entails(guitar, musical instrument) }\n"
        )

        self.assertEqual(
            _extract_lasha_kb_from_output(output),
            ["isa_wn(young lady, girl)", "isa_wn(guitar, musical instrument)"],
        )

    def test_extract_lasha_kb_from_non_entailment_output(self):
        self.assertEqual(_extract_lasha_kb_from_output("answer: non-entailment\n"), [])

    def test_extract_lasha_kb_from_empty_relation_set(self):
        output = "answer: entailment\nrelations: { }\n"
        self.assertEqual(_extract_lasha_kb_from_output(output), [])

    def test_lasha_prompt_is_registered_and_substituted(self):
        prompt = fill_prompt("lasha", ["A dog is running."], "An animal is moving.")

        self.assertIn("answer: entailment", prompt)
        self.assertIn("relations: {", prompt)
        self.assertIn("premise: A dog is running.", prompt)
        self.assertIn("hypothesis: An animal is moving.", prompt)
        self.assertNotIn("${PREMISE}", prompt)
        self.assertNotIn("${HYPOTHESIS}", prompt)


    def test_prompt_names_are_unique(self):
        self.assertEqual(len(list_prompts()), len(set(list_prompts())))

    def test_prompt_ablation_bases_and_explicit_variants(self):
        self.assertIn("isa_wn(strum, play)", ETTORE_BASE_PROMPT)
        self.assertEqual(get_prompt("icl"), ETTORE_BASE_PROMPT)
        self.assertNotIn("CALIBRATION UPDATE", ETTORE_BASE_PROMPT)
        self.assertIn("CALIBRATION UPDATE", get_prompt("ettore"))
        self.assertNotIn("Additional calibration", LASHA_BASE_PROMPT)
        self.assertNotIn("Additional calibration", get_prompt("lasha_uncalibrated"))
        self.assertIn("Additional calibration", get_prompt("lasha"))

    def test_calibrated_lasha_supports_predicate_overrides(self):
        prompt = fill_prompt(
            "lasha", ["A woman dances."], "A person moves.",
            variables={"predicates": {"entailment": "isa_wn"}},
        )
        self.assertIn("isa_wn(woman, person)", prompt)
        self.assertNotIn("${PREDICATE_ENTAILMENT}", prompt)
        self.assertNotIn("entails(", prompt)
        self.assertIn("Additional calibration", prompt)


if __name__ == "__main__":
    unittest.main()
