import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from kbprojection.llm import _extract_lasha_kb_from_output
from kbprojection.prompts import fill_prompt, get_prompt


class TestPrompts(unittest.TestCase):
    def test_icl_prompt_emphasizes_langpro_helpfulness(self):
        prompt = get_prompt("icl")

        self.assertIn("genuinely KB-helpful for LangPro", prompt)
        self.assertIn("Only output relations that are genuinely KB-helpful for LangPro", prompt)
        self.assertIn("Would each fact give LangPro a concrete new bridge", prompt)
        self.assertIn("isa_wn(walk, move around)", prompt)
        self.assertIn("disj(open, closed)", prompt)

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


if __name__ == "__main__":
    unittest.main()
