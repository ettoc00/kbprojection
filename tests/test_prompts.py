import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

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


if __name__ == "__main__":
    unittest.main()
