import unittest

from kbprojection.loaders.sick import SICKLoader
from kbprojection.loaders.snli import SNLILoader
from kbprojection.models import NLIProblem, NLILabel


class TestLoaderProblemIdNormalization(unittest.TestCase):
    def test_sick_loader_strips_problem_id(self):
        self.assertEqual(SICKLoader().normalize_problem_id(" 1792 "), "1792")

    def test_snli_loader_replaces_spaces_with_underscores(self):
        loader = SNLILoader()
        self.assertEqual(loader.normalize_problem_id("vg len26r4e"), "vg_len26r4e")
        self.assertEqual(loader.normalize_problem_id(" vg_len26r4e "), "vg_len26r4e")

    def test_get_problem_uses_normalized_id(self):
        loader = SNLILoader()
        problem = NLIProblem(
            id="vg_len26r4e",
            premises=["A person is outdoors."],
            hypothesis="Someone is outside.",
            gold_label=NLILabel.ENTAILMENT,
            dataset="snli",
            split="train",
        )
        loader._data["train"] = {"vg_len26r4e": problem}
        loader._loaded_splits.add("train")

        self.assertIs(loader.get_problem("vg len26r4e", split="train"), problem)


if __name__ == "__main__":
    unittest.main()
