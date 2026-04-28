import unittest

from kbprojection.langpro import parse_caty, parse_ccg_tree


class TestLangProParsing(unittest.TestCase):
    def test_parse_ccg_tree_accepts_bx_binary_combinator(self):
        tree = {
            "functor": "bx",
            "args": [
                "s:dcl",
                {
                    "functor": "t",
                    "args": ["np", "dogs", "dog", "NNS", "O", "O"],
                },
                {
                    "functor": "t",
                    "args": ["s:dcl\\np", "run", "run", "VBP", "O", "O"],
                },
            ],
        }

        parsed = parse_ccg_tree(tree)

        self.assertEqual(parsed.label(), "bx(s:dcl)")
        self.assertEqual(len(parsed), 2)

    def test_parse_ccg_tree_accepts_gfc_binary_combinator(self):
        tree = {
            "functor": "gfc",
            "args": [
                "s:dcl",
                {
                    "functor": "t",
                    "args": ["s:dcl/np", "find", "find", "VB", "O", "O"],
                },
                {
                    "functor": "t",
                    "args": ["np", "dogs", "dog", "NNS", "O", "O"],
                },
            ],
        }

        parsed = parse_ccg_tree(tree)

        self.assertEqual(parsed.label(), "gfc(s:dcl)")
        self.assertEqual(len(parsed), 2)

    def test_parse_ccg_tree_accepts_gbx_binary_combinator(self):
        tree = {
            "functor": "gbx",
            "args": [
                "s:dcl",
                {
                    "functor": "t",
                    "args": ["np", "dogs", "dog", "NNS", "O", "O"],
                },
                {
                    "functor": "t",
                    "args": ["s:dcl\\np", "run", "run", "VBP", "O", "O"],
                },
            ],
        }

        parsed = parse_ccg_tree(tree)

        self.assertEqual(parsed.label(), "gbx(s:dcl)")
        self.assertEqual(len(parsed), 2)

    def test_parse_caty_omits_empty_feature(self):
        parsed = parse_caty("s:")

        self.assertEqual(str(parsed), "s")


if __name__ == "__main__":
    unittest.main()
