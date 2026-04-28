import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from kbprojection.langpro import (
    _format_local_langpro_problem_id,
    _local_langpro_ccg_file,
    _local_langpro_goal_prefix,
    _relation_to_prolog_atom,
    _resolve_local_langpro_problem,
    _write_local_langpro_ccg_file,
)


class TestLocalLangProHelpers(unittest.IsolatedAsyncioTestCase):
    def test_relation_to_prolog_atom_quotes_multiword_arguments(self):
        self.assertEqual(
            _relation_to_prolog_atom("isa_wn(walk, move around)"),
            "isa_wn(walk, 'move around')",
        )
        self.assertEqual(
            _relation_to_prolog_atom("isa_wn(play drum, practice drum)"),
            "isa_wn('play drum', 'practice drum')",
        )
        self.assertEqual(
            _relation_to_prolog_atom("isa_wn(teenage, in one's teens)"),
            "isa_wn(teenage, 'in one\\'s teens')",
        )

    def test_local_goal_prefix_respects_langpro_flags(self):
        prefix = _local_langpro_goal_prefix(200, strong_align=True, intersective=True)
        self.assertEqual(prefix, "['prolog/main.pl'],parList([ral(200),mwe,aall,allInt]),")

    def test_resolve_local_problem_prefers_parser_specific_corpus(self):
        with TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir) / "ccg_sen_d"
            corpus_dir.mkdir(parents=True, exist_ok=True)

            (corpus_dir / "SICK_sen.pl").write_text(
                "sen_id(1, 1, 'p', 'unknown', 'A premise sentence').\n"
                "sen_id(2, 1, 'h', 'unknown', 'A hypothesis sentence').\n",
                encoding="utf-8",
            )
            (corpus_dir / "SICK_ccg.pl").write_text("ccg(1, noop).\n", encoding="utf-8")

            (corpus_dir / "SICK_train_trial_sen.pl").write_text(
                "sen_id(1, 11, 'p', 'unknown', 'A premise sentence').\n"
                "sen_id(2, 11, 'h', 'unknown', 'A hypothesis sentence').\n",
                encoding="utf-8",
            )
            (corpus_dir / "SICK_train_trial_eccg.pl").write_text("ccg(1, noop).\n", encoding="utf-8")

            resolved = _resolve_local_langpro_problem(
                ["A premise sentence"],
                "A hypothesis sentence",
                Path(tmp_dir),
                "easyccg",
            )

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.problem_id, "11")
            self.assertEqual(resolved.corpus.sen_relpath, "ccg_sen_d/SICK_train_trial_sen.pl")
            self.assertEqual(
                _local_langpro_ccg_file(resolved.corpus, "easyccg"),
                "ccg_sen_d/SICK_train_trial_eccg.pl",
            )

    def test_resolve_local_problem_supports_quoted_ids_for_snli(self):
        with TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir) / "ccg_sen_d"
            corpus_dir.mkdir(parents=True, exist_ok=True)

            (corpus_dir / "SNLI_dev_sen.pl").write_text(
                "sen_id(1, '386160015.jpg#3r1e', 'p', 'unknown', 'A dog runs').\n"
                "sen_id(2, '386160015.jpg#3r1e', 'h', 'unknown', 'An animal moves').\n",
                encoding="utf-8",
            )
            (corpus_dir / "SNLI_dev_eccg.pl").write_text("ccg(1, noop).\n", encoding="utf-8")

            resolved = _resolve_local_langpro_problem(
                ["A dog runs"],
                "An animal moves",
                Path(tmp_dir),
                "easyccg",
            )

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.problem_id, "386160015.jpg#3r1e")
            self.assertEqual(resolved.corpus.dataset_name, "snli")
            self.assertEqual(
                _format_local_langpro_problem_id(resolved.problem_id),
                "'386160015.jpg#3r1e'",
            )

    @patch("kbprojection.langpro.process_single_output")
    @patch("kbprojection.langpro._run_easyccg_async", new_callable=AsyncMock)
    async def test_write_easyccg_raw_file_preserves_sentence_numbering(
        self,
        mock_run_easyccg,
        mock_process_single_output,
    ):
        mock_run_easyccg.side_effect = ["raw-p1", "raw-p2", "raw-h"]
        mock_process_single_output.side_effect = ["term_p1", "term_p2", "term_h"]

        with TemporaryDirectory() as tmp_dir:
            ccg_path = Path(tmp_dir) / "ccg.pl"
            await _write_local_langpro_ccg_file(
                ccg_path,
                ["Premise one.", "Premise two."],
                "Hypothesis.",
                Path(tmp_dir),
                "en_core_web_sm",
                30,
            )

            self.assertEqual(
                [call.args[0] for call in mock_run_easyccg.call_args_list],
                ["Premise one.", "Premise two.", "Hypothesis."],
            )

            text = ccg_path.read_text(encoding="utf-8")
            self.assertIn("ccg(1,\nterm_p1\n).", text)
            self.assertIn("ccg(2,\nterm_p2\n).", text)
            self.assertIn("ccg(3,\nterm_h\n).", text)


if __name__ == "__main__":
    unittest.main()
