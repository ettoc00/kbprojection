import json
import unittest
from pathlib import Path


class TestShowcaseNotebookPublicSurface(unittest.TestCase):
    def test_notebook_uses_public_runner_api(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "nli_showcase_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )

        for internal_name in [
            "canonical_problem_id",
            "create_async_run_context",
            "AsyncRunLimits",
            "asyncio.Semaphore",
            "asyncio.as_completed",
            "ProblemConfig(",
            "model_dump_json",
        ]:
            self.assertNotIn(internal_name, source)

        for public_name in [
            "SHOWCASE_PROBLEM_IDS",
            "SICKLoader",
            "SNLILoader",
            "arun_problems",
        ]:
            self.assertIn(public_name, source)


if __name__ == "__main__":
    unittest.main()
