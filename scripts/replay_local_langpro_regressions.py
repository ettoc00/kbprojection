import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.settings import get_default_results_dir

DEFAULT_RESULTS = get_default_results_dir() / "results_6.4_openai_gpt-5-mini__legacy_icl.json"

REGRESSION_CASES: Dict[str, List[str]] = {
    "1792": ["isa_wn(walk, move around)"],
    "1497": ["isa_wn(strum, play)"],
    "2521878609.jpg#4r1e": ["isa_wn(kid, human)"],
    "1184967930.jpg#4r4e": ["isa_wn(field, grass)"],
    "2647049174.jpg#4r1e": ["isa_wn(toddler, baby)"],
    "2741990005.jpg#4r1e": ["isa_wn(vault, jump)"],
}

CONTROL_CASES: Dict[str, List[str]] = {
    "2281": ["isa_wn(opening, bite)", "isa_wn(open, bite)"],
    "4181": ["isa_wn(eat, bite)"],
    "4766": ["isa_wn(cook, roast)"],
    "5149": ["isa_wn(cord, rope)"],
    "3586": ["isa_wn(play drum, practice drum)", "isa_wn(play, practice)"],
}


def load_results(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Replay fixed local LangPro regression cases without any LLM call.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="Baseline results JSON to source exact problem texts from.")
    parser.add_argument("--regressions-only", action="store_true", help="Only run the known failing regression cases.")
    args = parser.parse_args()

    from kbprojection.langpro import clear_langpro_cache, langpro_api_call
    from kbprojection.models import NLILabel, NLIProblem

    os.environ["KBPROJECTION_LANGPRO_ENDPOINT"] = "local://auto"
    os.environ["KBPROJECTION_LANGPRO_CACHE_BACKEND"] = "memory"

    results_path = Path(args.results).resolve()
    if not results_path.exists():
        raise FileNotFoundError(
            f"Baseline results file not found: {results_path}. Pass --results to an existing JSON file."
        )
    items = load_results(results_path)
    problems = {
        item["problem"]["id"]: NLIProblem.model_validate(item["problem"])
        for item in items
    }

    selected_cases = dict(REGRESSION_CASES)
    if not args.regressions_only:
        selected_cases.update(CONTROL_CASES)

    clear_langpro_cache()

    failures = []
    for problem_id, kb in selected_cases.items():
        problem = problems[problem_id]
        result = await langpro_api_call(
            problem.premises,
            problem.hypothesis,
            kb=kb,
            report=True,
            timeout_seconds=180,
        )
        expected = problem.gold_label
        status = "PASS" if result.label == expected and not result.error else "FAIL"
        print(
            f"{status} {problem.dataset}/{problem.split}/{problem.id} "
            f"expected={expected.value} actual={result.label.value} kb={kb}"
        )
        if result.error or result.label != expected:
            failures.append(problem.id)

    if failures:
        print("\nFailures:")
        for problem_id in failures:
            print(problem_id)
        return 1

    print("\nAll selected local LangPro regression checks passed.")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
