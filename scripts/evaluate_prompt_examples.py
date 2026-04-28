import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.settings import get_default_results_dir

DEFAULT_DATASET = PROJECT_ROOT / "tests" / "fixtures" / "prompt_examples_icl.json"

def load_items(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def label_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return str(value)


async def evaluate_item(item: Dict[str, Any]) -> Dict[str, Any]:
    from kbprojection.filtering import pipeline_filter_kb_injections
    from kbprojection.langpro import langpro_api_call
    from kbprojection.models import NLIProblem

    problem = NLIProblem.model_validate(
        {
            "id": item["id"],
            "premises": item["premises"],
            "hypothesis": item["hypothesis"],
            "gold_label": item["gold_label"],
            "dataset": item.get("dataset", "prompt_examples"),
            "split": item.get("split", "icl"),
        }
    )
    kb_raw = list(item.get("kb_raw", []))

    no_kb = await langpro_api_call(problem.premises, problem.hypothesis, report=False)

    filtered_details = pipeline_filter_kb_injections(
        kb_raw,
        problem.premises,
        problem.hypothesis,
        post_process=True,
    )
    kb_filtered = [detail.relation for detail in filtered_details]

    raw_result = None
    if kb_raw:
        raw_result = await langpro_api_call(problem.premises, problem.hypothesis, kb=kb_raw, report=False)

    filtered_result = None
    if kb_filtered:
        if set(kb_filtered) == set(kb_raw) and raw_result is not None:
            filtered_result = raw_result
        else:
            filtered_result = await langpro_api_call(problem.premises, problem.hypothesis, kb=kb_filtered, report=False)

    gold = problem.gold_label.value
    pred_no_kb = label_value(no_kb.label)
    pred_raw = label_value(raw_result.label) if raw_result else None
    pred_filtered = label_value(filtered_result.label) if filtered_result else None

    baseline_correct = pred_no_kb == gold
    raw_useful = bool(kb_raw) and (not baseline_correct) and pred_raw == gold
    filtered_useful = bool(kb_filtered) and (not baseline_correct) and pred_filtered == gold

    if baseline_correct:
        verdict = "already_solved_without_kb"
    elif raw_useful or filtered_useful:
        verdict = "kb_helpful"
    elif not kb_raw:
        verdict = "no_example_kb"
    elif not kb_filtered:
        verdict = "filtered_out"
    else:
        verdict = "kb_not_helpful"

    return {
        "id": problem.id,
        "premises": problem.premises,
        "hypothesis": problem.hypothesis,
        "gold_label": gold,
        "kb_raw": kb_raw,
        "kb_filtered": kb_filtered,
        "pred_no_kb": pred_no_kb,
        "pred_with_raw_kb": pred_raw,
        "pred_with_kb": pred_filtered,
        "baseline_correct": baseline_correct,
        "raw_useful": raw_useful,
        "filtered_useful": filtered_useful,
        "verdict": verdict,
        "errors": {
            "no_kb": no_kb.error,
            "raw_kb": raw_result.error if raw_result else None,
            "filtered_kb": filtered_result.error if filtered_result else None,
        },
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "total": len(results),
        "already_solved_without_kb": 0,
        "kb_helpful": 0,
        "kb_not_helpful": 0,
        "filtered_out": 0,
        "no_example_kb": 0,
    }
    for item in results:
        summary[item["verdict"]] += 1
    return summary


async def main_async() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate whether the current ICL prompt examples are actually useful for LangPro."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the JSON dataset containing prompt examples.",
    )
    parser.add_argument(
        "--output",
        default=str(get_default_results_dir() / "prompt_examples_icl_eval.json"),
        help="Where to write the evaluation JSON.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()

    items = load_items(dataset_path)
    results = [await evaluate_item(item) for item in items]
    payload = {
        "dataset": str(dataset_path),
        "summary": summarize(results),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {output_path}")
    print(json.dumps(payload["summary"], indent=2))
    for item in results:
        print(
            f"{item['id']}: verdict={item['verdict']} "
            f"no_kb={item['pred_no_kb']} raw={item['pred_with_raw_kb']} filtered={item['pred_with_kb']}"
        )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
