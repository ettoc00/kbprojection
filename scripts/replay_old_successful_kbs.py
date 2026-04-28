import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.settings import get_default_results_dir


DEFAULT_BASELINE = PROJECT_ROOT / "results_anthropic_claude-opus-4.5.json"
DEFAULT_OUTPUT = get_default_results_dir() / "old_successful_kb_replay.json"
DEFAULT_PROBE_CACHE = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "kbprojection"
    / "langpro_replay_probe.sqlite3"
)

CASES = {
    "2543017787.jpg#4r1e": ["isa_wn(child, kid)", "isa_wn(ride, play)"],
    "3228793611.jpg#4r1e": ["isa_wn(dig, play)"],
    "326456451.jpg#4r1e": ["isa_wn(rottweiler, animal)"],
    "vg_len46r5e": ["disj(white, black)"],
    "vg_len47r4e": ["disj(white, black)"],
    "vg_len66r1e": ["isa_wn(shiny, glitter)"],
}

ENDPOINTS = [
    "local://auto",
    "https://langpro-annotator.hum.uu.nl/langpro-api/prove/",
]


def load_items(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_case_problems(path: Path) -> Dict[str, Dict[str, Any]]:
    items = load_items(path)
    by_id = {}
    for item in items:
        problem = item.get("problem", {})
        problem_id = problem.get("id")
        if problem_id in CASES:
            by_id[problem_id] = {
                "problem": problem,
                "old": {
                    "pred_no_kb": item.get("pred_no_kb"),
                    "pred_with_raw_kb": item.get("pred_with_raw_kb"),
                    "pred_with_kb": item.get("pred_with_kb"),
                    "final_status": item.get("final_status"),
                    "kb_raw": item.get("kb_raw"),
                    "kb_filtered": item.get("kb_filtered"),
                },
            }
    missing = sorted(set(CASES) - set(by_id))
    if missing:
        raise ValueError(f"Baseline is missing requested case IDs: {', '.join(missing)}")
    return by_id


def label_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return str(value)


async def call_langpro(
    premises: List[str],
    hypothesis: str,
    kb: List[str],
    endpoint: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    from kbprojection.langpro import langpro_api_call

    try:
        result = await asyncio.wait_for(
            langpro_api_call(
                premises,
                hypothesis,
                endpoint=endpoint,
                kb=kb,
                report=False,
                timeout_seconds=timeout_seconds,
            ),
            timeout=timeout_seconds + 5,
        )
    except asyncio.TimeoutError:
        return {"label": None, "error": "timeout", "has_prover_trees": False}
    return {
        "label": label_value(result.label),
        "error": result.error,
        "has_prover_trees": bool(result.proofs),
    }


async def replay_case(
    problem: Dict[str, Any],
    kb: List[str],
    endpoint: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    premises = list(problem["premises"])
    hypothesis = problem["hypothesis"]
    no_kb = await call_langpro(premises, hypothesis, [], endpoint, timeout_seconds)
    with_kb = await call_langpro(premises, hypothesis, kb, endpoint, timeout_seconds)
    return {
        "endpoint": endpoint,
        "no_kb": no_kb,
        "with_kb": with_kb,
        "proof_useful": (
            no_kb["label"] != problem["gold_label"]
            and with_kb["label"] == problem["gold_label"]
        ),
    }


def classify_case(problem: Dict[str, Any], endpoint_results: List[Dict[str, Any]]) -> str:
    if any(
        result["no_kb"]["error"]
        or result["with_kb"]["error"]
        or result["no_kb"]["label"] in {None, "-"}
        or result["with_kb"]["label"] in {None, "-"}
        for result in endpoint_results
    ):
        return "unresolved"

    useful_by_endpoint = {
        result["endpoint"]: result["proof_useful"] for result in endpoint_results
    }
    labels_by_endpoint = {
        result["endpoint"]: (
            result["no_kb"]["label"],
            result["with_kb"]["label"],
        )
        for result in endpoint_results
    }
    if len(set(labels_by_endpoint.values())) > 1:
        return "remote/local discrepancy"
    if all(useful_by_endpoint.values()):
        return "current prompt/generation issue"
    if not any(useful_by_endpoint.values()):
        return "current LangPro drift"
    return "unresolved"


async def main_async() -> int:
    parser = argparse.ArgumentParser(
        description="Replay exact old successful KBs against local and remote LangPro."
    )
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--timeout-per-call", type=float, default=60.0)
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        help="Endpoint to check. Repeatable. Defaults to local://auto and remote LangPro.",
    )
    args = parser.parse_args()

    os.environ.setdefault("KBPROJECTION_LANGPRO_CACHE_PATH", str(DEFAULT_PROBE_CACHE))

    cases = find_case_problems(Path(args.baseline).resolve())
    endpoints = args.endpoint or ENDPOINTS
    results = []

    for problem_id, case in cases.items():
        problem = case["problem"]
        endpoint_results = []
        for endpoint in endpoints:
            endpoint_results.append(
                await replay_case(
                    problem,
                    CASES[problem_id],
                    endpoint,
                    args.timeout_per_call,
                )
            )
        classification = classify_case(problem, endpoint_results)
        item = {
            "id": problem_id,
            "premises": problem["premises"],
            "hypothesis": problem["hypothesis"],
            "gold_label": problem["gold_label"],
            "kb": CASES[problem_id],
            "old": case["old"],
            "endpoint_results": endpoint_results,
            "classification": classification,
        }
        results.append(item)
        print(
            f"{problem_id}: {classification} "
            + " ".join(
                f"{r['endpoint']} no={r['no_kb']['label']} kb={r['with_kb']['label']}"
                for r in endpoint_results
            )
        )

    payload = {
        "cache_path": os.environ["KBPROJECTION_LANGPRO_CACHE_PATH"],
        "results": results,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {output}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
