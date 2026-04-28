import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.models import NLILabel
from kbprojection.settings import get_default_results_dir


DEFAULT_OUTPUT = get_default_results_dir() / "icl_example_audit.json"
DEFAULT_ENDPOINT = "local://auto"

SEMANTIC_NOTES = {
    1: "Reasonable paraphrase, but only worth keeping if LangPro uses it.",
    2: "Defensible event bridge: strumming a guitar is playing guitar.",
    3: "Over-specific phrase bridge; weak if the pair is already solved without KB.",
    4: "Useful prover bridge in some parses, but field is not strictly grass.",
    5: "Toddler/baby overlap is context-sensitive; weak as a general lexical rule.",
    6: "Defensible event bridge: vaulting entails jumping in this context.",
    7: "Eat does not generally entail bite, and the pair may already be solved.",
    8: "Defensible noun synonym/near-synonym bridge.",
    9: "Defensible disjointness for the open/closed state contrast.",
}


def extract_icl_examples() -> List[Dict[str, Any]]:
    from kbprojection.prompts import get_prompt

    prompt = get_prompt("icl")
    pattern = re.compile(
        r"### Example (?P<number>\d+): (?P<title>[^\n]+)\n"
        r"Premise: (?P<premise>.*?)\n"
        r"Hypothesis: (?P<hypothesis>.*?)\n"
        r"\[KB_START\]\n"
        r"(?P<kb>.*?)\n"
        r"\[KB_END\]",
        re.DOTALL,
    )
    examples = []
    for match in pattern.finditer(prompt):
        kb = [line.strip() for line in match.group("kb").splitlines() if line.strip()]
        number = int(match.group("number"))
        examples.append(
            {
                "example_number": number,
                "id": f"icl_example_{number}",
                "title": match.group("title").strip(),
                "premise": match.group("premise").strip(),
                "hypothesis": match.group("hypothesis").strip(),
                "kb": kb,
            }
        )
    return examples


def expected_label_for(example: Dict[str, Any]) -> str:
    if any(relation.startswith("disj(") for relation in example["kb"]):
        return NLILabel.CONTRADICTION.value
    return NLILabel.ENTAILMENT.value


def label_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return str(value)


def has_prover_trees(result: Any) -> bool:
    if result is None:
        return False
    return bool(getattr(result, "proofs", None))


def cached_langpro_result(
    premises: List[str],
    hypothesis: str,
    kb: List[str],
    endpoint: str,
    parser: str,
    ral: int,
    senses: str,
    strong_align: bool,
    intersective: bool,
) -> Any:
    from kbprojection.langpro import (
        _local_endpoint_cache_key,
        _make_langpro_cache_key,
        _parse_langpro_output,
        get_langpro_cache_backend,
    )
    from kbprojection.settings import get_langpro_settings

    settings = get_langpro_settings()
    cache_endpoint = _local_endpoint_cache_key(endpoint, settings)
    cache_key = _make_langpro_cache_key(
        premises,
        hypothesis,
        cache_endpoint,
        parser,
        ral,
        kb,
        senses,
        strong_align,
        intersective,
    )
    cached = get_langpro_cache_backend().get(cache_key)
    if cached is None:
        return None
    return _parse_langpro_output(json.loads(cached))


async def live_langpro_result(
    premises: List[str],
    hypothesis: str,
    kb: List[str],
    endpoint: str,
    parser: str,
    ral: int,
    senses: str,
    strong_align: bool,
    intersective: bool,
    timeout_seconds: float,
) -> Any:
    from kbprojection.langpro import langpro_api_call

    return await asyncio.wait_for(
        langpro_api_call(
            premises,
            hypothesis,
            endpoint=endpoint,
            parser=parser,
            ral=ral,
            kb=kb,
            senses=senses,
            strong_align=strong_align,
            intersective=intersective,
            report=False,
            timeout_seconds=timeout_seconds,
        ),
        timeout=timeout_seconds + 5,
    )


def recommendation(
    example_number: int,
    no_kb_label: Optional[str],
    with_kb_label: Optional[str],
    expected_label: str,
    cache_status: str,
) -> str:
    if cache_status in {"needs_live_check", "unverified"}:
        return "verify manually"
    if no_kb_label == expected_label:
        return "replace"
    if with_kb_label != expected_label:
        return "replace"
    if example_number in {4, 5}:
        return "verify manually"
    if example_number == 7:
        return "replace"
    return "keep"


def classify(
    example: Dict[str, Any],
    no_kb_result: Any,
    with_kb_result: Any,
    cache_status: str,
) -> Dict[str, Any]:
    expected = expected_label_for(example)
    no_kb_label = label_value(getattr(no_kb_result, "label", None))
    with_kb_label = label_value(getattr(with_kb_result, "label", None))
    proof_useful = (
        no_kb_label is not None
        and with_kb_label is not None
        and no_kb_label != expected
        and with_kb_label == expected
    )
    return {
        **example,
        "gold_label": expected,
        "no_kb_prediction": no_kb_label,
        "with_kb_prediction": with_kb_label,
        "proof_useful": proof_useful if cache_status == "checked" else None,
        "has_prover_trees": has_prover_trees(with_kb_result),
        "semantic_note": SEMANTIC_NOTES.get(example["example_number"], ""),
        "recommendation": recommendation(
            example["example_number"],
            no_kb_label,
            with_kb_label,
            expected,
            cache_status,
        ),
        "cache_status": cache_status,
        "errors": {
            "no_kb": getattr(no_kb_result, "error", None),
            "with_kb": getattr(with_kb_result, "error", None),
        },
    }


async def audit_example(
    example: Dict[str, Any],
    args: argparse.Namespace,
    allow_live: bool,
) -> Dict[str, Any]:
    premises = [example["premise"]]
    no_kb = cached_langpro_result(
        premises,
        example["hypothesis"],
        [],
        args.endpoint,
        args.parser,
        args.ral,
        args.senses,
        args.strong_align,
        args.intersective,
    )
    with_kb = cached_langpro_result(
        premises,
        example["hypothesis"],
        example["kb"],
        args.endpoint,
        args.parser,
        args.ral,
        args.senses,
        args.strong_align,
        args.intersective,
    )

    if no_kb is not None and with_kb is not None:
        return classify(example, no_kb, with_kb, "checked")

    if args.cache_only or not allow_live:
        return classify(example, no_kb, with_kb, "needs_live_check")

    try:
        if no_kb is None:
            no_kb = await live_langpro_result(
                premises,
                example["hypothesis"],
                [],
                args.endpoint,
                args.parser,
                args.ral,
                args.senses,
                args.strong_align,
                args.intersective,
                args.timeout_per_example,
            )
        if with_kb is None:
            with_kb = await live_langpro_result(
                premises,
                example["hypothesis"],
                example["kb"],
                args.endpoint,
                args.parser,
                args.ral,
                args.senses,
                args.strong_align,
                args.intersective,
                args.timeout_per_example,
            )
    except asyncio.TimeoutError:
        return classify(example, no_kb, with_kb, "unverified")

    return classify(example, no_kb, with_kb, "checked")


def print_summary(results: List[Dict[str, Any]]) -> None:
    for item in results:
        print(
            f"{item['example_number']}. {item['kb']} "
            f"no_kb={item['no_kb_prediction']} with_kb={item['with_kb_prediction']} "
            f"status={item['cache_status']} useful={item['proof_useful']} "
            f"recommendation={item['recommendation']}"
        )


async def main_async() -> int:
    parser = argparse.ArgumentParser(
        description="Audit examples embedded in get_prompt('icl') without replacing the prompt."
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--parser", default="easyccg")
    parser.add_argument("--ral", type=int, default=200)
    parser.add_argument("--senses", default="all")
    parser.add_argument("--strong-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intersective", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-live-checks", type=int, default=0)
    parser.add_argument("--timeout-per-example", type=float, default=60.0)
    parser.add_argument(
        "--live-example",
        type=int,
        action="append",
        default=[],
        help="Example number to live-check if missing from cache. Repeatable.",
    )
    parser.add_argument(
        "--expect-count",
        type=int,
        default=None,
        help="Fail if the prompt contains a different number of examples.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    examples = extract_icl_examples()
    live_budget = args.max_live_checks
    results = []
    live_targets = set(args.live_example)

    for example in examples:
        allow_live = False
        if not args.cache_only:
            if example["example_number"] in live_targets:
                allow_live = True
            elif not live_targets and live_budget > 0:
                allow_live = True

        result = await audit_example(example, args, allow_live)
        if allow_live and result["cache_status"] in {"checked", "unverified"}:
            live_budget -= 1
        results.append(result)

    payload = {
        "endpoint": args.endpoint,
        "cache_only": args.cache_only,
        "results": results,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {output}")
    print_summary(results)
    if args.expect_count is not None and len(results) != args.expect_count:
        print(f"Expected {args.expect_count} examples, found {len(results)}.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
