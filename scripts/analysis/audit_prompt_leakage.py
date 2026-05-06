import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.prompts import get_prompt
from kbprojection.settings import get_default_results_dir


EXAMPLE_RE = re.compile(
    r"Premise:\s*(?P<premise>.*?)\n"
    r"Hypothesis:\s*(?P<hypothesis>.*?)\n"
    r"\[KB_START\]",
    re.DOTALL,
)


def normalized_sentence(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def extract_prompt_examples(prompt: str) -> List[Dict[str, str]]:
    examples = []
    for index, match in enumerate(EXAMPLE_RE.finditer(prompt), start=1):
        examples.append(
            {
                "index": index,
                "premise": re.sub(r"\s+", " ", match.group("premise").strip()),
                "hypothesis": re.sub(r"\s+", " ", match.group("hypothesis").strip()),
            }
        )
    return examples


def prompt_without_examples(prompt: str) -> str:
    return EXAMPLE_RE.sub("Premise: \nHypothesis: \n[KB_START]", prompt)


def load_sick_rows(splits: Sequence[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    from kbprojection.loaders.sick import SICKLoader

    loader = SICKLoader()
    loader.load(splits=list(splits))
    rows = []
    errors = []
    for split in splits:
        try:
            for problem in loader.iter_problems(split=split):
                rows.append(
                    {
                        "dataset": "sick",
                        "split": split,
                        "id": problem.id,
                        "premise": problem.premises[0] if problem.premises else "",
                        "hypothesis": problem.hypothesis,
                    }
                )
        except Exception as exc:
            errors.append(f"sick:{split}: {type(exc).__name__}: {exc}")
    return rows, errors


def load_snli_rows(splits: Sequence[str], allow_download: bool) -> Tuple[List[Dict[str, str]], List[str]]:
    from kbprojection.loaders.snli import SNLILoader

    loader = SNLILoader()
    rows = []
    errors = []
    try:
        if allow_download:
            loader.load(splits=list(splits))
    except Exception as exc:
        errors.append(f"snli:load: {type(exc).__name__}: {exc}")
        return rows, errors

    for split in splits:
        try:
            if not loader._get_file_path(split).exists():
                errors.append(f"snli:{split}: file not available")
                continue
            if split not in loader._data:
                loader.load(splits=[split])
            for problem in loader.iter_problems(split=split):
                rows.append(
                    {
                        "dataset": "snli",
                        "split": split,
                        "id": problem.id,
                        "premise": problem.premises[0] if problem.premises else "",
                        "hypothesis": problem.hypothesis,
                    }
                )
        except Exception as exc:
            errors.append(f"snli:{split}: {type(exc).__name__}: {exc}")
    return rows, errors


def build_dataset_indexes(rows: Iterable[Dict[str, str]]) -> tuple[dict[tuple[str, str], List[Dict[str, str]]], dict[str, List[Dict[str, str]]]]:
    pairs: dict[tuple[str, str], List[Dict[str, str]]] = {}
    sentences: dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        premise = normalized_sentence(row["premise"])
        hypothesis = normalized_sentence(row["hypothesis"])
        pairs.setdefault((premise, hypothesis), []).append(row)
        if premise:
            sentences.setdefault(premise, []).append(row)
        if hypothesis:
            sentences.setdefault(hypothesis, []).append(row)
    return pairs, sentences


def find_example_leakage(
    examples: Sequence[Dict[str, str]],
    pair_index: dict[tuple[str, str], List[Dict[str, str]]],
    sentence_index: dict[str, List[Dict[str, str]]],
) -> List[Dict[str, Any]]:
    findings = []
    for example in examples:
        premise = normalized_sentence(example["premise"])
        hypothesis = normalized_sentence(example["hypothesis"])
        pair_matches = pair_index.get((premise, hypothesis), [])
        if pair_matches:
            findings.append({"type": "example_pair_match", "example": example, "matches": pair_matches[:10]})
        for role, sentence in (("premise", premise), ("hypothesis", hypothesis)):
            matches = sentence_index.get(sentence, [])
            if matches:
                findings.append(
                    {
                        "type": "example_sentence_match",
                        "role": role,
                        "example": example,
                        "matches": matches[:10],
                    }
                )
    return findings


def find_rule_text_leakage(prompt: str, rows: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    prompt_body = normalized_sentence(prompt_without_examples(prompt))
    findings = []
    seen = set()
    for row in rows:
        for role in ("premise", "hypothesis"):
            sentence = normalized_sentence(row[role])
            if len(sentence) < 15 or sentence in seen:
                continue
            seen.add(sentence)
            if sentence and sentence in prompt_body:
                findings.append({"type": "rule_text_sentence_match", "role": role, "sentence": sentence, "match": row})
                if len(findings) >= 50:
                    return findings
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit get_prompt('icl') examples for SICK/SNLI leakage.")
    parser.add_argument("--output", type=Path, default=get_default_results_dir() / "icl_prompt_leakage_audit.json")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], choices=["train", "dev", "test"])
    parser.add_argument(
        "--no-snli-download",
        action="store_true",
        help="Only check SNLI if local files already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = get_prompt("icl")
    examples = extract_prompt_examples(prompt)
    sick_rows, sick_errors = load_sick_rows(args.splits)
    snli_rows, snli_errors = load_snli_rows(args.splits, allow_download=not args.no_snli_download)
    rows = sick_rows + snli_rows
    pair_index, sentence_index = build_dataset_indexes(rows)

    findings = []
    findings.extend(find_example_leakage(examples, pair_index, sentence_index))
    findings.extend(find_rule_text_leakage(prompt, rows))

    payload = {
        "prompt": "icl",
        "example_count": len(examples),
        "dataset_rows_checked": {
            "sick": len(sick_rows),
            "snli": len(snli_rows),
            "total": len(rows),
        },
        "dataset_errors": sick_errors + snli_errors,
        "findings": findings,
        "passed": not findings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if findings:
        print(f"FAIL: found {len(findings)} prompt leakage finding(s). Wrote {args.output}")
        return 1

    print(
        "PASS: no prompt example leakage found "
        f"across {payload['dataset_rows_checked']['total']} rows. Wrote {args.output}"
    )
    if payload["dataset_errors"]:
        print("Dataset warnings:")
        for error in payload["dataset_errors"]:
            print(f"- {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
