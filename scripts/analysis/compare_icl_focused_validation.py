import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.settings import get_default_results_dir


WIN_STATUS_SETS = {
    "raw": {"raw_kb_solved"},
    "any_kb": {"raw_kb_solved", "normalised_kb_solved"},
}


def load_items(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["items_by_key"]


def is_win(item: Dict[str, Any], win_statuses: set[str]) -> bool:
    return item.get("final_status") in win_statuses


def count_status(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(item.get("final_status") or "unknown") for item in items).items()))


def compare_model(
    old_items: Dict[str, Dict[str, Any]],
    new_items: Dict[str, Dict[str, Any]],
    win_statuses: set[str],
) -> Dict[str, Any]:
    keys = set(old_items) & set(new_items)
    old_solved = {key for key in keys if is_win(old_items[key], win_statuses)}
    new_solved = {key for key in keys if is_win(new_items[key], win_statuses)}
    return {
        "completed_common_keys": len(keys),
        "old_kb_solved": len(old_solved),
        "new_kb_solved": len(new_solved),
        "delta_kb_solved": len(new_solved) - len(old_solved),
        "preserved_old_wins": len(old_solved & new_solved),
        "lost_old_wins": len(old_solved - new_solved),
        "new_wins": len(new_solved - old_solved),
        "old_status_counts": count_status(old_items[key] for key in keys),
        "new_status_counts": count_status(new_items[key] for key in keys),
        "lost_old_win_keys": sorted(old_solved - new_solved)[:50],
        "new_win_keys": sorted(new_solved - old_solved)[:50],
    }


def parse_args() -> argparse.Namespace:
    results_dir = get_default_results_dir()
    parser = argparse.ArgumentParser(description="Compare revised focused ICL validation runs against old full-run outcomes.")
    parser.add_argument(
        "--old-gpt54",
        type=Path,
        default=results_dir / "sick_full__icl__openrouter__openai_gpt-5.4.json",
    )
    parser.add_argument(
        "--old-opus47",
        type=Path,
        default=results_dir / "sick_full__icl__openrouter__anthropic_claude-opus-4.7.json",
    )
    parser.add_argument(
        "--new-gpt54",
        type=Path,
        default=results_dir / "icl_focused_validation__openrouter_safe__openrouter__openai_gpt-5.4.json",
    )
    parser.add_argument(
        "--new-opus47",
        type=Path,
        default=results_dir / "icl_focused_validation__openrouter_safe__openrouter__anthropic_claude-opus-4.7.json",
    )
    parser.add_argument("--output", type=Path, default=results_dir / "icl_focused_validation_comparison.json")
    parser.add_argument(
        "--win-mode",
        choices=sorted(WIN_STATUS_SETS),
        default="raw",
        help="Which statuses count as wins. Defaults to raw per current prompt-validation criterion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old_gpt = load_items(args.old_gpt54)
    old_opus = load_items(args.old_opus47)
    new_gpt = load_items(args.new_gpt54)
    new_opus = load_items(args.new_opus47)
    win_statuses = WIN_STATUS_SETS[args.win_mode]
    keys = set(new_gpt) & set(new_opus)
    old_union = {key for key in keys if is_win(old_gpt[key], win_statuses) or is_win(old_opus[key], win_statuses)}
    new_union = {key for key in keys if is_win(new_gpt[key], win_statuses) or is_win(new_opus[key], win_statuses)}
    payload = {
        "inputs": {name: str(value) for name, value in vars(args).items() if name != "output"},
        "win_mode": args.win_mode,
        "win_statuses": sorted(win_statuses),
        "gpt54": compare_model(old_gpt, new_gpt, win_statuses),
        "opus47": compare_model(old_opus, new_opus, win_statuses),
        "union": {
            "completed_common_keys": len(keys),
            "old_any_model_kb_solved": len(old_union),
            "new_any_model_kb_solved": len(new_union),
            "delta_any_model_kb_solved": len(new_union) - len(old_union),
            "preserved_old_union_wins": len(old_union & new_union),
            "lost_old_union_wins": len(old_union - new_union),
            "new_union_wins": len(new_union - old_union),
            "lost_old_union_win_keys": sorted(old_union - new_union)[:50],
            "new_union_win_keys": sorted(new_union - old_union)[:50],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps({k: payload[k] for k in ("gpt54", "opus47", "union")}, indent=2)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
