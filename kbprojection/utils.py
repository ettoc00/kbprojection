import re
from typing import List, Set
from .loaders.base import DatasetLoader


def sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def get_smallest_problems(dataset: DatasetLoader, split: str = "test", limit: int = 2000, label_filter: Set[str] = {"entailment"}) -> List[str]:
    """
    Returns a list of problem IDs from the specified split, sorted by
    total character length (premise + hypothesis), and filtered by label.
    """
    candidates = []
    print(f"[get_smallest] Filtering split='{split}' for labels={label_filter}...")

    for prob in dataset.iter_problems(split=split):
        if prob.gold_label not in label_filter:
            continue

        size = sum(len(p) for p in prob.premises) + len(prob.hypothesis)
        candidates.append((size, prob.id))

    candidates.sort(key=lambda x: x[0])

    selected_keys = [c[1] for c in candidates[:limit]]

    print(f"[get_smallest] Found {len(candidates)} matching problems. Returning smallest {len(selected_keys)}.")
    return selected_keys
