from typing import List
from pydantic import BaseModel, Field

class KB_injection(BaseModel):
    """Data model for KB Injection"""

    KB_injection : str = Field(description = "One single KB injection of style: disj(work, rest) or isa_wn(apple, fruit). Only use the relations disj and isa_wn. Don't use underscores inside the relation (disj(in_air, sitting) and use max. 2 words")


class KB_injections(BaseModel):
    """Data model for list of KB Injections"""

    output: List[KB_injection] = Field(description = "List of KB injections. Make it a list of strings")

# Note: The following functions depend on the SNLI data being loaded.
# In the notebook, this is done via:
# from assigntools.LoLa.read_nli import snli_jsonl2dict
# SNLI, S2A = snli_jsonl2dict('snli_1.0', clean_labels=False)
# This will need to be handled by the user of the library.

def get_snli_problem(SNLI, key: str, split: str = "dev"):
    """
    Given a SNLI key (e.g., '386160015.jpg#3r1e'),
    return the full SNLI problem dictionary.
    """

    if split not in SNLI:
        raise ValueError(f"Unknown split: {split}. Must be one of {list(SNLI.keys())}")

    if key not in SNLI[split]:
        raise KeyError(f"Key '{key}' not found in SNLI[{split}]")

    return SNLI[split][key]

def random_snli_key(SNLI, split="dev", label_filter=None, used_keys=None):
    """
    Returns a random key from SNLI[split] that:
      - matches the label_filter (if provided)
      - is not in used_keys.
    """
    keys = list(SNLI[split].keys())

    if label_filter is not None:
        keys = [k for k in keys if SNLI[split][k]["g"] in label_filter]

    if used_keys is not None:
        keys = [k for k in keys if k not in used_keys]

    if not keys:
        return None

    return random.choice(keys)
