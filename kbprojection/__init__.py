"""
Knowledge Base PROver inJECTION (kbprojection)
"""

__version__ = "0.4.0"

from .prompts import get_prompt, fill_prompt
from .models import (
    NLIProblem, 
    NLILabel, 
    LangProResult, 
    ExperimentResult,
    ExperimentStepStatus
)
from .loaders.base import DatasetLoader
from .loaders.snli import SNLILoader
from .loaders.sick import SICKLoader

from .langpro import langpro_api_call
from .llm import call_llm
from .filtering import (
    tokenize,
    remove_underscores,
    normalize_kb_args,
    drop_leading_preposition,
    parse_kb_injection,
    filter_kb_by_prem_hyp,
)
from .orchestration import collect_kb_helpful_examples_random
from .utils import get_smallest_problems