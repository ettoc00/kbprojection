"""
Knowledge Base PROver inJECTION (kbprojection)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kbprojection")
except PackageNotFoundError:
    __version__ = "0.5.0"

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
from .easyccg_vendor import install_local_easyccg

from .langpro import (
    clear_langpro_cache,
    get_langpro_cache_backend,
    langpro_api_call,
    set_langpro_cache_backend,
)
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
from .runners import (
    arun_problem,
    arun_problems,
    infer_provider,
    run_problem,
    run_problems,
    serialize_result_payload,
)
from .utils import get_smallest_problems, sanitize_filename_part
