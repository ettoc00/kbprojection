from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Forward references handled by Pydantic
from pydantic import BaseModel, Field
from enum import Enum

class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"
    UNKNOWN = "-"

class NLIProblem(BaseModel):
    """
    Represents a single NLI problem (Premises-Hypothesis pair).
    Normalized structure to be used across different datasets.
    """
    id: str
    premises: List[str]  # List of premise sentences
    hypothesis: str
    gold_label: NLILabel
    dataset: str
    split: str
    original_data: Optional[Dict[str, Any]] = Field(default=None, description="Original raw data from the dataset wrapper")



class LLMKBInjection(BaseModel):
    """
    Raw KB injection string from LLM response.
    Matches the prompt expectation of 'predicate(arg1, arg2)' string.
    """
    KB_injection: str = Field(description = "One single KB injection of style: disj(work, rest) or isa_wn(apple, fruit).")

class LLMKBResponse(BaseModel):
    """
    Structure for LLM output parsing.
    """
    output: List[LLMKBInjection] = Field(description = "List of KB injections.")

class LangProResult(BaseModel):
    """
    Result returned by the LangPro prover.
    """
    label: NLILabel
    kb: Any = None
    ccg_trees: List[Any] = []
    ccg_terms: List[Any] = []
    terms: List[Any] = []
    llfs: List[Any] = []
    proofs: Dict[str, Any] = {}
    error: Optional[str] = None

class ExperimentStepStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    ERROR = "error"

class ExperimentStatus(str, Enum):
    UNKNOWN = "unknown"
    ALREADY_CORRECT = "already_correct"
    FIXED = "fixed"
    FIXED_RAW_KB = "fixed_raw_kb"
    STILL_WRONG = "still_wrong"
    STILL_WRONG_RAW_KB = "still_wrong_raw_kb"
    ERROR_NO_KB = "error_no_kb"
    ERROR_WITH_KB = "error_with_kb"
    LLM_ERROR = "llm_error"
    EMPTY_KB_AFTER_FILTER = "empty_kb_after_filter"


class ExperimentResult(BaseModel):
    """
    The result of running the projection pipeline on a single problem.
    """
    problem: NLIProblem
    
    # Step 1: No KB
    pred_no_kb: Optional[NLILabel] = None
    status_no_kb: ExperimentStepStatus = ExperimentStepStatus.PENDING
    
    # Step 2: Generation
    kb_raw: Optional[List[str]] = None # List of strings as returned by LLM
    
    # Step 3: Filtering
    kb_filtered: Optional[List[str]] = None # List of formatted strings ready for LangPro
    
    # Step 4: With raw KB (unfiltered)
    pred_with_raw_kb: Optional[NLILabel] = None
    status_with_raw_kb: ExperimentStepStatus = ExperimentStepStatus.PENDING
    
    # Step 5: With filtered KB
    pred_with_kb: Optional[NLILabel] = None
    status_with_kb: ExperimentStepStatus = ExperimentStepStatus.PENDING
    
    # Provenance details
    kb_details: Optional[List["KBResult"]] = None

    # Ablation results
    essential_kb: Optional[List[str]] = None  # Best minimal subset (by token count)
    ablation_subsets: Optional[List[List[str]]] = None  # All minimal sufficient subsets
    ablation_results: Optional[Dict[str, NLILabel]] = None  # Detailed log of tested subsets

    # Which KB type fixed the problem
    fixed_by: Optional[str] = None  # "raw_kb", "filtered_kb", "both", or None

    # Prover call history
    prover_calls: Optional[List["LangProResult"]] = None

    # Overall Outcome
    final_status: ExperimentStatus = ExperimentStatus.UNKNOWN


class TestMode(str, Enum):
    """
    Controls which KB injection stages to test.
    """
    NO_KB = "no_kb"           # Only test without KB
    RAW_KB = "raw_kb"         # Test with raw LLM output (unfiltered)
    FILTERED_KB = "filtered"  # Test with filtered KB only
    BOTH = "both"             # Test both raw and filtered KB
    FULL = "full"             # Full pipeline (default, same as current behavior)


class ProblemConfig(BaseModel):
    """
    Configuration for processing a single NLI problem.
    Encapsulates all options for the KB injection pipeline.
    """
    # LLM settings
    llm_provider: str = "openai"
    model: str = "gpt-5-mini"
    prompt_style: str = "icl"
    
    # Processing options
    post_process: bool = True
    
    # Test mode
    test_mode: TestMode = TestMode.FULL
    
    # Ablation
    run_ablation: bool = False
    
    # Output options
    verbose: bool = True

class KBResult(BaseModel):
    """
    Detailed result of a KB injection with provenance.
    """
    relation: str
    provenance: str = "llm"  # "llm", "post_process", "derived_swap", "derived_diff"
    original_text: Optional[str] = None

    def __str__(self):
        return self.relation