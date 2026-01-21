# Knowledge Base PROver inJECTION (kbprojection)

This library is designed to facilitate the use of Large Language Models (LLMs) to generate Knowledge Base (KB) injections for the LangPro prover. It provides tools for prompting LLMs, processing the generated KBs, and orchestrating experiments to evaluate the effectiveness of these injections.

## Installation

```bash
# Using uv (recommended)
uv sync

# Or install in development mode
uv pip install -e .
```

## Usage

The library is divided into several modules:

* `kbprojection.loaders`: Data loaders for SNLI and SICK datasets.
* `kbprojection.models`: Pydantic models for type safety across the pipeline.
* `kbprojection.prompts`: Manage and fill prompt templates.
* `kbprojection.langpro`: Interface with the LangPro API.
* `kbprojection.llm`: A unified interface for calling various LLMs (OpenAI, Anthropic, Gemini).
* `kbprojection.filtering`: Functions to normalize and filter the generated KB injections.
* `kbprojection.orchestration`: High-level functions to run experiments.

### Automatic Data Downloading

The dataset loaders (`SNLILoader` and `SICKLoader`) will automatically download the necessary data if it is not found in the specified directory. If no directory is specified, a temporary directory is used.

### Example: Loading a Single Problem

```python
from kbprojection import SNLILoader

# Initialize loader pointing to your data directory
# Ensure data is downloaded (runs automatically if not present)
loader = SNLILoader(data_dir="./data")

# Get a specific problem by ID (e.g., from SNLI dev set)
problem = loader.get_problem("4705552913.jpg#2r1n", split="dev")

print(f"Problem ID: {problem.id}")
print(f"Premise: {problem.premise}")
print(f"Hypothesis: {problem.hypothesis}")
print(f"Gold Label: {problem.gold_label}")
```

### Example: Full Experiment Orchestration

```python
from pathlib import Path
from kbprojection import collect_kb_helpful_examples_random, SNLILoader, SICKLoader
from kbprojection.models import ProblemConfig, TestMode

# Initialize dataset loader
# If data is not present in ./data/snli, it will be downloaded automatically.
snli_data = SNLILoader(data_dir="./data/snli")

# Configure the experiment
config = ProblemConfig(
    llm_provider="openai",
    model="gpt-4o",
    prompt_style="icl",
    test_mode=TestMode.BOTH,  # Test both raw LLM KB and filtered KB
    run_ablation=False,       # Set to True to find minimal set of injections
    verbose=True
)

results = collect_kb_helpful_examples_random(
    dataset=snli_data,
    config=config,
    split="dev",
    label_filter={"entailment", "contradiction"},
    max_matches=1,
    max_checked=10,
    cache_dir=Path("./cache")
)

# Inspect results (List[ExperimentResult])
for res in results:
    print(f"Problem {res.problem.id}: Fixed with KB: {res.kb_filtered}")
```

### Example: Manually Creating and Executing a Problem

You can also create a problem instance manually and process it through the pipeline.

```python
from kbprojection.models import NLIProblem, NLILabel, ProblemConfig
from kbprojection.orchestration import process_single_problem

# 1. Create a manual problem
manual_problem = NLIProblem(
    id="manual-test-1",
    premise="A dog is running in the park.",
    hypothesis="An animal is moving.",
    gold_label=NLILabel.ENTAILMENT,
    dataset="manual",
    split="test"
)

# 2. Process the problem
# This runs the full pipeline: No-KB check -> LLM generation -> Filtering -> Re-check
config = ProblemConfig(
    llm_provider="openai",  # or "anthropic", "gemini"
    model="gpt-4o",
    verbose=True
)

result = process_single_problem(manual_problem, config=config)

print(f"Final Status: {result.final_status}")
if result.kb_filtered:
    print(f"Generated KB: {result.kb_filtered}")
```

## Core Models

### NLIProblem

Represents a single NLI problem instance.

* `id`: Unique identifier for the problem.
* `premise`: The premise text.
* `hypothesis`: The hypothesis text.
* `gold_label`: The ground truth label (`entailment`, `contradiction`, or `neutral`).
* `dataset`: Source dataset name (e.g., "snli", "sick").
* `split`: Dataset split key (e.g., "train", "dev", "test").
* `original_data`: Dictionary containing original raw data from the dataset wrapper.

### ProblemConfig

Configuration object for the pipeline.

* `llm_provider`: String identifier for the LLM provider (e.g., "openai").
* `model`: Model identifier (e.g., "gpt-4o").
* `prompt_style`: Identifier for the prompt template style.
* `post_process`: Boolean; if `True`, applies post-processing to LLM output.
* `test_mode`: `TestMode` enum controlling which stages to run (`no_kb`, `raw_kb`, `filtered`, `both`, `full`).
* `run_ablation`: Boolean; if `True`, runs ablation to find all minimal sufficient KB subsets.
* `verbose`: Boolean; enables detailed logging.

### ExperimentResult

Encapsulates the results of running the pipeline on a problem.

* `problem`: The `NLIProblem` instance being processed.

* `kb_raw`: List of raw KB strings generated by the LLM.
* `kb_filtered`: List of filtered/formatted KB strings ready for LangPro.
* `kb_details`: List of `KBResult` objects containing detailed provenance for each injection.

* `pred_no_kb`: Prediction from LangPro without any KB injection.
* `status_no_kb`: Status of the baseline step.

* `pred_with_raw_kb`: Prediction using the raw (unfiltered) KB.
* `status_with_raw_kb`: Status of the raw KB evaluation step.
* `pred_with_kb`: Prediction using the filtered KB.
* `status_with_kb`: Status of the filtered KB evaluation step.

* `final_status`: `ExperimentStatus` enum summarizing the overall outcome (e.g., `FIXED`, `STILL_WRONG`).
* `fixed_by`: String indicating which KB version fixed the problem (`"raw_kb"`, `"filtered_kb"`, or `"both"`).
* `essential_kb`: Best minimal sufficient KB subset (ranked by token count). If ablation was run and multiple KB entries are redundant, this contains the simplest subset that alone fixes the problem.
* `ablation_subsets`: List of all minimal sufficient subsets found during ablation. Each subset is a list of KB strings that independently can fix the problem.
* `ablation_results`: Dictionary mapping tested subsets (as tuples) to their resulting label.
