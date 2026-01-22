# Knowledge Base PROver inJECTION (kbprojection)

This library is designed to facilitate the use of Large Language Models (LLMs) to generate Knowledge Base (KB) injections for the LangPro prover. It provides tools for prompting LLMs, processing the generated KBs, and orchestrating experiments to evaluate the effectiveness of these injections.

## Installation

```bash
# Using poetry (recommended)
poetry install

# Or using pip
pip install .
```

## Usage

The library is divided into several modules:

*   `kbprojection.loaders`: Data loaders for SNLI and SICK datasets.
*   `kbprojection.models`: Pydantic models for type safety across the pipeline.
*   `kbprojection.prompts`: Manage and fill prompt templates.
*   `kbprojection.langpro`: Interface with the LangPro API.
*   `kbprojection.llm`: A unified interface for calling various LLMs (OpenAI, Anthropic, Gemini).
*   `kbprojection.filtering`: Functions to normalize and filter the generated KB injections.
*   `kbprojection.orchestration`: High-level functions to run experiments.

### Automatic Data Downloading

The dataset loaders (`SNLILoader` and `SICKLoader`) will automatically download the necessary data if it is not found in the specified directory. If no directory is specified, a temporary directory is used.

### Example

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