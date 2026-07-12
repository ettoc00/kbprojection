# Knowledge Base PROver inJECTION (kbprojection)

This library is designed to facilitate the use of Large Language Models (LLMs) to generate Knowledge Base (KB) injections for the LangPro prover. It provides tools for prompting LLMs, processing the generated KBs, and orchestrating experiments to evaluate the effectiveness of these injections.

## Installation

```bash
# Using uv (recommended)
uv sync

# Or install in development mode
uv pip install -e .
```

## Runtime configuration: local or Google Colab

The package can run with local paths or Drive-backed Colab paths through one
shared runtime helper. The switch is controlled by environment variables:

* `KBPROJECTION_RUNTIME`: `local` or `colab`. If unset, Colab is auto-detected.
* `KBPROJECTION_PROJECT_ROOT`: base project directory.
* `KBPROJECTION_DATA_DIR`: optional override for datasets.
* `KBPROJECTION_CACHE_ROOT`: optional override for experiment JSON caches.
* `KBPROJECTION_RESULTS_DIR`: optional override for CSV/JSONL outputs.
* `KBPROJECTION_THIRD_PARTY_CACHE`: optional override for NLTK, Hugging Face,
  sentence-transformers, and Torch caches.

Local usage usually needs no extra setup:

```python
from kbprojection.runtime import configure_runtime

paths = configure_runtime()
DATA_DIR = paths.data_dir
CACHE_ROOT = paths.cache_root
RESULTS_DIR = paths.results_dir
```

In Google Colab, mount Drive and point the project root at your copied repo:

```python
from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/kbprojection")
os.environ["KBPROJECTION_RUNTIME"] = "colab"
os.environ["KBPROJECTION_PROJECT_ROOT"] = str(PROJECT_ROOT)
```

Then install and configure:

```python
%cd /content/drive/MyDrive/kbprojection
%pip install -r requirements-colab.txt -e .

from kbprojection.runtime import configure_runtime
paths = configure_runtime(project_root=PROJECT_ROOT)
```

This keeps datasets, experiment caches, result files, NLTK data, Hugging Face
models, sentence-transformer models, and Torch cache files under the project
root instead of Colab's temporary VM storage. See
`kbprojection_colab_setup.ipynb` for a ready-to-run Colab bootstrap notebook.

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
from kbprojection.runtime import configure_runtime

paths = configure_runtime()

# Initialize loader pointing to your data directory
# Ensure data is downloaded (runs automatically if not present)
loader = SNLILoader(data_dir=paths.data_dir / "snli")

# Get a specific problem by ID (e.g., from SNLI dev set)
problem = loader.get_problem("4705552913.jpg#2r1n", split="dev")

print(f"Problem ID: {problem.id}")
print(f"Premises: {problem.premises}")
print(f"Hypothesis: {problem.hypothesis}")
print(f"Gold Label: {problem.gold_label}")
```

### Example: Full Experiment Orchestration

```python
from kbprojection import collect_kb_helpful_examples_random, SNLILoader
from kbprojection.models import ProblemConfig, TestMode
from kbprojection.runtime import configure_runtime

paths = configure_runtime()

# Initialize dataset loader
# If data is not present, it will be downloaded automatically.
snli_data = SNLILoader(data_dir=paths.data_dir / "snli")

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
    cache_dir=paths.cache_root / "readme_example"
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

## Multi-reference KB experiment workflow

The paper experiments compare LLM-generated KB relations with multiple human
KB annotations. In this working copy, the experiment scripts and annotation
CSVs live in the parent project directory, so run these commands from
`../` relative to this package repository.

### Input data

The experiment input CSV must contain one row per NLI item. The runner uses:

* `ID`: stable item identifier. Keep this non-blank for real experiment rows.
* `premise`: premise sentence shown to the model.
* `hypothesis`: hypothesis sentence shown to the model.
* `gold_label`: NLI label, such as `entailment`, `neutral`, or
  `contradiction`.
* `dataset`: source dataset, for example `sick` or `snli`.
* `split`: source split, for example `train`, `dev`, or `test`.

The multi-reference evaluator also needs the human KB reference columns:

* `Ettore_KB`
* `Jorryt_KB`
* `Lasha_KB`
* `Stefan_KB`

Blank KB cells mean the annotator did not provide an annotation. `NO_RELATION`
means an explicit annotation that no KB relation is needed. Do not convert
blank cells into `NO_RELATION`.

### Run an LLM experiment

Always run a small live smoke test before a full model run:

```bash
.venv/bin/python run_multi_reference_llm_experiment.py \
  --input-csv "all_usable_items_362.csv" \
  --output-csv "llm_outputs_smoke.csv" \
  --provider openrouter \
  --prompts ettore lasha \
  --models openai/gpt-5.4 \
  --limit 2 \
  --write-every 1
```

Then run the full experiment:

```bash
.venv/bin/python run_multi_reference_llm_experiment.py \
  --input-csv "all_usable_items_362.csv" \
  --output-csv "llm_outputs_sonnet45_gpt54_gemini35flash_all_usable.csv" \
  --provider openrouter \
  --prompts ettore lasha \
  --models \
    anthropic/claude-sonnet-4.5 \
    openai/gpt-5.4 \
    google/gemini-3.5-flash \
  --write-every 1
```

The output file contains three columns per prompt/model pair:

* `*_raw_response`: exact provider response.
* `*_KB`: parsed KB relations used for scoring.
* `*_error`: API or parsing error.

Resume is enabled by default. If a run stops, rerun the same command with the
same output file. Existing KB/error cells are skipped.

### Calculate multi-reference micro-F1

Evaluate every generated `LLM__*_KB` column against the human references:

```bash
.venv/bin/python calculate_multi_reference_f1.py \
  --csv "llm_outputs_sonnet45_gpt54_gemini35flash_all_usable.csv" \
  --reference-columns \
    Ettore_KB Jorryt_KB Lasha_KB Stefan_KB \
  --summary-csv \
    "multi_reference_f1_sonnet45_gpt54_gemini35flash_all_usable_summary.csv"
```

The evaluator works item by item:

1. Parse the model KB as a set of relations.
2. Compare it to every non-blank human KB reference.
3. Select the best-matching reference for that item.
4. Accumulate TP, FP, and FN over all items.
5. Compute micro-precision, micro-recall, and micro-F1.

`Exact best match` means the model's full KB set exactly equals at least one
available human KB reference.

### Position-sensitive relation-sequence micro-F1

The default micro-F1 treats each KB as an unordered set of relations. Use the
position-sensitive variant when the order in which relations are written should
also affect the score. A relation only matches when it is identical and appears
at the same position in both KB sequences.

For example, these KBs receive a perfect default set-based score because they
contain the same two relations:

```text
Prediction: (isa, cat, animal); (entails, cat, sleeps)
Reference:  (entails, cat, sleeps); (isa, cat, animal)
```

With position-sensitive scoring, neither relation is in the same position, so
this example has `TP=0`, `FP=2`, and `FN=2`.

Add `--position-sensitive` to calculate both metrics in one run:

```bash
.venv/bin/python calculate_multi_reference_f1.py \
  --csv "llm_outputs_sonnet45_gpt54_gemini35flash_all_usable.csv" \
  --reference-columns \
    Ettore_KB Jorryt_KB Lasha_KB Stefan_KB \
  --position-sensitive \
  --summary-csv \
    "multi_reference_f1_sonnet45_gpt54_gemini35flash_all_usable_position_sensitive_summary.csv"
```

The summary retains the default `micro_f1` columns and adds
`position_sensitive_precision`, `position_sensitive_recall`, and
`position_sensitive_micro_f1`. The position-sensitive metric independently
selects the best available human reference per item under the ordered scoring
rule.

### Calculate inter-annotator agreement

Rebuild the agreement overview from the assignment JSON files with:

```bash
.venv/bin/python calculate_inter_annotator_agreement.py \
  "annotated files" \
  --csv "inter_annotator_agreement_overview.csv" \
  --tsv "inter_annotator_agreement_overview.tsv"
```

This reports pairwise exact KB-set agreement, weighted Cohen's kappa over the
number of relations, pairwise relation-level micro-F1, all-annotator exact
agreement, and nominal Krippendorff's alpha.
