# Knowledge Base PROver inJECTION (kbprojection)

This library is designed to facilitate the use of Large Language Models (LLMs) to generate Knowledge Base (KB) injections for the LangPro prover. It provides tools for prompting LLMs, processing the generated KBs, and orchestrating experiments to evaluate the effectiveness of these injections.

## Installation

```bash
# Using uv (recommended)
uv sync

# Or install the project into an existing environment
uv pip install -e .
```

`uv sync` installs the project and its dependencies into `.venv`. Use Python
3.10 or newer; the repository's type annotations require it.

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

The dataset loaders (`SNLILoader` and `SICKLoader`) will automatically download the necessary data if it is not found in the specified directory. If no directory is specified, data is stored persistently under the kbprojection app-data directory.

### Runtime Requirements

The library depends on:

* Access to the remote LangPro API, or a local LangPro checkout for corpora-backed runs.
* An API key for the LLM provider you want to use.
* NLTK data packages, which are downloaded on demand.
* For local EasyCCG parsing: the `spacy` package plus an installed English model such as `en_core_web_sm`.

Provider detection is environment-variable based:

* `OPENAI_API_KEY` -> OpenAI
* `OPENROUTER_API_KEY` -> OpenRouter
* `GEMINI_API_KEY` -> Gemini
* `ANTHROPIC_API_KEY` -> Claude

If a repo-local `.env` file is present, missing provider variables are loaded from it automatically.

Generated runtime data defaults to an app-data directory:

* Windows: `%LOCALAPPDATA%\kbprojection`
* Linux/macOS: `$XDG_CACHE_HOME/kbprojection`, or `~/.cache/kbprojection` when `XDG_CACHE_HOME` is unset

By default this contains:

* `datasets/` for downloaded SNLI/SICK data.
* `results/` for generated script outputs.
* `cache/` for resumable per-problem caches when scripts opt into them.
* `vendor/LangPro` for explicitly auto-cloned LangPro checkouts.
* `langpro_cache.sqlite3` for the SQLite LangPro result cache.

Runtime behavior can also be configured with:

* `KBPROJECTION_APP_DIR` to override the generated data root.
* `KBPROJECTION_DATA_DIR` to override the downloaded dataset root.
* `KBPROJECTION_RESULTS_DIR` to override generated script output defaults.
* `KBPROJECTION_CACHE_DIR` to override generated cache defaults.
* `KBPROJECTION_LANGPRO_ENDPOINT` to override the LangPro API endpoint.
  Set this to `local://auto` to use a local LangPro checkout instead of the remote API.
  Set this to `hybrid://auto` to use both the default remote endpoint and `local://auto` in the same run.
* `KBPROJECTION_LANGPRO_TIMEOUT_SECONDS` to control LangPro request timeouts.
* `KBPROJECTION_DOWNLOAD_TIMEOUT_SECONDS` to control dataset download timeouts.
* `KBPROJECTION_LANGPRO_CACHE_BACKEND` to choose `memory` or `sqlite`.
* `KBPROJECTION_LANGPRO_CACHE_PATH` to set the SQLite cache file path.
* `KBPROJECTION_LANGPRO_LOCAL_ROOT` to point at a local LangPro checkout.
  If unset, `local://auto` searches repo-local `vendor/LangPro`, sibling `../LangPro`, then the app-data `vendor/LangPro`.
* `KBPROJECTION_LANGPRO_VENDOR_DIR` to override the app-data LangPro clone path.
* `KBPROJECTION_LANGPRO_AUTO_CLONE=1` to let `local://auto` clone LangPro when no checkout is found.
* `KBPROJECTION_LANGPRO_REPO` to override the auto-clone repository. Defaults to `https://github.com/kovvalsky/LangPro.git`.
* `KBPROJECTION_LANGPRO_REF` to override the auto-clone branch/ref. Defaults to `nl`.
* `KBPROJECTION_LANGPRO_LOCAL_SWIPL` to override the `swipl` executable used for local runs.

In hybrid LangPro mode, `--langpro-concurrency` limits remote API calls and
`--local-langpro-concurrency` limits local subprocess calls. If one backend fails
repeatedly, kbprojection temporarily routes normal traffic to the other backend
and probes the failed backend with exponential backoff, so short remote outages
are routed around without permanently disabling the remote endpoint.

In Python, local LangPro can also be enabled for the current process with:

```python
from kbprojection.settings import enable_local

enable_local()  # optional: local_root=..., easyccg_dir=..., auto_clone=True
```

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
KB annotations. Run the experiment scripts from the repository root; input data and saved
results are tracked alongside the scripts.

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

* `Alternative_KB`
* `Ettore_KB`
* `Jorryt_KB`
* `Lasha_KB`
* `Stefan_KB`

Blank KB cells mean the annotator did not provide an annotation. `NO_RELATION`
means an explicit annotation that no KB relation is needed. Do not convert
blank cells into `NO_RELATION`.

### Canonical experiment input

The reproducible 362-item input is committed at
`data/all_usable_items_362.csv`. It contains the quality-controlled SNLI/SICK
entailment problems and the five reference LEX annotation columns used by the
multi-reference evaluations. The file has 362 data rows and its SHA-256 is:

```text
7be06d326bdaff587368b06c86e4b28ee0d8e642fda3cbf676a5baffd816e77e
```

The experiment scripts validate the required columns and row count before
making model calls. This prevents accidentally running the paper evaluation
on a different or incomplete CSV.

### Run an LLM experiment

Always run a small live smoke test before a full model run:

```bash
.venv/bin/python run_multi_reference_llm_experiment.py \
  --input-csv "data/all_usable_items_362.csv" \
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
  --input-csv "data/all_usable_items_362.csv" \
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

### Repeated-run model evaluation

Use `scripts/experiments/run_repeated_multi_reference_experiment.py` when each
model should process every item multiple times. Unlike the standard experiment
runner, this script writes one row per item, prompt, model, and repetition. This
long format allows accuracy and output stability to be evaluated separately.

The following experiment runs the improved Lasha prompt five times with eight
models on all 362 usable annotation items:

```bash
.venv/bin/python scripts/experiments/run_repeated_multi_reference_experiment.py \
  --input-csv "data/all_usable_items_362.csv" \
  --sample-size 362 \
  --repeats 5 \
  --prompts lasha \
  --models \
    openai/gpt-5.4-mini \
    anthropic/claude-haiku-4.5 \
    google/gemini-3.1-flash-lite \
    openai/gpt-oss-20b \
    google/gemma-3-4b-it \
    anthropic/claude-sonnet-4.5 \
    openai/gpt-5.4 \
    google/gemini-3.5-flash \
  --reference-columns \
    Alternative_KB Ettore_KB Jorryt_KB Lasha_KB Stefan_KB \
  --temperature 0 \
  --concurrency 4 \
  --write-every-jobs 40 \
  --request-timeout 120 \
  --max-retries 2
```

This creates `362 x 8 x 5 = 14,480` model calls. The production prompt name
`lasha` refers to the original Lasha prompt with the selected precision
calibration added.

Generated files are written to:

```text
experiment_results/lasha_all362_5runs/
```

The files have distinct purposes:

* `consistency_sample.csv`: frozen copy of the exact evaluated input items.
* `consistency_outputs.csv`: raw response, parsed KB, error, and repetition for
  every model call.
* `consistency_metrics.csv`: repeatability statistics for each prompt-model
  combination.
* `consistency_f1_by_run.csv`: standard and position-sensitive multi-reference
  F1 for every individual repetition.
* `consistency_f1_summary.csv`: mean, sample standard deviation, minimum, and
  maximum F1 across repetitions.

The consistency metrics include:

* `all_runs_identical_rate`: fraction of complete items for which every
  repetition produced the same KB.
* `mean_pairwise_kb_f1`: average KB similarity between every pair of
  repetitions.
* `no_relation_flip_rate`: frequency with which a model alternated between
  `NO_RELATION` and a non-empty KB.
* `mean_unique_kb_sets_per_item`: average number of distinct KB answers per
  item.
* `error_rate`: fraction of unsuccessful model calls.

Both F1 variants use the same multi-reference best-match procedure as
`calculate_multi_reference_f1.py`. For each item, the model prediction is
compared with every available non-blank human reference, and the best reference
is selected before TP, FP, and FN are accumulated.

The standard metric treats relations as an unordered set. The
position-sensitive metric requires matching relations to appear in the same
sequence positions and selects the best reference independently under that
rule.

Blank predictions and errors are skipped and counted; they are not interpreted
as `NO_RELATION`.

The output is resumable. Reusing the same output path skips rows that already
contain a KB or an error. Requests are retried during their initial execution,
but persisted error rows are not automatically retried on a later resume.

Temperature zero is requested for all models, but OpenRouter may ignore it when
the selected model does not support temperature control. At the time of this
experiment, it was unsupported for GPT-5.4 and GPT-5.4 Mini. Even where
supported, temperature zero does not guarantee identical hosted-model output.

A completed five-run experiment is available in
[`experiment_results/lasha_all362_5runs`](experiment_results/lasha_all362_5runs).

### Reproduce the committed LEX prediction scores (no API calls)

The saved long-format raw responses for all 5 runs, the 362 annotated items,
and the expected score files are committed. Reparse the raw responses and
recompute the set-based and position-sensitive scores with:

```bash
mkdir -p /tmp/kbprojection-lex-replay
.venv/bin/python scripts/experiments/recompute_repeated_no_filter_scores.py \
  --input-csv experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_outputs.csv \
  --sample-csv data/all_usable_items_362.csv \
  --output-csv /tmp/kbprojection-lex-replay/no_filter_outputs.csv \
  --metrics-csv /tmp/kbprojection-lex-replay/no_filter_stability.csv \
  --f1-metrics-csv /tmp/kbprojection-lex-replay/no_filter_f1_by_run.csv \
  --f1-summary-csv /tmp/kbprojection-lex-replay/no_filter_f1_summary.csv \
  --filtered-f1-summary-csv experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_f1_summary.csv \
  --comparison-csv /tmp/kbprojection-lex-replay/filtered_vs_no_filter.csv \
  --repeats 5

diff -u \
  experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_no_filter_f1_by_run.csv \
  /tmp/kbprojection-lex-replay/no_filter_f1_by_run.csv
diff -u \
  experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_no_filter_f1_summary.csv \
  /tmp/kbprojection-lex-replay/no_filter_f1_summary.csv
```

Both `diff` commands should produce no output and exit with status 0. The
committed `*_no_filter_f1_by_run.csv` contains the precision, recall,
micro-F1, exact-best-match, and position-sensitive scores for every model and
repeat; `*_no_filter_f1_summary.csv` contains their five-run mean and sample
standard deviation. This procedure makes no network or model API calls.

### Multi-reference scoring method

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

The replay procedure above calculates both metrics in one run. Its summaries
retain the default `micro_f1` columns and add
`position_sensitive_precision`, `position_sensitive_recall`, and
`position_sensitive_micro_f1`. The position-sensitive metric independently
selects the best available human reference per item under the ordered scoring
rule.

### Calculate inter-annotator agreement

The paper's Table 2 reports post-adjudication agreement, not the initial JSON
submissions. Its source is the tracked export of the edited project sheet:
[`data/annotator_agreement/iaa_overview_edit.csv`](data/annotator_agreement/iaa_overview_edit.csv).
The raw individual submissions remain available in
[`data/annotator_assignments`](data/annotator_assignments), but they represent
the pre-adjudication stage and do not reproduce the Table 2 scores.

Recompute Table 2 without network or API calls with:

```bash
mkdir -p /tmp/kbprojection-iaa
.venv/bin/python scripts/experiments/recompute_paper_iaa_scores.py \
  --iaa-csv data/annotator_agreement/iaa_overview_edit.csv \
  --final-items-csv data/all_usable_items_362.csv \
  --summary-csv /tmp/kbprojection-iaa/paper_table2_iaa_scores.csv

diff -u experiment_results/iaa/paper_table2_iaa_scores.csv \
  /tmp/kbprojection-iaa/paper_table2_iaa_scores.csv
```

For every annotator, the script filters to the final 362 IDs, selects the
best-matching KB from the other three original annotators, and reports exact
match and relation-level micro-F1. `Alternative_KB` is excluded because it is
an adjudication-created fourth valid variant rather than an independent
annotator submission. The recomputed micro-F1 values match Table 2. Ettore's
exact rate is 205/271 = 75.645...%, which displays as 75.6% at one decimal
place; Table 2 prints 75.7%.

To analyze the initial, pre-adjudication submissions separately, run
`calculate_inter_annotator_agreement.py data/annotator_assignments`.

### Prepare the prompt-ablation comparison (no API calls)

The prompt ablation uses the 223 unanimous items from the pre-finalization
edited IAA overview. This is intentional: two of these items were excluded
later from the 362-item final paper dataset, so do not replace the source with
`data/all_usable_items_362.csv`.

```bash
.venv/bin/python prompt_engineering/run_lex_prompt_ablation_overnight.py \
  --output-dir /tmp/kbprojection-prompt-ablation \
  --prepare-only
```

This writes the derived `agreed_subset.csv` (223 items), `primary_sample.csv`,
the deterministic stability sample, and the exact prompt templates into the
chosen output directory. It makes no model or network calls. A later live run
uses the same command without `--prepare-only`; it requires configured API
credentials and may vary because hosted model outputs can change.

## Notes On Current API Shape

* `NLIProblem` uses `premises: List[str]`, even for single-premise datasets.
* Use `run_problem(...)` and `run_problems(...)` from synchronous scripts.
* Use `arun_problem(...)` and `arun_problems(...)` from notebooks or async applications.
* `process_single_problem(...)` remains available for advanced callers that need direct `ProblemConfig` or `AsyncRunContext` control.
* `ProblemConfig.model` currently defaults to `gpt-5-mini`.
* `prompt_style` defaults to `icl`; available prompt names include `legacy_cot`, `legacy_least_to_most`, `legacy_icl`, `icl`, and `cot`.
* `collect_kb_helpful_examples_random(...)` yields only results whose final status is `normalised_kb_solved` or `raw_kb_solved`.
* `process_kb_examples(...)` yields all processed results, including failures and already-correct baselines.
* Identical KB relation sets share the same LangPro cache entry, even when they come from different models, arrive in a different order, or are run through local, remote, or hybrid LangPro endpoints.
* LangPro caching can be backed by in-memory storage or SQLite persistence. The default SQLite path is `<app-data>/langpro_cache.sqlite3`; legacy repo-local `.kbprojection/` files are ignored but no longer created by default.
* Local LangPro mode first tries sentence match against any preparsed corpora discovered under `<LangPro>/ccg_sen_d` with `*_sen.pl` companion files. It prefers parser-specific `*_eccg.pl` or `*_depccg*.pl` files when they exist, and falls back to generic `*_ccg.pl` when needed.
* If no bundled or generated corpus match is found, local mode falls back to raw-text parsing through EasyCCG, then runs LangPro locally on temporary `ccg.pl`/`sen.pl` files.
* The raw-text fallback requires EasyCCG plus the configured spaCy model.

## Repository utility scripts

* `scripts/2x2_50_problems.py` to replay the fixed 50-problem benchmark across `legacy_icl` and `icl`.
* `scripts/evaluate_sick_dataset.py` to run one model/prompt across the full SICK dataset with resumable JSON output.
* `scripts/evaluate_snli_dataset.py` to run one model/prompt across the full SNLI dataset with resumable JSON output.
* `scripts/build_local_langpro_corpus.py` to generate preparsed `*_sen.pl` and `*_eccg.pl` corpora for `SNLI` or `SICK` inside a local LangPro checkout so repeated local runs can skip raw-text parsing.
