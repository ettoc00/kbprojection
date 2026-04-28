# Knowledge Base PROver inJECTION (kbprojection)

This library is designed to facilitate the use of Large Language Models (LLMs) to generate Knowledge Base (KB) injections for the LangPro prover. It provides tools for prompting LLMs, processing the generated KBs, and orchestrating experiments to evaluate the effectiveness of these injections.

At a high level, the pipeline is:

1. Load an NLI problem from SNLI or SICK.
2. Run LangPro without any KB injection.
3. If the baseline is wrong, ask an LLM to propose `isa_wn(...)` and `disj(...)` relations.
4. Normalise the proposed relations.
5. Re-run LangPro with the raw and/or normalised KB.
6. Optionally run ablation to find minimal helpful subsets.

[![Open NLI showcase in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ettoc00/kbprojection/blob/main/notebooks/nli_showcase_colab.ipynb)

## Installation

```bash
# Using uv (recommended)
uv sync

# Or install in development mode
uv pip install -e .
```

If you use local EasyCCG parsing, also install the default spaCy English model:

```bash
uv run python -m spacy download en_core_web_sm
```

## Usage

The library is divided into several modules:

* `kbprojection.loaders`: Data loaders for SNLI and SICK datasets.
* `kbprojection.models`: Pydantic models for type safety across the pipeline.
* `kbprojection.prompts`: Manage and fill prompt templates.
* `kbprojection.langpro`: Interface with the LangPro API.
* `kbprojection.llm`: A unified interface for calling various LLMs (OpenAI, Claude/Anthropic, Gemini, OpenRouter).
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

In Python, local LangPro can also be enabled for the current process with:

```python
from kbprojection.settings import enable_local

enable_local()  # optional: local_root=..., easyccg_dir=..., auto_clone=True
```

### Example: Loading a Single Problem

```python
from kbprojection import SNLILoader

# Initialize loader pointing to your data directory
# Ensure data is downloaded (runs automatically if not present)
loader = SNLILoader(data_dir="./data")

# Get a specific problem by ID (e.g., from SNLI dev set)
problem = loader.get_problem("4705552913.jpg#2r1n", split="dev")

print(f"Problem ID: {problem.id}")
print(f"Premises: {problem.premises}")
print(f"Hypothesis: {problem.hypothesis}")
print(f"Gold Label: {problem.gold_label}")
```

### Example: Full Experiment Orchestration

```python
from kbprojection import run_problems, SNLILoader

# Initialize dataset loader
# If no data_dir is passed, data is downloaded under the app-data datasets directory.
snli_data = SNLILoader()
snli_data.load(splits=["dev"])
problems = [
    snli_data.get_problem("4705552913.jpg#2r1n", split="dev"),
]

results = run_problems(
    problems,
    provider="openai",
    model="gpt-5-mini",
    prompt_style="icl",
    test_mode="both",
)

for item in results:
    print(f"Problem {item['problem']['id']}: {item['final_status']}")
```

In notebooks or async applications, use the async equivalent:

```python
from kbprojection import arun_problems

results = await arun_problems(
    problems,
    model="openai/gpt-5.4-mini",
    provider="openrouter",
    prompt_style="icl",
)
```

### Example: Manually Creating and Executing a Problem

You can also create a problem instance manually and process it through the pipeline.

```python
from kbprojection import run_problem
from kbprojection.models import NLIProblem, NLILabel

# 1. Create a manual problem
manual_problem = NLIProblem(
    id="manual-test-1",
    premises=["A dog is running in the park."],
    hypothesis="An animal is moving.",
    gold_label=NLILabel.ENTAILMENT,
    dataset="manual",
    split="test"
)

# 2. Process the problem
# This runs the full pipeline: No-KB check -> LLM generation -> normalisation -> Re-check
result = run_problem(
    manual_problem,
    provider="openai",  # or "claude", "gemini", "openrouter"
    model="gpt-5-mini",
    prompt_style="icl",
)

print(f"Final Status: {result.final_status}")
if result.kb_filtered:
    print(f"Generated KB: {result.kb_filtered}")
```

## Core Models

### NLIProblem

Represents a single NLI problem instance.

* `id`: Unique identifier for the problem.
* `premises`: List of premise sentences.
* `hypothesis`: The hypothesis text.
* `gold_label`: The ground truth label (`entailment`, `contradiction`, or `neutral`).
* `dataset`: Source dataset name (e.g., "snli", "sick").
* `split`: Dataset split key (e.g., "train", "dev", "test").
* `original_data`: Dictionary containing original raw data from the dataset wrapper.

### ProblemConfig

Configuration object for the pipeline.

* `llm_provider`: String identifier for the LLM provider (e.g., "openai", "claude", "gemini", "openrouter").
* `model`: Model identifier (e.g., "gpt-5-mini").
* `prompt_style`: Identifier for the prompt template style.
* `post_process`: Boolean; if `True`, applies post-processing to LLM output.
* `test_mode`: `TestMode` enum controlling which stages to run (`no_kb`, `raw_kb`, `normalised`, `both`, `full`; `filtered` is a legacy alias).
* `run_ablation`: Boolean; if `True`, runs ablation to find all minimal sufficient KB subsets.
* `verbose`: Boolean; enables detailed logging.

### ExperimentResult

Encapsulates the results of running the pipeline on a problem.

* `problem`: The `NLIProblem` instance being processed.

* `kb_raw`: List of raw KB strings generated by the LLM.
* `kb_filtered`: List of normalised/formatted KB strings ready for LangPro.
* `kb_details`: List of `KBResult` objects containing detailed provenance for each injection.

* `pred_no_kb`: Prediction from LangPro without any KB injection.
* `status_no_kb`: Status of the baseline step.

* `pred_with_raw_kb`: Prediction using the raw (unfiltered) KB.
* `status_with_raw_kb`: Status of the raw KB evaluation step.
* `pred_with_kb`: Prediction using the normalised KB.
* `status_with_kb`: Status of the normalised KB evaluation step.

* `final_status`: `ExperimentStatus` enum summarizing the overall outcome (e.g., `NORMALISED_KB_SOLVED`, `KB_NOT_SOLVED`).
* `fixed_by`: String indicating which KB version solved the problem (`"raw_kb"`, `"normalised_kb"`, or `"both"`).
* `essential_kb`: Best minimal sufficient KB subset (ranked by token count). If ablation was run and multiple KB entries are redundant, this contains the simplest subset that alone fixes the problem.
* `ablation_subsets`: List of all minimal sufficient subsets found during ablation. Each subset is a list of KB strings that independently can fix the problem.
* `ablation_results`: Dictionary mapping tested subsets (as tuples) to their resulting label.

## Notes On Current API Shape

Some implementation details matter when using the library directly:

* `NLIProblem` uses `premises: List[str]`, even for single-premise datasets.
* Use `run_problem(...)` and `run_problems(...)` from synchronous scripts.
* Use `arun_problem(...)` and `arun_problems(...)` from notebooks or async applications.
* `process_single_problem(...)` remains available for advanced callers that need direct `ProblemConfig` or `AsyncRunContext` control.
* `ProblemConfig.model` currently defaults to `gpt-5-mini`.
* `prompt_style` defaults to `icl`; available prompt names include `legacy_cot`, `legacy_least_to_most`, `legacy_icl`, `icl`, and `cot`.
* `collect_kb_helpful_examples_random(...)` yields only results whose final status is `normalised_kb_solved` or `raw_kb_solved`.
* `process_kb_examples(...)` yields all processed results, including failures and already-correct baselines.
* Identical KB relation sets share the same LangPro cache entry, even when they come from different models or arrive in a different order.
* LangPro caching can be backed by in-memory storage or SQLite persistence. The default SQLite path is `<app-data>/langpro_cache.sqlite3`; legacy repo-local `.kbprojection/` files are ignored but no longer created by default.
* Local LangPro mode first tries sentence match against any preparsed corpora discovered under `<LangPro>/ccg_sen_d` with `*_sen.pl` companion files. It prefers parser-specific `*_eccg.pl` or `*_depccg*.pl` files when they exist, and falls back to generic `*_ccg.pl` when needed.
* If no bundled or generated corpus match is found, local mode falls back to raw-text parsing through EasyCCG, then runs LangPro locally on temporary `ccg.pl`/`sen.pl` files.
* The raw-text fallback requires EasyCCG plus the configured spaCy model.

## Scripts

Repository-local utilities live under `scripts/` rather than the package root.

Useful scripts include:

* `scripts/2x2_50_problems.py` to replay the fixed 50-problem benchmark across `legacy_icl` and `icl`.
* `scripts/evaluate_sick_dataset.py` to run one model/prompt across the full SICK dataset with resumable JSON output.
* `scripts/build_local_langpro_corpus.py` to generate preparsed `*_sen.pl` and `*_eccg.pl` corpora for `SNLI` or `SICK` inside a local LangPro checkout so repeated local runs can skip raw-text parsing.
