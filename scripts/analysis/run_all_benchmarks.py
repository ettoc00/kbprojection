import argparse
import subprocess
import sys
from pathlib import Path

# All required models
MODELS = [
    "anthropic/claude-opus-4.7",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.5",
    "openai/gpt-oss-120b",
    "google/gemini-3-flash-preview",
    "google/gemma-4-31b-it",
    "openai/gpt-5.4-mini",
]

# Models to use for the quick test
TEST_MODELS = [
    "openai/gpt-5.4-mini",
    "google/gemini-3-flash-preview",
]

def run_benchmarks(is_test: bool = False):
    models_to_run = TEST_MODELS if is_test else MODELS
    project_root = Path(__file__).resolve().parents[2]

    # Base arguments for both scripts
    base_args = [
        sys.executable,
        "--prompt-style", "icl",
        "--llm-concurrency", "4",
        "--job-concurrency", "8",
        "--local-langpro-concurrency", "2",
        "--langpro-concurrency", "2"
    ]

    if is_test:
        base_args.extend(["--limit", "10"])

    for model in models_to_run:
        print(f"\n==========================================")
        print(f"Running benchmarks for model: {model}")
        print(f"==========================================\n")

        # 1. Run SICK (all splits)
        sick_script = project_root / "scripts" / "evaluate_sick_dataset.py"
        sick_cmd = base_args.copy()
        sick_cmd.insert(1, str(sick_script))
        sick_cmd.extend(["--model", model])

        print(f"--> Running SICK benchmark for {model}...")
        subprocess.run(sick_cmd, check=True)

        # 2. Run SNLI (dev split)
        snli_script = project_root / "scripts" / "evaluate_snli_dataset.py"
        snli_cmd = base_args.copy()
        snli_cmd.insert(1, str(snli_script))
        snli_cmd.extend(["--model", model, "--splits", "dev"])

        print(f"--> Running SNLI benchmark for {model}...")
        subprocess.run(snli_cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate SICK and SNLI benchmarks.")
    parser.add_argument("--test", action="store_true", help="Run a quick test with 10 limits and 2 models.")
    args = parser.parse_args()

    run_benchmarks(is_test=args.test)
