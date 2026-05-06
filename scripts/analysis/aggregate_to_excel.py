import json
import re
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pandas is not installed. Please install it using: uv pip install pandas openpyxl")
    exit(1)

from kbprojection.settings import get_default_results_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = get_default_results_dir()
OUTPUT_FILE = PROJECT_ROOT / "scripts" / "analysis" / "benchmark_results.xlsx"

def extract_provider_model(filename: str):
    # Filenames match: {dataset}__{prompt}__{provider}__{model}.json
    # or sick_full__{prompt}__{provider}__{model}.json
    parts = filename.replace(".json", "").split("__")
    if len(parts) >= 4:
        provider = parts[-2]
        model = parts[-1]
        dataset = parts[0]
        prompt = parts[1]
        return dataset, prompt, provider, model
    return "unknown", "unknown", "unknown", "unknown"

def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory {RESULTS_DIR} does not exist.")
        return

    data = []

    for json_file in RESULTS_DIR.glob("*.json"):
        dataset, prompt, provider, model = extract_provider_model(json_file.name)
        if dataset == "unknown":
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

        summary = content.get("summary", {})
        if not summary:
            # Maybe it's a file with just summary at root
            if "completed" in content and "total" in content:
                summary = content
            else:
                continue

        total = summary.get("total", 0)
        completed = summary.get("completed", 0)

        correct_counts = summary.get("correct_counts", {})
        correct_no_kb = correct_counts.get("no_kb", 0)
        correct_raw_kb = correct_counts.get("raw_kb", 0)
        correct_normalised_kb = correct_counts.get("normalised_kb", 0)

        accuracy_rates = summary.get("accuracy_on_successful_calls", {})
        acc_no_kb = accuracy_rates.get("no_kb", 0.0)
        acc_raw_kb = accuracy_rates.get("raw_kb", 0.0)
        acc_normalised_kb = accuracy_rates.get("normalised_kb", 0.0)

        row = {
            "Dataset": dataset,
            "Prompt": prompt,
            "Provider": provider,
            "Model": model,
            "Total": total,
            "Completed": completed,
            "No KB Correct": correct_no_kb,
            "Raw KB Correct": correct_raw_kb,
            "Norm KB Correct": correct_normalised_kb,
            "No KB Acc": acc_no_kb,
            "Raw KB Acc": acc_raw_kb,
            "Norm KB Acc": acc_normalised_kb,
        }
        data.append(row)

    if not data:
        print("No valid benchmark JSON results found.")
        return

    df = pd.DataFrame(data)
    # Sort for better readability
    df = df.sort_values(by=["Dataset", "Provider", "Model"])

    # Save to Excel
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Aggregated results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
