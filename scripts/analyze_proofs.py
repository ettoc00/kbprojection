import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kbprojection.filtering import pipeline_filter_kb_injections
from kbprojection.langpro import langpro_api_call
from kbprojection.llm import call_llm
from kbprojection.loaders.sick import SICKLoader


DEBUG_IDS = ["5448", "1801", "2190", "2693"]

MOCK_KBS = {
    "5448": [],
    "1801": ["isa_wn(pace, walk)", "isa_wn(tiredly, slowly)"],
    "2190": ["isa_wn(men, people)"],
    "2693": [],
}


async def analyze():
    print("Loading SICK problems...")
    loader = SICKLoader()

    problems = {}
    for split in ["test", "dev", "train"]:
        try:
            for prob in loader.iter_problems(split=split):
                if prob.id in DEBUG_IDS and prob.id not in problems:
                    problems[prob.id] = prob
        except FileNotFoundError:
            pass

    for problem_id in DEBUG_IDS:
        if problem_id not in problems:
            print(f"Problem {problem_id} not found in any split.")

    for problem_id in DEBUG_IDS:
        prob = problems.get(problem_id)
        if prob is None:
            continue

        print(f"\n{'=' * 80}")
        print(f"ANALYZING PROBLEM {problem_id}")
        print(f"{'=' * 80}")
        print(f"Premises:   {prob.premises}")
        print(f"Hypothesis: {prob.hypothesis}")
        print(f"Gold Label: {prob.gold_label}")

        print("\n--- 1. Baseline (No KB) ---")
        try:
            res_no_kb = await langpro_api_call(prob.premises, prob.hypothesis, kb=[])
            print(f"Label: {res_no_kb.label}")
            if res_no_kb.error:
                print(f"Error: {res_no_kb.error}")
            elif "entailment" in res_no_kb.proofs:
                tree = res_no_kb.proofs["entailment"]
                print("Entailment Proof Tree (Top 20 lines):")
                if hasattr(tree, "pformat"):
                    print("\n".join(tree.pformat().split("\n")[:20]))
                else:
                    print(tree)
        except Exception as e:
            print(f"Error running baseline: {e}")

        print("\n--- 2. With KB ---")
        try:
            if problem_id in MOCK_KBS:
                kb_list = list(MOCK_KBS[problem_id])
                print(f"Using mock KB for {problem_id}: {kb_list}")
            else:
                kb_list = []
                print("Attempting to generate KB via LLM...")
                kb_raw = await call_llm(
                    provider="openai",
                    model="gpt-5-mini",
                    prompt_style="icl",
                    prob=prob,
                )
                print(f"Raw KB: {kb_raw}")
                kb_results = pipeline_filter_kb_injections(
                    kb_raw,
                    prob.premises,
                    prob.hypothesis,
                    post_process=True,
                )
                kb_list = [res.relation for res in kb_results]
                print(f"Filtered KB: {kb_list}")

            res_kb = await langpro_api_call(prob.premises, prob.hypothesis, kb=kb_list)
            print(f"Label: {res_kb.label}")

            target_proof = "entailment"
            if prob.gold_label.value == "contradiction":
                target_proof = "contradiction"

            if target_proof in res_kb.proofs:
                tree = res_kb.proofs[target_proof]
                print(f"{target_proof.capitalize()} Proof Tree (Top 30 lines):")
                if hasattr(tree, "pformat"):
                    print("\n".join(tree.pformat().split("\n")[:30]))
                else:
                    print(tree)
            else:
                print(f"No {target_proof} proof found.")
        except Exception as e:
            print(f"Error running with KB: {e}")


if __name__ == "__main__":
    asyncio.run(analyze())
