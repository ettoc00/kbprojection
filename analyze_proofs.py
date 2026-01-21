
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kbprojection.loaders.sick import SICKLoader
from kbprojection.models import ProblemConfig, TestMode
from kbprojection.orchestration import process_single_problem

# IDs to analyze
# 5448: Passive/Active (Fish being fried)
# 1801: Multi-word (pacing tiredly -> slowly walking)
# 2190: Reverse Entailment (People -> men)
# 2693: (New request from user)
DEBUG_IDS = ["5448", "1801", "2190", "2693"]

def analyze():
    print("Loading SICK problems...")
    loader = SICKLoader()
    # iter_problems() returns a generator of problems for a split
    # Note: 2693 might be in train/dev/test, let's check test first then try others if needed?
    # Actually, let's just load them all into a dict
    
    problems = {}
    for split in ['test', 'dev', 'train']:
        try:
            for prob in loader.iter_problems(split=split):
                if prob.id in DEBUG_IDS and prob.id not in problems:
                    problems[prob.id] = prob
        except FileNotFoundError:
            pass
            
    # Check if we found them all
    for pid in DEBUG_IDS:
        if pid not in problems:
            print(f"Problem {pid} not found in any split!")
            
    config = ProblemConfig(
        test_mode=TestMode.BOTH, # Run both raw and filtered to see proof diffs
        run_ablation=False,
        verbose=True
    )
    
    for pid in DEBUG_IDS:
        if pid not in problems:
            continue
            
        print(f"\n{'='*80}")
        print(f"ANALYZING PROBLEM {pid}")
        print(f"{'='*80}")
        
        prob = problems[pid]
        print(f"Premise:    {prob.premise}")
        print(f"Hypothesis: {prob.hypothesis}")
        print(f"Gold Label: {prob.gold_label}")
        
        # We need to capture the LangProResult from the orchestration
        # The orchestration returns ExperimentResult which has steps strings/enums
        # But wait, ExperimentResult stores the actual result?
        # Checking models.py... it seems we don't store the full LangProResult object in ExperimentResult, 
        # only the label and status. 
        # Wait, looking at models.py again in previous steps...
        # ExperimentResult has:
        # pred_no_kb: Optional[NLILabel]
        # pred_with_kb: Optional[NLILabel]
        # But NO field for the proof object itself!
        
        # To analyze the tree, I should probably call langpro directly like I told the user,
        # OR I can modify the orchestration to return it.
        # Calling locally is easier for a script.
        
        from kbprojection.langpro import langpro_api_call
        try:
            from kbprojection.llm import call_llm
            from kbprojection.filtering import pipeline_filter_kb_injections
            HAS_LLM = True
        except ImportError:
            HAS_LLM = False
        
        
        # 1. Run Baseline (No KB)
        print("\n--- 1. Baseline (No KB) ---")
        try:
            res_no_kb = langpro_api_call([prob.premise], prob.hypothesis, kb=[])
            print(f"Label: {res_no_kb.label}")
            if res_no_kb.error:
                print(f"Error: {res_no_kb.error}")
            else:
                # Print the tree for Entailment attempt
                if 'entailment' in res_no_kb.proofs:
                    print("Entailment Proof Tree (Top 20 lines):")
                    tree = res_no_kb.proofs['entailment']
                    if hasattr(tree, 'pformat'):
                        print('\n'.join(tree.pformat().split('\n')[:20]))
                    else: 
                        print(tree)
        except Exception as e:
            print(f"Error running baseline: {e}")
        
        # 2. Generate KB (or use Mock)
        print("\n--- 2. With KB ---")
        
        try:
            # Hardcoded KBs for structural testing without LLM
            MOCK_KBS = {
                "5448": [], # Lemmas match (fry/woman/fish), testing structure
                "1801": ["isa(pace, walk)", "isa(tiredly, slowly)"],
                "2190": ["isa(men, people)"],
                "2693": []
            }
        
            kb_list = []
            
            # Use Mock KB if defined
            if pid in MOCK_KBS:
                 kb_list = MOCK_KBS[pid]
                 print(f"Using Mock KB for {pid}: {kb_list}")
            # Else try LLM if available
            elif HAS_LLM:
                 print("Attempting to generate KB via LLM...")
                 try:
                    kb_raw = call_llm(provider="openai", model="gpt-4o", prompt_style="icl", prob=prob)
                    print(f"Raw KB: {kb_raw}")
                    kb_results = pipeline_filter_kb_injections(kb_raw, prob.premise, prob.hypothesis, post_process=True)
                    kb_list = [res.relation for res in kb_results]
                    print(f"Filtered KB: {kb_list}")
                 except Exception as e:
                     print(f"LLM Generation failed: {e}")
            else:
                 print("No KB source available.")
            
            # Always run LangPro even if KB is empty to confirm baseline consistency check
            res_kb = langpro_api_call([prob.premise], prob.hypothesis, kb=kb_list)
            print(f"Label: {res_kb.label}")
            
            # Print tree
            target_proof = 'entailment' if prob.gold_label == 'entailment' else 'contradiction'
            # If neutral, we check Entailment tree
            if prob.gold_label == 'neutral':
                    target_proof = 'entailment'

            if target_proof in res_kb.proofs:
                print(f"{target_proof.capitalize()} Proof Tree (Top 30 lines):")
                tree = res_kb.proofs[target_proof]
                if hasattr(tree, 'pformat'):
                    print('\n'.join(tree.pformat().split('\n')[:30]))
                else:
                    print(tree)
            else:
                print(f"No {target_proof} proof found.")

        except Exception as e:
            print(f"Error running with KB: {e}")

if __name__ == "__main__":
    analyze()
