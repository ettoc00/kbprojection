
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from kbprojection.loaders.sick import SICKLoader
from kbprojection.langpro import langpro_api_call

def run_debug():
    print("1. Initializing SICKLoader...")
    loader = SICKLoader()
    
    print("2. Searching for Problem 5448...")
    target_prob = None
    # Just check test split first for speed
    try:
        for prob in loader.iter_problems(split='test'):
            if prob.id == "5448":
                target_prob = prob
                break
        
        if not target_prob:
             print("Not found in test, checking train...")
             for prob in loader.iter_problems(split='train'):
                if prob.id == "5448":
                    target_prob = prob
                    break
    except Exception as e:
        print(f"Error iterating problems: {e}")
        return

    if not target_prob:
        print("ERROR: Problem 5448 not found in SICK dataset.")
        return
        
    print(f"3. Found Problem 5448:")
    print(f"   Premise: {target_prob.premise}")
    print(f"   Hypothesis: {target_prob.hypothesis}")
    
    print("\n4. Attempting LangPro API Call (Baseline)...")
    try:
        # 30 second timeout implicitly in requests? langpro.py doesn't set timeout.
        # Let's see what happens.
        res = langpro_api_call(
            premises=[target_prob.premise], 
            hypothesis=target_prob.hypothesis, 
            report=True
        )
        print(f"   Result Label: {res.label}")
        if res.error:
            print(f"   API Error: {res.error}")
        else:
            print(f"   Success! Proofs keys: {res.proofs.keys()}")
            
    except Exception as e:
        print(f"   CRITICAL EXCEPTION during API call: {e}")

if __name__ == "__main__":
    run_debug()
