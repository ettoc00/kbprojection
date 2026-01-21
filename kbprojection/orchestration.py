import json
from pathlib import Path
from typing import List, Set, Optional, Union, Iterator
from .models import ExperimentResult, ExperimentStepStatus, ExperimentStatus, NLIProblem, NLILabel, ProblemConfig, TestMode, LangProResult
from .loaders.base import DatasetLoader
from .langpro import langpro_api_call
from .llm import call_llm
from .filtering import filter_kb_by_prem_hyp

from .filtering import pipeline_filter_kb_injections

def _save_cache_result(cache_path: Path, result: ExperimentResult):
    """Helper to save a result object to a JSON file."""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        print(f"[cache] Result cached to {cache_path}")
    except Exception as e:
        print(f"[Warning] Failed to save cache to {cache_path}: {e}")


def process_single_problem(
    prob: NLIProblem,
    config: Optional[ProblemConfig] = None,
    cache_file: Optional[Path] = None
) -> ExperimentResult:
    """
    Process a single NLI problem through the KB injection pipeline.
    
    Args:
        prob: The NLI problem to process.
        config: Configuration for processing. If None, uses defaults.
        cache_file: Optional path to save/load cached results.
    
    Returns:
        ExperimentResult with all pipeline steps populated.
    """
    if config is None:
        config = ProblemConfig()
    
    def log(msg: str):
        if config.verbose:
            print(msg)
    
    test_mode = config.test_mode.value if isinstance(config.test_mode, TestMode) else config.test_mode
    
    log(f"\n[process] Key: {prob.id} | Gold: {prob.gold_label.value}")
    log(f"[process] Premises: {prob.premises}")
    log(f"[process] Hypothesis: {prob.hypothesis}")
    log(f"[process] Mode: {test_mode} | Ablation: {config.run_ablation}")

    exp_result = ExperimentResult(problem=prob, prover_calls=[])

    # ----- Step 1: No-KB baseline -----
    log("  [no-KB] Calling LangPro...")
    lp_res_no_kb = langpro_api_call(prob.premises, prob.hypothesis, report=False)

    exp_result.pred_no_kb = lp_res_no_kb.label
    exp_result.prover_calls.append(lp_res_no_kb)
    log(f"  [no-KB] Predicted: {lp_res_no_kb.label.value}")

    if lp_res_no_kb.error:
        log("  [no-KB] Call failed.")
        exp_result.status_no_kb = ExperimentStepStatus.ERROR
        exp_result.final_status = ExperimentStatus.ERROR_NO_KB
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result
         
    exp_result.status_no_kb = ExperimentStepStatus.SUCCESS

    # Early exit if only testing no-KB
    if test_mode == "no_kb":
        if lp_res_no_kb.label == prob.gold_label:
            exp_result.final_status = ExperimentStatus.ALREADY_CORRECT
        else:
            exp_result.final_status = ExperimentStatus.UNKNOWN # Or a specific WRONG status
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result

    if lp_res_no_kb.label == prob.gold_label:
        log("  [no-KB] Already correct.")
        exp_result.final_status = ExperimentStatus.ALREADY_CORRECT
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result

    if lp_res_no_kb.label != NLILabel.NEUTRAL:
        log("  [no-KB] Wrong and not neutral.")
        exp_result.final_status = ExperimentStatus.STILL_WRONG # Should be distinct if needed
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result

    # ----- Step 2: Generate KB -----
    try:
        kb_raw = call_llm(config.llm_provider, config.model, config.prompt_style, prob)
    except Exception as e:
        log(f"  [KB] LLM Call failed: {e}")
        exp_result.final_status = ExperimentStatus.LLM_ERROR
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result
        
    exp_result.kb_raw = kb_raw
    log(f"  [KB raw] Proposed: {kb_raw}")

    if not kb_raw:
        log("  [KB] Empty KB.")
        exp_result.final_status = ExperimentStatus.LLM_ERROR # Treat empty as error-like or just empty
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result

    # ----- Step 3: Test with RAW KB (if requested) -----
    raw_kb_fixed = False
    if test_mode in ("raw_kb", "both"):
        log("  [raw-KB] Calling LangPro with raw (unfiltered) KB...")
        lp_res_raw = langpro_api_call(prob.premises, prob.hypothesis, kb=kb_raw)
        
        if lp_res_raw.error:
            log("  [raw-KB] Call failed.")
            exp_result.status_with_raw_kb = ExperimentStepStatus.ERROR
        else:
            exp_result.pred_with_raw_kb = lp_res_raw.label
            exp_result.status_with_raw_kb = ExperimentStepStatus.SUCCESS
            exp_result.prover_calls.append(lp_res_raw)
            log(f"  [raw-KB] Predicted: {lp_res_raw.label.value}")
            
            if lp_res_raw.label == prob.gold_label:
                raw_kb_fixed = True
                log(f"  [raw-KB] ✔ RAW LLM KB fixed it!")
        
        # If only testing raw KB, we're done
        if test_mode == "raw_kb":
            if exp_result.pred_with_raw_kb == prob.gold_label:
                exp_result.final_status = ExperimentStatus.FIXED_RAW_KB
                exp_result.fixed_by = "raw_kb"
            else:
                exp_result.final_status = ExperimentStatus.STILL_WRONG_RAW_KB
            if cache_file:
                _save_cache_result(cache_file, exp_result)
            return exp_result

    # ----- Step 4: Filter KB -----
    kb_results = pipeline_filter_kb_injections(
        kb_raw,
        prob.premises,
        prob.hypothesis,
        post_process=config.post_process
    )
    
    kb_strings = [res.relation for res in kb_results]
    
    exp_result.kb_filtered = kb_strings
    exp_result.kb_details = kb_results

    log(f"  [KB filtered] {kb_strings}")

    if not kb_strings:
        log("  [KB] All KB filtered out.")
        exp_result.final_status = ExperimentStatus.EMPTY_KB_AFTER_FILTER
        # If raw KB fixed it but filtered is empty, raw_kb was the fixer
        if raw_kb_fixed:
            exp_result.fixed_by = "raw_kb"
            exp_result.final_status = ExperimentStatus.FIXED_RAW_KB
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result

    # ----- Step 5: Test with FILTERED KB -----
    # Check if filtered KB is identical to raw KB
    kb_identical = set(kb_strings) == set(kb_raw)
    if kb_identical and test_mode in ("raw_kb", "both"):
        # Filtered KB is same as raw KB - reuse the raw KB result
        log("  [KB] Filtered KB identical to raw KB - reusing raw KB result.")
        exp_result.pred_with_kb = exp_result.pred_with_raw_kb
        exp_result.status_with_kb = exp_result.status_with_raw_kb
        filtered_kb_fixed = raw_kb_fixed
    else:
        log("  [KB] Calling LangPro with filtered KB...")
        lp_res_with_kb = langpro_api_call(prob.premises, prob.hypothesis, kb=kb_strings)
        
        if lp_res_with_kb.error:
            log("  [KB] Call failed.")
            exp_result.status_with_kb = ExperimentStepStatus.ERROR
            exp_result.final_status = ExperimentStatus.ERROR_WITH_KB
            if cache_file:
                _save_cache_result(cache_file, exp_result)
            return exp_result

        exp_result.pred_with_kb = lp_res_with_kb.label
        exp_result.status_with_kb = ExperimentStepStatus.SUCCESS
        exp_result.prover_calls.append(lp_res_with_kb)
        filtered_kb_fixed = lp_res_with_kb.label == prob.gold_label
    
    log(f"  [KB] Predicted: {exp_result.pred_with_kb.value}")

    # Determine fixed_by based on what worked
    # Note: "both" only applies when filtering actually changed the KB
    if filtered_kb_fixed and raw_kb_fixed:
        if kb_identical:
            # Filtering was a no-op, so credit goes to raw KB only
            exp_result.fixed_by = "raw_kb"
            log(f"  [KB] ✔ RAW LLM KB fixed it (filtering made no changes).")
        else:
            exp_result.fixed_by = "both"
            log(f"  [KB] ✔ BOTH raw LLM KB and processed KB fixed it!")
        exp_result.final_status = ExperimentStatus.FIXED
    elif filtered_kb_fixed:
        exp_result.fixed_by = "filtered_kb"
        if test_mode in ("raw_kb", "both"):
            log(f"  [KB] ✔ PROCESSED KB fixed it! (raw LLM KB did NOT)")
        else:
            log(f"  [KB] ✔ PROCESSED KB fixed it! (raw KB not tested)")
        exp_result.final_status = ExperimentStatus.FIXED
    elif raw_kb_fixed:
        exp_result.fixed_by = "raw_kb"
        log(f"  [KB] ✘ Processed KB did NOT fix it, but RAW LLM KB did!")
        exp_result.final_status = ExperimentStatus.FIXED_RAW_KB
    else:
        log("  [KB] ✘ Neither raw nor processed KB fixed it.")
        exp_result.final_status = ExperimentStatus.STILL_WRONG
        
    # ----- Step 6: Ablation (if requested and something fixed it) -----
    if exp_result.fixed_by and config.run_ablation:
        # Run ablation on whichever KB fixed it
        if exp_result.fixed_by in ("filtered_kb", "both") and len(kb_strings) > 1:
            log("  [ablation] Running ablation on filtered KB to find minimal sufficient subsets...")
            essential_kb, all_minimals, ablation_results = _run_ablation(
                prob, kb_strings, prob.gold_label, config.verbose
            )
            exp_result.essential_kb = essential_kb
            exp_result.ablation_subsets = all_minimals
            exp_result.ablation_results = ablation_results
            log(f"  [ablation] Best minimal subset: {essential_kb}")
            log(f"  [ablation] All minimal subsets ({len(all_minimals)}): {all_minimals}")

    if cache_file:
        _save_cache_result(cache_file, exp_result)

    return exp_result


def _run_ablation(
    prob: NLIProblem,
    kb_list: List[str],
    gold_label: NLILabel,
    verbose: bool = True
) -> tuple:
    """
    Find all minimal sufficient KB subsets using breadth-first search.
    
    A subset is "minimal sufficient" if:
    1. It produces the correct gold_label when injected
    2. No proper subset of it also produces the correct label
    
    Returns:
        Tuple of (best_minimal_subset, all_minimal_subsets, ablation_log)
        - best_minimal_subset: The minimal subset with lowest token count
        - all_minimal_subsets: All minimal sufficient subsets found
        - ablation_log: Dict mapping tested subsets (as tuple) to resulting label
    """
    from itertools import combinations
    
    minimal_subsets: List[frozenset] = []
    ablation_log = {}
    
    def token_count(subset) -> int:
        """Count total tokens across all KB entries in subset."""
        return sum(len(s.split()) for s in subset)
    
    def test_subset(subset: List[str]) -> Optional[NLILabel]:
        """Test a KB subset and return the resulting label, or None on error."""
        if not subset:
            # Empty subset - call LangPro with no KB
            res = langpro_api_call(prob.premises, prob.hypothesis, report=False)
        else:
            res = langpro_api_call(prob.premises, prob.hypothesis, kb=subset)
        
        if res.error:
            return None
        return res.label
    
    # BFS by subset size: test all subsets of size 1, then 2, etc.
    for size in range(1, len(kb_list) + 1):
        if verbose:
            print(f"    [ablation] Testing subsets of size {size}...")
        
        for subset_tuple in combinations(kb_list, size):
            subset_set = frozenset(subset_tuple)
            
            # Skip if this subset is a superset of an already-found minimal
            if any(m.issubset(subset_set) and m != subset_set for m in minimal_subsets):
                if verbose:
                    print(f"    [ablation] Skipping {list(subset_tuple)} (superset of known minimal)")
                continue
            
            # Test this subset
            result_label = test_subset(list(subset_tuple))
            ablation_log[subset_tuple] = result_label
            
            if result_label is None:
                if verbose:
                    print(f"    [ablation] {list(subset_tuple)} -> ERROR")
                continue
            
            if result_label == gold_label:
                # This subset is sufficient - check if it's minimal
                # (no proper subset also works, but we already checked smaller sizes first)
                minimal_subsets.append(subset_set)
                if verbose:
                    print(f"    [ablation] {list(subset_tuple)} -> {result_label.value} (MINIMAL)")
            else:
                if verbose:
                    print(f"    [ablation] {list(subset_tuple)} -> {result_label.value}")
    
    # Convert frozensets to lists
    all_minimals = [list(s) for s in minimal_subsets]
    
    # Convert tuple keys to strings for serialization (join with |)
    ablation_log_str = {"|".join(k): v for k, v in ablation_log.items()}
    
    # Sort by token count and select best
    if all_minimals:
        sorted_minimals = sorted(all_minimals, key=token_count)
        best = sorted_minimals[0]
    else:
        best = []
    
    return best, all_minimals, ablation_log_str

def process_kb_examples(
    dataset: DatasetLoader,
    config: Optional[ProblemConfig] = None,
    split: str = "dev",
    label_filter: Set[str] = {"entailment", "contradiction"},
    max_matches: Optional[int] = None, # Renamed conceptual limit, acts as max_yields if needed
    max_checked: Optional[int] = 500,
    problem_ids: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None
) -> Iterator[ExperimentResult]:
    """
    Iterates through dataset examples and yields experiment results for ALL processed problems,
    regardless of outcome.
    
    Args:
        dataset: The dataset loader to use.
        config: Configuration for processing each problem.
        split: Dataset split to use.
        label_filter: Set of gold labels to consider.
        max_matches: Stop after yielding this many results (any outcome).
        max_checked: Maximum number of problems to check/process.
        problem_ids: Optional list of specific problem IDs to process.
        cache_dir: Optional directory for caching results.
    """
    if config is None:
        config = ProblemConfig()

    print(f"[kb-processor] Processing split='{split}', labels={label_filter}")
    print(f"[kb-processor] Config: mode={config.test_mode.value}, ablation={config.run_ablation}")

    checked_count = 0
    yielded_count = 0
    
    used_keys: Set[str] = set()
    
    problem_iterator: Iterator[str]
    is_random = False
    
    if problem_ids:
        print(f"[kb-processor] Using provided list of {len(problem_ids)} problems.")
        problem_iterator = iter(problem_ids)
    else:
        print(f"[kb-processor] Using random sampling.")
        is_random = True
        problem_iterator = iter([])

    while True:
        # Check limits
        if max_matches is not None and yielded_count >= max_matches:
            print(f"[kb-processor] Reached target of {max_matches} results. Stopping.")
            break
        if max_checked is not None and checked_count >= max_checked:
            print(f"[kb-processor] Reached max_checked limit ({max_checked}). Stopping.")
            break
        
        prob: Optional[NLIProblem] = None
        
        if is_random:
            prob = dataset.random_problem(split=split, label_filter=label_filter, exclude_keys=used_keys)
            if not prob:
                print("[kb-processor] No more valid random problems found.")
                break
        else:
            try:
                next_id = next(problem_iterator)
                try:
                    prob = dataset.get_problem(next_id, split=split)
                except KeyError:
                    print(f"[Warning] Problem ID {next_id} not found in dataset.")
                    continue
            except StopIteration:
                print("[kb-processor] Finished processing provided list.")
                break

        if prob.id in used_keys:
            continue
        used_keys.add(prob.id)
        
        cache_file = None
        if cache_dir:
            safe_id = prob.id.replace("/", "_").replace("#", "_")
            filename = f"{prob.dataset}_{split}_{safe_id}.json"
            cache_file = cache_dir / filename

            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_result = ExperimentResult.model_validate_json(f.read())

                    checked_count += 1
                    print(f"\n[kb-processor] #{checked_count} | Key: {prob.id} | Gold: {prob.gold_label.value} [CACHED]")
                    
                    if cached_result.final_status.startswith("fixed"):
                        print(f"  [KB] ✔ KB FIXED IT ({cached_result.fixed_by})")
                    
                    yielded_count += 1
                    yield cached_result
                    continue

                except Exception as e:
                    print(f"[cache] Error reading {cache_file}: {e}. Reprocessing.")

        checked_count += 1
        print(f"\n[kb-processor] #{checked_count} | Key: {prob.id} | Gold: {prob.gold_label.value}")

        # Delegate to single-problem processor
        exp_result = process_single_problem(
            prob,
            config=config,
            cache_file=cache_file
        )
        
        yielded_count += 1
        yield exp_result


def collect_kb_helpful_examples_random(
    dataset: DatasetLoader,
    config: Optional[ProblemConfig] = None,
    split: str = "dev",
    label_filter: Set[str] = {"entailment", "contradiction"},
    max_matches: Optional[int] = 10,
    max_checked: Optional[int] = 500,
    problem_ids: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None
) -> Iterator[ExperimentResult]:
    """
    Wrapper around process_kb_examples that ONLY yields results where KB was helpful
    (i.e., final_status starts with "fixed").
    """
    matches_found = 0
    
    # We pass max_matches=None to the inner loop because we want to control the count 
    # based on *matches*, not total processed.
    generator = process_kb_examples(
        dataset=dataset,
        config=config,
        split=split,
        label_filter=label_filter,
        max_matches=None, 
        max_checked=max_checked,
        problem_ids=problem_ids,
        cache_dir=cache_dir
    )
    
    for result in generator:
        if result.final_status.startswith("fixed"):
            matches_found += 1
            yield result
            
            if max_matches is not None and matches_found >= max_matches:
                print(f"[kb-collector] Reached target of {max_matches} helpful matches. Stopping.")
                break
                
    print(f"\n[kb-collector] Finished. Found {matches_found} KB-helpful examples.")