import re
from typing import Tuple, List
import nltk
from nltk.stem import WordNetLemmatizer
from .downloads import check_nltk

def tokenize(text: str):
    check_nltk('punkt_tab')
    return [t.lower() for t in nltk.word_tokenize(text)]

def remove_underscores(s: str) -> str:
    """
    Replace underscores with spaces.
    """
    return s.replace("_", " ")


def normalize_kb_args(kb: str) -> str:
    pred, a, b = parse_kb_injection(kb)
    a = remove_underscores(a)
    b = remove_underscores(b)
    return f"{pred}({a}, {b})"


def drop_leading_preposition(phrase: str) -> str:
    """
    If the phrase has exactly 3 tokens and the first is a preposition (IN),
    drop the first token. Otherwise, return the phrase unchanged.
    """
    check_nltk('punkt_tab')
    check_nltk('averaged_perceptron_tagger_eng')
    tokens = nltk.word_tokenize(phrase)

    if len(tokens) == 3:
        tagged = nltk.pos_tag(tokens)
        word, tag = tagged[0]

        if tag == "IN":   # preposition
            return " ".join(tokens[1:])

    return phrase


def parse_kb_injection(kb: str) -> Tuple[str, str, str]:
    """
    Parse predicate(arg1, arg2) into (predicate, arg1, arg2)
    Works for isa_wn, disj, and any other binary predicate.
    """
    m = re.fullmatch(r'\s*(\w+)\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)\s*', kb)
    if not m:
        raise ValueError(f"Invalid KB format: {kb}")
    return m.group(1), m.group(2), m.group(3)


from .models import KBResult

ALLOWED_PREDICATES = {"isa_wn", "disj"}

def filter_kb_by_prem_hyp(
    kb_list,
    premise,
    hypothesis,
    st_model=None,
    threshold=0.60,
    swap_args=True,
    strict=True,
    use_token_level=True,
) -> List[KBResult]:
    """
    Filters KB injections using sentence-transformers soft matching and optionally adds swapped relations.

    Args:
        kb_list: List of strings (raw) or List of KBResult (pre-processed).
    """
    check_nltk('wordnet')
    lemmatizer = WordNetLemmatizer()
    from sentence_transformers import SentenceTransformer, util

    if st_model is None:
        st_model = SentenceTransformer("all-MiniLM-L6-v2")

    prem_tokens = tokenize(premise)
    hyp_tokens = tokenize(hypothesis)
    all_tokens = prem_tokens + hyp_tokens

    filtered_kb: List[KBResult] = []
    seen_relations = set()

    for kb_item in kb_list:
        # Normalize input to KBResult if it's a string
        if isinstance(kb_item, str):
            item_obj = KBResult(relation=kb_item, provenance="llm", original_text=kb_item)
        else:
            item_obj = kb_item

        try:
            parsed = parse_kb_injection(item_obj.relation)
        except ValueError:
            continue
            
        if not parsed:
            continue

        pred, arg1, arg2 = parsed

        # Generate candidates: original and optionally "diff-only" pair + swapped for disj
        # Store metadata about how each candidate was derived
        candidates = [] # Tuple[pred, arg1, arg2, provenance_type]
        
        candidates.append((pred, arg1, arg2, item_obj.provenance))

        lemma_a1 = " ".join([lemmatizer.lemmatize(t, pos='v') for t in tokenize(arg1)])
        lemma_a2 = " ".join([lemmatizer.lemmatize(t, pos='v') for t in tokenize(arg2)])
        
        if lemma_a1 != arg1.lower() or lemma_a2 != arg2.lower():
             candidates.append((pred, lemma_a1, lemma_a2, "derived_lemma"))

        t1, t2 = arg1.split(), arg2.split()
        if len(t1) == len(t2):
            diffs = [(w1, w2) for w1, w2 in zip(t1, t2) if w1.lower() != w2.lower()]
            if diffs:
                diff_a1 = " ".join(d[0] for d in diffs)
                diff_a2 = " ".join(d[1] for d in diffs)
                candidates.append((pred, diff_a1, diff_a2, "derived_diff"))

        final_candidates = []
        for p, a1, a2, prov in candidates:
            if a1.lower() == a2.lower():
                continue
            final_candidates.append((p, a1, a2, prov))
            if swap_args and p == "disj":
                final_candidates.append((p, a2, a1, "derived_swap"))

        for p, a1, a2, prov in final_candidates:
            left_tokens  = prem_tokens if strict else all_tokens
            right_tokens = hyp_tokens  if strict else all_tokens

            if is_arg_supported(a1, left_tokens,  st_model, threshold, use_token_level=use_token_level) and \
               is_arg_supported(a2, right_tokens, st_model, threshold, use_token_level=use_token_level):

                valid_rel_str = create_rel(p, a1, a2)
                
                if valid_rel_str not in seen_relations:
                    seen_relations.add(valid_rel_str)
                    
                    # Create result object with provenance
                    # If the relation string changed from the original input item, it's a derived form
                    res = KBResult(
                        relation=valid_rel_str,
                        provenance=prov,
                        original_text=item_obj.original_text
                    )
                    filtered_kb.append(res)

    return filtered_kb


def is_arg_supported(arg_str, tokens, st_model=None, threshold=0.60, use_token_level=True):
    """
    Checks if a (possibly multi-word) argument is supported by the text tokens.

    Strategy:
      1) Exact match / lemma match (fast).
      2) Sentence-transformers similarity:
         - If use_token_level=True: compare each arg part to the best matching token.
         - Else: compare full arg phrase to the full text (joined tokens).
    """
    check_nltk('wordnet')
    lemmatizer = WordNetLemmatizer()
    text_str = " ".join(tokens).lower()
    arg_str = (arg_str or "").strip().lower()

    if not arg_str:
        return False

    # Fallback if no model
    if st_model is None:
        return arg_str in text_str

    token_set = {t.lower() for t in tokens}
    token_lemmas = {lemmatizer.lemmatize(t.lower()) for t in tokens}
    token_lemmas_v = {lemmatizer.lemmatize(t.lower(), pos='v') for t in tokens}

    # For multi-word args like "blue shirt", check if all parts are supported
    parts = tokenize(arg_str)
    if not parts:
        return False

    # --- Fast checks first ---
    for part in parts:
        part_l = part.lower()

        # exact token match
        if part_l in token_set:
            continue

        # lemma match (Noun and Verb)
        if lemmatizer.lemmatize(part_l) in token_lemmas:
            continue
        if lemmatizer.lemmatize(part_l, pos='v') in token_lemmas_v:
            continue

        # --- Semantic soft match ---
        if use_token_level:
            # Compare this part against all tokens; take best cosine similarity
            # (Small optimization: deduplicate tokens)
            cand_tokens = list({t.lower() for t in tokens if t})
            if not cand_tokens:
                return False

            emb = st_model.encode([part_l] + cand_tokens, convert_to_tensor=True)
            part_emb = emb[0]
            token_embs = emb[1:]

            from sentence_transformers import util
            sims = util.cos_sim(part_emb, token_embs)  # shape (1, N)
            best_score = float(sims.max().item())

            if best_score < threshold:
                return False
        else:
            # Compare the whole argument phrase to the whole text
            emb = st_model.encode([arg_str, text_str], convert_to_tensor=True)
            from sentence_transformers import util
            score = float(util.cos_sim(emb[0], emb[1]).item())
            if score < threshold:
                return False

    return True


def create_rel(pred, arg1, arg2):
    return f"{pred}({arg1}, {arg2})"

def pipeline_filter_kb_injections(
    kb_list: List[str],
    premise: str,
    hypothesis: str,
    st_model=None,
    post_process: bool = True
) -> List[KBResult]:
    """
    Pipeline:
      1) parse predicate(arg1, arg2)
      2) keep only isa_wn / disj
      3) (Optional) remove underscores in args
      4) (Optional) drop leading preposition for 3-token phrases
      5) keep injections where arg1 in premise and arg2 in hypothesis (exact string match check)
      6) Soft filter (Word2Vec/SentenceTransformer)
    """
    
    # Prepare lemmatized versions of premise and hypothesis for containment check
    check_nltk('wordnet')
    lemmatizer = WordNetLemmatizer()
    
    # helper to lemmatize a full sentence
    def get_lemma_str(text):
        return " ".join([lemmatizer.lemmatize(t.lower(), pos='v') for t in tokenize(text)])

    # helper to remove determiners for loose matching
    def remove_determiners(text):
        tokens = tokenize(text)
        return " ".join([t for t in tokens if t not in {"a", "an", "the"}])

    prem_lemma = get_lemma_str(premise)
    hyp_lemma = get_lemma_str(hypothesis)
    
    current_premise_lower = premise.lower()
    current_hypothesis_lower = hypothesis.lower()
    
    # Normalized versions (no determiners)
    prem_norm = remove_determiners(premise)
    hyp_norm = remove_determiners(hypothesis)

    pre_processed_items: List[KBResult] = []

    for inj in kb_list:
        try:
            pred, a, b = parse_kb_injection(inj)
        except ValueError:
            continue  # skip malformed entries

        # 🔒 1) FILTER predicate type
        if pred not in ALLOWED_PREDICATES:
            continue

        provenance = "llm"
        original_a, original_b = a, b

        if post_process:
            # 2) underscores → spaces
            a = remove_underscores(a)
            b = remove_underscores(b)

            # 3) drop leading preposition (3-token phrases only)
            a = drop_leading_preposition(a)
            b = drop_leading_preposition(b)
            
            if a != original_a or b != original_b:
                provenance = "post_process"

        # 4) prem / hyp containment
        # Check if arguments are present in either formatted text or lemmatized text
        a_lower = a.lower()
        b_lower = b.lower()
        
        # Lemmatize the args too for comparison against lemmatized sentences
        a_lemma = get_lemma_str(a)
        b_lemma = get_lemma_str(b)
        
        # Normalize args (remove determiners)
        a_norm = remove_determiners(a)
        b_norm = remove_determiners(b)

        in_premise = (a_lower in current_premise_lower) or \
                     (a_lemma in prem_lemma) or \
                     (a_norm in prem_norm)

        in_hypothesis = (b_lower in current_hypothesis_lower) or \
                        (b_lemma in hyp_lemma) or \
                        (b_norm in hyp_norm)

        if in_premise and in_hypothesis:
            # Reconstruct the relation string (potentially modified)
            clean_rel = create_rel(pred, a, b)
            
            pre_processed_items.append(
                KBResult(
                    relation=clean_rel,
                    provenance=provenance,
                    original_text=inj 
                )
            )

    # 6) Final Soft Filter
    return filter_kb_by_prem_hyp(
        pre_processed_items,
        premise,
        hypothesis,
        st_model=st_model,
        threshold=0.60,
        strict=True
    )
