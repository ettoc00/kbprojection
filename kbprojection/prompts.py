"""
KB Injection Prompts for NLI Problems

Contains both legacy prompts (preserved for backward compatibility) and 
improved v2 prompts based on cross-analysis findings.
"""

# =============================================================================
# LEGACY PROMPTS (Preserved for backward compatibility)
# =============================================================================

legacy_prompts = {
    "prompts": [
        {
            "name": "legacy_cot",
            "description": "Chain-of-thought: reason internally then emit only KB facts.",
            "template": "Convert the premise and hypothesis into concise KB injections. Think step by step and show your reasoning clearly. Break the problem into intermediate steps, explain each step, and then give the final answer. Reason step by step internally and return ONLY facts as predicate(arg1, arg2). Don't use underscores in the arguments and use max 2 words per relation\nPremise: ${premise}\nHypothesis: ${hypothesis}\nKBinjections:"
        },
        {
            "name": "legacy_least_to_most",
            "description": "Least-to-most: decompose the task into minimal facts before composing the KB.",
            "template": "Your job is to break complex questions into a sequence of simpler subproblems that can be solved one by one.\nEach subproblem should be small, explicit, and should not require reasoning about later steps.\nAfter planning, output ONLY the final KB injection as predicate(arg1, arg2). Avoid natural language.Don't use underscores in the arguments and use max 2 words per relation \nPremise: ${premise}\nHypothesis: ${hypothesis}\nKBinjection:"
        },
        {
            "name": "legacy_icl",
            "description": "In-context learning with a few exemplars.",
            "template": "You are learning from the examples to emit KB injections only.\nRules:\n\n* Output ONLY KB facts as predicate(arg1, arg2) on separate lines (no extra text).\n* Only use the relations: isa_wn and disj.\n* Don't use underscores in the arguments.\n* Use max 2 words per argument.\n* PREFER BASE FORMS (LEMMAS) for single words (e.g., use \"run\" instead of \"running\", \"eat\" instead of \"eating\").\n* Focus on the meaning shift between Premise and Hypothesis.\n\nExample 1:\nPremise: A little girl in pink boots runs down the street.\nHypothesis: A human is running outdoors.\nKnowledge injection:\nisa_wn(girl, human)\nisa_wn(street, outdoors)\n\nExample 2:\nPremise: A woman soaks her feet in a natural pool in a landscape of rocks, with green covered tents in the background.\nHypothesis: A lady is outdoors\nKnowledge injection:\nisa_wn(woman, lady)\n\nExample 3:\nPremise: A man and a woman hug on a grassy hillside overlooking the countryside in the distance.\nHypothesis: The man and woman are outdoors.\nKnowledge injection:\nisa_wn(hillside, outdoors)\n\nExample 4:\nPremise: A female swimmer getting out of the pool still dripping wet.\nHypothesis: A woman gets out of the pool.\nKnowledge injection:\nisa_wn(female swimmer, woman)\n\nExample 5:\nPremise: A lady looks to her right and holds a mini camera.\nHypothesis: A lady is holding a device.\nKnowledge injection:\nisa_wn(mini camera, device)\n\nExample 6:\nPremise: A dog runs along the shore of a pond with two elegant geese swimming.\nHypothesis: A dog runs along the edge of a pond outdoors.\nKnowledge injection:\nisa_wn(shore, edge)\nisa_wn(pond, outdoors)\n\nExample 7:\nPremise: A man wearing a white shirt is playing the drums.\nHypothesis: A man is playing a musical instrument.\nKnowledge injection:\nisa_wn(drum, musical instrument)\n\nExample 8:\nPremise: A little boy wearing a blue striped shirt has a party hat on his head and is playing in a puddle.\nHypothesis: The party boy is playing in a puddle.\nKnowledge injection:\nisa_wn(little boy, party boy)\n\nExample 9:\nPremise: A little girl is sitting on the counter dangling one foot in the sink whilst holding a dish jet washer.\nHypothesis: A human sitting\nKnowledge injection:\nisa_wn(girl, human)\n\nNow do the next one. You can think of more injections than one, if needed.\nPremise: ${premise}\nHypothesis: ${hypothesis}\nKnowledge injection:\n"
        }
    ]
}

# =============================================================================
# IMPROVED V2 PROMPTS
# Based on cross-analysis findings:
# 1. Stricter format constraints with explicit delimiters
# 2. ALWAYS use base forms (lemmas) - not just "prefer"
# 3. Examples showing both isa_wn AND disj usage
# 4. Negative examples for common mistakes
# 5. Reverse-direction examples (hypothesis more specific than premise)
# =============================================================================

ICL_V2_TEMPLATE = """You are a knowledge base injection assistant. Your task is to generate semantic relations that bridge the meaning gap between a Premise and Hypothesis.

## STRICT RULES (Must follow exactly)

1. **Output format**: Your output MUST be enclosed in delimiters:
   [KB_START]
   predicate(arg1, arg2)
   [KB_END]

2. **Allowed predicates only**:
   - `isa_wn(X, Y)` = X is a type/kind of Y (hypernymy). Example: isa_wn(dog, animal)
   - `disj(X, Y)` = X and Y are mutually exclusive (cannot both be true). Example: disj(sit, stand)

3. **ALWAYS use base forms (lemmas)**:
   - CORRECT: isa_wn(run, move)
   - WRONG: isa_wn(running, moving)
   - CORRECT: disj(walk, ride)
   - WRONG: disj(walking, riding)

4. **Max 2 words per argument**, no underscores.

5. **When to use each predicate**:
   - Use `isa_wn` when: one term is a more specific version of another
   - Use `disj` ONLY for clear opposites: sit/stand, hot/cold, red/blue, smile/frown
   - Do NOT use `disj` when unsure - it's better to omit than hallucinate

## EXAMPLES

### Example 1: Basic hypernymy (Premise → Hypothesis generalization)
Premise: A little girl runs down the street.
Hypothesis: A human is running outdoors.
[KB_START]
isa_wn(girl, human)
isa_wn(street, outdoors)
[KB_END]

### Example 2: Reverse direction (Hypothesis has the specific term)
Premise: An animal is running in the park.
Hypothesis: A dog is running in the park.
[KB_START]
isa_wn(dog, animal)
[KB_END]

### Example 3: Multiple relations
Premise: A man plays the drums on stage.
Hypothesis: A person plays a musical instrument indoors.
[KB_START]
isa_wn(man, person)
isa_wn(drum, musical instrument)
isa_wn(stage, indoors)
[KB_END]

### Example 4: Using disj for clear opposites
Premise: A woman is smiling at the camera.
Hypothesis: A woman is frowning at the camera.
[KB_START]
disj(smile, frown)
[KB_END]

### Example 5: Using disj for action opposites
Premise: People sit on the curb watching an event.
Hypothesis: People stand while watching an event.
[KB_START]
disj(sit, stand)
[KB_END]

### Example 6: Using disj for color opposites
Premise: A car painted red drives down the highway.
Hypothesis: A car painted blue drives down the highway.
[KB_START]
disj(red, blue)
[KB_END]

### Example 7: When NOT to use disj (uncertain relationship)
Premise: A man is aiming a gun.
Hypothesis: A man is drawing a gun.
[KB_START]
isa_wn(aim, draw)
[KB_END]
Note: "aim" and "draw" are NOT opposites - aiming can follow drawing. Use isa_wn if one action implies the other.

### Example 8: Compound noun hypernymy
Premise: A female swimmer exits the pool.
Hypothesis: A woman gets out of the pool.
[KB_START]
isa_wn(female swimmer, woman)
[KB_END]

## YOUR TASK

Premise: ${premise}
Hypothesis: ${hypothesis}

Generate the knowledge injections:
"""

COT_V2_TEMPLATE = """You are a knowledge base injection assistant using chain-of-thought reasoning.

## TASK
Analyze the semantic gap between Premise and Hypothesis, then generate bridging KB facts.

## RULES
1. Output MUST be enclosed in [KB_START] and [KB_END] delimiters
2. Only use: isa_wn(X, Y) for "X is a type of Y", disj(X, Y) for "X and Y are mutually exclusive"
3. ALWAYS use base forms: "run" not "running", "eat" not "eating"
4. Max 2 words per argument, no underscores
5. Use disj ONLY for clear opposites (sit/stand, hot/cold, red/blue)

## REASONING STEPS

1. Identify key terms in Premise
2. Identify key terms in Hypothesis  
3. Find semantic gaps (terms that don't match but should connect)
4. For each gap: decide if it's a hypernymy (isa_wn) or opposition (disj)
5. Generate lemmatized KB facts

## EXAMPLE

Premise: A young girl is running through a field.
Hypothesis: A child moves through a meadow.

Reasoning:
- Premise terms: girl, run, field
- Hypothesis terms: child, move, meadow
- Gaps: girl→child, run→move, field→meadow
- Relations: girl IS A child (isa_wn), run IS A type of move (isa_wn), field IS A meadow (isa_wn)

[KB_START]
isa_wn(girl, child)
isa_wn(run, move)
isa_wn(field, meadow)
[KB_END]

## YOUR TASK

Premise: ${premise}
Hypothesis: ${hypothesis}

First reason step-by-step, then output KB facts in delimiters:
"""

# Combined prompt registry (for backward compatibility, maps old names to legacy)
prompts = {
    "prompts": [
        # Legacy prompts
        {
            "name": "legacy_cot",
            "description": "[LEGACY] Chain-of-thought prompt (use cot for improved version)",
            "template": legacy_prompts["prompts"][0]["template"]
        },
        {
            "name": "legacy_least_to_most",
            "description": "[LEGACY] Least-to-most decomposition prompt",
            "template": legacy_prompts["prompts"][1]["template"]
        },
        {
            "name": "legacy_icl",
            "description": "[LEGACY] In-context learning prompt (use icl for improved version)",
            "template": legacy_prompts["prompts"][2]["template"]
        },
        # Improved prompts (now default)
        {
            "name": "icl",
            "description": "Improved ICL with strict formatting, lemma rules, and disj examples",
            "template": ICL_V2_TEMPLATE
        },
        {
            "name": "cot", 
            "description": "Improved CoT with strict formatting and structured reasoning",
            "template": COT_V2_TEMPLATE
        }
    ]
}


def get_prompt(prompt_name: str) -> str:
    """Get a prompt template by name from the in-memory dictionary `prompts`."""
    for prompt in prompts.get("prompts", []):
        if prompt.get("name") == prompt_name:
            return prompt.get("template", "")
    raise KeyError(f"Prompt '{prompt_name}' not found. Available: {[p['name'] for p in prompts['prompts']]}")


def fill_prompt(prompt_name: str, premises: list, hypothesis: str) -> str:
    """Get a prompt template and fill in the premises and hypothesis."""
    template_str = get_prompt(prompt_name)
    # Join multiple premises with newlines
    premise_text = "\n".join(premises) if isinstance(premises, list) else premises
    return template_str.replace("${premise}", premise_text).replace("${hypothesis}", hypothesis)


def list_prompts() -> list:
    """Return a list of available prompt names."""
    return [p["name"] for p in prompts.get("prompts", [])]
