prompts = {
  "prompts": [
    {
      "name": "cot",
      "description": "Chain-of-thought: reason internally then emit only KB facts.",
      "template": "Convert the premise and hypothesis into concise KB injections. Think step by step and show your reasoning clearly. Break the problem into intermediate steps, explain each step, and then give the final answer. Reason step by step internally and return ONLY facts as predicate(arg1, arg2). Don't use underscores in the arguments and use max 2 words per relation\nPremise: ${premise}\nHypothesis: ${hypothesis}\nKBinjections:"
    },
    {
      "name": "least_to_most",
      "description": "Least-to-most: decompose the task into minimal facts before composing the KB.",
      "template": "Your job is to break complex questions into a sequence of simpler subproblems that can be solved one by one.\nEach subproblem should be small, explicit, and should not require reasoning about later steps.\nAfter planning, output ONLY the final KB injection as predicate(arg1, arg2). Avoid natural language.Don't use underscores in the arguments and use max 2 words per relation \nPremise: ${premise}\nHypothesis: ${hypothesis}\nKBinjection:"
    },
    {
      "name": "icl",
      "description": "In-context learning with a few exemplars.",
      "template": "You are learning from the examples to emit KB injections only.\nRules:\n\n* Output ONLY KB facts as predicate(arg1, arg2) on separate lines (no extra text).\n* Only use the relations: isa_wn and disj.\n* Don't use underscores in the arguments.\n* Use max 2 words per argument.\n* PREFER BASE FORMS (LEMMAS) for single words (e.g., use \"run\" instead of \"running\", \"eat\" instead of \"eating\").\n* Focus on the meaning shift between Premise and Hypothesis.\n\nExample 1:\nPremise: A little girl in pink boots runs down the street.\nHypothesis: A human is running outdoors.\nKnowledge injection:\nisa_wn(girl, human)\nisa_wn(street, outdoors)\n\nExample 2:\nPremise: A woman soaks her feet in a natural pool in a landscape of rocks, with green covered tents in the background.\nHypothesis: A lady is outdoors\nKnowledge injection:\nisa_wn(woman, lady)\n\nExample 3:\nPremise: A man and a woman hug on a grassy hillside overlooking the countryside in the distance.\nHypothesis: The man and woman are outdoors.\nKnowledge injection:\nisa_wn(hillside, outdoors)\n\nExample 4:\nPremise: A female swimmer getting out of the pool still dripping wet.\nHypothesis: A woman gets out of the pool.\nKnowledge injection:\nisa_wn(female swimmer, woman)\n\nExample 5:\nPremise: A lady looks to her right and holds a mini camera.\nHypothesis: A lady is holding a device.\nKnowledge injection:\nisa_wn(mini camera, device)\n\nExample 6:\nPremise: A dog runs along the shore of a pond with two elegant geese swimming.\nHypothesis: A dog runs along the edge of a pond outdoors.\nKnowledge injection:\nisa_wn(shore, edge)\nisa_wn(pond, outdoors)\n\nExample 7:\nPremise: A man wearing a white shirt is playing the drums.\nHypothesis: A man is playing a musical instrument.\nKnowledge injection:\nisa_wn(drum, musical instrument)\n\nExample 8:\nPremise: A little boy wearing a blue striped shirt has a party hat on his head and is playing in a puddle.\nHypothesis: The party boy is playing in a puddle.\nKnowledge injection:\nisa_wn(little boy, party boy)\n\nExample 9:\nPremise: A little girl is sitting on the counter dangling one foot in the sink whilst holding a dish jet washer.\nHypothesis: A human sitting\nKnowledge injection:\nisa_wn(girl, human)\n\nNow do the next one. You can think of more injections than one, if needed.\nPremise: ${premise}\nHypothesis: ${hypothesis}\nKnowledge injection:\n"
    }
  ]
}

def get_prompt(prompt_name: str) -> str:
    """Get a prompt template by name from the in-memory dictionary `prompts`. """
    for prompt in prompts.get("prompts", []):
        if prompt.get("name") == prompt_name:
            return prompt.get("template", "")
    raise KeyError(f"Prompt '{prompt_name}' not found")

def fill_prompt(prompt_name: str, premise: str, hypothesis: str) -> str:
    """Get a prompt template and fill in the premise and hypothesis."""
    template_str = get_prompt(prompt_name)
    return template_str.replace("${premise}", premise).replace("${hypothesis}", hypothesis)
