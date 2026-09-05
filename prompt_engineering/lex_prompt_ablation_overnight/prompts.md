# LEX prompt ablation

The original final-query paragraph is excluded from every condition.
The premise and hypothesis are appended through the shared runtime input.

## lex_zero_shot_base

LEX task section with the constraint and example sections removed; the item is supplied separately.

```text
You are an expert in linguistic semantics and logic. You will receive a Natural Language Inference (NLI) problem in English, consisting of a premise sentence and a hypothesis sentence.
You will reason carefully and decide whether the premise entails the hypothesis, which means that if the premise is true, then the hypothesis must also be true under ordinary English meaning and widely accepted background knowledge.
If the answer is "entailment", output a structured explanation that is a set of lexical entailment relations over short phrases that explain why the hypothesis is entailed from the premise.
Lexical entailment should be defined over short phrases that are lemmatized or normalized versions of short phrases occurring in the premise and the hypothesis, e.g., entails(phrase_1, phrase_2), and it means that phrase_1 is a type of phrase_2, for example, entails(woman, person), entails(dog, domestic animal), entails(huge, very big), and entails(run, move fast).
Use lexical entailment relations only when they are needed to explain the entailment. If the entailment follows without any non-trivial lexical relation, output an empty set.

input:
	premise: ${PREMISE}
	hypothesis: ${HYPOTHESIS}
```

## lex_zero_shot_constrained

LEX task and relation-formatting sections with only the worked examples and original final-query paragraph removed.

```text
You are an expert in linguistic semantics and logic. You will receive a Natural Language Inference (NLI) problem in English, consisting of a premise sentence and a hypothesis sentence.
You will reason carefully and decide whether the premise entails the hypothesis, which means that if the premise is true, then the hypothesis must also be true under ordinary English meaning and widely accepted background knowledge.
If the answer is "entailment", output a structured explanation that is a set of lexical entailment relations over short phrases that explain why the hypothesis is entailed from the premise.
Lexical entailment should be defined over short phrases that are lemmatized or normalized versions of short phrases occurring in the premise and the hypothesis, e.g., entails(phrase_1, phrase_2), and it means that phrase_1 is a type of phrase_2, for example, entails(woman, person), entails(dog, domestic animal), entails(huge, very big), and entails(run, move fast).
Use lexical entailment relations only when they are needed to explain the entailment. If the entailment follows without any non-trivial lexical relation, output an empty set.

Relation formatting rules:
The meaning of a lexical entailment relation has to be acceptable based on common sense, e.g., entails(woman, blond person) is not acceptable.
A lexical entailment may not express a trivial relation that is obtainable by discarding modifiers, e.g., entails(blond woman, woman) is not acceptable.
A lexical entailment may not contain redundant words such as auxiliary verbs and the infinitive "to", e.g., entails(will walk, will move), entails(is red, is colored), and entails(to walk, to move) are not acceptable.
Phrases in a lexical entailment have to contain lemmatized words, e.g., entails(dogs, domestic animals) is not acceptable.
Phrases in a lexical entailment may not contain determiners, e.g., entails(a dog, a domestic animal) is not acceptable.
Phrases in a lexical entailment may not contain prepositional phrases, e.g., entails(dog with spots, a domestic animal) is not acceptable.

input:
	premise: ${PREMISE}
	hypothesis: ${HYPOTHESIS}
```

## lex_few_shot_base

LEX task, constraint, and example sections; the original final-query paragraph is removed and the item is supplied separately.

```text
You are an expert in linguistic semantics and logic. You will receive a Natural Language Inference (NLI) problem in English, consisting of a premise sentence and a hypothesis sentence.
You will reason carefully and decide whether the premise entails the hypothesis, which means that if the premise is true, then the hypothesis must also be true under ordinary English meaning and widely accepted background knowledge.
If the answer is "entailment", output a structured explanation that is a set of lexical entailment relations over short phrases that explain why the hypothesis is entailed from the premise.
Lexical entailment should be defined over short phrases that are lemmatized or normalized versions of short phrases occurring in the premise and the hypothesis, e.g., entails(phrase_1, phrase_2), and it means that phrase_1 is a type of phrase_2, for example, entails(woman, person), entails(dog, domestic animal), entails(huge, very big), and entails(run, move fast).
Use lexical entailment relations only when they are needed to explain the entailment. If the entailment follows without any non-trivial lexical relation, output an empty set.

Relation formatting rules:
The meaning of a lexical entailment relation has to be acceptable based on common sense, e.g., entails(woman, blond person) is not acceptable.
A lexical entailment may not express a trivial relation that is obtainable by discarding modifiers, e.g., entails(blond woman, woman) is not acceptable.
A lexical entailment may not contain redundant words such as auxiliary verbs and the infinitive "to", e.g., entails(will walk, will move), entails(is red, is colored), and entails(to walk, to move) are not acceptable.
Phrases in a lexical entailment have to contain lemmatized words, e.g., entails(dogs, domestic animals) is not acceptable.
Phrases in a lexical entailment may not contain determiners, e.g., entails(a dog, a domestic animal) is not acceptable.
Phrases in a lexical entailment may not contain prepositional phrases, e.g., entails(dog with spots, a domestic animal) is not acceptable.

Examples:
Example 1:
	input:
		premise: Young ladies are playing the guitar.
		hypothesis: A musical instrument is being played by girls.
	correct output:
		answer: entailment
		relations: { entails(young lady, girl), entails(guitar, musical instrument) }
	unwanted output:
		relations: { entails(young ladies, girls), entails(the guitar, a musical instrument) }
		explanation: phrases in relations may not have determiners, e.g., "a" and "the". "ladies" and "girls" have to use lemmas "lady" and "girl", respectively.

Example 2:
	input:
		premise: A female swimmer getting out of the pool still dripping wet.
		hypothesis: A woman gets out of the pool.
	correct output:
		answer: entailment
		relations: { entails(female swimmer, woman) }
	unwanted output:
		relations: { entails(swimmer, woman) }
		explanation: it is a factually wrong relation because not every swimmer is a woman

Example 3:
	input:
		premise: A young girl wearing a pink coat plays with a yellow toy.
		hypothesis: A kid is swinging a toy golf club.
	correct output:
		answer: non-entailment

Example 4:
	input:
		premise: A black race car starts up in front of a crowd of people.
		hypothesis: A car is running.
	correct output:
		answer: entailment
		relations: { entails(start up, run) }
	unwanted output:
		relations: { entails(starts up, is running) }
		explanation: "starts up" and "is running" do not contain lemmatized words, and "is" is unnecessary in "is running".

Example 5:
	input:
		premise: A woman dressed in red clothing is dancing inside a crowd of people.
		hypothesis: A woman in red is dancing in a crowd.
	correct output:
		answer: entailment
		relations: { }
	unwanted output:
		relations: { entails(in red clothing, in red) }
		explanation: The phrases in the relation may not include prepositional phrases, e.g., "in red clothing"

Example 6:
	input:
		premise: A tall man with a cap is climbing a cord.
		hypothesis: The man in a hat is climbing a rope.
	correct output:
		answer: entailment
		relations: { entails(cap, hat), entails(cord, rope) }
	unwanted output:
		relations: { entails(cap, hat), entails(tall man, man) }
		explanation: "entails(cord, rope)" is missing. entails(tall man, man) is trivial since it includes dropping the adjective "tall".

Example 7:
	input:
		premise: No person is cooking.
		hypothesis: No cook is cooking in the kitchen.
	correct output:
		answer: entailment
		relations: { entails(cook, person) }
	unwanted output:
		relations: { entails(person, cook) }
		explanation: The relation does not help to explain "entailment", taking into account that negation reverses a lexical entailment direction. It is also a factually wrong relation because not every person is a cook.

Example 8:
	input:
		premise: A person who is obese is holding a chinchilla.
		hypothesis: A fat person is holding a small animal.
	correct output:
		answer: entailment
		relations: { entails(chinchilla, small animal), entails(obese, fat) }
	unwanted output:
		relations: { entails(fat, obese), entails(chinchilla, animal) }
		explanation: "entails(fat, obese)" needs to reverse its arguments to align with the entailment direction. "entails(chinchilla, animal)" is not sufficient to explain "entailment" since it misses "small", which is crucial.

Example 9:
	input:
		premise: A little boy is laughing and happily bouncing on a trampoline outside.
		hypothesis: The child is jumping outdoors.
	correct output:
		answer: entailment
		relations: { entails(little boy, child), entails(bounce, jump), entails(outside, outdoors) }
	unwanted output:
		relations: { entails(boy, child), entails(outdoors, outside) }
		explanation: "entails(bounce, jump)" is missing. "entails(little boy, child)" is preferred over "entails(boy, child)" as the former is more acceptable. "entails(outdoors, outside)" needs to reverse its arguments to align it to the entailment direction.

input:
	premise: ${PREMISE}
	hypothesis: ${HYPOTHESIS}
```

## lex_few_shot_precision

The few-shot condition plus the precision calibration block.

```text
You are an expert in linguistic semantics and logic. You will receive a Natural Language Inference (NLI) problem in English, consisting of a premise sentence and a hypothesis sentence.
You will reason carefully and decide whether the premise entails the hypothesis, which means that if the premise is true, then the hypothesis must also be true under ordinary English meaning and widely accepted background knowledge.
If the answer is "entailment", output a structured explanation that is a set of lexical entailment relations over short phrases that explain why the hypothesis is entailed from the premise.
Lexical entailment should be defined over short phrases that are lemmatized or normalized versions of short phrases occurring in the premise and the hypothesis, e.g., entails(phrase_1, phrase_2), and it means that phrase_1 is a type of phrase_2, for example, entails(woman, person), entails(dog, domestic animal), entails(huge, very big), and entails(run, move fast).
Use lexical entailment relations only when they are needed to explain the entailment. If the entailment follows without any non-trivial lexical relation, output an empty set.

Relation formatting rules:
The meaning of a lexical entailment relation has to be acceptable based on common sense, e.g., entails(woman, blond person) is not acceptable.
A lexical entailment may not express a trivial relation that is obtainable by discarding modifiers, e.g., entails(blond woman, woman) is not acceptable.
A lexical entailment may not contain redundant words such as auxiliary verbs and the infinitive "to", e.g., entails(will walk, will move), entails(is red, is colored), and entails(to walk, to move) are not acceptable.
Phrases in a lexical entailment have to contain lemmatized words, e.g., entails(dogs, domestic animals) is not acceptable.
Phrases in a lexical entailment may not contain determiners, e.g., entails(a dog, a domestic animal) is not acceptable.
Phrases in a lexical entailment may not contain prepositional phrases, e.g., entails(dog with spots, a domestic animal) is not acceptable.

Examples:
Example 1:
	input:
		premise: Young ladies are playing the guitar.
		hypothesis: A musical instrument is being played by girls.
	correct output:
		answer: entailment
		relations: { entails(young lady, girl), entails(guitar, musical instrument) }
	unwanted output:
		relations: { entails(young ladies, girls), entails(the guitar, a musical instrument) }
		explanation: phrases in relations may not have determiners, e.g., "a" and "the". "ladies" and "girls" have to use lemmas "lady" and "girl", respectively.

Example 2:
	input:
		premise: A female swimmer getting out of the pool still dripping wet.
		hypothesis: A woman gets out of the pool.
	correct output:
		answer: entailment
		relations: { entails(female swimmer, woman) }
	unwanted output:
		relations: { entails(swimmer, woman) }
		explanation: it is a factually wrong relation because not every swimmer is a woman

Example 3:
	input:
		premise: A young girl wearing a pink coat plays with a yellow toy.
		hypothesis: A kid is swinging a toy golf club.
	correct output:
		answer: non-entailment

Example 4:
	input:
		premise: A black race car starts up in front of a crowd of people.
		hypothesis: A car is running.
	correct output:
		answer: entailment
		relations: { entails(start up, run) }
	unwanted output:
		relations: { entails(starts up, is running) }
		explanation: "starts up" and "is running" do not contain lemmatized words, and "is" is unnecessary in "is running".

Example 5:
	input:
		premise: A woman dressed in red clothing is dancing inside a crowd of people.
		hypothesis: A woman in red is dancing in a crowd.
	correct output:
		answer: entailment
		relations: { }
	unwanted output:
		relations: { entails(in red clothing, in red) }
		explanation: The phrases in the relation may not include prepositional phrases, e.g., "in red clothing"

Example 6:
	input:
		premise: A tall man with a cap is climbing a cord.
		hypothesis: The man in a hat is climbing a rope.
	correct output:
		answer: entailment
		relations: { entails(cap, hat), entails(cord, rope) }
	unwanted output:
		relations: { entails(cap, hat), entails(tall man, man) }
		explanation: "entails(cord, rope)" is missing. entails(tall man, man) is trivial since it includes dropping the adjective "tall".

Example 7:
	input:
		premise: No person is cooking.
		hypothesis: No cook is cooking in the kitchen.
	correct output:
		answer: entailment
		relations: { entails(cook, person) }
	unwanted output:
		relations: { entails(person, cook) }
		explanation: The relation does not help to explain "entailment", taking into account that negation reverses a lexical entailment direction. It is also a factually wrong relation because not every person is a cook.

Example 8:
	input:
		premise: A person who is obese is holding a chinchilla.
		hypothesis: A fat person is holding a small animal.
	correct output:
		answer: entailment
		relations: { entails(chinchilla, small animal), entails(obese, fat) }
	unwanted output:
		relations: { entails(fat, obese), entails(chinchilla, animal) }
		explanation: "entails(fat, obese)" needs to reverse its arguments to align with the entailment direction. "entails(chinchilla, animal)" is not sufficient to explain "entailment" since it misses "small", which is crucial.

Example 9:
	input:
		premise: A little boy is laughing and happily bouncing on a trampoline outside.
		hypothesis: The child is jumping outdoors.
	correct output:
		answer: entailment
		relations: { entails(little boy, child), entails(bounce, jump), entails(outside, outdoors) }
	unwanted output:
		relations: { entails(boy, child), entails(outdoors, outside) }
		explanation: "entails(bounce, jump)" is missing. "entails(little boy, child)" is preferred over "entails(boy, child)" as the former is more acceptable. "entails(outdoors, outside)" needs to reverse its arguments to align it to the entailment direction.

Additional calibration while preserving all rules above:
- Only output lexical entailment relations that are both factually acceptable and needed to explain the entailment.
- Do not output relations for entailments that follow without a non-trivial lexical bridge.
- Do not use event or social implications as lexical entailment unless the phrase relation is a direct paraphrase.
- Do not output modifier-dropping relations such as old woman -> woman or military men -> men.
- Do not output both an inflected form and a lemmatized form for the same relation.
- If the best relation would violate any existing formatting rule, omit it instead of approximating it.

input:
	premise: ${PREMISE}
	hypothesis: ${HYPOTHESIS}
```
