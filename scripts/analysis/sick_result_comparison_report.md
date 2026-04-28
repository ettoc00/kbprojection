# SICK Full-Run Model Comparison

Inputs:

- `sick_full__icl__openrouter__openai_gpt-5.4.json`
- `sick_full__icl__openrouter__openai_gpt-5.4-mini.json`
- `sick_full__icl__openrouter__anthropic_claude-opus-4.7.json`

All three files contain 9,927 completed unique problem keys.

## Status Semantics

Each entry has one terminal `final_status`, so `raw_kb_solved + normalised_kb_solved` is safe as a count of `raw_kb_solved union normalised_kb_solved`.

Important caveat: `kb_not_solved` is overloaded in the current orchestration. In `kbprojection/orchestration.py`, if no-KB prediction is wrong and non-neutral, the run exits before KB generation and sets `KB_NOT_SOLVED`. Separately, after KB generation/testing, the code also sets `KB_NOT_SOLVED` when neither raw nor normalised KB solves.

Observed split:

| model | kb_not_solved total | early no-KB wrong non-neutral | generated KB available |
|---|---:|---:|---:|
| GPT-5.4 | 939 | 78 | 861 |
| GPT-5.4-mini | 1014 | 78 | 936 |
| Opus 4.7 | 1077 | 78 | 999 |

## Overall Results

| model | baseline | raw solved | normalised solved | any KB solved | final solved |
|---|---:|---:|---:|---:|---:|
| GPT-5.4 | 7659 | 20 | 621 | 641 | 8300 |
| GPT-5.4-mini | 7659 | 586 | 50 | 636 | 8295 |
| Opus 4.7 | 7659 | 652 | 41 | 693 | 8352 |

Baseline solved sets are exactly identical across all three models.

Opus is the best single model by KB-assisted wins, but the absolute lift over GPT-5.4 is modest: +52 final solved cases, or about +0.52 percentage points on the full dataset. Its distinctive advantage is that, under the current pipeline decision logic, it reaches `raw_kb_solved` far more often.

## GPT-5.4-mini vs GPT-5.4

Formal sets:

- `mini_kb_solved = mini.raw_kb_solved union mini.normalised_kb_solved = 636`
- `full_kb_solved = full.raw_kb_solved union full.normalised_kb_solved = 641`

| set comparison | count |
|---|---:|
| overlap | 580 |
| mini only | 56 |
| GPT-5.4 only | 61 |
| mini included in GPT-5.4 | 91.2% |

Mini is not a strict subset of GPT-5.4 because 56 mini KB wins are not GPT-5.4 KB wins. It is mostly covered by GPT-5.4 at the any-KB-solved level.

Exact status-level comparison:

| status | mini | GPT-5.4 | overlap | mini only | GPT-5.4 only |
|---|---:|---:|---:|---:|---:|
| raw_kb_solved | 586 | 20 | 10 | 576 | 10 |
| normalised_kb_solved | 50 | 621 | 42 | 8 | 579 |
| kb_not_solved | 1014 | 939 | 810 | 204 | 129 |
| kb_generation_empty | 488 | 655 | 422 | 66 | 233 |
| kb_normalisation_empty | 130 | 33 | 17 | 113 | 16 |

## Pairwise KB Wins

| comparison | overlap | left only | right only |
|---|---:|---:|---:|
| GPT-5.4 vs mini | 580 | 61 | 56 |
| GPT-5.4 vs Opus | 621 | 20 | 72 |
| mini vs Opus | 592 | 44 | 101 |

Oracle upper bounds across all three models:

| oracle set | count |
|---|---:|
| any model raw KB solved | 700 |
| any model normalised KB solved | 637 |
| any model any-KB solved | 745 |
| baseline plus any model any-KB solved | 8404 |

This is an oracle ceiling, not an achievable selector result. A practical ensemble needs a gold-free selection signal.

## Combined-KB Experiment

For GPT-5.4 and Opus:

- overlapping `kb_not_solved`: 872
- same raw and normalised KB sets: 519
- materially different KB sets: 353
- different both raw and normalised: 326
- different raw only: 27
- same after normalisation: 27

Combined-KB mechanics:

- `combined_raw = ordered deduplicated union(gpt54.kb_raw, opus.kb_raw)`
- `combined_normalised = ordered deduplicated union(gpt54.kb_filtered, opus.kb_filtered)`
- no re-normalisation after union
- raw union is not passed through the normaliser

Full 353-case result:

| combined path | solved | errors |
|---|---:|---:|
| combined raw | 1 / 353 | 0 |
| combined normalised | 1 / 353 | 0 |
| either combined path | 1 / 353 | 0 |

The only solved combined case was `test:8485`.

Premise: A young man on a bmx bicycle is jumping on a masonry pyramid  
Hypothesis: A bicyclist is jumping on a pyramid-shaped ramp  
Gold: entailment

GPT-5.4 KB:

- `isa_wn(young man, bicyclist)`
- `isa_wn(masonry pyramid, ramp)`

Opus KB:

- `isa_wn(man on bicycle, bicyclist)`
- `isa_wn(masonry pyramid, pyramid-shaped ramp)`

The union solves because the two models contribute complementary alignments.

## Examples

### Mini-only: `test:52`

Premise: Kids in red shirts are playing in the leaves  
Hypothesis: Children in red shirts are playing in the leaves  
Gold: entailment

| model | status | raw pred | norm pred | KB |
|---|---|---|---|---|
| GPT-5.4 | kb_not_solved | neutral | neutral | `isa_wn(kid, child)` |
| mini | normalised_kb_solved | entailment | entailment | `isa_wn(kids, children)`, `isa_wn(kid, children)` |
| Opus | kb_not_solved | neutral | neutral | `isa_wn(kid, child)` |

The plural lexical relation appears to matter for LangPro here.

### Normalisation Harm: `test:3058`

Premise: There is no man pouring oil into a pan  
Hypothesis: A man is pouring oil into a skillet  
Gold: contradiction

| model | status | raw pred | norm pred | raw KB | normalised KB |
|---|---|---|---|---|---|
| GPT-5.4 | raw_kb_solved | contradiction | neutral | `isa_wn(skillet, pan)` | `isa_wn(pan, skillet)` |
| mini | kb_not_solved | neutral | neutral | `isa_wn(pan, skillet)` | `isa_wn(pan, skillet)` |
| Opus | raw_kb_solved | contradiction | neutral | `isa_wn(skillet, pan)` | `isa_wn(pan, skillet)` |

The normaliser flips direction and loses the contradiction proof.

### Opus-only: `test:178`

Premise: A large group of Asian people is eating at a restaurant  
Hypothesis: A group of people from Asia is eating at a restaurant  
Gold: entailment

| model | status | raw pred | norm pred | KB |
|---|---|---|---|---|
| GPT-5.4 | kb_not_solved | neutral | neutral | `isa_wn(Asian, from Asia)` |
| mini | kb_not_solved | neutral | neutral | `isa_wn(asian, asia)` |
| Opus | raw_kb_solved | entailment | entailment | `isa_wn(asian, from asia)` |

This suggests surface casing/phrase normalization can affect prover compatibility.

## Recommendation

Plain union of GPT-5.4 and Opus KBs is not worth scaling as the main ensemble strategy: the full shared-failure differing-KB run solved only 1 of 353 cases.

The better next target is a gold-free selector or verifier for model-generated KBs. The oracle ceiling across all three models is 745 KB-assisted wins, versus 693 for Opus alone, so there is potential headroom, but it is selection headroom rather than evidence that simple KB composition is broadly effective.
