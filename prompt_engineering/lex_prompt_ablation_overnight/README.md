# LEX prompt ablation overnight experiment

## Design

- Primary data: all high-agreement items
- Primary prompts: 4
- Models: 4
- Stability sample: 100 items
- Stability repetitions: 3
- KB filtering/lemmatization pipeline enabled: no
- Relation source: directly parsed model response
- Temperature explicitly supplied: no

## Selected prompts

- `lex_few_shot_precision`
- `lex_few_shot_base`

## Resume behavior

Rerun the same command. Completed KB or error cells are skipped.

## Main artifacts

- `prompts.md`: exact tested prompt templates
- `primary_outputs.csv`: raw responses, parsed KBs, and errors
- `primary_leaderboard.csv`: model-level quality scores
- `primary_prompt_summary.csv`: prompt means used for selection
- `stability_sample.csv`: balanced follow-up sample
- `repeat_{1,2,3}_outputs.csv`: repeated outputs
- `quality_across_repeats.csv`: repeat-level quality summary
- `stability_metrics.csv`: agreement between repeated outputs
