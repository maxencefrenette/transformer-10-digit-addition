# Agent Rules

## Experiment Naming
- Do not append `_mps` to experiment names.
- Assume experiments run on MPS by default in this repository unless explicitly stated otherwise.

## Experiment Logging
- Always record every experiment result in `EXPERIMENTS.md`.
- Add the command, key outputs/paths, and findings after each run.

## Experiment Execution
- Always run 3 seeds for every experiment by default.

## Multi-Model Default
- Use `num_models=8` by default unless explicitly overridden.

## Experiment Success Criterion
- Standard success criterion: at least 4 of the 8 models must converge to `>99%` validation exact-match accuracy.
