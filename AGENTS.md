# Agent Rules

## Experiment Naming
- Do not append `_mps` to experiment names.
- Assume experiments run on MPS by default in this repository unless explicitly stated otherwise.

## Experiment Logging
- Always record every experiment result in `EXPERIMENTS.md`.
- Add the command, key outputs/paths, and findings after each run.

## Experiment Execution
- Do not run separate multi-seed repeats by default.
- Run each experiment as a single `num_models=8` training run.
- Treat that one run as the 8-seed experiment result.

## Multi-Model Default
- Use `num_models=8` by default unless explicitly overridden.
- Treat this as the baseline configuration.

## Baseline Hyperparameters
- Baseline optimization defaults are `batch_size=512` and `lr=0.015`.
- Keep these as the baseline unless explicitly experimenting with alternatives.

## Experiment Success Criterion
- Standard success criterion: at least 4 of the 8 models must converge to `>99%` validation exact-match accuracy.
