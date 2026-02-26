# A 456-Parameter Transformer Solves 10-Digit Addition

A **456-parameter** transformer that solves 10-digit integer addition. Given two integers A, B in [0, 10^10), the model predicts C = A + B autoregressively, achieving **100% exact-match accuracy** on 100,000 test examples.

Current code focus: baseline transformer with optional rank factorization (`pos_rank`, `qkv_rank`, `attn_out_rank`, `ffn_rank`). The model path uses LayerNorm and standard QKV projections (full-rank or low-rank), with only embedding/output weight tying supported.

The repo also includes historical results from earlier variants explored in the ecosystem (including RMSNorm and QKV-tying ideas), but those are not part of the current trainable code path.



## Grokking

All successful models exhibit **grokking**: prolonged near-zero accuracy followed by a sudden phase transition.


## Leaderboard

| Params | Model | Accuracy | Reference |
|---|---|---|---|
| 1,644 | Codex baseline | 99.04% | [Papailiopoulos](https://github.com/anadim/smallest-addition-transformer-codex) |
| 777 | gpt-acc-jax (pico-7d-ff14-lr02) | 99.69% | [Havinga](https://github.com/yhavinga/gpt-acc-jax) |
| 512 | + Low-rank factorization (rank 3) | 99.988% | Ours |
| 491 | Historical RMSNorm variant | 99.97% | [rezabyt](https://github.com/rezabyt/digit-addition-491p) |
| **456** | **Historical 456p variant** | **100%** | **Ours** |

## Architecture

Current code path: single-layer, single-head, decoder-only transformer with vocabulary size 14 (digits 0-9, `+`, `=`, `<PAD>`, `<EOS>`). Norms are `LayerNorm`; QKV/attention-output/FFN/position projections are full-rank when rank=0 and low-rank when the corresponding `*_rank` is >0.

| Component | Implementation |
|---|---|
| Token embedding + LM head | Tied (`lm_head.weight = token_emb.weight`) |
| Position embedding | Full-rank embedding or low-rank factorized embedding (`pos_rank`) |
| Attention QKV | Full-rank linear or low-rank factorized linear (`qkv_rank`) |
| Attention output | Full-rank linear or low-rank factorized linear (`attn_out_rank`) |
| FFN up/down | Full-rank linear or low-rank factorized linear (`ffn_rank`) |
| Normalization | `LayerNorm` |

Reference parameter counts in current code:
- `d_model=7`, `d_ff=14`, all ranks `0`: `763` params.
- `d_model=7`, `d_ff=14`, `pos_rank=3`, `qkv_rank=3`, `attn_out_rank=2`, `ffn_rank=3`: `498` params.

## Results

Evaluated on 10 independent test sets of 10,000 examples each (100,000 total), with seeds disjoint from training:

| Model | Params | Exact Match | Errors / 100K |
|---|---|---|---|
| 582p (seed 42) | 582 | 99.999% | 1 |
| 512p (seed 42) | 512 | 99.988% | 12 |
| **456p (seed 43)** | **456** | **100%** | **0** |
| 456p (seed 44) | 456 | 99.958% | 42 |



## Quick Start

### Install

```bash
uv sync
```

Training runs auto-log to Weights & Biases at `maxence-frenette/transformer-10-digit-addition`.
Use `--no-wandb` to disable logging for a run.
Only embedding/output weight tying is supported in this code path.

### Evaluate Pre-trained Checkpoint

```bash
uv run python evaluate_checkpoints.py \
  results/runs/<run_name>/checkpoints/best.pt --device cuda
```

### Single Prediction

```bash
uv run python -m src.eval predict \
  --ckpt results/runs/<run_name>/checkpoints/best.pt \
  --a 1234567890 --b 9876543210
```

### Train from Scratch

```bash
# d=8 baseline (reliable on this repo/hardware)
uv run python -m src.train \
  --run-name baseline_d8_s42_30k \
  --d-model 8 --d-ff 28 \
  --train-steps 30000 --device cuda --seed 42

# Optional low-rank variant
uv run python -m src.train \
  --run-name rank_variant_s42 \
  --d-model 7 --d-ff 14 \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3 \
  --train-steps 54000 --device cuda --seed 42
```

## Training

3-phase curriculum following [gpt-acc-jax](https://github.com/yhavinga/gpt-acc-jax):
1. Steps 0-2,000: 1-3 digit operands
2. Steps 2,000-7,000: 1-6 digit operands
3. Steps 7,000-27,000: 1-10 digit operands (full range)

AdamW optimizer, peak LR = 0.02, linear warmup (1,350 steps) + cosine decay, min LR = 0.002, weight decay = 0.01, batch size = 512, total steps = 27,000 by default.

All successful models exhibit **grokking**: prolonged near-zero accuracy followed by a sudden phase transition. For the 456-parameter model, 2 out of 5 random seeds (43 and 44) grokked within 54K steps.

## Files

```
src/
  model.py    # Baseline transformer with optional rank factorization
  data.py     # Raw digit tokenization pipeline
  train.py    # Training with curriculum learning
  eval.py     # Evaluation and inference
results/runs/<run_name>/checkpoints/
  best.pt / last.pt   # Runtime checkpoints written by training
evaluate_checkpoints.py  # Multi-seed evaluation script
report.pdf               # Technical report
```

## References

- D. Papailiopoulos, "Glove box challenge: smallest transformer for 10-digit addition," 2026. [GitHub](https://github.com/anadim/smallest-addition-transformer-codex)
- Y. Havinga, "gpt-acc-jax: Smallest GPT for 10-digit addition," 2026. [GitHub](https://github.com/yhavinga/gpt-acc-jax)
- rezabyt, "digit-addition-491p," 2026. [GitHub](https://github.com/rezabyt/digit-addition-491p)
- yinglunz, "A-456-Parameter-Transformer-Solves-10-Digit-Addition," 2026. [GitHub](https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition)
