# Experiments Log

## 2026-02-26

### Environment
- Repo: `transformer-10-digit-addition`
- Python: `3.13.0` (venv: `.venv`)
- PyTorch: `2.10.0`
- Device: `mps` (`cuda` unavailable)

### Experiment E1: 456p Training Reproduction (Seed 43)
- Command:
```bash
.venv/bin/python -m src.train \
  --run-name repro_456_s43 \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 2 --ffn-rank 3 \
  --use-rmsnorm --tie-qkv shareA_tieKV \
  --train-steps 54000 --device mps --seed 43
```
- Outputs:
  - `results/runs/repro_456_s43/summary.json`
  - `results/runs/repro_456_s43/metrics.csv`
- Finding: Did not grok. `best_val_exact = 0.0` after 54,000 steps.

### Experiment E2: 456p Training Reproduction (Seed 44)
- Command:
```bash
.venv/bin/python -m src.train \
  --run-name repro_456_s44 \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 2 --ffn-rank 3 \
  --use-rmsnorm --tie-qkv shareA_tieKV \
  --train-steps 54000 --device mps --seed 44
```
- Outputs:
  - `results/runs/repro_456_s44/summary.json`
  - `results/runs/repro_456_s44/metrics.csv`
- Finding: Did not grok. `best_val_exact = 0.0` after 54,000 steps.

### Experiment E3: Evaluate Reproduced Checkpoints
- Commands:
```bash
.venv/bin/python evaluate_checkpoints.py results/runs/repro_456_s43/checkpoints/last.pt --device mps --output results/repro_456_s43_eval.json
.venv/bin/python evaluate_checkpoints.py results/runs/repro_456_s44/checkpoints/last.pt --device mps --output results/repro_456_s44_eval.json
```
- Findings:
  - `results/repro_456_s43_eval.json`: aggregate exact match `0.0` (100,000/100,000 errors)
  - `results/repro_456_s44_eval.json`: aggregate exact match `0.0` (100,000/100,000 errors)

### Experiment E4: Evaluation Sanity Check (Provided Checkpoint)
- Command:
```bash
.venv/bin/python evaluate_checkpoints.py checkpoints/best_456p_s43.pt --device mps --output results/reference_456_s43_eval.json
```
- Finding: Passes fully on this machine. Aggregate exact match `1.0` (0/100,000 errors).

### Experiment E5: 512p Baseline Training Reproduction (Seed 42)
- Command:
```bash
uv run python -m src.train \
  --run-name repro_512_s42_mps \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3 \
  --train-steps 27000 --seed 42 --device mps
```
- Outputs:
  - `results/runs/repro_512_s42_mps/summary.json`
  - `results/runs/repro_512_s42_mps/metrics.csv`
- Finding: Did not grok. `best_val_exact = 0.0` after 27,000 steps.

### Experiment E6: 512p Evaluation (Reproduced vs Provided Checkpoint)
- Commands:
```bash
uv run python evaluate_checkpoints.py results/runs/repro_512_s42_mps/checkpoints/last.pt --device mps --output results/repro_512_s42_mps_eval.json
uv run python evaluate_checkpoints.py checkpoints/best_512params.pt --device mps --output results/reference_512_eval.json
```
- Findings:
  - Reproduced checkpoint: `results/repro_512_s42_mps_eval.json` aggregate exact match `0.0` (100,000/100,000 errors).
  - Provided checkpoint: `results/reference_512_eval.json` aggregate exact match `0.99988` (12/100,000 errors), matching the repo baseline.

### Experiment E7: 512p Long Run (100k steps, Seed 42)
- Command:
```bash
uv run python -m src.train \
  --run-name repro_512_100k_s42_mps \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3 \
  --train-steps 100000 --seed 42 --device mps
```
- Outputs:
  - `results/runs/repro_512_100k_s42_mps/summary.json`
  - `results/runs/repro_512_100k_s42_mps/metrics.csv`
  - `results/runs/repro_512_100k_s42_mps/train_val_loss_per_step.png`
  - `results/runs/repro_512_100k_s42_mps/val_accuracy_per_step.png`
- Finding: No successful grokking. `best_val_exact = 0.0` at 100,000 steps.

### Experiment E8: ~777 Baseline Attempt (50k steps, Seed 42)
- Command:
```bash
uv run python -m src.train \
  --run-name baseline_777_s42_mps_50k \
  --d-model 7 --d-ff 14 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 \
  --lr 0.02 --weight-decay 0.01 \
  --train-steps 50000 --seed 42 --device mps --eval-interval 500
```
- Outputs:
  - `results/runs/baseline_777_s42_mps_50k/summary.json`
  - `results/runs/baseline_777_s42_mps_50k/metrics.csv`
  - `results/baseline_777_s42_mps_50k_eval.json`
- Findings:
  - Model size in this codebase is `763` params (same tiny config at seq_len=33).
  - Late grokking-like phase appeared around step ~33k-37.5k.
  - Best validation exact match during training: `0.203` at step `49000`.
  - Held-out aggregate exact match: `0.1933` (80,670 errors / 100,000).

### Experiment E9: ~973 Baseline Attempt (27k steps, Seed 42)
- Command:
```bash
uv run python -m src.train \
  --run-name baseline_973_s42_mps \
  --d-model 7 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 \
  --lr 0.01 --train-steps 27000 --seed 42 --device mps --eval-interval 500
```
- Outputs:
  - `results/runs/baseline_973_s42_mps/summary.json`
  - `results/runs/baseline_973_s42_mps/metrics.csv`
  - `results/baseline_973_s42_mps_eval.json`
- Findings:
  - Model size in this codebase is `959` params (same architecture idea at seq_len=33).
  - Best validation exact match during training: `0.0002` at step `19000`.
  - Held-out aggregate exact match: `0.00002` (99,998 errors / 100,000).

### Experiment E10: d=8 Baseline from ~973 Setup (30k steps, Seed 42)
- Command:
```bash
uv run python -m src.train \
  --run-name baseline_d8_from_973_s42_30k \
  --d-model 8 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 \
  --lr 0.01 --train-steps 30000 --seed 42 --device mps --eval-interval 500
```
- Outputs:
  - `results/runs/baseline_d8_from_973_s42_30k/summary.json`
  - `results/runs/baseline_d8_from_973_s42_30k/metrics.csv`
  - `results/baseline_d8_from_973_s42_30k_eval.json`
  - W&B run: `maxence-frenette/transformer-10-digit-addition/runs/qjs0xesr`
- Findings:
  - Model size in this codebase is `1128` params.
  - Best validation exact match during training: `1.0` (first reached at step `8500`).
  - Training mostly remained at `val_exact=1.0` through 30k, with brief transient drops.
  - Held-out aggregate exact match: `0.99967` (33 errors / 100,000).

### Experiment E11: Baseline Cleanup Validation (Smoke + Eval Flow)
- Commands:
```bash
uv run python -m src.train \
  --run-name smoke_full_rank_cleanup \
  --train-steps 2 --eval-interval 1 \
  --batch-size 8 --val-size 16 --test-size 16 --eval-batch-size 16 \
  --device cpu --wandb --wandb-mode disabled \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0

uv run python -m src.train \
  --run-name smoke_low_rank_cleanup \
  --train-steps 2 --eval-interval 1 \
  --batch-size 8 --val-size 16 --test-size 16 --eval-batch-size 16 \
  --device cpu --wandb --wandb-mode disabled \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 2 --ffn-rank 3

uv run python - <<'PY'
import torch
from pathlib import Path
from src.model import ModelConfig, TinyDecoderLM
cfg = ModelConfig()
model = TinyDecoderLM(cfg)
print('fresh_tied', model.lm_head.weight is model.token_emb.weight)
ckpt = torch.load(Path('results/runs/smoke_full_rank_cleanup/checkpoints/best.pt'), map_location='cpu', weights_only=False)
model2 = TinyDecoderLM(ModelConfig(**ckpt['model_config']))
model2.load_state_dict(ckpt['model_state'])
print('loaded_tied', model2.lm_head.weight is model2.token_emb.weight)
PY

uv run python evaluate_checkpoints.py \
  results/runs/smoke_full_rank_cleanup/checkpoints/best.pt \
  --device cpu --batch-size 256 \
  --output results/smoke_full_rank_cleanup_eval.json
```
- Outputs:
  - `results/runs/smoke_full_rank_cleanup/summary.json`
  - `results/runs/smoke_low_rank_cleanup/summary.json`
  - `results/smoke_full_rank_cleanup_eval.json`
- Findings:
  - Full-rank smoke path runs successfully after cleanup (`params=763`).
  - Low-rank smoke path runs successfully after cleanup (`params=498`), confirming `*_rank` paths remain functional.
  - Embedding/output tying invariant holds (`fresh_tied=True`, `loaded_tied=True`).
  - Evaluation flow works with cleaned config schema.

### Conclusion
- Reproduction training attempts with seeds 43 and 44 on `mps` did not reproduce the published 100% result.
- Evaluation pipeline is valid locally (provided checkpoint reproduces expected 100%).
- 512-parameter baseline training also failed to reproduce on `mps`, while the provided 512 checkpoint evaluates as expected.
- Extending the 512 baseline to 100k steps on `mps` still failed to produce a strong grokking solution.
- 763-param and 959-param baseline attempts also failed to reach reliable exact-match accuracy on `mps`.
- Increasing the baseline to `d_model=8` (1128 params) produced a robust high-accuracy run on `mps` (`0.99967` aggregate exact match).
- Model cleanup now keeps only embedding/output weight tying while preserving functional `*_rank` controls.

### Experiment E12: Baseline Cleanup Finalization + Artifact Pruning
- Commands:
```bash
uv run python -m src.train --help

uv run python - <<'PY'
from src.model import ModelConfig, TinyDecoderLM
payload = {
    'n_layer': 1, 'd_model': 7, 'n_head': 1, 'd_ff': 14,
    'max_seq_len': 33, 'vocab_size': 14,
    'pos_rank': 3, 'qkv_rank': 3, 'attn_out_rank': 2, 'ffn_rank': 3,
    'dropout': 0.0, 'use_rmsnorm': False, 'tie_qkv': 'none',
}
cfg = ModelConfig.from_dict(payload)
model = TinyDecoderLM(cfg)
print(cfg)
print('tied', model.lm_head.weight is model.token_emb.weight)
PY

find . -type f \( -path './checkpoints/*.pt' -o -path './results/runs/*/checkpoints/*.pt' \) -delete
/bin/rm -f results/grokking_plot.png
```
- Findings:
  - Train CLI now exposes only baseline + rank controls; removed flags (`dropout`, `use_rmsnorm`, `tie_qkv`) are absent.
  - `ModelConfig.from_dict(...)` correctly drops removed legacy keys while keeping rank fields functional.
  - Embedding/output tying invariant holds (`tied=True`).
  - Saved checkpoint artifacts and `results/grokking_plot.png` were removed from the workspace.

### Experiment E13: Shared-Depth Compute Scaling (Repeat Same Layer 2x)
- Goal:
  - Increase compute without increasing parameter count by reusing the same transformer block twice (`n_layer=2` with shared weights).
- Code change:
  - `TinyDecoderLM` now uses one shared `Block` and applies it `n_layer` times in forward pass.
  - Parameter count stays fixed as `n_layer` increases.
- Sanity check:
```bash
uv run python - <<'PY'
from src.model import ModelConfig, TinyDecoderLM, count_parameters
for n in [1, 2, 3]:
    m = TinyDecoderLM(ModelConfig(n_layer=n, d_model=8, d_ff=28, n_head=1, max_seq_len=33, vocab_size=14))
    print(n, count_parameters(m))
PY
```
  - Output: `1 1128`, `2 1128`, `3 1128`.
- Training command:
```bash
uv run python -m src.train \
  --run-name baseline_d8_shared2_s42_30k \
  --n-layer 2 --d-model 8 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 \
  --lr 0.01 --train-steps 30000 --seed 42 --device mps --eval-interval 500
```
- Outputs:
  - `results/runs/baseline_d8_shared2_s42_30k/summary.json`
  - `results/runs/baseline_d8_shared2_s42_30k/metrics.csv`
  - `results/runs/baseline_d8_shared2_s42_30k/checkpoints/best.pt`
  - W&B run: `maxence-frenette/transformer-10-digit-addition/runs/lzf2rf8c`
- Training findings:
  - `params=1128` (unchanged from d8 baseline with `n_layer=1`).
  - `best_val_exact=0.9984` at step `26500`.
- Evaluation command:
```bash
uv run python evaluate_checkpoints.py \
  results/runs/baseline_d8_shared2_s42_30k/checkpoints/best.pt \
  --device mps --output results/baseline_d8_shared2_s42_30k_eval.json
```
- Evaluation findings:
  - Aggregate exact match: `0.99825` (175 errors / 100,000).
  - Per-seed exact match range: `0.9977` to `0.9986`.
  - Compared to Experiment E10 (`n_layer=1` d8 baseline, `0.99967`), shared-depth 2x still works but is materially worse on held-out aggregate accuracy.

### Experiment E14: Learning-Rate Sweep for Shared-Depth (`n_layer=2`)
- Goal:
  - Tune learning rate for the 2x repeated shared-layer baseline at fixed parameter count.
- Setup:
  - `n_layer=2`, `d_model=8`, `d_ff=28`, full-rank (`*_rank=0`), `train_steps=30000`, `seed=42`, `device=mps`, `eval_interval=500`.
- Sweep commands:
```bash
uv run python -m src.train --run-name baseline_d8_shared2_lr0p006_s42_30k --n-layer 2 --d-model 8 --d-ff 28 --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 --lr 0.006 --train-steps 30000 --seed 42 --device mps --eval-interval 500
uv run python -m src.train --run-name baseline_d8_shared2_lr0p008_s42_30k --n-layer 2 --d-model 8 --d-ff 28 --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 --lr 0.008 --train-steps 30000 --seed 42 --device mps --eval-interval 500
uv run python -m src.train --run-name baseline_d8_shared2_lr0p01_s42_30k --n-layer 2 --d-model 8 --d-ff 28 --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 --lr 0.01 --train-steps 30000 --seed 42 --device mps --eval-interval 500
uv run python -m src.train --run-name baseline_d8_shared2_lr0p012_s42_30k --n-layer 2 --d-model 8 --d-ff 28 --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 --lr 0.012 --train-steps 30000 --seed 42 --device mps --eval-interval 500
uv run python -m src.train --run-name baseline_d8_shared2_lr0p015_s42_30k --n-layer 2 --d-model 8 --d-ff 28 --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 --lr 0.015 --train-steps 30000 --seed 42 --device mps --eval-interval 500
```
- Evaluation command pattern:
```bash
uv run python evaluate_checkpoints.py \
  results/runs/<run_name>/checkpoints/best.pt \
  --device mps --output results/<run_name>_eval.json
```
- Outputs:
  - `results/runs/baseline_d8_shared2_lr0p006_s42_30k/summary.json`
  - `results/runs/baseline_d8_shared2_lr0p008_s42_30k/summary.json`
  - `results/runs/baseline_d8_shared2_lr0p01_s42_30k/summary.json`
  - `results/runs/baseline_d8_shared2_lr0p012_s42_30k/summary.json`
  - `results/runs/baseline_d8_shared2_lr0p015_s42_30k/summary.json`
  - `results/baseline_d8_shared2_lr0p006_s42_30k_eval.json`
  - `results/baseline_d8_shared2_lr0p008_s42_30k_eval.json`
  - `results/baseline_d8_shared2_lr0p01_s42_30k_eval.json`
  - `results/baseline_d8_shared2_lr0p012_s42_30k_eval.json`
  - `results/baseline_d8_shared2_lr0p015_s42_30k_eval.json`
- Findings:
  - `lr=0.006`: `best_val_exact=0.0116`, aggregate exact `0.01208` (98,792 errors / 100,000).
  - `lr=0.008`: `best_val_exact=0.1114`, aggregate exact `0.09908` (90,092 errors / 100,000).
  - `lr=0.010`: `best_val_exact=0.0114`, aggregate exact `0.00970` (99,030 errors / 100,000).
  - `lr=0.012`: `best_val_exact=0.9828`, aggregate exact `0.97991` (2,009 errors / 100,000).
  - `lr=0.015`: `best_val_exact=0.3176`, aggregate exact `0.32465` (67,535 errors / 100,000).
- Conclusion:
  - Best LR from this sweep is `0.012`.
  - Training is unstable on MPS even with fixed seed (`lr=0.01` previously produced a strong run in E13 but collapsed in this sweep), so LR tuning should be paired with multi-seed repeats for robust selection.

### Experiment E15: Learning-Rate Sweep for Regular 1-Layer Baseline (`n_layer=1`)
- Goal:
  - Run the same LR sweep as E14 for the regular 1-layer transformer and compare stability region width.
- Setup:
  - `n_layer=1`, `d_model=8`, `d_ff=28`, full-rank (`*_rank=0`), `train_steps=30000`, `seed=42`, `device=mps`, `eval_interval=500`.
- Sweep commands:
```bash
uv run python -m src.train --run-name baseline_d8_layer1_lr0p006_s42_30k --n-layer 1 --d-model 8 --d-ff 28 --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 --lr 0.006 --train-steps 30000 --seed 42 --device mps --eval-interval 500
uv run python -m src.train --run-name baseline_d8_layer1_lr0p008_s42_30k --n-layer 1 --d-model 8 --d-ff 28 --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 --lr 0.008 --train-steps 30000 --seed 42 --device mps --eval-interval 500
uv run python -m src.train --run-name baseline_d8_layer1_lr0p01_s42_30k --n-layer 1 --d-model 8 --d-ff 28 --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 --lr 0.01 --train-steps 30000 --seed 42 --device mps --eval-interval 500
uv run python -m src.train --run-name baseline_d8_layer1_lr0p012_s42_30k --n-layer 1 --d-model 8 --d-ff 28 --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 --lr 0.012 --train-steps 30000 --seed 42 --device mps --eval-interval 500
uv run python -m src.train --run-name baseline_d8_layer1_lr0p015_s42_30k --n-layer 1 --d-model 8 --d-ff 28 --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 0 --lr 0.015 --train-steps 30000 --seed 42 --device mps --eval-interval 500
```
- Evaluation command pattern:
```bash
uv run python evaluate_checkpoints.py \
  results/runs/<run_name>/checkpoints/best.pt \
  --device mps --output results/<run_name>_eval.json
```
- Outputs:
  - `results/runs/baseline_d8_layer1_lr0p006_s42_30k/summary.json`
  - `results/runs/baseline_d8_layer1_lr0p008_s42_30k/summary.json`
  - `results/runs/baseline_d8_layer1_lr0p01_s42_30k/summary.json`
  - `results/runs/baseline_d8_layer1_lr0p012_s42_30k/summary.json`
  - `results/runs/baseline_d8_layer1_lr0p015_s42_30k/summary.json`
  - `results/baseline_d8_layer1_lr0p006_s42_30k_eval.json`
  - `results/baseline_d8_layer1_lr0p008_s42_30k_eval.json`
  - `results/baseline_d8_layer1_lr0p01_s42_30k_eval.json`
  - `results/baseline_d8_layer1_lr0p012_s42_30k_eval.json`
  - `results/baseline_d8_layer1_lr0p015_s42_30k_eval.json`
- Findings:
  - `lr=0.006`: `best_val_exact=0.0`, aggregate exact `0.00000` (100,000 errors / 100,000).
  - `lr=0.008`: `best_val_exact=0.0006`, aggregate exact `0.00017` (99,983 errors / 100,000).
  - `lr=0.010`: `best_val_exact=0.0`, aggregate exact `0.00000` (100,000 errors / 100,000).
  - `lr=0.012`: `best_val_exact=1.0`, aggregate exact `0.99960` (40 errors / 100,000).
  - `lr=0.015`: `best_val_exact=1.0`, aggregate exact `0.99999` (1 error / 100,000).
- Stability comparison vs E14 (`n_layer=2`):
  - Shared-2 best was `lr=0.012` with aggregate exact `0.97991`; `lr=0.015` degraded to `0.32465`.
  - Regular 1-layer reaches near-perfect performance at both `lr=0.012` and `lr=0.015`.
  - In this sweep, the high-performing LR region is larger for `n_layer=1` than for `n_layer=2`.

### Experiment E16: Fixed Full-Rank Random Residual in `LowRankLinear`
- Goal:
  - Add a non-trainable full-rank random matrix to each `LowRankLinear` and test whether low-rank training improves.
- Code change:
  - `LowRankLinear` now computes `x @ A @ B + x @ W_fixed`, where `W_fixed` is a buffer (non-trainable).
  - Trainable parameter counts are unchanged; only fixed buffer tensors were added.
- Parameter sanity check:
```bash
uv run python - <<'PY'
from src.model import ModelConfig, TinyDecoderLM, count_parameters
cfg = ModelConfig(
    n_layer=1, d_model=7, n_head=1, d_ff=14,
    max_seq_len=33, vocab_size=14,
    pos_rank=3, qkv_rank=3, attn_out_rank=3, ffn_rank=3,
)
m = TinyDecoderLM(cfg)
print('trainable_params', count_parameters(m))
print('buffer_params', sum(b.numel() for _, b in m.named_buffers()))
PY
```
  - Output: `trainable_params=512`, `buffer_params=1481`.
- Smoke command:
```bash
uv run python -m src.train \
  --run-name smoke_lowrank_fixedfullrank \
  --train-steps 2 --eval-interval 1 \
  --batch-size 8 --val-size 16 --test-size 16 --eval-batch-size 16 \
  --device cpu --wandb --wandb-mode disabled \
  --d-model 7 --d-ff 14 \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3
```
- Main training command:
```bash
uv run python -m src.train \
  --run-name lowrank_fixedfullrank_512_s42_27k \
  --n-layer 1 --d-model 7 --d-ff 14 \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3 \
  --train-steps 27000 --seed 42 --device mps --eval-interval 500
```
- Evaluation command:
```bash
uv run python evaluate_checkpoints.py \
  results/runs/lowrank_fixedfullrank_512_s42_27k/checkpoints/best.pt \
  --device mps --output results/lowrank_fixedfullrank_512_s42_27k_eval.json
```
- Outputs:
  - `results/runs/lowrank_fixedfullrank_512_s42_27k/summary.json`
  - `results/runs/lowrank_fixedfullrank_512_s42_27k/metrics.csv`
  - `results/runs/lowrank_fixedfullrank_512_s42_27k/checkpoints/best.pt`
  - `results/lowrank_fixedfullrank_512_s42_27k_eval.json`
  - W&B run: `maxence-frenette/transformer-10-digit-addition/runs/bczb65jd`
- Findings:
  - Training reached `best_val_exact=1.0` at step `18000`.
  - Multi-seed aggregate exact match: `1.0` (0 errors / 100,000).
  - On this run, adding fixed full-rank random residuals to low-rank layers produced a strong, fully-correct 100k evaluation.

### Experiment E17: Fixed Full-Rank Residual Robustness Check (Additional Seeds)
- Goal:
  - Re-run the same fixed-full-rank low-rank configuration with two additional seeds.
- Config:
  - `n_layer=1`, `d_model=7`, `d_ff=14`, `pos_rank=3`, `qkv_rank=3`, `attn_out_rank=3`, `ffn_rank=3`, `train_steps=27000`, `device=mps`.
- Commands:
```bash
uv run python -m src.train \
  --run-name lowrank_fixedfullrank_512_s43_27k \
  --n-layer 1 --d-model 7 --d-ff 14 \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3 \
  --train-steps 27000 --seed 43 --device mps --eval-interval 500

uv run python -m src.train \
  --run-name lowrank_fixedfullrank_512_s44_27k \
  --n-layer 1 --d-model 7 --d-ff 14 \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3 \
  --train-steps 27000 --seed 44 --device mps --eval-interval 500

uv run python evaluate_checkpoints.py \
  results/runs/lowrank_fixedfullrank_512_s43_27k/checkpoints/best.pt \
  --device mps --output results/lowrank_fixedfullrank_512_s43_27k_eval.json

uv run python evaluate_checkpoints.py \
  results/runs/lowrank_fixedfullrank_512_s44_27k/checkpoints/best.pt \
  --device mps --output results/lowrank_fixedfullrank_512_s44_27k_eval.json
```
- Outputs:
  - `results/runs/lowrank_fixedfullrank_512_s43_27k/summary.json`
  - `results/runs/lowrank_fixedfullrank_512_s44_27k/summary.json`
  - `results/lowrank_fixedfullrank_512_s43_27k_eval.json`
  - `results/lowrank_fixedfullrank_512_s44_27k_eval.json`
  - W&B runs:
    - `maxence-frenette/transformer-10-digit-addition/runs/0t5gpfhr`
    - `maxence-frenette/transformer-10-digit-addition/runs/ntiyhj9r`
- Findings:
  - Seed 43:
    - `best_val_exact=0.0` (step 0)
    - aggregate exact `0.0` (100,000 errors / 100,000)
  - Seed 44:
    - `best_val_exact=0.0` (step 0)
    - aggregate exact `0.0` (100,000 errors / 100,000)
  - Combined with seed 42 from E16:
    - seed 42: aggregate exact `1.0` (0 errors)
    - seeds 43/44: aggregate exact `0.0` (complete failure)
- Conclusion:
  - The fixed full-rank residual modification is extremely seed-sensitive in current form on this hardware; the seed-42 success does not generalize to these additional seeds.

### Experiment E18: `fixed_full_rank` Toggle as Hyperparameter (6 Runs, `ffn_rank=3`)
- Goal:
  - Compare `fixed_full_rank=false` vs `fixed_full_rank=true` across 3 seeds each (6 runs total), with `ffn_rank=3`.
- Code updates:
  - Added `fixed_full_rank` to `ModelConfig`.
  - Added CLI flag: `--fixed-full-rank/--no-fixed-full-rank`.
  - `LowRankLinear` now gates the fixed matrix contribution by this flag.
  - Included `fixed_full_rank` in `evaluate_checkpoints.py` output metadata.
- Shared training setup:
  - `n_layer=1`, `d_model=8`, `d_ff=28`, `n_head=1`
  - `pos_rank=0`, `qkv_rank=0`, `attn_out_rank=0`, `ffn_rank=3`
  - `lr=0.015`, `train_steps=30000`, `device=mps`, `eval_interval=500`
  - seeds: `42`, `43`, `44`
- Commands (pattern):
```bash
uv run python -m src.train \
  --run-name ffnr3_fixed{0|1}_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --no-fixed-full-rank|--fixed-full-rank \
  --lr 0.015 --train-steps 30000 --seed {SEED} --device mps --eval-interval 500

uv run python evaluate_checkpoints.py \
  results/runs/ffnr3_fixed{0|1}_s{SEED}_30k/checkpoints/best.pt \
  --device mps --output results/ffnr3_fixed{0|1}_s{SEED}_30k_eval.json
```
- Per-run results:
  - `ffnr3_fixed0_s42_30k`: `best_val_exact=0.0000`, aggregate exact `0.00000` (100,000 errors)
  - `ffnr3_fixed0_s43_30k`: `best_val_exact=0.0180`, aggregate exact `0.01374` (98,626 errors)
  - `ffnr3_fixed0_s44_30k`: `best_val_exact=0.0004`, aggregate exact `0.00012` (99,988 errors)
  - `ffnr3_fixed1_s42_30k`: `best_val_exact=1.0000`, aggregate exact `0.99998` (2 errors)
  - `ffnr3_fixed1_s43_30k`: `best_val_exact=1.0000`, aggregate exact `0.99996` (4 errors)
  - `ffnr3_fixed1_s44_30k`: `best_val_exact=1.0000`, aggregate exact `0.99999` (1 error)
- Grouped comparison:
  - `fixed_full_rank=false`:
    - mean best validation exact: `0.00613`
    - mean aggregate exact: `0.00462`
    - mean aggregate errors: `99,538`
  - `fixed_full_rank=true`:
    - mean best validation exact: `1.00000`
    - mean aggregate exact: `0.99998`
    - mean aggregate errors: `2.33`
- Conclusion:
  - For this `ffn_rank=3` setup, enabling `fixed_full_rank` is decisively better and robust across all three tested seeds.

### Experiment E19: Replace LayerNorm with RMSNorm (`fixed_full_rank=true`, `ffn_rank=3`, 3 seeds)
- Goal:
  - Replace all model LayerNorms with RMSNorm and test training on 3 seeds.
- Code change:
  - `src/model.py`: added `RMSNorm` and replaced `ln1`, `ln2`, and `ln_f`.
  - Parameter count changed from `896` to `872` (RMSNorm has scale only, no bias).
- Setup:
  - `n_layer=1`, `d_model=8`, `d_ff=28`, `n_head=1`
  - `pos_rank=0`, `qkv_rank=0`, `attn_out_rank=0`, `ffn_rank=3`
  - `fixed_full_rank=true`, `lr=0.015`, `train_steps=30000`, `device=mps`, `eval_interval=500`
  - seeds: `42`, `43`, `44`
- Commands (pattern):
```bash
uv run python -m src.train \
  --run-name rmsnorm_fixed1_ffnr3_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --lr 0.015 --train-steps 30000 --seed {SEED} --device mps --eval-interval 500

uv run python evaluate_checkpoints.py \
  results/runs/rmsnorm_fixed1_ffnr3_s{SEED}_30k/checkpoints/best.pt \
  --device mps --output results/rmsnorm_fixed1_ffnr3_s{SEED}_30k_eval.json
```
- Outputs:
  - `results/runs/rmsnorm_fixed1_ffnr3_s42_30k/summary.json`
  - `results/runs/rmsnorm_fixed1_ffnr3_s43_30k/summary.json`
  - `results/runs/rmsnorm_fixed1_ffnr3_s44_30k/summary.json`
  - `results/rmsnorm_fixed1_ffnr3_s42_30k_eval.json`
  - `results/rmsnorm_fixed1_ffnr3_s43_30k_eval.json`
  - `results/rmsnorm_fixed1_ffnr3_s44_30k_eval.json`
- Findings:
  - Seed 42:
    - `best_val_exact=0.0` (step `0`)
    - aggregate exact `0.0` (100,000 errors / 100,000)
  - Seed 43:
    - `best_val_exact=0.0116` (step `28000`)
    - aggregate exact `0.00899` (99,101 errors / 100,000)
  - Seed 44:
    - `best_val_exact=1.0` (step `17500`)
    - aggregate exact `0.99997` (3 errors / 100,000)
- Conclusion:
  - RMSNorm in this setup is highly seed-sensitive on this hardware: 1 strong seed, 2 failed seeds.

### Experiment E20: Remove Normalization Completely (`fixed_full_rank=true`, `ffn_rank=3`, 3 seeds)
- Goal:
  - Remove normalization from all blocks and the final layer, then test 3 seeds.
- Code change:
  - `src/model.py`:
    - Replaced `ln1`, `ln2`, and `ln_f` with `nn.Identity()`.
    - Removed `RMSNorm` usage entirely.
  - Parameter count changed from `872` (RMSNorm variant) to `848`.
- Setup:
  - `n_layer=1`, `d_model=8`, `d_ff=28`, `n_head=1`
  - `pos_rank=0`, `qkv_rank=0`, `attn_out_rank=0`, `ffn_rank=3`
  - `fixed_full_rank=true`, `lr=0.015`, `train_steps=30000`, `device=mps`, `eval_interval=500`
  - seeds: `42`, `43`, `44`
- Commands (pattern):
```bash
uv run python -m src.train \
  --run-name nonorm_fixed1_ffnr3_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --lr 0.015 --train-steps 30000 --seed {SEED} --device mps --eval-interval 500

uv run python evaluate_checkpoints.py \
  results/runs/nonorm_fixed1_ffnr3_s{SEED}_30k/checkpoints/best.pt \
  --device mps --output results/nonorm_fixed1_ffnr3_s{SEED}_30k_eval.json
```
- Outputs:
  - `results/runs/nonorm_fixed1_ffnr3_s42_30k/summary.json`
  - `results/runs/nonorm_fixed1_ffnr3_s43_30k/summary.json`
  - `results/runs/nonorm_fixed1_ffnr3_s44_30k/summary.json`
  - `results/nonorm_fixed1_ffnr3_s42_30k_eval.json`
  - `results/nonorm_fixed1_ffnr3_s43_30k_eval.json`
  - `results/nonorm_fixed1_ffnr3_s44_30k_eval.json`
- Findings:
  - Seed 42:
    - `best_val_exact=0.0` (step `0`)
    - aggregate exact `0.0` (100,000 errors / 100,000)
  - Seed 43:
    - `best_val_exact=0.0002` (step `22000`)
    - aggregate exact `0.0` (100,000 errors / 100,000)
  - Seed 44:
    - `best_val_exact=0.0018` (step `10000`)
    - aggregate exact `0.0009` (99,910 errors / 100,000)
- Conclusion:
  - Removing normalization entirely is unstable and unsuccessful in this baseline regime on this hardware (0/3 successful seeds).

### Experiment E21: Lower Positional Embedding Rank to 6 (`pos_rank=6`, 3 seeds)
- Goal:
  - Test whether lowering positional embedding rank to `6` remains reliable under the current baseline setup.
- Setup:
  - `n_layer=1`, `d_model=8`, `d_ff=28`, `n_head=1`
  - `pos_rank=6`, `qkv_rank=0`, `attn_out_rank=0`, `ffn_rank=3`
  - `fixed_full_rank=true`, `lr=0.015`, `train_steps=30000`, `device=mps`, `eval_interval=500`
  - seeds: `42`, `43`, `44`
- Commands (pattern):
```bash
uv run python -m src.train \
  --run-name posrank6_fixed1_ffnr3_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 6 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --lr 0.015 --train-steps 30000 --seed {SEED} --device mps --eval-interval 500

uv run python evaluate_checkpoints.py \
  results/runs/posrank6_fixed1_ffnr3_s{SEED}_30k/checkpoints/best.pt \
  --device mps --output results/posrank6_fixed1_ffnr3_s{SEED}_30k_eval.json
```
- Outputs:
  - `results/runs/posrank6_fixed1_ffnr3_s42_30k/summary.json`
  - `results/runs/posrank6_fixed1_ffnr3_s43_30k/summary.json`
  - `results/runs/posrank6_fixed1_ffnr3_s44_30k/summary.json`
  - `results/posrank6_fixed1_ffnr3_s42_30k_eval.json`
  - `results/posrank6_fixed1_ffnr3_s43_30k_eval.json`
  - `results/posrank6_fixed1_ffnr3_s44_30k_eval.json`
  - W&B runs:
    - `maxence-frenette/transformer-10-digit-addition/runs/axxpl9vl`
    - `maxence-frenette/transformer-10-digit-addition/runs/adbcbgfv`
    - `maxence-frenette/transformer-10-digit-addition/runs/mvi1x4jn`
- Findings:
  - All runs used `878` parameters.
  - Seed 42:
    - `best_val_exact=0.5916` (step `29999`)
    - aggregate exact `0.59502` (40,498 errors / 100,000)
  - Seed 43:
    - `best_val_exact=1.0` (step `15000`)
    - aggregate exact `0.99993` (7 errors / 100,000)
  - Seed 44:
    - `best_val_exact=1.0` (step `24500`)
    - aggregate exact `0.99986` (14 errors / 100,000)
- Conclusion:
  - `pos_rank=6` can reach near-perfect performance, but remains seed-sensitive in this setup (2 strong seeds, 1 partial-failure seed).
