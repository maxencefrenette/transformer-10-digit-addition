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

### Experiment E22: `pos_rank=6` with Fixed Full-Rank Term Also Applied to Low-Rank Positional Embeddings (3 seeds)
- Goal:
  - Re-run the `pos_rank=6` setting, but extend `fixed_full_rank` behavior to `LowRankEmbedding` (not only `LowRankLinear`).
- Code change:
  - `src/model.py`:
    - `LowRankEmbedding` now accepts `fixed_full_rank`.
    - Added non-trainable buffer `W_fixed` in `LowRankEmbedding`.
    - Forward path now uses: `A[idx] @ B + W_fixed[idx]` when `fixed_full_rank=true`.
    - `TinyDecoderLM` now passes `fixed_full_rank=cfg.fixed_full_rank` when constructing low-rank positional embeddings.
- Setup:
  - `n_layer=1`, `d_model=8`, `d_ff=28`, `n_head=1`
  - `pos_rank=6`, `qkv_rank=0`, `attn_out_rank=0`, `ffn_rank=3`
  - `fixed_full_rank=true`, `lr=0.015`, `train_steps=30000`, `device=mps`, `eval_interval=500`
  - seeds: `42`, `43`, `44`
- Commands (pattern):
```bash
uv run python -m src.train \
  --run-name posrank6_embfixed_fixed1_ffnr3_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 6 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --lr 0.015 --train-steps 30000 --seed {SEED} --device mps --eval-interval 500

uv run python evaluate_checkpoints.py \
  results/runs/posrank6_embfixed_fixed1_ffnr3_s{SEED}_30k/checkpoints/best.pt \
  --device mps --output results/posrank6_embfixed_fixed1_ffnr3_s{SEED}_30k_eval.json
```
- Outputs:
  - `results/runs/posrank6_embfixed_fixed1_ffnr3_s42_30k/summary.json`
  - `results/runs/posrank6_embfixed_fixed1_ffnr3_s43_30k/summary.json`
  - `results/runs/posrank6_embfixed_fixed1_ffnr3_s44_30k/summary.json`
  - `results/posrank6_embfixed_fixed1_ffnr3_s42_30k_eval.json`
  - `results/posrank6_embfixed_fixed1_ffnr3_s43_30k_eval.json`
  - `results/posrank6_embfixed_fixed1_ffnr3_s44_30k_eval.json`
  - W&B runs:
    - `maxence-frenette/transformer-10-digit-addition/runs/d0lqkh92`
    - `maxence-frenette/transformer-10-digit-addition/runs/irizk0oi`
    - `maxence-frenette/transformer-10-digit-addition/runs/0ct2fv0a`
- Findings:
  - All runs used `878` parameters.
  - Seed 42:
    - `best_val_exact=0.0082` (step `28500`)
    - aggregate exact `0.00655` (99,345 errors / 100,000)
  - Seed 43:
    - `best_val_exact=0.0` (step `0`)
    - aggregate exact `0.0` (100,000 errors / 100,000)
  - Seed 44:
    - `best_val_exact=0.1080` (step `29999`)
    - aggregate exact `0.10881` (89,119 errors / 100,000)
- Conclusion:
  - Applying the fixed full-rank residual to low-rank positional embeddings significantly degraded this `pos_rank=6` setup on all 3 seeds.
  - Compared with E21 (same hyperparameters except this embedding change), performance regressed from 2/3 near-perfect seeds to 0/3 successful seeds.

### Experiment E23: Split Positional Embedding `A` into 4 Trainable + 2 Fixed Random Weights (`pos_rank=6`, `pos_fixed_a_rank=2`, 3 seeds)
- Goal:
  - Keep low-rank positional embedding at rank 6, but make each row of `A` use 4 trainable values and 2 non-trainable random values.
- Code change:
  - `src/model.py`:
    - Added `pos_fixed_a_rank` to `ModelConfig`.
    - Updated `LowRankEmbedding` to use:
      - trainable `A` with shape `(num_embeddings, rank - fixed_a_rank)`
      - non-trainable buffer `A_fixed` with shape `(num_embeddings, fixed_a_rank)`
      - trainable `B` with shape `(rank, embedding_dim)`
    - Forward now computes `concat(A[idx], A_fixed[idx]) @ B`.
    - `TinyDecoderLM` now passes `fixed_a_rank=cfg.pos_fixed_a_rank` for low-rank positional embeddings.
  - `src/train.py`:
    - Added CLI flag `--pos-fixed-a-rank` and validation (`<= --pos-rank`).
  - `evaluate_checkpoints.py`:
    - Added `pos_fixed_a_rank` to output config metadata.
- Setup:
  - `n_layer=1`, `d_model=8`, `d_ff=28`, `n_head=1`
  - `pos_rank=6`, `pos_fixed_a_rank=2`, `qkv_rank=0`, `attn_out_rank=0`, `ffn_rank=3`
  - `fixed_full_rank=true`, `lr=0.015`, `train_steps=30000`, `device=mps`, `eval_interval=500`
  - seeds: `42`, `43`, `44`
- Commands (pattern):
```bash
uv run python -m src.train \
  --run-name posrank6_afixed2_fixed1_ffnr3_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 6 --pos-fixed-a-rank 2 \
  --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --lr 0.015 --train-steps 30000 --seed {SEED} --device mps --eval-interval 500

uv run python evaluate_checkpoints.py \
  results/runs/posrank6_afixed2_fixed1_ffnr3_s{SEED}_30k/checkpoints/best.pt \
  --device mps --output results/posrank6_afixed2_fixed1_ffnr3_s{SEED}_30k_eval.json
```
- Outputs:
  - `results/runs/posrank6_afixed2_fixed1_ffnr3_s42_30k/summary.json`
  - `results/runs/posrank6_afixed2_fixed1_ffnr3_s43_30k/summary.json`
  - `results/runs/posrank6_afixed2_fixed1_ffnr3_s44_30k/summary.json`
  - `results/posrank6_afixed2_fixed1_ffnr3_s42_30k_eval.json`
  - `results/posrank6_afixed2_fixed1_ffnr3_s43_30k_eval.json`
  - `results/posrank6_afixed2_fixed1_ffnr3_s44_30k_eval.json`
  - W&B runs:
    - `maxence-frenette/transformer-10-digit-addition/runs/efehhwdh`
    - `maxence-frenette/transformer-10-digit-addition/runs/e49jnkfa`
    - `maxence-frenette/transformer-10-digit-addition/runs/48xk0ayr`
- Findings:
  - All runs used `812` parameters.
  - Seed 42:
    - `best_val_exact=0.0` (step `0`)
    - aggregate exact `0.0` (100,000 errors / 100,000)
  - Seed 43:
    - `best_val_exact=0.0` (step `0`)
    - aggregate exact `0.0` (100,000 errors / 100,000)
  - Seed 44:
    - `best_val_exact=0.1084` (step `25000`)
    - aggregate exact `0.09949` (90,051 errors / 100,000)
- Conclusion:
  - This clean 4-trainable/2-fixed `A` split for positional embeddings underperformed strongly on this setup (0/3 successful seeds).

### Experiment E24: `EmbeddingsXsLora` (LoRA-XS-Style Positional Embedding, Random Fixed Factors, `r=16`, 3 seeds)
- Goal:
  - Add a dedicated positional embedding module inspired by LoRA-XS where only a small square matrix is trainable.
  - Use fully random fixed factors (no SVD init) and no fixed full-rank positional embedding addend.
- Code change:
  - `src/model.py`:
    - Added `pos_xs_rank` to `ModelConfig`.
    - Added new module `EmbeddingsXsLora`:
      - fixed non-trainable `B_fixed` of shape `(33, r)`
      - trainable `R` of shape `(r, r)`
      - fixed non-trainable `A_fixed` of shape `(r, d_model)`
      - positional embedding computed as `B_fixed[idx] @ R @ A_fixed`
    - `TinyDecoderLM` uses `EmbeddingsXsLora` when `pos_xs_rank > 0`.
  - `src/train.py`:
    - Added CLI flag `--pos-xs-rank`.
    - Added guard so `--pos-rank` and `--pos-xs-rank` cannot both be enabled.
  - `evaluate_checkpoints.py`:
    - Added `pos_xs_rank` to emitted config metadata.
- Setup:
  - `n_layer=1`, `d_model=8`, `d_ff=28`, `n_head=1`
  - `pos_rank=0`, `pos_xs_rank=16`, `qkv_rank=0`, `attn_out_rank=0`, `ffn_rank=3`
  - `fixed_full_rank=true` for linear layers, `lr=0.015`, `train_steps=30000`, `device=mps`, `eval_interval=500`
  - seeds: `42`, `43`, `44`
- Parameter note:
  - Total parameters: `888` (close to baseline; trainable positional matrix is `R` with `16x16=256` params vs `33x8=264` in full learned positional embedding).
- Commands (pattern):
```bash
uv run python -m src.train \
  --run-name posxs16_fixed1_ffnr3_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 0 --pos-xs-rank 16 \
  --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --lr 0.015 --train-steps 30000 --seed {SEED} --device mps --eval-interval 500

uv run python evaluate_checkpoints.py \
  results/runs/posxs16_fixed1_ffnr3_s{SEED}_30k/checkpoints/best.pt \
  --device mps --output results/posxs16_fixed1_ffnr3_s{SEED}_30k_eval.json
```
- Outputs:
  - `results/runs/posxs16_fixed1_ffnr3_s42_30k/summary.json`
  - `results/runs/posxs16_fixed1_ffnr3_s43_30k/summary.json`
  - `results/runs/posxs16_fixed1_ffnr3_s44_30k/summary.json`
  - `results/posxs16_fixed1_ffnr3_s42_30k_eval.json`
  - `results/posxs16_fixed1_ffnr3_s43_30k_eval.json`
  - `results/posxs16_fixed1_ffnr3_s44_30k_eval.json`
  - W&B runs:
    - `maxence-frenette/transformer-10-digit-addition/runs/l12siwlw`
    - `maxence-frenette/transformer-10-digit-addition/runs/u2eh9xd6`
    - `maxence-frenette/transformer-10-digit-addition/runs/6hzu9lrh`
- Findings:
  - Seed 42:
    - `best_val_exact=0.0` (step `0`)
    - aggregate exact `0.0` (100,000 errors / 100,000)
  - Seed 43:
    - `best_val_exact=0.0` (step `0`)
    - aggregate exact `0.0` (100,000 errors / 100,000)
  - Seed 44:
    - `best_val_exact=0.0` (step `0`)
    - aggregate exact `0.0` (100,000 errors / 100,000)
- Conclusion:
  - This randomized LoRA-XS-style positional embedding (`r=16`) failed on all 3 seeds in the current training setup.

### Experiment E25: Add `grad_norm` Logging
- Goal:
  - Track gradient norm for every evaluation point to diagnose optimization stability.
- Code change:
  - `src/train.py`:
    - Compute grad norm each step via `torch.nn.utils.clip_grad_norm_`.
    - Add `grad_norm` column to `results/runs/*/metrics.csv`.
    - Print `grad_norm` in step logs.
    - Log `optim/grad_norm` to W&B.
- Command:
```bash
uv run python -m src.train \
  --run-name smoke_gradnorm_log \
  --train-steps 1 --eval-interval 1 \
  --batch-size 16 --val-size 64 --test-size 64 --eval-batch-size 32 \
  --device cpu --wandb false
```
- Outputs:
  - `results/runs/smoke_gradnorm_log/metrics.csv`
  - `results/runs/smoke_gradnorm_log/summary.json`
- Findings:
  - `metrics.csv` now includes `grad_norm`.
  - Console and W&B metrics include gradient norm without changing model shape (`896` params).

### Experiment E26: LR/WD/Warmup Retune `tune_fastlr1` (3 seeds)
- Goal:
  - Improve time-to-99% by using a more aggressive schedule on the 896-parameter baseline.
- Setup:
  - `n_layer=1`, `d_model=8`, `d_ff=28`, `ffn_rank=3`, `fixed_full_rank=true`
  - `batch_size=512`, `lr=0.02`, `weight_decay=0.005`, `warmup_steps=500`, `min_lr_ratio=0.2`
  - `train_steps=30000`, `eval_interval=500`, seeds `42/43/44`
- Command (pattern):
```bash
uv run python -m src.train \
  --run-name tune_fastlr1_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --batch-size 512 --lr 0.02 --weight-decay 0.005 \
  --warmup-steps 500 --min-lr-ratio 0.2 \
  --train-steps 30000 --seed {SEED} --device mps --eval-interval 500
```
- Outputs:
  - `results/runs/tune_fastlr1_s42_30k/summary.json`
  - `results/runs/tune_fastlr1_s43_30k/summary.json`
  - `results/runs/tune_fastlr1_s44_30k/summary.json`
  - W&B runs:
    - `maxence-frenette/transformer-10-digit-addition/runs/yc0bjcv3`
    - `maxence-frenette/transformer-10-digit-addition/runs/ct4anphs`
    - `maxence-frenette/transformer-10-digit-addition/runs/ouqboulz`
- Findings:
  - Seed 42: `best_val_exact=0.1136` (step `21500`)
  - Seed 43: `best_val_exact=0.4586` (step `27500`)
  - Seed 44: `best_val_exact=0.0128` (step `27500`)
- Conclusion:
  - Failed stability constraint (`0/3` seeds reached `>=0.99`).

### Experiment E27: LR/WD/Warmup Retune `tune_stable1` (3 seeds)
- Goal:
  - Test a less aggressive variant after `tune_fastlr1`.
- Setup:
  - Same model shape (`896` params)
  - `batch_size=512`, `lr=0.015`, `weight_decay=0.01`, `warmup_steps=800`, `min_lr_ratio=0.2`
  - `train_steps=30000`, seeds `42/43/44`
- Command (pattern):
```bash
uv run python -m src.train \
  --run-name tune_stable1_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --batch-size 512 --lr 0.015 --weight-decay 0.01 \
  --warmup-steps 800 --min-lr-ratio 0.2 \
  --train-steps 30000 --seed {SEED} --device mps --eval-interval 500
```
- Outputs:
  - `results/runs/tune_stable1_s42_30k/summary.json`
  - `results/runs/tune_stable1_s43_30k/summary.json`
  - `results/runs/tune_stable1_s44_30k/summary.json`
  - W&B runs:
    - `maxence-frenette/transformer-10-digit-addition/runs/qmiw8tgt`
    - `maxence-frenette/transformer-10-digit-addition/runs/dhdx4tgj`
    - `maxence-frenette/transformer-10-digit-addition/runs/rmc8it54`
- Findings:
  - Seed 42: `best_val_exact=0.0986` (step `29999`)
  - Seed 43: `best_val_exact=0.0012` (step `29000`)
  - Seed 44: `best_val_exact=0.0002` (step `27000`)
- Conclusion:
  - Failed stability constraint (`0/3` seeds reached `>=0.99`).

### Experiment E28: Warmup Sweep `tune_warmup1000` (3 seeds)
- Goal:
  - Recover baseline behavior by moving warmup to `1000` and `min_lr_ratio=0.1`.
- Setup:
  - Same model shape (`896` params)
  - `batch_size=512`, `lr=0.015`, `weight_decay=0.01`, `warmup_steps=1000`, `min_lr_ratio=0.1`
  - `train_steps=30000`, seeds `42/43/44`
- Command (pattern):
```bash
uv run python -m src.train \
  --run-name tune_warmup1000_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --batch-size 512 --lr 0.015 --weight-decay 0.01 \
  --warmup-steps 1000 --min-lr-ratio 0.1 \
  --train-steps 30000 --seed {SEED} --device mps --eval-interval 500
```
- Outputs:
  - `results/runs/tune_warmup1000_s42_30k/summary.json`
  - `results/runs/tune_warmup1000_s43_30k/summary.json`
  - `results/runs/tune_warmup1000_s44_30k/summary.json`
  - W&B runs:
    - `maxence-frenette/transformer-10-digit-addition/runs/cgxh6jvf`
    - `maxence-frenette/transformer-10-digit-addition/runs/zafn5ilm`
    - `maxence-frenette/transformer-10-digit-addition/runs/mpyc4vo0`
- Findings:
  - Seed 42: `best_val_exact=0.9872` (step `29500`) [near miss]
  - Seed 43: `best_val_exact=0.0` (step `0`)
  - Seed 44: `best_val_exact=0.0154` (step `27500`)
- Conclusion:
  - Failed stability constraint (`0/3` seeds reached `>=0.99`).

### Experiment E29: Higher Batch Size `1024` (`tune_bs1024_lr015`, 3 seeds)
- Goal:
  - Test whether higher batch size improves wall-clock time-to-99%.
- Setup:
  - Same model shape (`896` params)
  - `batch_size=1024`, `lr=0.015`, `weight_decay=0.01`, `warmup_steps=1350`, `min_lr_ratio=0.1`
  - `train_steps=30000`, seeds `42/43/44`
- Command (pattern):
```bash
uv run python -m src.train \
  --run-name tune_bs1024_lr015_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --batch-size 1024 --lr 0.015 --weight-decay 0.01 \
  --warmup-steps 1350 --min-lr-ratio 0.1 \
  --train-steps 30000 --seed {SEED} --device mps --eval-interval 500
```
- Outputs:
  - `results/runs/tune_bs1024_lr015_s42_30k/summary.json`
  - `results/runs/tune_bs1024_lr015_s43_30k/summary.json`
  - `results/runs/tune_bs1024_lr015_s44_30k/summary.json`
  - W&B runs:
    - `maxence-frenette/transformer-10-digit-addition/runs/ceahzy62`
    - `maxence-frenette/transformer-10-digit-addition/runs/s63zjc45`
    - `maxence-frenette/transformer-10-digit-addition/runs/j0vy20nt`
- Findings:
  - Seed 42: `best_val_exact=0.0016` (step `29500`)
  - Seed 43: `best_val_exact=0.0` (step `0`)
  - Seed 44: `best_val_exact=0.1082` (step `13000`)
- Conclusion:
  - Failed stability constraint (`0/3` seeds reached `>=0.99`).
  - Also slower per-step wall clock than `batch_size=512`.

### Experiment E30: Higher Batch Size `768` (`tune_bs768_lr015`, 3 seeds)
- Goal:
  - Try a milder batch-size increase after `1024` failure.
- Setup:
  - Same model shape (`896` params)
  - `batch_size=768`, `lr=0.015`, `weight_decay=0.01`, `warmup_steps=1350`, `min_lr_ratio=0.1`
  - `train_steps=30000`, seeds `42/43/44`
- Command (pattern):
```bash
uv run python -m src.train \
  --run-name tune_bs768_lr015_s{SEED}_30k \
  --n-layer 1 --d-model 8 --d-ff 28 \
  --pos-rank 0 --qkv-rank 0 --attn-out-rank 0 --ffn-rank 3 \
  --fixed-full-rank \
  --batch-size 768 --lr 0.015 --weight-decay 0.01 \
  --warmup-steps 1350 --min-lr-ratio 0.1 \
  --train-steps 30000 --seed {SEED} --device mps --eval-interval 500
```
- Outputs:
  - `results/runs/tune_bs768_lr015_s42_30k/summary.json`
  - `results/runs/tune_bs768_lr015_s43_30k/summary.json`
  - `results/runs/tune_bs768_lr015_s44_30k/summary.json`
  - W&B runs:
    - `maxence-frenette/transformer-10-digit-addition/runs/rbh4e277`
    - `maxence-frenette/transformer-10-digit-addition/runs/pzuz5ryq`
    - `maxence-frenette/transformer-10-digit-addition/runs/4249p23t`
- Findings:
  - Seed 42: `best_val_exact=0.0754` (step `29999`)
  - Seed 43: `best_val_exact=1.0` (step `13500`), first `>=0.99` at step `13000` (`156.3s`)
  - Seed 44: `best_val_exact=0.4558` (step `29999`)
- Conclusion:
  - Failed stability constraint (`1/3` seeds reached `>=0.99`).
  - Despite one fast seed, variance is too high for reliable time-to-99 optimization.

### Experiment E31: Vectorized Multi-Model Training (`num_models`) Implementation Validation
- Goal:
  - Validate model-axis (`m`) vectorized training of multiple independent models in one process.
  - Verify per-model gradient isolation, per-model clipping, checkpoint compatibility, W&B multi-run logging, and backward compatibility (`num_models=1`).

- Code change summary:
  - `src/model.py`:
    - Added `num_models` to `ModelConfig`.
    - Refactored embedding/linear/layernorm/attention/MLP to model-axis aware tensors.
    - Added per-model CE reduction helper and vectorized `generate` support.
    - Kept embedding-output tying (`lm_head.weight` tied to `token_emb.weight`) per model.
  - `src/train.py`:
    - Added `--num-models` CLI flag.
    - Added multi-model data sampling (`seed+i` streams), per-model grad clipping, per-model checkpoints, and per-model metrics CSVs.
    - Added one W&B run per model (grouped by parent run name).
  - `src/eval.py`:
    - Added `evaluate_exact_match_multi(...)` for vectorized per-model eval in training.
    - Kept single-model `evaluate_exact_match(...)` behavior unchanged for checkpoint evaluation flows.
  - `README.md`:
    - Documented `--num-models`, seed policy, output layout, and W&B grouping behavior.

- Validation commands:
```bash
# 1) Model-shape/backward/isolation unit checks
uv run python - <<'PY'
import torch
from src.model import ModelConfig, TinyDecoderLM
from src.train import clip_grad_norm_per_model

cfg = ModelConfig(num_models=3)
model = TinyDecoderLM(cfg)
idx = torch.randint(0, 14, (3, 4, 33), dtype=torch.long)
tgt = torch.randint(0, 14, (3, 4, 33), dtype=torch.long)
tgt[:, :, :21] = -100
logits, loss, per_model = model(idx, tgt, return_per_model_loss=True)
loss.backward()
clip_grad_norm_per_model(model.parameters(), 1.0, 3)

model.zero_grad(set_to_none=True)
_, _, pm = model(idx, tgt, return_per_model_loss=True)
pm[0].backward()
max_other = 0.0
for p in model.parameters():
    if p.grad is not None and p.grad.shape[0] == 3:
        max_other = max(max_other, float(p.grad[1:].abs().max().item()))
print("independence_max_other_grad", max_other)
PY

# 2) Multi-model trainer smoke
uv run python -m src.train \
  --run-name smoke_num_models3_cpu \
  --num-models 3 \
  --train-steps 2 --eval-interval 1 \
  --batch-size 8 --val-size 16 --test-size 16 --eval-batch-size 16 \
  --device cpu --wandb --wandb-mode disabled

# 3) Checkpoint compatibility (existing eval pipeline)
uv run python evaluate_checkpoints.py \
  results/runs/smoke_num_models3_cpu/model_000/checkpoints/best.pt \
  --device cpu --batch-size 2048 \
  --output results/smoke_num_models3_cpu_model000_eval.json

# 4) W&B multi-run offline smoke
uv run python -m src.train \
  --run-name smoke_num_models3_wandb_offline \
  --num-models 3 \
  --train-steps 1 --eval-interval 1 \
  --batch-size 8 --val-size 16 --test-size 16 --eval-batch-size 16 \
  --device cpu --wandb --wandb-mode offline

# 5) Backward compatibility smoke (num_models=1)
uv run python -m src.train \
  --run-name smoke_num_models1_compat \
  --num-models 1 \
  --train-steps 1 --eval-interval 1 \
  --batch-size 8 --val-size 16 --test-size 16 --eval-batch-size 16 \
  --device cpu --wandb --wandb-mode disabled

# 6) Performance sanity on MPS (short benchmark)
uv run python -m src.train \
  --run-name perf_num_models1_mps \
  --num-models 1 \
  --train-steps 501 --eval-interval 500 \
  --batch-size 64 --val-size 256 --test-size 256 --eval-batch-size 256 \
  --device mps --wandb --wandb-mode disabled

uv run python -m src.train \
  --run-name perf_num_models3_mps \
  --num-models 3 \
  --train-steps 501 --eval-interval 500 \
  --batch-size 64 --val-size 256 --test-size 256 --eval-batch-size 256 \
  --device mps --wandb --wandb-mode disabled
```

- Outputs:
  - `results/runs/smoke_num_models3_cpu/summary.json`
  - `results/runs/smoke_num_models3_cpu/model_000/checkpoints/best.pt`
  - `results/runs/smoke_num_models3_cpu/model_001/checkpoints/best.pt`
  - `results/runs/smoke_num_models3_cpu/model_002/checkpoints/best.pt`
  - `results/smoke_num_models3_cpu_model000_eval.json`
  - `results/runs/smoke_num_models3_wandb_offline/summary.json`
  - `results/runs/smoke_num_models1_compat/summary.json`
  - `results/runs/perf_num_models1_mps/metrics.csv`
  - `results/runs/perf_num_models3_mps/metrics.csv`
  - W&B offline runs:
    - `wandb/offline-run-20260226_143027-micoxv8i`
    - `wandb/offline-run-20260226_143028-ls2jdpd7`
    - `wandb/offline-run-20260226_143028-j5ldm9fl`

- Findings:
  - Model-axis shape and backward smoke checks pass:
    - logits shape: `(3, 4, 33, 14)`
    - loss is scalar
    - per-model loss shape: `(3,)`
  - Gradient independence check passes:
    - `independence_max_other_grad = 0.0` when optimizing only model-0 loss.
  - Per-model clipping behaves independently (norms reported per model, clipping not global).
  - Multi-model trainer smoke writes expected per-model directory layout and checkpoints.
  - `evaluate_checkpoints.py` can load sliced per-model checkpoints (compatibility preserved).
  - W&B offline mode now creates one active run per model without run-finalization conflicts.
  - Backward-compatible `num_models=1` run still emits single-run summary/checkpoint schema.
  - Short MPS performance sanity:
    - `num_models=1`: ~15.57s to step 500
    - `num_models=3`: ~16.73s to step 500
    - Observed near-flat wall-clock increase in this short benchmark, indicating good vectorization utilization on this hardware.

- Conclusion:
  - Vectorized multi-model training is functional with independent optimization behavior per model and compatible per-model checkpoint artifacts.

### Experiment E32: MPS Throughput Benchmark (`num_models=8`, step 500)
- Goal:
  - Benchmark short-run wall-clock performance for `num_models=8` on this hardware, comparable to prior `num_models=1` and `num_models=3` checks.
- Command:
```bash
uv run python -m src.train \
  --run-name perf_num_models8_mps \
  --num-models 8 \
  --train-steps 501 --eval-interval 500 \
  --batch-size 64 --val-size 256 --test-size 256 --eval-batch-size 256 \
  --device mps --wandb --wandb-mode disabled
```
- Outputs:
  - `results/runs/perf_num_models8_mps/summary.json`
  - `results/runs/perf_num_models8_mps/model_000/metrics.csv`
  - `results/runs/perf_num_models8_mps/model_001/metrics.csv`
  - `results/runs/perf_num_models8_mps/model_002/metrics.csv`
  - `results/runs/perf_num_models8_mps/model_003/metrics.csv`
  - `results/runs/perf_num_models8_mps/model_004/metrics.csv`
  - `results/runs/perf_num_models8_mps/model_005/metrics.csv`
  - `results/runs/perf_num_models8_mps/model_006/metrics.csv`
  - `results/runs/perf_num_models8_mps/model_007/metrics.csv`
- Findings:
  - `params_total=7168` (`896` per model x `8` models)
  - Step-500 elapsed wall-clock: `15.875s`
  - Short benchmark comparison:
    - `num_models=1`: `15.572s` (E31)
    - `num_models=3`: `16.729s` (E31)
    - `num_models=8`: `15.875s` (this run)
- Conclusion:
  - On this short MPS benchmark, `num_models=8` maintained nearly flat wall-clock relative to `num_models=1`, indicating strong hardware utilization for vectorized parallel models in this setup.

### Experiment E33: Baseline Run with Latest Multi-Model Defaults (`num_models=8`, 30k steps, base seed 42)
- Goal:
  - Run the current baseline end-to-end with all recent code changes and defaults.
  - Validate against the current success standard (`>=4/8` models with `best_val_exact > 0.99`).
- Command:
```bash
uv run python -m src.train \
  --run-name baseline_num8_default_s42_30k \
  --device mps --seed 42
```
- Outputs:
  - `results/runs/baseline_num8_default_s42_30k/summary.json`
  - `results/runs/baseline_num8_default_s42_30k/model_000/metrics.csv`
  - `results/runs/baseline_num8_default_s42_30k/model_001/metrics.csv`
  - `results/runs/baseline_num8_default_s42_30k/model_002/metrics.csv`
  - `results/runs/baseline_num8_default_s42_30k/model_003/metrics.csv`
  - `results/runs/baseline_num8_default_s42_30k/model_004/metrics.csv`
  - `results/runs/baseline_num8_default_s42_30k/model_005/metrics.csv`
  - `results/runs/baseline_num8_default_s42_30k/model_006/metrics.csv`
  - `results/runs/baseline_num8_default_s42_30k/model_007/metrics.csv`
- Findings:
  - Runtime:
    - `elapsed_sec = 1684.3319` (~28.1 min)
  - Aggregate:
    - `best_val_exact_mean = 0.4061`
    - `best_val_exact_min = 0.0`
    - `best_val_exact_max = 1.0`
  - Per model (`model_index`, `seed`, `best_val_exact`, `best_step`):
    - `0, 42, 0.1106, 15000`
    - `1, 43, 0.9988, 29500`
    - `2, 44, 0.1042, 12000`
    - `3, 45, 0.0352, 17000`
    - `4, 46, 0.0, 0`
    - `5, 47, 1.0, 14000`
    - `6, 48, 0.0, 0`
    - `7, 49, 1.0, 25500`
  - Models with `best_val_exact > 0.99`: `3 / 8`
- Conclusion:
  - Baseline run completed successfully, but **did not meet** current success criterion (`>=4/8` converged to `>99%`).

### Experiment E34: Targeted Follow-up Sweep2 (3 configs, `num_models=8`, 30k steps, base seed 42)
- Goal:
  - Run the requested targeted follow-up configs sequentially and compare convergence count (`best_val_exact > 0.99`).
- Command:
```bash
set -euo pipefail
for spec in "384 0.0162" "384 0.0168" "448 0.0165"; do
  bs=$(echo "$spec" | awk '{print $1}')
  lr=$(echo "$spec" | awk '{print $2}')
  lr_tag=${lr/./p}
  run_name="sweep2_num8_bs${bs}_lr${lr_tag}_s42_30k"
  echo "=== START ${run_name} ==="
  uv run python -m src.train \
    --run-name "${run_name}" \
    --num-models 8 \
    --batch-size "${bs}" \
    --lr "${lr}" \
    --train-steps 30000 \
    --device mps \
    --seed 42

  uv run python - <<PY
import json
from pathlib import Path
p = Path('results/runs') / '${run_name}' / 'summary.json'
obj = json.loads(p.read_text())
cnt = sum(1 for m in obj.get('models', []) if float(m.get('best_val_exact', 0.0)) > 0.99)
print(f"SWEEP2_RESULT run={obj['run_name']} elapsed_sec={obj.get('elapsed_sec')} success_gt099={cnt}/8")
PY
done

echo "=== SWEEP2 COMPLETE ==="
```
- Outputs:
  - `results/runs/sweep2_num8_bs384_lr0p0162_s42_30k/summary.json`
  - `results/runs/sweep2_num8_bs384_lr0p0168_s42_30k/summary.json`
  - `results/runs/sweep2_num8_bs448_lr0p0165_s42_30k/summary.json`
- Findings:
  - `sweep2_num8_bs384_lr0p0162_s42_30k`: `elapsed_sec=1184.9450`, `best_val_exact > 0.99` = `2/8`
  - `sweep2_num8_bs384_lr0p0168_s42_30k`: `elapsed_sec=1189.1019`, `best_val_exact > 0.99` = `2/8`
  - `sweep2_num8_bs448_lr0p0165_s42_30k`: `elapsed_sec=1272.3717`, `best_val_exact > 0.99` = `2/8`
- Conclusion:
  - None of the three targeted follow-up configs met the success criterion (`>=4/8` models with `best_val_exact > 0.99`).

### Experiment E35: Requested Batch Size / LR Sweep (`num_models=8`, 30k steps, base seed 42)
- Goal:
  - Sweep `batch_size ∈ {384, 512}` and `lr ∈ {0.0135, 0.0150, 0.0165}` to improve success rate from baseline `3/8`.
- Command:
```bash
set -euo pipefail
for bs in 384 512; do
  for lr in 0.0135 0.0150 0.0165; do
    lr_tag=${lr/./p}
    run_name="sweep_num8_bs${bs}_lr${lr_tag}_s42_30k"
    uv run python -m src.train \
      --run-name "${run_name}" \
      --num-models 8 \
      --batch-size "${bs}" \
      --lr "${lr}" \
      --train-steps 30000 \
      --device mps \
      --seed 42
  done
done
```
- Outputs:
  - `results/runs/sweep_num8_bs384_lr0p0135_s42_30k/summary.json`
  - `results/runs/sweep_num8_bs384_lr0p0150_s42_30k/summary.json`
  - `results/runs/sweep_num8_bs384_lr0p0165_s42_30k/summary.json`
  - `results/runs/sweep_num8_bs512_lr0p0135_s42_30k/summary.json`
  - `results/runs/sweep_num8_bs512_lr0p0150_s42_30k/summary.json`
  - `results/runs/sweep_num8_bs512_lr0p0165_s42_30k/summary.json`
- Findings:
  - `sweep_num8_bs384_lr0p0135_s42_30k`: `elapsed_sec=1598.1071`, success `3/8`
  - `sweep_num8_bs384_lr0p0150_s42_30k`: `elapsed_sec=1475.3998`, success `2/8`
  - `sweep_num8_bs384_lr0p0165_s42_30k`: `elapsed_sec=1547.0577`, success `6/8`
  - `sweep_num8_bs512_lr0p0135_s42_30k`: `elapsed_sec=1905.1928`, success `3/8`
  - `sweep_num8_bs512_lr0p0150_s42_30k`: `elapsed_sec=2000.0991`, success `4/8`
  - `sweep_num8_bs512_lr0p0165_s42_30k`: `elapsed_sec=1620.6442`, success `2/8`
- Conclusion:
  - Best configuration in the requested sweep: **`batch_size=384, lr=0.0165` with `6/8` success**, improving over baseline `3/8`.
  - Secondary viable configuration: `batch_size=512, lr=0.0150` with `4/8`.

### Experiment E36: Runtime Bottleneck Profiling for `num_models=8`
- Goal:
  - Measure where wall-clock time goes for vectorized 8-model training.
- Command:
```bash
uv run python - <<'PY'
import time, torch
from pathlib import Path
from src.model import ModelConfig, TinyDecoderLM
from src.train import CurriculumBatchSampler, CURRICULUM_PHASES, clip_grad_norm_per_model
from src.eval import evaluate_exact_match_multi
from src.data import build_holdout_splits, pair_hash, encode_batch

cfg = ModelConfig(num_models=8)
model = TinyDecoderLM(cfg).to('mps')
opt = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.01)

val_size=5000
test_size=10000
val_a=[]; val_b=[]; reserved=[]
for i in range(cfg.num_models):
    sp = build_holdout_splits(val_size, test_size, 42+i, Path(f'results/data/holdout_v{val_size}_t{test_size}_seed{42+i}.pt'))
    va=sp['val_a']; vb=sp['val_b']
    val_a.append(va); val_b.append(vb)
    rs=set()
    for ai,bi in zip(va.tolist(),vb.tolist()): rs.add(pair_hash(int(ai),int(bi)))
    for ai,bi in zip(sp['test_a'].tolist(),sp['test_b'].tolist()): rs.add(pair_hash(int(ai),int(bi)))
    reserved.append(rs)

samplers=[CurriculumBatchSampler(512, 1337+i, reserved[i], CURRICULUM_PHASES) for i in range(cfg.num_models)]

def do_sample(step=20000):
    a=[]; b=[]
    for s in samplers:
        ai,bi=s.sample_operands(step)
        a.append(ai); b.append(bi)
    a=torch.stack(a,0); b=torch.stack(b,0)
    x,y=encode_batch(a,b)
    return x,y

def do_step(x,y):
    x=x.to('mps'); y=y.to('mps')
    opt.zero_grad(set_to_none=True)
    _,loss,_ = model(x,y,return_per_model_loss=True)
    loss.backward()
    clip_grad_norm_per_model(model.parameters(), 1.0, cfg.num_models)
    opt.step()

for _ in range(10):
    x,y = do_sample(); do_step(x,y)

torch.mps.synchronize()
t0=time.time()
for _ in range(20):
    _x,_y = do_sample()
torch.mps.synchronize()
sample_sec=time.time()-t0

x,y = do_sample()
torch.mps.synchronize()
t0=time.time()
for _ in range(20):
    do_step(x,y)
torch.mps.synchronize()
step_sec=time.time()-t0

torch.mps.synchronize()
t0=time.time()
em,tok = evaluate_exact_match_multi(model, val_a, val_b, 512, torch.device('mps'))
torch.mps.synchronize()
eval_sec=time.time()-t0

print('sample_sec_total',sample_sec,'sample_sec_per_iter',sample_sec/20)
print('step_sec_total',step_sec,'step_sec_per_iter',step_sec/20)
print('eval_sec_full',eval_sec)
print('val_exact_mean',sum(em)/len(em),'val_tok_mean',sum(tok)/len(tok))
PY
```
- Outputs:
  - Profiling numbers printed to stdout (no artifact file).
- Findings:
  - `sample_sec_per_iter = 0.00118s`
  - `step_sec_per_iter = 0.03369s`
  - `eval_sec_full = 1.4390s` (full validation across all 8 models)
  - Bottleneck is training-step compute, with sampling overhead now much smaller.
- Conclusion:
  - The remaining dominant cost is model forward/backward/optimizer on MPS; further gains will require numerical/algorithmic tradeoffs (precision, eval cadence, logging cadence, or compile strategy).

### Experiment E37: End-to-End 500-Step Runtime Benchmark After Speed Optimizations
- Goal:
  - Validate real wall-clock impact of the runtime optimization changes for `num_models=8`.
- Command:
```bash
uv run python -m src.train \
  --device mps \
  --wandb-mode disabled \
  --run-name bench_num8_speedopt_s42_500 \
  --run-dir results/runs/bench_num8_speedopt_s42_500 \
  --train-steps 500 \
  --eval-interval 500 \
  --batch-size 512 \
  --lr 0.015 \
  --seed 42
```
- Outputs:
  - `results/runs/bench_num8_speedopt_s42_500/summary.json`
  - `results/runs/bench_num8_speedopt_s42_500/model_000/metrics.csv`
  - `results/runs/bench_num8_speedopt_s42_500/model_001/metrics.csv`
  - `results/runs/bench_num8_speedopt_s42_500/model_002/metrics.csv`
  - `results/runs/bench_num8_speedopt_s42_500/model_003/metrics.csv`
  - `results/runs/bench_num8_speedopt_s42_500/model_004/metrics.csv`
  - `results/runs/bench_num8_speedopt_s42_500/model_005/metrics.csv`
  - `results/runs/bench_num8_speedopt_s42_500/model_006/metrics.csv`
  - `results/runs/bench_num8_speedopt_s42_500/model_007/metrics.csv`
- Follow-up command (sampler correctness-preserving variant):
```bash
uv run python -m src.train \
  --device mps \
  --wandb-mode disabled \
  --run-name bench_num8_speedopt_v2_s42_500 \
  --run-dir results/runs/bench_num8_speedopt_v2_s42_500 \
  --train-steps 500 \
  --eval-interval 500 \
  --batch-size 512 \
  --lr 0.015 \
  --seed 42
```
- Follow-up outputs:
  - `results/runs/bench_num8_speedopt_v2_s42_500/summary.json`
  - `results/runs/bench_num8_speedopt_v2_s42_500/model_000/metrics.csv`
  - `results/runs/bench_num8_speedopt_v2_s42_500/model_001/metrics.csv`
  - `results/runs/bench_num8_speedopt_v2_s42_500/model_002/metrics.csv`
  - `results/runs/bench_num8_speedopt_v2_s42_500/model_003/metrics.csv`
  - `results/runs/bench_num8_speedopt_v2_s42_500/model_004/metrics.csv`
  - `results/runs/bench_num8_speedopt_v2_s42_500/model_005/metrics.csv`
  - `results/runs/bench_num8_speedopt_v2_s42_500/model_006/metrics.csv`
  - `results/runs/bench_num8_speedopt_v2_s42_500/model_007/metrics.csv`
- 3-seed timing commands (same settings, seeds 43 and 44):
```bash
for seed in 43 44; do
  run_name="bench_num8_speedopt_v2_s${seed}_500"
  uv run python -m src.train \
    --device mps \
    --wandb-mode disabled \
    --run-name "${run_name}" \
    --run-dir "results/runs/${run_name}" \
    --train-steps 500 \
    --eval-interval 500 \
    --batch-size 512 \
    --lr 0.015 \
    --seed "${seed}"
done
```
- 3-seed additional outputs:
  - `results/runs/bench_num8_speedopt_v2_s43_500/summary.json`
  - `results/runs/bench_num8_speedopt_v2_s44_500/summary.json`
- Findings:
  - `elapsed_sec = 19.7193` for 500 steps (`bench_num8_speedopt_s42_500`).
  - `elapsed_sec = 19.4721` for 500 steps (`bench_num8_speedopt_v2_s42_500`).
  - `elapsed_sec = 19.6068` for 500 steps (`bench_num8_speedopt_v2_s43_500`).
  - `elapsed_sec = 19.8226` for 500 steps (`bench_num8_speedopt_v2_s44_500`).
  - `bench_num8_speedopt_v2_*` 3-seed mean wall-clock: `19.6338s` per 500 steps.
  - Previous baseline-family timing reference (`sweep_num8_bs512_lr0p0150_s42_30k`, model_000): `33.2019s` for one 500-step window (`step 0 -> step 500`).
  - Optimized timing (`bench_num8_speedopt_v2_s42_500`, model_000): `17.7391s` for one 499-step window (`step 0 -> step 499`).
  - Approximate runtime reduction versus previous reference: `~46.6%` (`1.87x` faster).
  - Compared with previous observed ~`33s` per 500-step window in this config family, this is a substantial speedup.
- Conclusion:
  - Runtime optimization succeeded; 8-model experiments now complete markedly faster with unchanged model shape/parameter count.

### Experiment E38: TODO Batch (Single `num_models=8` Run per Experiment)
- Goal:
  - Execute all experiments listed in `TODO.md` using one `num_models=8` run per experiment.
  - Evaluate against TODO success rules relative to baseline reference:
    - baseline reference run: `todo_baseline_ref_s42_30k`
    - baseline metrics: `success_gt099=5/8`, `params_per_model=896`, `elapsed_sec=1132.235`
  - Success checks applied:
    1. Fewer params and `>50%` success (`>=5/8`).
    2. Same params and higher success than baseline (`>5/8`).
    3. Same params and same success as baseline (`5/8`) with lower wall-clock.
- Commands:
```bash
# baseline reference used for comparison
uv run python -m src.train \
  --run-name todo_baseline_ref_s42_30k \
  --run-dir results/runs/todo_baseline_ref_s42_30k \
  --device mps --seed 42 --train-steps 30000 --num-models 8 --batch-size 512 --lr 0.015

# Ablation of fixed_full_rank for mlp
uv run python -m src.train \
  --run-name todo_ablate_fixed_mlp_s42_30k \
  --run-dir results/runs/todo_ablate_fixed_mlp_s42_30k \
  --device mps --seed 42 --train-steps 30000 --num-models 8 --no-fixed-full-rank-mlp

# Ablation of fixed_full_rank for attention (rerun after interruption)
uv run python -m src.train \
  --run-name todo_ablate_fixed_attn_s42_30k_rerun \
  --run-dir results/runs/todo_ablate_fixed_attn_s42_30k_rerun \
  --device mps --seed 42 --train-steps 30000 --num-models 8 --no-fixed-full-rank-attn

# Lower rank for attention
uv run python -m src.train \
  --run-name todo_lowrank_attn_q3_o3_s42_30k \
  --run-dir results/runs/todo_lowrank_attn_q3_o3_s42_30k \
  --device mps --seed 42 --train-steps 30000 --num-models 8 --qkv-rank 3 --attn-out-rank 3

# Run for more steps
uv run python -m src.train \
  --run-name todo_more_steps_s42_50k \
  --run-dir results/runs/todo_more_steps_s42_50k \
  --device mps --seed 42 --train-steps 50000 --num-models 8

# Lower d_model to 7
uv run python -m src.train \
  --run-name todo_dmodel7_s42_30k \
  --run-dir results/runs/todo_dmodel7_s42_30k \
  --device mps --seed 42 --train-steps 30000 --num-models 8 --d-model 7

# Lower ffn expansion to 2x equivalent at d_model=8
uv run python -m src.train \
  --run-name todo_ffexp2x_dff16_s42_30k \
  --run-dir results/runs/todo_ffexp2x_dff16_s42_30k \
  --device mps --seed 42 --train-steps 30000 --num-models 8 --d-ff 16

# Larger weight decay with LR sweep
uv run python -m src.train \
  --run-name todo_wd005_lr0p015_s42_30k \
  --run-dir results/runs/todo_wd005_lr0p015_s42_30k \
  --device mps --seed 42 --train-steps 30000 --num-models 8 --weight-decay 0.05 --lr 0.015
uv run python -m src.train \
  --run-name todo_wd005_lr0p018_s42_30k \
  --run-dir results/runs/todo_wd005_lr0p018_s42_30k \
  --device mps --seed 42 --train-steps 30000 --num-models 8 --weight-decay 0.05 --lr 0.018
uv run python -m src.train \
  --run-name todo_wd005_lr0p021_s42_30k \
  --run-dir results/runs/todo_wd005_lr0p021_s42_30k \
  --device mps --seed 42 --train-steps 30000 --num-models 8 --weight-decay 0.05 --lr 0.021
```
- Outputs:
  - `results/runs/todo_baseline_ref_s42_30k/summary.json`
  - `results/runs/todo_ablate_fixed_mlp_s42_30k/summary.json`
  - `results/runs/todo_ablate_fixed_attn_s42_30k_rerun/summary.json`
  - `results/runs/todo_lowrank_attn_q3_o3_s42_30k/summary.json`
  - `results/runs/todo_more_steps_s42_50k/summary.json`
  - `results/runs/todo_dmodel7_s42_30k/summary.json`
  - `results/runs/todo_ffexp2x_dff16_s42_30k/summary.json`
  - `results/runs/todo_wd005_lr0p015_s42_30k/summary.json`
  - `results/runs/todo_wd005_lr0p018_s42_30k/summary.json`
  - `results/runs/todo_wd005_lr0p021_s42_30k/summary.json`
- Findings:
  - `todo_ablate_fixed_mlp_s42_30k`: `success=1/8`, `params=896`, `elapsed_sec=770.393` -> **Fail**
  - `todo_ablate_fixed_attn_s42_30k_rerun`: `success=3/8`, `params=896`, `elapsed_sec=839.646` -> **Fail**
  - `todo_lowrank_attn_q3_o3_s42_30k`: `success=0/8`, `params=784`, `elapsed_sec=1000.504` -> **Fail**
  - `todo_more_steps_s42_50k`: `success=5/8`, `params=896`, `elapsed_sec=1935.829` -> **Fail** (no success gain, slower)
  - `todo_dmodel7_s42_30k`: `success=0/8`, `params=777`, `elapsed_sec=837.817` -> **Fail**
  - `todo_ffexp2x_dff16_s42_30k`: `success=4/8`, `params=824`, `elapsed_sec=800.781` -> **Fail** (`4/8` is not `>50%`)
  - `todo_wd005_lr0p015_s42_30k`: `success=5/8`, `params=896`, `elapsed_sec=1375.145` -> **Fail** (same success, slower)
  - `todo_wd005_lr0p018_s42_30k`: `success=4/8`, `params=896`, `elapsed_sec=1332.575` -> **Fail**
  - `todo_wd005_lr0p021_s42_30k`: `success=5/8`, `params=896`, `elapsed_sec=1322.769` -> **Fail** (same success, slower)
- Conclusion:
  - No experiment in this batch met TODO success criteria.
  - Baseline defaults are unchanged based on this batch.

### Experiment E39: TODO Custom LR Schedule (Low-LR Tail) and Revert
- Goal:
  - Test a custom schedule that spends more time at low learning rate near training end.
- Temporary code change:
  - Added an optional schedule mode (`cosine_low_tail`) in `src/train.py`, ran experiment, then reverted the code after evaluation.
- Command:
```bash
uv run python -m src.train \
  --run-name todo_lr_schedule_lowtail03_s42_30k \
  --run-dir results/runs/todo_lr_schedule_lowtail03_s42_30k \
  --device mps --seed 42 \
  --train-steps 30000 --num-models 8 \
  --lr-schedule cosine_low_tail \
  --low-lr-tail-ratio 0.3
```
- Outputs:
  - `results/runs/todo_lr_schedule_lowtail03_s42_30k/summary.json`
- Findings:
  - `success=3/8`, `params=896`, `elapsed_sec=1375.892`
  - Versus baseline reference (`5/8`, `1132.235s`), this is worse in both success and wall-clock.
- Conclusion:
  - **Fail** by TODO criteria.
  - Temporary schedule code was reverted; baseline training code remains on the original cosine schedule.
