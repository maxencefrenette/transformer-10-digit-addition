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

### Conclusion
- Reproduction training attempts with seeds 43 and 44 on `mps` did not reproduce the published 100% result.
- Evaluation pipeline is valid locally (provided checkpoint reproduces expected 100%).
- 512-parameter baseline training also failed to reproduce on `mps`, while the provided 512 checkpoint evaluates as expected.
- Extending the 512 baseline to 100k steps on `mps` still failed to produce a strong grokking solution.
- 763-param and 959-param baseline attempts also failed to reach reliable exact-match accuracy on `mps`.
