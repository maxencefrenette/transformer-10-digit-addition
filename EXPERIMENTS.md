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

### Conclusion
- Reproduction training attempts with seeds 43 and 44 on `mps` did not reproduce the published 100% result.
- Evaluation pipeline is valid locally (provided checkpoint reproduces expected 100%).
