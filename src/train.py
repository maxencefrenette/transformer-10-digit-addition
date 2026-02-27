"""Training entrypoint for 10-digit addition transformer.

Incorporates gpt-acc-jax techniques:
  - Curriculum learning (3 phases: 1-3 digits, 1-6 digits, 1-10 digits)
  - High learning rate (0.02) with cosine decay
  - 27K total training steps
  - AdamW optimizer with gradient clipping

Usage:
  python -m src.train --run-name baseline_num8_default --device mps --seed 42
"""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from src.data import (
    INPUT_LEN,
    VOCAB_SIZE,
    build_holdout_splits,
    encode_batch,
    pair_hash,
)
from src.eval import evaluate_exact_match, evaluate_exact_match_multi
from src.model import ModelConfig, TinyDecoderLM, count_parameters


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class TrainConfig:
    seed: int
    train_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    min_lr_ratio: float
    grad_clip: float
    eval_interval: int
    val_size: int
    test_size: int
    eval_batch_size: int
    run_name: str
    run_dir: str
    split_dir: str
    best_ckpt_out: str
    last_ckpt_out: str
    device: str
    dtype: str = "fp32"  # fp32, fp16, bf16
    wandb: bool = True
    wandb_project: str = "maxence-frenette/transformer-10-digit-addition"
    wandb_mode: str = "online"
    # Curriculum phases: list of (min_digits, max_digits, steps)
    curriculum: str = ""  # serialized; parsed below


CURRICULUM_PHASES = [
    (1, 3, 2000),    # Phase 1: easy
    (1, 6, 5000),    # Phase 2: medium
    (1, 10, 20000),  # Phase 3: full range
]
TOTAL_CURRICULUM_STEPS = sum(s for _, _, s in CURRICULUM_PHASES)  # 27000


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class CurriculumBatchSampler:
    """Samples training batches with curriculum learning and holdout avoidance."""

    def __init__(self, batch_size: int, seed: int, reserved_hashes: Set[int],
                 curriculum_phases: List[Tuple[int, int, int]]):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes
        self.phases = curriculum_phases
        # Build cumulative step boundaries
        self.boundaries = []
        cum = 0
        for _, _, steps in self.phases:
            cum += steps
            self.boundaries.append(cum)

    def _phase_for_step(self, step: int) -> Tuple[int, int]:
        for i, boundary in enumerate(self.boundaries):
            if step < boundary:
                return self.phases[i][0], self.phases[i][1]
        return self.phases[-1][0], self.phases[-1][1]

    def sample_batch(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        min_dig, max_dig = self._phase_for_step(step)

        a = torch.zeros(self.batch_size, dtype=torch.int64)
        b = torch.zeros(self.batch_size, dtype=torch.int64)

        for i in range(self.batch_size):
            n_dig = int(torch.randint(min_dig, max_dig + 1, (1,), generator=self.g).item())
            max_val = 10 ** n_dig
            ai = int(torch.randint(0, max_val, (1,), generator=self.g, dtype=torch.int64).item())
            bi = int(torch.randint(0, max_val, (1,), generator=self.g, dtype=torch.int64).item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, max_val, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, max_val, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi

        return encode_batch(a, b)


def cosine_lr(step: int, max_steps: int, base_lr: float,
              warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv_header(path: Path, header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def append_csv(path: Path, row: List) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def parse_wandb_project_path(value: str) -> Tuple[Optional[str], str]:
    """Accept either 'project' or 'entity/project'."""
    path = value.strip()
    if not path:
        raise ValueError("W&B project path cannot be empty")
    if "/" not in path:
        return None, path
    entity, project = path.split("/", 1)
    if not entity or not project:
        raise ValueError(f"Invalid W&B project path: '{value}'")
    return entity, project


def clip_grad_norm_per_model(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    num_models: int,
) -> torch.Tensor:
    """Compute and clip gradient norms independently per model axis entry."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(num_models)

    device = grads[0].device
    total_sq = torch.zeros(num_models, device=device)

    for g in grads:
        if g.dim() == 0:
            raise ValueError("Encountered scalar gradient; expected model-axis tensors")
        if g.shape[0] != num_models:
            raise ValueError(
                "All gradients must have model axis in dim 0 for per-model clipping; "
                f"got shape {tuple(g.shape)} with num_models={num_models}"
            )
        flat = g.detach().reshape(num_models, -1)
        total_sq += (flat * flat).sum(dim=1)

    total_norm = torch.sqrt(total_sq)

    if max_norm > 0 and not math.isinf(max_norm):
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        for g in grads:
            view = [num_models] + [1] * (g.dim() - 1)
            g.mul_(clip_coef.view(*view))

    return total_norm


def _model_axis_state_keys(model: TinyDecoderLM, num_models: int) -> Set[str]:
    keys: Set[str] = set()
    for name, param in model.named_parameters():
        if param.dim() > 0 and param.shape[0] == num_models:
            keys.add(name)
    for name, buf in model.named_buffers():
        # Causal mask is shared, not model-axis specific.
        if name.endswith("mask"):
            continue
        if buf.dim() > 0 and buf.shape[0] == num_models:
            keys.add(name)
    return keys


def _slice_state_dict_for_model(
    state_dict: Dict[str, torch.Tensor],
    model_axis_keys: Set[str],
    model_index: int,
    num_models: int,
) -> Dict[str, torch.Tensor]:
    sliced: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            sliced[key] = value
            continue
        # lm_head.weight is tied to token_emb.weight and may be omitted from
        # named_parameters() deduplicated traversal, so keep a shape-based fallback.
        has_model_axis = (
            key in model_axis_keys
            or (value.dim() > 0 and value.shape[0] == num_models and not key.endswith("mask"))
        )
        if has_model_axis:
            sliced[key] = value[model_index:model_index + 1].clone()
        else:
            sliced[key] = value.clone()
    return sliced


def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Dict:
    if model_cfg.num_models < 1:
        raise ValueError(f"num_models must be >= 1, got {model_cfg.num_models}")

    device = torch.device(train_cfg.device)
    dtype = DTYPE_MAP.get(train_cfg.dtype, torch.float32)
    run_dir = Path(train_cfg.run_dir)
    split_dir = Path(train_cfg.split_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    num_models = model_cfg.num_models
    model_seeds = [train_cfg.seed + i for i in range(num_models)]
    set_seed(train_cfg.seed)

    val_a_by_model: List[torch.Tensor] = []
    val_b_by_model: List[torch.Tensor] = []
    reserved_hashes_by_model: List[Set[int]] = []
    for seed_i in model_seeds:
        split_path = split_dir / f"holdout_v{train_cfg.val_size}_t{train_cfg.test_size}_seed{seed_i}.pt"
        splits = build_holdout_splits(train_cfg.val_size, train_cfg.test_size, seed_i, split_path)

        reserved_hashes: Set[int] = set()
        for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
            reserved_hashes.add(pair_hash(int(ai), int(bi)))
        for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
            reserved_hashes.add(pair_hash(int(ai), int(bi)))

        val_a_by_model.append(splits["val_a"])
        val_b_by_model.append(splits["val_b"])
        reserved_hashes_by_model.append(reserved_hashes)

    samplers = [
        CurriculumBatchSampler(
            train_cfg.batch_size,
            train_cfg.seed + 1337 + i,
            reserved_hashes_by_model[i],
            CURRICULUM_PHASES,
        )
        for i in range(num_models)
    ]

    if num_models == 1:
        model_run_dirs = [run_dir]
        model_run_names = [train_cfg.run_name]
        best_ckpt_paths = [Path(train_cfg.best_ckpt_out)]
        last_ckpt_paths = [Path(train_cfg.last_ckpt_out)]
    else:
        model_run_dirs = [run_dir / f"model_{i:03d}" for i in range(num_models)]
        model_run_names = [f"{train_cfg.run_name}_model{i:03d}" for i in range(num_models)]
        best_ckpt_paths = [d / "checkpoints" / "best.pt" for d in model_run_dirs]
        last_ckpt_paths = [d / "checkpoints" / "last.pt" for d in model_run_dirs]

    metrics_paths = [d / "metrics.csv" for d in model_run_dirs]
    metrics_header = ["step", "train_loss", "val_exact", "val_token_acc", "grad_norm", "lr", "elapsed_sec"]
    for path in metrics_paths:
        save_csv_header(path, metrics_header)

    # Mixed precision: model stays fp32, autocast handles fp16/bf16 in forward pass.
    use_amp = (dtype != torch.float32 and device.type == "cuda")
    model = TinyDecoderLM(model_cfg).to(device=device)  # always fp32

    params_total = count_parameters(model)
    params_per_model = params_total // num_models

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    best_val = [-1.0 for _ in range(num_models)]
    best_step = [-1 for _ in range(num_models)]
    t0 = time.time()

    # GradScaler for fp16 (not needed for bf16)
    use_scaler = (dtype == torch.float16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    print(f"Run: {train_cfg.run_name}")
    if num_models == 1:
        print(f"Params: {params_total}")
    else:
        print(
            f"Params: total={params_total} "
            f"({params_per_model} per model x {num_models} models)"
        )
    print(f"Model config: {model_cfg}")
    print(f"Model seeds: {model_seeds}")
    print(f"Device: {device}, amp_dtype: {dtype}, use_amp: {use_amp}, scaler: {use_scaler}")
    print(f"Curriculum: {CURRICULUM_PHASES}")

    wandb_runs: List[Optional[object]] = [None for _ in range(num_models)]
    if train_cfg.wandb and train_cfg.wandb_mode != "disabled":
        try:
            import wandb

            entity, project = parse_wandb_project_path(train_cfg.wandb_project)
            for i in range(num_models):
                cfg_for_run = replace(model_cfg, num_models=1)
                wandb_init = {
                    "project": project,
                    "name": model_run_names[i],
                    "group": train_cfg.run_name if num_models > 1 else None,
                    "mode": train_cfg.wandb_mode,
                    "reinit": "create_new",
                    "config": {
                        "model": asdict(cfg_for_run),
                        "train": asdict(train_cfg),
                        "curriculum_phases": CURRICULUM_PHASES,
                        "params": params_per_model if num_models > 1 else params_total,
                        "parallel": {
                            "num_models": num_models,
                            "model_index": i,
                            "model_seed": model_seeds[i],
                            "base_seed": train_cfg.seed,
                        },
                    },
                }
                if wandb_init["group"] is None:
                    del wandb_init["group"]
                if entity is not None:
                    wandb_init["entity"] = entity
                wandb_runs[i] = wandb.init(**wandb_init)
            print(f"W&B: logging to {train_cfg.wandb_project} (mode={train_cfg.wandb_mode})")
        except Exception as exc:
            print(f"W&B: disabled ({exc})")
            wandb_runs = [None for _ in range(num_models)]

    model_axis_keys = _model_axis_state_keys(model, num_models)
    single_model_cfg = replace(model_cfg, num_models=1)

    for step in range(train_cfg.train_steps):
        model.train()

        x_batches: List[torch.Tensor] = []
        y_batches: List[torch.Tensor] = []
        for sampler in samplers:
            x_i, y_i = sampler.sample_batch(step)
            x_batches.append(x_i)
            y_batches.append(y_i)

        x = torch.stack(x_batches, dim=0).to(device)  # [M, B, T]
        y = torch.stack(y_batches, dim=0).to(device)  # [M, B, T]

        lr_now = cosine_lr(step, train_cfg.train_steps, train_cfg.lr,
                           train_cfg.warmup_steps, train_cfg.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=dtype):
                _, loss, per_model_loss = model(x, y, return_per_model_loss=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_max_norm = train_cfg.grad_clip if train_cfg.grad_clip > 0 else float("inf")
                grad_norm = clip_grad_norm_per_model(model.parameters(), clip_max_norm, num_models)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_max_norm = train_cfg.grad_clip if train_cfg.grad_clip > 0 else float("inf")
                grad_norm = clip_grad_norm_per_model(model.parameters(), clip_max_norm, num_models)
                optimizer.step()
        else:
            _, loss, per_model_loss = model(x, y, return_per_model_loss=True)
            loss.backward()
            clip_max_norm = train_cfg.grad_clip if train_cfg.grad_clip > 0 else float("inf")
            grad_norm = clip_grad_norm_per_model(model.parameters(), clip_max_norm, num_models)
            optimizer.step()

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
            if num_models == 1:
                val_exact_single, val_tok_single = evaluate_exact_match(
                    model,
                    val_a_by_model[0],
                    val_b_by_model[0],
                    train_cfg.eval_batch_size,
                    device,
                )
                val_exact = [val_exact_single]
                val_tok = [val_tok_single]
            else:
                val_exact, val_tok = evaluate_exact_match_multi(
                    model,
                    val_a_by_model,
                    val_b_by_model,
                    train_cfg.eval_batch_size,
                    device,
                )

            elapsed = time.time() - t0
            per_model_loss_vals = [float(v) for v in per_model_loss.detach().cpu().tolist()]
            grad_norm_vals = [float(v) for v in grad_norm.detach().cpu().tolist()]

            if num_models == 1:
                append_csv(
                    metrics_paths[0],
                    [step, per_model_loss_vals[0], val_exact[0], val_tok[0], grad_norm_vals[0], lr_now, elapsed],
                )
                print(
                    f"step={step:6d} loss={per_model_loss_vals[0]:.4f} val_exact={val_exact[0]:.4f} "
                    f"val_tok={val_tok[0]:.5f} grad_norm={grad_norm_vals[0]:.4f} "
                    f"lr={lr_now:.2e} t={elapsed:.1f}s"
                )
            else:
                for i in range(num_models):
                    append_csv(
                        metrics_paths[i],
                        [step, per_model_loss_vals[i], val_exact[i], val_tok[i], grad_norm_vals[i], lr_now, elapsed],
                    )
                print(
                    f"step={step:6d} "
                    f"loss_mean={sum(per_model_loss_vals)/num_models:.4f} "
                    f"val_exact_mean={sum(val_exact)/num_models:.4f} "
                    f"val_exact_min={min(val_exact):.4f} "
                    f"val_exact_max={max(val_exact):.4f} "
                    f"lr={lr_now:.2e} t={elapsed:.1f}s"
                )

            for i in range(num_models):
                if wandb_runs[i] is not None:
                    wandb_runs[i].log(
                        {
                            "train/loss": per_model_loss_vals[i],
                            "val/exact_match": val_exact[i],
                            "val/token_acc": val_tok[i],
                            "optim/grad_norm": grad_norm_vals[i],
                            "optim/lr": lr_now,
                            "time/elapsed_sec": elapsed,
                        },
                        step=step,
                    )

            improved = [i for i in range(num_models) if val_exact[i] > best_val[i]]
            if improved:
                full_state = model.state_dict()
                for i in improved:
                    best_val[i] = val_exact[i]
                    best_step[i] = step

                    if num_models == 1:
                        model_state = full_state
                        model_config_payload = asdict(model_cfg)
                        train_config_payload = asdict(train_cfg)
                        params_payload = params_total
                    else:
                        model_state = _slice_state_dict_for_model(full_state, model_axis_keys, i, num_models)
                        model_config_payload = asdict(single_model_cfg)
                        train_config_payload = asdict(train_cfg)
                        train_config_payload["base_seed"] = train_cfg.seed
                        train_config_payload["seed"] = model_seeds[i]
                        train_config_payload["model_index"] = i
                        train_config_payload["num_models"] = num_models
                        train_config_payload["run_name"] = model_run_names[i]
                        train_config_payload["run_dir"] = str(model_run_dirs[i])
                        train_config_payload["best_ckpt_out"] = str(best_ckpt_paths[i])
                        train_config_payload["last_ckpt_out"] = str(last_ckpt_paths[i])
                        params_payload = params_per_model

                    best_ckpt_paths[i].parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state": model_state,
                            "model_config": model_config_payload,
                            "train_config": train_config_payload,
                            "step": step,
                            "val_exact": val_exact[i],
                            "params": params_payload,
                        },
                        best_ckpt_paths[i],
                    )

                    if wandb_runs[i] is not None:
                        wandb_runs[i].summary["best_val_exact"] = best_val[i]
                        wandb_runs[i].summary["best_step"] = best_step[i]

    full_state = model.state_dict()
    for i in range(num_models):
        if num_models == 1:
            model_state = full_state
            model_config_payload = asdict(model_cfg)
            train_config_payload = asdict(train_cfg)
            params_payload = params_total
        else:
            model_state = _slice_state_dict_for_model(full_state, model_axis_keys, i, num_models)
            model_config_payload = asdict(single_model_cfg)
            train_config_payload = asdict(train_cfg)
            train_config_payload["base_seed"] = train_cfg.seed
            train_config_payload["seed"] = model_seeds[i]
            train_config_payload["model_index"] = i
            train_config_payload["num_models"] = num_models
            train_config_payload["run_name"] = model_run_names[i]
            train_config_payload["run_dir"] = str(model_run_dirs[i])
            train_config_payload["best_ckpt_out"] = str(best_ckpt_paths[i])
            train_config_payload["last_ckpt_out"] = str(last_ckpt_paths[i])
            params_payload = params_per_model

        last_ckpt_paths[i].parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model_state,
                "model_config": model_config_payload,
                "train_config": train_config_payload,
                "step": train_cfg.train_steps - 1,
                "val_exact": best_val[i],
                "params": params_payload,
            },
            last_ckpt_paths[i],
        )

    elapsed_sec = time.time() - t0

    if num_models == 1:
        summary = {
            "run_name": train_cfg.run_name,
            "params": params_total,
            "best_val_exact": best_val[0],
            "best_step": best_step[0],
            "train_steps": train_cfg.train_steps,
            "elapsed_sec": elapsed_sec,
            "model_config": asdict(model_cfg),
        }
        save_json(run_dir / "summary.json", summary)
        save_json(run_dir / "config.json", {"model": asdict(model_cfg), "train": asdict(train_cfg)})
    else:
        model_summaries = []
        for i in range(num_models):
            per_model_summary = {
                "run_name": model_run_names[i],
                "model_index": i,
                "seed": model_seeds[i],
                "params": params_per_model,
                "best_val_exact": best_val[i],
                "best_step": best_step[i],
                "train_steps": train_cfg.train_steps,
                "elapsed_sec": elapsed_sec,
                "model_config": asdict(single_model_cfg),
            }
            model_summaries.append(per_model_summary)
            save_json(model_run_dirs[i] / "summary.json", per_model_summary)
            save_json(
                model_run_dirs[i] / "config.json",
                {
                    "model": asdict(single_model_cfg),
                    "train": {
                        **asdict(train_cfg),
                        "base_seed": train_cfg.seed,
                        "seed": model_seeds[i],
                        "model_index": i,
                        "num_models": num_models,
                        "run_name": model_run_names[i],
                        "run_dir": str(model_run_dirs[i]),
                    },
                },
            )

        summary = {
            "run_name": train_cfg.run_name,
            "num_models": num_models,
            "params_per_model": params_per_model,
            "params_total": params_total,
            "train_steps": train_cfg.train_steps,
            "elapsed_sec": elapsed_sec,
            "model_config": asdict(model_cfg),
            "aggregate": {
                "best_val_exact_mean": sum(best_val) / num_models,
                "best_val_exact_min": min(best_val),
                "best_val_exact_max": max(best_val),
            },
            "models": model_summaries,
        }
        save_json(run_dir / "summary.json", summary)
        save_json(
            run_dir / "config.json",
            {
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "model_seeds": model_seeds,
            },
        )

    for i in range(num_models):
        if wandb_runs[i] is not None:
            wandb_runs[i].summary["final_val_exact"] = best_val[i]
            wandb_runs[i].summary["final_best_step"] = best_step[i]
            wandb_runs[i].summary["elapsed_sec"] = elapsed_sec
            wandb_runs[i].finish()

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train addition transformer")

    # run/output
    p.add_argument("--run-name", type=str, default="baseline_num8_default")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--split-dir", type=Path, default=Path("results/data"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="Training precision (fp32, fp16, bf16)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed. Model i uses seed+i when --num-models > 1")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str,
                   default="maxence-frenette/transformer-10-digit-addition",
                   help="W&B project path: 'entity/project' or 'project'")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="W&B logging mode")

    # model (gpt-acc-jax defaults)
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--n-head", type=int, default=1)
    p.add_argument("--d-ff", type=int, default=28)
    p.add_argument("--num-models", type=int, default=8,
                   help="Number of independent models trained in parallel")
    # low-rank options
    p.add_argument("--pos-rank", type=int, default=0, help="Position embedding rank (0=full)")
    p.add_argument("--qkv-rank", type=int, default=0, help="QKV projection rank (0=full)")
    p.add_argument("--attn-out-rank", type=int, default=0, help="Attn output rank (0=full)")
    p.add_argument("--ffn-rank", type=int, default=3, help="FFN rank (0=full)")
    p.add_argument(
        "--fixed-full-rank-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add fixed full-rank random term in attention low-rank linear layers",
    )
    p.add_argument(
        "--fixed-full-rank-mlp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add fixed full-rank random term in MLP low-rank linear layers",
    )

    # optimization (gpt-acc-jax defaults)
    p.add_argument("--train-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.015)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1350)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=500)

    # eval
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--test-size", type=int, default=10000)
    p.add_argument("--eval-batch-size", type=int, default=512)

    args = p.parse_args()

    if args.run_dir is None:
        args.run_dir = Path(f"results/runs/{args.run_name}")

    model_cfg = ModelConfig(
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        max_seq_len=INPUT_LEN,
        vocab_size=VOCAB_SIZE,
        pos_rank=args.pos_rank,
        qkv_rank=args.qkv_rank,
        attn_out_rank=args.attn_out_rank,
        ffn_rank=args.ffn_rank,
        fixed_full_rank_attn=args.fixed_full_rank_attn,
        fixed_full_rank_mlp=args.fixed_full_rank_mlp,
        num_models=args.num_models,
    )

    run_dir = Path(args.run_dir)
    train_cfg = TrainConfig(
        seed=args.seed,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        val_size=args.val_size,
        test_size=args.test_size,
        eval_batch_size=args.eval_batch_size,
        run_name=args.run_name,
        run_dir=str(run_dir),
        split_dir=str(args.split_dir),
        best_ckpt_out=str(run_dir / "checkpoints" / "best.pt"),
        last_ckpt_out=str(run_dir / "checkpoints" / "last.pt"),
        device=args.device,
        dtype=args.dtype,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
    )

    summary = train(model_cfg, train_cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
