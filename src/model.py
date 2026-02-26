"""Transformer for 10-digit addition with optional low-rank factorization.

Supports vectorized multi-model training via a leading model axis (m).
"""

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_layer: int = 1
    d_model: int = 8
    n_head: int = 1
    d_ff: int = 28
    max_seq_len: int = 33   # 22 prompt + 11 target digits
    vocab_size: int = 14    # 0-9, +, =, <PAD>, <EOS>
    # Optional low-rank factors (0 = full rank)
    pos_rank: int = 0
    qkv_rank: int = 0
    attn_out_rank: int = 0
    ffn_rank: int = 3
    fixed_full_rank_attn: bool = True
    fixed_full_rank_mlp: bool = True
    # Number of independent models trained in parallel.
    num_models: int = 8

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelConfig":
        """Create config from checkpoint dict while ignoring unknown keys."""
        keys = {k: payload[k] for k in cls.__annotations__ if k in payload}
        return cls(**keys)


class ModelLinear(nn.Module):
    """Model-axis aware full-rank linear: y = x @ W (no bias)."""

    def __init__(self, num_models: int, in_features: int, out_features: int):
        super().__init__()
        self.num_models = num_models
        self.weight = nn.Parameter(torch.empty(num_models, in_features, out_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [m, b, t, in], weight: [m, in, out]
        return torch.einsum("mbti,mio->mbto", x, self.weight)


class LowRankLinear(nn.Module):
    """Model-axis aware low-rank linear: y = x @ A @ B (+ fixed full-rank term)."""

    def __init__(
        self,
        num_models: int,
        in_features: int,
        out_features: int,
        rank: int,
        fixed_full_rank: bool = False,
    ):
        super().__init__()
        self.num_models = num_models
        self.fixed_full_rank = fixed_full_rank
        self.A = nn.Parameter(torch.empty(num_models, in_features, rank))
        self.B = nn.Parameter(torch.empty(num_models, rank, out_features))
        nn.init.normal_(self.A, std=math.sqrt(2.0 / (in_features + rank)))
        nn.init.normal_(self.B, std=math.sqrt(2.0 / (rank + out_features)))

        # Always keep this buffer for checkpoint compatibility; usage is gated.
        self.register_buffer("W_fixed", torch.empty(num_models, in_features, out_features))
        nn.init.normal_(self.W_fixed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [m, b, t, in]
        y = torch.einsum("mbti,mir,mro->mbto", x, self.A, self.B)
        if self.fixed_full_rank:
            y = y + torch.einsum("mbti,mio->mbto", x, self.W_fixed)
        return y


class ModelEmbedding(nn.Module):
    """Model-axis aware embedding table."""

    def __init__(self, num_models: int, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_models = num_models
        self.weight = nn.Parameter(torch.empty(num_models, num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [m, b, t]
        if idx.dim() != 3:
            raise ValueError(f"idx must be 3D [m,b,t], got shape {tuple(idx.shape)}")
        if idx.shape[0] != self.num_models:
            raise ValueError(
                f"idx model axis ({idx.shape[0]}) must equal num_models ({self.num_models})"
            )
        midx = torch.arange(self.num_models, device=idx.device).view(self.num_models, 1, 1)
        return self.weight[midx, idx]


class LowRankEmbedding(nn.Module):
    """Model-axis aware low-rank embedding: E[i] = A[i] @ B."""

    def __init__(self, num_models: int, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        self.num_models = num_models
        self.A = nn.Parameter(torch.empty(num_models, num_embeddings, rank))
        self.B = nn.Parameter(torch.empty(num_models, rank, embedding_dim))
        nn.init.normal_(self.A, std=0.02)
        nn.init.normal_(self.B, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [m, b, t]
        if idx.dim() != 3:
            raise ValueError(f"idx must be 3D [m,b,t], got shape {tuple(idx.shape)}")
        if idx.shape[0] != self.num_models:
            raise ValueError(
                f"idx model axis ({idx.shape[0]}) must equal num_models ({self.num_models})"
            )
        midx = torch.arange(self.num_models, device=idx.device).view(self.num_models, 1, 1)
        a_rows = self.A[midx, idx]  # [m, b, t, r]
        return torch.einsum("mbtr,mrd->mbtd", a_rows, self.B)


class ModelLayerNorm(nn.Module):
    """Model-axis aware LayerNorm over the last dimension."""

    def __init__(self, num_models: int, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.num_models = num_models
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_models, d_model))
        self.bias = nn.Parameter(torch.zeros(num_models, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [m, b, t, d]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return (
            xhat * self.weight[:, None, None, :]
            + self.bias[:, None, None, :]
        )


class TiedLMHead(nn.Module):
    """Model-axis aware tied output projection using token embeddings."""

    def __init__(self, token_emb: ModelEmbedding):
        super().__init__()
        # Keep only embedding-output weight tying.
        self.weight = token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [m, b, t, d], weight: [m, vocab, d]
        return torch.einsum("mbtd,mvd->mbtv", x, self.weight)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        num_models: int,
        d_model: int,
        n_head: int,
        max_seq_len: int,
        qkv_rank: int = 0,
        attn_out_rank: int = 0,
        fixed_full_rank: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        if qkv_rank > 0:
            self.qkv = LowRankLinear(
                num_models,
                d_model,
                3 * d_model,
                qkv_rank,
                fixed_full_rank=fixed_full_rank,
            )
        else:
            self.qkv = ModelLinear(num_models, d_model, 3 * d_model)

        if attn_out_rank > 0:
            self.proj = LowRankLinear(
                num_models,
                d_model,
                d_model,
                attn_out_rank,
                fixed_full_rank=fixed_full_rank,
            )
        else:
            self.proj = ModelLinear(num_models, d_model, d_model)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [m, b, t, d]
        msz, bsz, seqlen, d_model = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(msz, bsz, seqlen, self.n_head, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.view(msz, bsz, seqlen, self.n_head, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.view(msz, bsz, seqlen, self.n_head, self.head_dim).permute(0, 1, 3, 2, 4)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:seqlen, :seqlen], float("-inf"))
        att = F.softmax(att, dim=-1)

        y = (att @ v).permute(0, 1, 3, 2, 4).contiguous().view(msz, bsz, seqlen, d_model)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(
        self,
        num_models: int,
        d_model: int,
        d_ff: int,
        ffn_rank: int = 0,
        fixed_full_rank: bool = False,
    ):
        super().__init__()
        if ffn_rank > 0:
            self.fc1 = LowRankLinear(
                num_models,
                d_model,
                d_ff,
                ffn_rank,
                fixed_full_rank=fixed_full_rank,
            )
            self.fc2 = LowRankLinear(
                num_models,
                d_ff,
                d_model,
                ffn_rank,
                fixed_full_rank=fixed_full_rank,
            )
        else:
            self.fc1 = ModelLinear(num_models, d_model, d_ff)
            self.fc2 = ModelLinear(num_models, d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = ModelLayerNorm(cfg.num_models, cfg.d_model)
        self.attn = CausalSelfAttention(
            cfg.num_models,
            cfg.d_model,
            cfg.n_head,
            cfg.max_seq_len,
            qkv_rank=cfg.qkv_rank,
            attn_out_rank=cfg.attn_out_rank,
            fixed_full_rank=cfg.fixed_full_rank_attn,
        )
        self.ln2 = ModelLayerNorm(cfg.num_models, cfg.d_model)
        self.mlp = MLP(
            cfg.num_models,
            cfg.d_model,
            cfg.d_ff,
            ffn_rank=cfg.ffn_rank,
            fixed_full_rank=cfg.fixed_full_rank_mlp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def per_model_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy reduced to one scalar per model axis entry."""
    if logits.dim() != 4:
        raise ValueError(f"logits must be 4D [m,b,t,v], got {tuple(logits.shape)}")
    if targets.dim() != 3:
        raise ValueError(f"targets must be 3D [m,b,t], got {tuple(targets.shape)}")

    msz, bsz, tlen, vocab = logits.shape
    flat_loss = F.cross_entropy(
        logits.reshape(msz * bsz * tlen, vocab),
        targets.reshape(msz * bsz * tlen),
        ignore_index=ignore_index,
        reduction="none",
    )
    token_loss = flat_loss.view(msz, bsz, tlen)
    valid = targets.ne(ignore_index)
    denom = valid.sum(dim=(1, 2)).clamp_min(1).to(token_loss.dtype)
    return (token_loss * valid.to(token_loss.dtype)).sum(dim=(1, 2)) / denom


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = ModelEmbedding(cfg.num_models, cfg.vocab_size, cfg.d_model)
        if cfg.pos_rank > 0:
            self.pos_emb = LowRankEmbedding(
                cfg.num_models,
                cfg.max_seq_len,
                cfg.d_model,
                cfg.pos_rank,
            )
        else:
            self.pos_emb = ModelEmbedding(cfg.num_models, cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = ModelLayerNorm(cfg.num_models, cfg.d_model)
        self.lm_head = TiedLMHead(self.token_emb)

    def _prepare_idx(self, idx: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Normalize input to [m,b,t]. Returns (idx3d, squeeze_back)."""
        if idx.dim() == 2:
            if self.cfg.num_models != 1:
                raise ValueError(
                    "2D [b,t] input is only valid when num_models=1; "
                    f"got num_models={self.cfg.num_models}"
                )
            return idx.unsqueeze(0), True
        if idx.dim() == 3:
            if idx.shape[0] != self.cfg.num_models:
                raise ValueError(
                    f"idx model axis ({idx.shape[0]}) must equal num_models ({self.cfg.num_models})"
                )
            return idx, False
        raise ValueError(f"idx must be 2D or 3D, got shape {tuple(idx.shape)}")

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_per_model_loss: bool = False,
    ):
        idx3, squeeze_back = self._prepare_idx(idx)
        msz, bsz, seqlen = idx3.shape

        pos = torch.arange(seqlen, device=idx3.device).view(1, 1, seqlen)
        pos = pos.expand(msz, bsz, seqlen)

        x = self.token_emb(idx3) + self.pos_emb(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        per_model_loss = None
        if targets is not None:
            tgt3, _ = self._prepare_idx(targets)
            per_model_loss = per_model_cross_entropy(logits, tgt3, ignore_index=-100)
            loss = per_model_loss.sum()

        if squeeze_back:
            logits = logits.squeeze(0)
            if per_model_loss is not None:
                per_model_loss = per_model_loss.squeeze(0)

        if return_per_model_loss:
            return logits, loss, per_model_loss
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        out, squeeze_back = self._prepare_idx(prompt)
        for _ in range(max_new_tokens):
            idx = out[:, :, -self.cfg.max_seq_len:]
            logits, _ = self.forward(idx)
            next_tok = torch.argmax(logits[:, :, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=-1)
        if squeeze_back:
            return out.squeeze(0)
        return out


def count_parameters(model: nn.Module) -> int:
    """Count unique parameters (respects embedding/output weight tying)."""
    seen = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total
