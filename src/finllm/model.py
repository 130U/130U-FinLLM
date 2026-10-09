"""Small dense decoder, adapted from MiniMind's Apache-2.0 implementation.

Reference: jingyaogong/minimind @ f659b55761b754d306bd140573493a6543cafd7f,
model/model_minimind.py. Modified for 130U-FinLLM in 2026: native PyTorch
configuration, fixed dense architecture, explicit padding-safe causal attention,
no Transformers, generation cache, MoE, hidden downloads, or internal loss shift.
See THIRD_PARTY_NOTICES.md and LICENSE.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 1664
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    dropout: float = 0.0
    qk_norm: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        for name in (
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "intermediate_size", "max_position_embeddings",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("rms_norm_eps and rope_theta must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulating the mean square in FP32 is essential under FP16 AMP.
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.repeat = self.heads // self.kv_heads
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, allowed=None, query_valid=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q = (q * cos + _rotate_half(q) * sin).to(q.dtype)
        k = (k * cos + _rotate_half(k) * sin).to(k.dtype)
        # Explicit replication works with both CPU and T4 SDPA backends.
        k = k.repeat_interleave(self.repeat, dim=1)
        v = v.repeat_interleave(self.repeat, dim=1)
        value = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=allowed is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        value = value.transpose(1, 2).contiguous().view(batch, length, -1)
        if query_valid is not None:
            value = value * query_valid.unsqueeze(-1)
        return self.residual_dropout(self.o_proj(value))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x, cos, sin, allowed=None, query_valid=None):
        x = x + self.attention(self.attention_norm(x), cos, sin, allowed, query_valid)
        return x + self.ffn(self.ffn_norm(x))


class FinLLM(nn.Module):
    """Return raw [batch, length, vocabulary] logits without shifting or loss.

    Training code owns the single next-token shift. ``attention_mask`` uses
    True/1 for real tokens and False/0 for padding. There is no automatic loading
    of pretrained weights: construction initializes every parameter randomly
    except the conventional unit RMSNorm scales.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        frequency = cfg.rope_theta ** (-torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim)
        angles = torch.outer(torch.arange(cfg.max_position_embeddings).float(), frequency)
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("rope_cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("rope_sin", angles.sin()[None, None], persistent=False)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, length] tensor")
        length = input_ids.shape[1]
        if length > self.config.max_position_embeddings:
            raise ValueError("input exceeds max_position_embeddings; truncate explicitly or reject it")
        allowed, valid = None, None
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask must contain only 0/1 or boolean values")
            valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            causal = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).tril()
            allowed = causal[None, None] & valid[:, None, None, :]
            # A padded query may have no permitted keys (left/all padding).
            # Permit its diagonal, then zero its output. This prevents NaNs on
            # SDPA backends while never exposing a padded key to a real query.
            diagonal = torch.eye(length, dtype=torch.bool, device=input_ids.device)
            allowed = allowed | ((~valid)[:, None, :, None] & diagonal[None, None])
        x = self.dropout(self.embed_tokens(input_ids))
        if valid is not None:
            x = x * valid.unsqueeze(-1)
        cos, sin = self.rope_cos[:, :, :length], self.rope_sin[:, :, :length]
        for layer in self.layers:
            x = layer(x, cos, sin, allowed, valid)
        return self.lm_head(self.norm(x))
