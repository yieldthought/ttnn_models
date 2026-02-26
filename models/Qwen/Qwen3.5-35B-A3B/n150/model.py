# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Qwen3.5-35B-A3B path for n150.

Execution policy:
- All attention (full + linear) runs on TTNN device.
- Embedding, RMSNorm, residual path, and lm_head run on TTNN device.
- Only sparse MoE expert execution runs on host torch.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import ttnn
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


TILE_SIZE = 32
PAGED_BLOCK_SIZE = 64
USE_DECODE_TRACE = os.environ.get("QWEN35_USE_DECODE_TRACE", "1") != "0"


def pad_to_tile(x: int) -> int:
    """Pad to tile boundary (32)."""
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def ensure_4d_hidden(x: ttnn.Tensor) -> ttnn.Tensor:
    """Normalize hidden tensors to [batch, 1, seq, hidden]."""
    if len(x.shape) == 4:
        return x
    if len(x.shape) == 3:
        return ttnn.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
    raise ValueError(f"Expected rank-3 or rank-4 hidden tensor, got shape {tuple(x.shape)}")


@dataclass
class ModelConfig:
    """Model configuration extracted from HuggingFace text config."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    attention_bias: bool
    hidden_act: str

    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int

    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int

    layer_types: list

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        text_config = getattr(hf_config, "text_config", hf_config)
        rope_parameters = dict(getattr(text_config, "rope_parameters", {}) or {})

        layer_types = list(getattr(text_config, "layer_types", []))
        if not layer_types:
            interval = getattr(text_config, "full_attention_interval", 4)
            layer_types = [
                "linear_attention" if (i + 1) % interval else "full_attention"
                for i in range(text_config.num_hidden_layers)
            ]

        return cls(
            text_config.vocab_size,
            text_config.hidden_size,
            text_config.num_hidden_layers,
            text_config.num_attention_heads,
            text_config.num_key_value_heads,
            text_config.head_dim,
            text_config.rms_norm_eps,
            rope_parameters.get("rope_theta", getattr(text_config, "rope_theta", 10000.0)),
            rope_parameters.get("partial_rotary_factor", getattr(text_config, "partial_rotary_factor", 1.0)),
            text_config.attention_bias,
            text_config.hidden_act,
            text_config.linear_conv_kernel_dim,
            text_config.linear_key_head_dim,
            text_config.linear_value_head_dim,
            text_config.linear_num_key_heads,
            text_config.linear_num_value_heads,
            text_config.num_experts,
            text_config.num_experts_per_tok,
            text_config.moe_intermediate_size,
            text_config.shared_expert_intermediate_size,
            layer_types,
        )


@dataclass
class PagedAttentionConfig:
    """Paged KV cache configuration."""

    block_size: int
    max_num_blocks: int


def compute_partial_rope_cache(config: ModelConfig, max_seq_len: int) -> tuple:
    """
    Precompute RoPE cache for the rotary sub-dimension.

    Returns cos/sin tensors of shape [1, 1, max_seq_len, rotary_dim].
    """
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    rotary_dim = (rotary_dim // 2) * 2
    if rotary_dim <= 0:
        raise ValueError("rotary_dim must be positive")

    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin = emb.sin().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    return cos, sin


def resolve_max_seq_len(hf_config, max_seq_len: Optional[int]) -> int:
    """Resolve max sequence length from HF text config when not provided."""
    text_config = getattr(hf_config, "text_config", hf_config)
    config_max = getattr(text_config, "max_position_embeddings", None)
    if config_max is None:
        config_max = getattr(text_config, "seq_length", None)
    if config_max is None:
        config_max = getattr(text_config, "max_seq_len", None)
    if max_seq_len is None:
        if config_max is None:
            raise ValueError("max_seq_len is required when config has no max_position_embeddings")
        return config_max
    if config_max is not None and max_seq_len > config_max:
        raise ValueError(f"max_seq_len {max_seq_len} exceeds config max {config_max}")
    return max_seq_len


class TTRMSNorm:
    """TT RMSNorm layer."""

    def __init__(self, weight: torch.Tensor, eps: float, tt_device, add_unit_offset: bool = False):
        self.eps = eps
        norm_weight = weight.to(torch.bfloat16)
        if add_unit_offset:
            norm_weight = norm_weight + 1
        self.weight = ttnn.as_tensor(
            norm_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)


class FullAttentionTT:
    """Qwen3.5 full-attention block running on TTNN."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        base_prefix: str,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        tt_device,
        paged_attention_config: PagedAttentionConfig,
        page_table: ttnn.Tensor,
    ):
        self.tt_device = tt_device
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.q_hidden_dim = self.n_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.paged_attention_config = paged_attention_config
        self.page_table = page_table

        self.partial_rotary_factor = config.partial_rotary_factor
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.rotary_dim = (self.rotary_dim // 2) * 2
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim:
            raise ValueError(f"Invalid rotary_dim={self.rotary_dim} for head_dim={self.head_dim}")

        if config.attention_bias:
            raise ValueError("attention_bias=True is not supported in this bringup")

        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

        p = f"{base_prefix}layers.{layer_idx}.self_attn."
        self.q_proj = self._load_q_proj_weight(state_dict[f"{p}q_proj.weight"])
        self.k_proj = self._load_weight(state_dict[f"{p}k_proj.weight"])
        self.v_proj = self._load_weight(state_dict[f"{p}v_proj.weight"])
        self.o_proj = self._load_weight(state_dict[f"{p}o_proj.weight"])
        self.q_norm = TTRMSNorm(state_dict[f"{p}q_norm.weight"], config.rms_norm_eps, tt_device, add_unit_offset=True)
        self.k_norm = TTRMSNorm(state_dict[f"{p}k_norm.weight"], config.rms_norm_eps, tt_device, add_unit_offset=True)

        cache_shape = (
            self.paged_attention_config.max_num_blocks,
            self.n_kv_heads,
            self.paged_attention_config.block_size,
            self.head_dim,
        )
        self.k_cache = ttnn.as_tensor(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_cache = ttnn.as_tensor(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_weight(self, w: torch.Tensor) -> ttnn.Tensor:
        """Load transposed weight for ttnn.linear: [out, in] -> [1, 1, in, out]."""
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_q_proj_weight(self, w: torch.Tensor) -> ttnn.Tensor:
        """
        Reorder Q projection rows from per-head [q|gate] layout into [all_q | all_gate].

        HF packs q_proj as:
        - head0: q(0:head_dim), gate(head_dim:2*head_dim)
        - head1: q, gate
        ...
        This reorder keeps the existing contiguous slice split in __call__ correct.
        """
        w_heads = w.reshape(self.n_heads, 2 * self.head_dim, self.hidden_size)
        q_rows = w_heads[:, : self.head_dim, :].reshape(self.q_hidden_dim, self.hidden_size)
        gate_rows = w_heads[:, self.head_dim :, :].reshape(self.q_hidden_dim, self.hidden_size)
        w_reordered = torch.cat([q_rows, gate_rows], dim=0).contiguous()
        return self._load_weight(w_reordered)

    def _apply_partial_rope_prefill(self, q: ttnn.Tensor, k: ttnn.Tensor, padded_seq: int) -> tuple:
        cos = self.cos_cache[:, :, :padded_seq, :]
        sin = self.sin_cache[:, :, :padded_seq, :]

        if self.rotary_dim == self.head_dim:
            q = ttnn.experimental.rotary_embedding(q, cos, sin)
            k = ttnn.experimental.rotary_embedding(k, cos, sin)
            return q, k

        q_rot = ttnn.slice(q, (0, 0, 0, 0), (q.shape[0], q.shape[1], q.shape[2], self.rotary_dim))
        q_pass = ttnn.slice(q, (0, 0, 0, self.rotary_dim), (q.shape[0], q.shape[1], q.shape[2], self.head_dim))
        k_rot = ttnn.slice(k, (0, 0, 0, 0), (k.shape[0], k.shape[1], k.shape[2], self.rotary_dim))
        k_pass = ttnn.slice(k, (0, 0, 0, self.rotary_dim), (k.shape[0], k.shape[1], k.shape[2], self.head_dim))

        q_rot = ttnn.experimental.rotary_embedding(q_rot, cos, sin)
        k_rot = ttnn.experimental.rotary_embedding(k_rot, cos, sin)
        q = ttnn.concat([q_rot, q_pass], dim=-1)
        k = ttnn.concat([k_rot, k_pass], dim=-1)
        return q, k

    def _apply_partial_rope_decode(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        start_pos: int,
        decode_cos_q: Optional[ttnn.Tensor] = None,
        decode_sin_q: Optional[ttnn.Tensor] = None,
        decode_cos_k: Optional[ttnn.Tensor] = None,
        decode_sin_k: Optional[ttnn.Tensor] = None,
    ) -> tuple:
        q_cos = decode_cos_q if decode_cos_q is not None else self.cos_cache
        q_sin = decode_sin_q if decode_sin_q is not None else self.sin_cache
        k_cos = decode_cos_k if decode_cos_k is not None else self.cos_cache
        k_sin = decode_sin_k if decode_sin_k is not None else self.sin_cache

        if self.rotary_dim == self.head_dim:
            if decode_cos_q is None or decode_sin_q is None:
                q = ttnn.experimental.rotary_embedding(q, q_cos, q_sin, start_pos)
            else:
                q = ttnn.experimental.rotary_embedding(q, q_cos, q_sin)
            if decode_cos_k is None or decode_sin_k is None:
                k = ttnn.experimental.rotary_embedding(k, k_cos, k_sin, start_pos)
            else:
                k = ttnn.experimental.rotary_embedding(k, k_cos, k_sin)
            return q, k

        q_rot = ttnn.slice(q, (0, 0, 0, 0), (q.shape[0], q.shape[1], q.shape[2], self.rotary_dim))
        q_pass = ttnn.slice(q, (0, 0, 0, self.rotary_dim), (q.shape[0], q.shape[1], q.shape[2], self.head_dim))
        k_rot = ttnn.slice(k, (0, 0, 0, 0), (k.shape[0], k.shape[1], k.shape[2], self.rotary_dim))
        k_pass = ttnn.slice(k, (0, 0, 0, self.rotary_dim), (k.shape[0], k.shape[1], k.shape[2], self.head_dim))

        if decode_cos_q is None or decode_sin_q is None:
            q_rot = ttnn.experimental.rotary_embedding(q_rot, q_cos, q_sin, start_pos)
        else:
            q_rot = ttnn.experimental.rotary_embedding(q_rot, q_cos, q_sin)
        if decode_cos_k is None or decode_sin_k is None:
            k_rot = ttnn.experimental.rotary_embedding(k_rot, k_cos, k_sin, start_pos)
        else:
            k_rot = ttnn.experimental.rotary_embedding(k_rot, k_cos, k_sin)
        q = ttnn.concat([q_rot, q_pass], dim=-1)
        k = ttnn.concat([k_rot, k_pass], dim=-1)
        return q, k

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_cos_q: Optional[ttnn.Tensor] = None,
        decode_sin_q: Optional[ttnn.Tensor] = None,
        decode_cos_k: Optional[ttnn.Tensor] = None,
        decode_sin_k: Optional[ttnn.Tensor] = None,
        trace_decode: bool = False,
    ) -> ttnn.Tensor:
        """Forward pass for prefill (seq_len > 1) or decode (seq_len == 1)."""
        x = ensure_4d_hidden(x)
        is_prefill = seq_len > 1
        padded_seq = x.shape[2]

        q_and_gate = ttnn.linear(x, self.q_proj)
        q = ttnn.slice(
            q_and_gate,
            (0, 0, 0, 0),
            (q_and_gate.shape[0], q_and_gate.shape[1], q_and_gate.shape[2], self.q_hidden_dim),
        )
        gate = ttnn.slice(
            q_and_gate,
            (0, 0, 0, self.q_hidden_dim),
            (q_and_gate.shape[0], q_and_gate.shape[1], q_and_gate.shape[2], 2 * self.q_hidden_dim),
        )
        ttnn.deallocate(q_and_gate)

        k = ttnn.linear(x, self.k_proj)
        v = ttnn.linear(x, self.v_proj)
        qkv = ttnn.concat([q, k, v], dim=-1)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        if is_prefill:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=self.n_heads,
                num_kv_heads=self.n_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)

            q_mem = ttnn.get_memory_config(q)
            k_mem = ttnn.get_memory_config(k)
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = ttnn.to_memory_config(q, q_mem)
            k = ttnn.to_memory_config(k, k_mem)

            q, k = self._apply_partial_rope_prefill(q, k, padded_seq)

            ttnn.experimental.paged_fill_cache(self.k_cache, k, self.page_table, batch_idx=0)
            ttnn.experimental.paged_fill_cache(self.v_cache, v, self.page_table, batch_idx=0)

            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.scale,
            )
            attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            if cur_pos_tensor is None:
                raise ValueError("cur_pos_tensor is required for decode")

            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv,
                num_heads=self.n_heads,
                num_kv_heads=self.n_kv_heads,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if not trace_decode:
                ttnn.deallocate(qkv)

            q_mem = ttnn.get_memory_config(q)
            k_mem = ttnn.get_memory_config(k)
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
            q = self.q_norm(q)
            k = self.k_norm(k)

            q = ttnn.reshape(q, (1, 1, q.shape[1] * self.n_heads, self.head_dim))
            k = ttnn.reshape(k, (1, 1, k.shape[1] * self.n_kv_heads, self.head_dim))

            q, k = self._apply_partial_rope_decode(
                q,
                k,
                start_pos,
                decode_cos_q,
                decode_sin_q,
                decode_cos_k,
                decode_sin_k,
            )

            q = ttnn.reshape(q, (1, q.shape[2] // self.n_heads, self.n_heads, self.head_dim))
            k = ttnn.reshape(k, (1, k.shape[2] // self.n_kv_heads, self.n_kv_heads, self.head_dim))

            q = ttnn.to_memory_config(q, q_mem)
            k = ttnn.to_memory_config(k, k_mem)

            ttnn.experimental.paged_update_cache(
                self.k_cache,
                k,
                update_idxs_tensor=cur_pos_tensor,
                page_table=self.page_table,
            )
            ttnn.experimental.paged_update_cache(
                self.v_cache,
                v,
                update_idxs_tensor=cur_pos_tensor,
                page_table=self.page_table,
            )

            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                self.k_cache,
                self.v_cache,
                page_table_tensor=self.page_table,
                cur_pos_tensor=cur_pos_tensor,
                scale=self.scale,
            )
            attn_out = ttnn.transpose(attn_out, 1, 2)
            attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if attn_out.shape[-1] != self.q_hidden_dim:
            attn_out = ttnn.slice(
                attn_out,
                (0, 0, 0, 0),
                (attn_out.shape[0], attn_out.shape[1], attn_out.shape[2], self.q_hidden_dim),
            )

        gate = ttnn.sigmoid(gate)
        attn_out = ttnn.mul(attn_out, gate)
        ttnn.deallocate(gate)

        out = ttnn.linear(attn_out, self.o_proj)
        return ensure_4d_hidden(out)


class LinearAttentionTT:
    """Qwen3.5 linear-attention block running on TTNN."""

    def __init__(self, config: ModelConfig, layer_idx: int, state_dict: dict, base_prefix: str, tt_device):
        self.tt_device = tt_device
        self.hidden_size = config.hidden_size

        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_eps = config.rms_norm_eps

        self.kv_repeat = self.num_v_heads // self.num_k_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")

        self.query_scale = 1.0 / math.sqrt(self.head_k_dim)
        self.inv_head_v_dim = 1.0 / self.head_v_dim
        self.l2norm_eps = 1e-6

        p = f"{base_prefix}layers.{layer_idx}.linear_attn."
        self.in_proj_qkv = self._load_weight(state_dict[f"{p}in_proj_qkv.weight"])
        self.in_proj_z = self._load_weight(state_dict[f"{p}in_proj_z.weight"])
        self.in_proj_b = self._load_weight(state_dict[f"{p}in_proj_b.weight"])
        self.in_proj_a = self._load_weight(state_dict[f"{p}in_proj_a.weight"])
        self.out_proj = self._load_weight(state_dict[f"{p}out_proj.weight"])

        conv_weight = state_dict[f"{p}conv1d.weight"].squeeze(1).transpose(0, 1).contiguous()
        self.conv1d_weight = ttnn.as_tensor(
            conv_weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.dt_bias = ttnn.as_tensor(
            state_dict[f"{p}dt_bias"].reshape(1, 1, 1, self.num_v_heads).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.neg_exp_a = ttnn.as_tensor(
            (-state_dict[f"{p}A_log"].float().exp()).reshape(1, 1, 1, self.num_v_heads).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.norm_weight = ttnn.as_tensor(
            state_dict[f"{p}norm.weight"].reshape(1, 1, 1, self.head_v_dim).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.conv_state = None
        self.recurrent_state = None
        self.reset()

    def _load_weight(self, w: torch.Tensor) -> ttnn.Tensor:
        """Load transposed weight for ttnn.linear: [out, in] -> [1, 1, in, out]."""
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def reset(self) -> None:
        if self.conv_state is not None:
            ttnn.deallocate(self.conv_state)
        if self.recurrent_state is not None:
            ttnn.deallocate(self.recurrent_state)

        self.conv_state = ttnn.from_torch(
            torch.zeros((1, 1, self.conv_kernel_size, self.conv_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.recurrent_state = ttnn.from_torch(
            torch.zeros((1, self.num_v_heads, self.head_k_dim, self.head_v_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _l2norm(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_sq = ttnn.mul(x, x)
        x_sq_sum = ttnn.sum(x_sq, dim=3, keepdim=True)
        ttnn.deallocate(x_sq)
        x_sq_sum = ttnn.add(x_sq_sum, self.l2norm_eps)
        inv = ttnn.rsqrt(x_sq_sum)
        ttnn.deallocate(x_sq_sum)
        out = ttnn.mul(x, inv)
        ttnn.deallocate(inv)
        return out

    def _apply_rms_norm_gated(self, core: ttnn.Tensor, gate: ttnn.Tensor) -> ttnn.Tensor:
        core_sq = ttnn.mul(core, core)
        variance = ttnn.sum(core_sq, dim=3, keepdim=True)
        ttnn.deallocate(core_sq)

        variance = ttnn.mul(variance, self.inv_head_v_dim)
        variance = ttnn.add(variance, self.layer_norm_eps)
        inv_std = ttnn.rsqrt(variance)
        ttnn.deallocate(variance)

        core = ttnn.mul(core, inv_std)
        ttnn.deallocate(inv_std)
        core = ttnn.mul(core, self.norm_weight)

        gate = ttnn.silu(gate)
        out = ttnn.mul(core, gate)
        ttnn.deallocate(core)
        ttnn.deallocate(gate)
        return out

    def _conv_step(self, mixed_qkv_input: ttnn.Tensor) -> ttnn.Tensor:
        conv_cat = ttnn.concat([self.conv_state, mixed_qkv_input], dim=2)
        start = conv_cat.shape[2] - self.conv_kernel_size
        new_state = ttnn.slice(
            conv_cat,
            (0, 0, start, 0),
            (conv_cat.shape[0], conv_cat.shape[1], conv_cat.shape[2], conv_cat.shape[3]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(conv_cat)
        ttnn.deallocate(self.conv_state)
        self.conv_state = new_state

        conv_mul = ttnn.mul(self.conv_state, self.conv1d_weight)
        conv_out = ttnn.sum(conv_mul, dim=2, keepdim=True)
        ttnn.deallocate(conv_mul)
        conv_out = ttnn.silu(conv_out)
        return conv_out

    def _token_forward(self, x_t: ttnn.Tensor) -> ttnn.Tensor:
        mixed_qkv_input = ttnn.linear(x_t, self.in_proj_qkv)
        mixed_qkv = self._conv_step(mixed_qkv_input)
        ttnn.deallocate(mixed_qkv_input)

        z = ttnn.linear(x_t, self.in_proj_z)
        b = ttnn.linear(x_t, self.in_proj_b)
        a = ttnn.linear(x_t, self.in_proj_a)

        query = ttnn.slice(mixed_qkv, (0, 0, 0, 0), (1, 1, 1, self.key_dim))
        key = ttnn.slice(mixed_qkv, (0, 0, 0, self.key_dim), (1, 1, 1, 2 * self.key_dim))
        value = ttnn.slice(mixed_qkv, (0, 0, 0, 2 * self.key_dim), (1, 1, 1, self.conv_dim))
        ttnn.deallocate(mixed_qkv)

        query = ttnn.reshape(query, (1, 1, self.num_k_heads, self.head_k_dim))
        key = ttnn.reshape(key, (1, 1, self.num_k_heads, self.head_k_dim))
        value = ttnn.reshape(value, (1, 1, self.num_v_heads, self.head_v_dim))

        beta = ttnn.sigmoid(b)
        ttnn.deallocate(b)
        a = ttnn.add(a, self.dt_bias)
        g = ttnn.softplus(a)
        ttnn.deallocate(a)
        g = ttnn.mul(g, self.neg_exp_a)

        if self.kv_repeat > 1:
            query = ttnn.repeat_interleave(query, repeats=self.kv_repeat, dim=2)
            key = ttnn.repeat_interleave(key, repeats=self.kv_repeat, dim=2)

        query = self._l2norm(query)
        key = self._l2norm(key)
        query = ttnn.mul(query, self.query_scale)

        g_state = ttnn.exp(g)
        ttnn.deallocate(g)
        g_state = ttnn.reshape(g_state, (1, self.num_v_heads, 1, 1))
        beta_state = ttnn.reshape(beta, (1, self.num_v_heads, 1, 1))
        ttnn.deallocate(beta)

        q_state = ttnn.reshape(query, (1, self.num_v_heads, self.head_k_dim, 1))
        k_state = ttnn.reshape(key, (1, self.num_v_heads, self.head_k_dim, 1))
        v_state = ttnn.reshape(value, (1, self.num_v_heads, 1, self.head_v_dim))
        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        rec_scaled = ttnn.mul(self.recurrent_state, g_state)
        ttnn.deallocate(g_state)

        kv_mem = ttnn.mul(rec_scaled, k_state)
        kv_mem = ttnn.sum(kv_mem, dim=2, keepdim=True)

        delta = ttnn.sub(v_state, kv_mem)
        ttnn.deallocate(v_state)
        ttnn.deallocate(kv_mem)
        delta = ttnn.mul(delta, beta_state)
        ttnn.deallocate(beta_state)

        rec_update = ttnn.mul(k_state, delta)
        ttnn.deallocate(k_state)
        ttnn.deallocate(delta)

        rec_next = ttnn.add(rec_scaled, rec_update)
        ttnn.deallocate(rec_scaled)
        ttnn.deallocate(rec_update)
        ttnn.deallocate(self.recurrent_state)
        self.recurrent_state = rec_next

        core = ttnn.mul(self.recurrent_state, q_state)
        ttnn.deallocate(q_state)
        core = ttnn.sum(core, dim=2, keepdim=True)

        core = ttnn.reshape(core, (1, 1, self.num_v_heads, self.head_v_dim))
        z = ttnn.reshape(z, (1, 1, self.num_v_heads, self.head_v_dim))
        core = self._apply_rms_norm_gated(core, z)
        core = ttnn.reshape(core, (1, 1, 1, self.value_dim))

        out = ttnn.linear(core, self.out_proj)
        ttnn.deallocate(core)
        return ensure_4d_hidden(out)

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_cos_q: Optional[ttnn.Tensor] = None,
        decode_sin_q: Optional[ttnn.Tensor] = None,
        decode_cos_k: Optional[ttnn.Tensor] = None,
        decode_sin_k: Optional[ttnn.Tensor] = None,
        trace_decode: bool = False,
    ) -> ttnn.Tensor:
        del start_pos
        del cur_pos_tensor
        del decode_cos_q
        del decode_sin_q
        del decode_cos_k
        del decode_sin_k
        del trace_decode

        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        x = ensure_4d_hidden(x)
        padded_seq = x.shape[2]
        outputs = []

        for token_idx in range(seq_len):
            x_t = ttnn.slice(
                x,
                (0, 0, token_idx, 0),
                (x.shape[0], x.shape[1], token_idx + 1, x.shape[3]),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out_t = self._token_forward(x_t)
            ttnn.deallocate(x_t)
            outputs.append(out_t)

        if len(outputs) == 1:
            out_real = outputs[0]
        else:
            out_real = ttnn.concat(outputs, dim=2)
            for out in outputs:
                ttnn.deallocate(out)

        if seq_len < padded_seq:
            out_pad = ttnn.from_torch(
                torch.zeros((1, 1, padded_seq - seq_len, self.hidden_size), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.tt_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = ttnn.concat([out_real, out_pad], dim=2)
            ttnn.deallocate(out_real)
            ttnn.deallocate(out_pad)
            return out

        return out_real


class SparseMoEHost:
    """Qwen3.5 sparse MoE with TTNN router/shared path and host-only expert execution."""

    def __init__(self, config: ModelConfig, layer_idx: int, state_dict: dict, base_prefix: str, tt_device):
        self.tt_device = tt_device
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        decode_top_k = int(os.environ.get("QWEN35_DECODE_TOP_K", "6"))
        self.decode_top_k = max(1, min(self.top_k, decode_top_k))

        p = f"{base_prefix}layers.{layer_idx}.mlp."
        self.router_weight = self._load_weight(state_dict[f"{p}gate.weight"])

        self.expert_gate_up = state_dict[f"{p}experts.gate_up_proj"].to(torch.bfloat16)
        self.expert_down = state_dict[f"{p}experts.down_proj"].to(torch.bfloat16)

        self.shared_gate_proj = self._load_weight(state_dict[f"{p}shared_expert.gate_proj.weight"])
        self.shared_up_proj = self._load_weight(state_dict[f"{p}shared_expert.up_proj.weight"])
        self.shared_down_proj = self._load_weight(state_dict[f"{p}shared_expert.down_proj.weight"])
        self.shared_expert_gate = self._load_weight(state_dict[f"{p}shared_expert_gate.weight"])
        self.router_softmax_config = ttnn.init_device_compute_kernel_config(
            tt_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _load_weight(self, w: torch.Tensor) -> ttnn.Tensor:
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _router_on_device(self, hidden_states: ttnn.Tensor, route_top_k: int) -> tuple:
        router_logits = ttnn.linear(hidden_states, self.router_weight)
        routing_logits, selected_experts = ttnn.topk(router_logits, k=route_top_k, dim=3, sorted=True)
        ttnn.deallocate(router_logits)
        routing_weights = ttnn.softmax(
            routing_logits,
            dim=3,
            numeric_stable=True,
            compute_kernel_config=self.router_softmax_config,
        )
        ttnn.deallocate(routing_logits)
        return routing_weights, selected_experts

    def _shared_on_device(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        shared_gate = ttnn.linear(hidden_states, self.shared_gate_proj)
        shared_gate = ttnn.silu(shared_gate)

        shared_up = ttnn.linear(hidden_states, self.shared_up_proj)
        shared = ttnn.mul(shared_gate, shared_up)
        ttnn.deallocate(shared_gate)
        ttnn.deallocate(shared_up)

        shared = ttnn.linear(shared, self.shared_down_proj)
        shared_expert_gate = ttnn.linear(hidden_states, self.shared_expert_gate)
        shared_expert_gate = ttnn.sigmoid(shared_expert_gate)
        shared = ttnn.mul(shared, shared_expert_gate)
        ttnn.deallocate(shared_expert_gate)
        return shared

    def __call__(self, x: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        x = ensure_4d_hidden(x)
        padded_seq = x.shape[2]
        if seq_len < padded_seq:
            hidden_states_tt = ttnn.slice(
                x,
                (0, 0, 0, 0),
                (x.shape[0], x.shape[1], seq_len, x.shape[3]),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            hidden_states_tt = x

        route_top_k = self.decode_top_k if seq_len == 1 else self.top_k
        routing_weights_tt, selected_experts_tt = self._router_on_device(hidden_states_tt, route_top_k)
        shared_tt = self._shared_on_device(hidden_states_tt)

        hidden_states = ttnn.to_torch(hidden_states_tt).reshape(seq_len, self.hidden_size).to(torch.bfloat16)
        routing_weights = ttnn.to_torch(routing_weights_tt).reshape(seq_len, route_top_k).to(torch.bfloat16)
        selected_experts = ttnn.to_torch(selected_experts_tt).reshape(seq_len, route_top_k).to(torch.long)

        ttnn.deallocate(routing_weights_tt)
        ttnn.deallocate(selected_experts_tt)
        if hidden_states_tt is not x:
            ttnn.deallocate(hidden_states_tt)

        expert_hidden_states = torch.zeros((seq_len, self.hidden_size), dtype=torch.bfloat16)

        expert_hitted = torch.unique(selected_experts)
        for expert_idx_tensor in expert_hitted:
            expert_idx = int(expert_idx_tensor.item())
            token_idx, route_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue

            current_state = hidden_states.index_select(0, token_idx).contiguous()
            gate, up = F.linear(current_state, self.expert_gate_up[expert_idx]).chunk(2, dim=-1)
            current_hidden = F.silu(gate) * up
            current_hidden = F.linear(current_hidden, self.expert_down[expert_idx])
            current_hidden = current_hidden * routing_weights[token_idx, route_idx].unsqueeze(-1).to(current_hidden.dtype)
            expert_hidden_states.index_add_(0, token_idx, current_hidden)

        expert_hidden_states = expert_hidden_states.reshape(1, 1, seq_len, self.hidden_size)
        expert_hidden_states_tt = ttnn.from_torch(
            expert_hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_real = ttnn.add(shared_tt, expert_hidden_states_tt)
        ttnn.deallocate(shared_tt)
        ttnn.deallocate(expert_hidden_states_tt)

        if seq_len < padded_seq:
            out_pad = ttnn.from_torch(
                torch.zeros((1, 1, padded_seq - seq_len, self.hidden_size), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.tt_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = ttnn.concat([out_real, out_pad], dim=2)
            ttnn.deallocate(out_real)
            ttnn.deallocate(out_pad)
            return out

        return out_real


class DecoderLayer:
    """Single Qwen3.5 decoder layer."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        base_prefix: str,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        tt_device,
        paged_attention_config: PagedAttentionConfig,
        page_table: ttnn.Tensor,
    ):
        self.layer_type = config.layer_types[layer_idx]

        p = f"{base_prefix}layers.{layer_idx}."
        self.input_norm = TTRMSNorm(state_dict[f"{p}input_layernorm.weight"], config.rms_norm_eps, tt_device, add_unit_offset=True)
        self.post_norm = TTRMSNorm(
            state_dict[f"{p}post_attention_layernorm.weight"],
            config.rms_norm_eps,
            tt_device,
            add_unit_offset=True,
        )

        if self.layer_type == "full_attention":
            self.attn = FullAttentionTT(
                config,
                layer_idx,
                state_dict,
                base_prefix,
                cos_cache,
                sin_cache,
                tt_device,
                paged_attention_config,
                page_table,
            )
        elif self.layer_type == "linear_attention":
            self.attn = LinearAttentionTT(config, layer_idx, state_dict, base_prefix, tt_device)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

        self.mlp = SparseMoEHost(config, layer_idx, state_dict, base_prefix, tt_device)

    def reset_cache(self) -> None:
        if hasattr(self.attn, "reset"):
            self.attn.reset()

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_cos_q: Optional[ttnn.Tensor] = None,
        decode_sin_q: Optional[ttnn.Tensor] = None,
        decode_cos_k: Optional[ttnn.Tensor] = None,
        decode_sin_k: Optional[ttnn.Tensor] = None,
        trace_decode: bool = False,
    ) -> ttnn.Tensor:
        x = ensure_4d_hidden(x)
        h = self.input_norm(x)
        h = ensure_4d_hidden(h)
        attn_out = self.attn(
            h,
            start_pos,
            seq_len,
            cur_pos_tensor,
            decode_cos_q,
            decode_sin_q,
            decode_cos_k,
            decode_sin_k,
            trace_decode,
        )
        attn_out = ensure_4d_hidden(attn_out)
        ttnn.deallocate(h)

        x = ttnn.add(x, attn_out)
        ttnn.deallocate(attn_out)

        h = self.post_norm(x)
        h = ensure_4d_hidden(h)
        mlp_out = self.mlp(h, seq_len)
        mlp_out = ensure_4d_hidden(mlp_out)
        ttnn.deallocate(h)

        x = ttnn.add(x, mlp_out)
        ttnn.deallocate(mlp_out)
        return x


class TtnnQwen35MoeForCausalLM(torch.nn.Module, GenerationMixin):
    """TTNN Qwen3.5-35B-A3B model with expert-only host fallback."""

    def __init__(self, hf_model, tt_device, max_seq_len: Optional[int] = None):
        super().__init__()

        self.tt_device = tt_device
        self.hf_config = getattr(hf_model, "config", None)
        self.tt_config = ModelConfig.from_hf(self.hf_config)
        self.max_seq_len = resolve_max_seq_len(self.hf_config, max_seq_len)
        self._pos = 0

        if self.tt_config.hidden_act != "silu":
            raise ValueError(f"hidden_act {self.tt_config.hidden_act} is not supported in this bringup")
        if self.tt_config.num_experts_per_tok > self.tt_config.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")

        text_config = getattr(self.hf_config, "text_config", self.hf_config)
        self.config = text_config
        self.generation_config = GenerationConfig.from_model_config(self.config)
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = self.generation_config.eos_token_id
        self._supports_cache_class = False
        self.main_input_name = "input_ids"
        self.register_buffer("_torch_dummy", torch.empty(0, dtype=torch.float32), persistent=False)

        param_dtype = next(hf_model.parameters()).dtype
        if param_dtype != torch.bfloat16:
            print("  Converting HuggingFace weights to bfloat16 for TT bringup memory headroom...")
            hf_model.to(torch.bfloat16)

        state_dict = hf_model.state_dict()

        if "model.language_model.embed_tokens.weight" in state_dict:
            self.base_prefix = "model.language_model."
        elif "model.embed_tokens.weight" in state_dict:
            self.base_prefix = "model."
        elif "embed_tokens.weight" in state_dict:
            self.base_prefix = ""
        else:
            raise ValueError("Failed to detect text-model weight prefix in state_dict")

        print("  Loading embedding + final norm + lm_head...")
        self.embed = ttnn.as_tensor(
            state_dict[f"{self.base_prefix}embed_tokens.weight"].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.norm = TTRMSNorm(state_dict[f"{self.base_prefix}norm.weight"], self.tt_config.rms_norm_eps, tt_device, add_unit_offset=True)

        lm_head_weight = state_dict.get("lm_head.weight")
        if lm_head_weight is None and f"{self.base_prefix}lm_head.weight" in state_dict:
            lm_head_weight = state_dict[f"{self.base_prefix}lm_head.weight"]
        if lm_head_weight is None:
            lm_head_weight = state_dict[f"{self.base_prefix}embed_tokens.weight"]
        self.lm_head = ttnn.as_tensor(
            lm_head_weight.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print("  Computing RoPE cache...")
        cos, sin = compute_partial_rope_cache(self.tt_config, self.max_seq_len)
        self.cos_cache_host = cos
        self.sin_cache_host = sin
        self.decode_rope_dim = int(cos.shape[-1])
        self.cos_cache = ttnn.as_tensor(
            cos,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_cache = ttnn.as_tensor(
            sin,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        max_num_blocks = math.ceil(self.max_seq_len / PAGED_BLOCK_SIZE)
        self.paged_attention_config = PagedAttentionConfig(PAGED_BLOCK_SIZE, max_num_blocks)
        page_table = torch.arange(max_num_blocks, dtype=torch.int32).repeat(TILE_SIZE, 1)
        self.page_table = ttnn.as_tensor(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.decode_token_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 1, TILE_SIZE), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.decode_pos_buffer = ttnn.from_torch(
            torch.zeros((TILE_SIZE,), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.decode_q_rope_seq = self.tt_config.num_attention_heads * TILE_SIZE
        self.decode_k_rope_seq = self.tt_config.num_key_value_heads * TILE_SIZE
        self.decode_cos_q_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.decode_q_rope_seq, self.decode_rope_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.decode_sin_q_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.decode_q_rope_seq, self.decode_rope_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.decode_cos_k_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.decode_k_rope_seq, self.decode_rope_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.decode_sin_k_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.decode_k_rope_seq, self.decode_rope_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.argmax_output_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 1, 1), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.decode_hidden_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 1, self.tt_config.hidden_size), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.use_decode_trace = USE_DECODE_TRACE
        self.decode_trace_id = None
        self.decode_trace_logits = None
        self._decode_trace_capture_logged = False
        self._decode_trace_execute_logged = False

        print(f"  Loading {self.tt_config.num_hidden_layers} layers...")
        self.layers = [
            DecoderLayer(
                self.tt_config,
                i,
                state_dict,
                self.base_prefix,
                self.cos_cache,
                self.sin_cache,
                tt_device,
                self.paged_attention_config,
                self.page_table,
            )
            for i in range(self.tt_config.num_hidden_layers)
        ]

        self._tt_past_key_values = object()

    @property
    def device(self) -> torch.device:
        return self._torch_dummy.device

    def _release_decode_trace(self) -> None:
        if self.decode_trace_id is None:
            return
        ttnn.release_trace(self.tt_device, self.decode_trace_id)
        self.decode_trace_id = None
        self.decode_trace_logits = None
        self._decode_trace_capture_logged = False
        self._decode_trace_execute_logged = False

    def reset(self):
        """Reset position counter and per-layer linear-attention caches."""
        self._pos = 0
        self._release_decode_trace()
        for layer in self.layers:
            layer.reset_cache()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor],
        decode_cos_q: Optional[ttnn.Tensor] = None,
        decode_sin_q: Optional[ttnn.Tensor] = None,
        decode_cos_k: Optional[ttnn.Tensor] = None,
        decode_sin_k: Optional[ttnn.Tensor] = None,
        trace_decode: bool = False,
    ) -> ttnn.Tensor:
        tokens = ttnn.from_torch(
            input_ids.reshape(1, 1, 1, -1),
            dtype=ttnn.uint32,
            device=self.tt_device,
        )
        h = ttnn.embedding(tokens, self.embed, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(tokens)
        h = ensure_4d_hidden(h)

        for layer in self.layers:
            h = layer(
                h,
                start_pos,
                seq_len,
                cur_pos_tensor,
                decode_cos_q,
                decode_sin_q,
                decode_cos_k,
                decode_sin_k,
                trace_decode,
            )
            h = ensure_4d_hidden(h)

        h = self.norm(h)
        return h

    def _update_decode_token_buffer(self, input_ids: torch.Tensor) -> None:
        token_ids = torch.zeros((TILE_SIZE,), dtype=torch.int32)
        token_ids[: input_ids.numel()] = input_ids.view(-1).to(torch.int32)
        token_ids = token_ids.reshape(1, 1, 1, -1)
        host_tokens = ttnn.from_torch(
            token_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_tokens, self.decode_token_buffer)

    def _update_decode_pos_buffer(self, start_pos: int) -> None:
        pos = torch.full((TILE_SIZE,), -1, dtype=torch.int32)
        pos[0] = start_pos
        host_pos = ttnn.from_torch(
            pos,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_pos, self.decode_pos_buffer)

    def _update_decode_rope_buffers(self, start_pos: int) -> None:
        cos_token = self.cos_cache_host[:, :, start_pos : start_pos + 1, :]
        sin_token = self.sin_cache_host[:, :, start_pos : start_pos + 1, :]
        cos_q = cos_token.repeat(1, 1, self.decode_q_rope_seq, 1)
        sin_q = sin_token.repeat(1, 1, self.decode_q_rope_seq, 1)
        cos_k = cos_token.repeat(1, 1, self.decode_k_rope_seq, 1)
        sin_k = sin_token.repeat(1, 1, self.decode_k_rope_seq, 1)

        host_cos_q = ttnn.from_torch(
            cos_q,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        host_sin_q = ttnn.from_torch(
            sin_q,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        host_cos_k = ttnn.from_torch(
            cos_k,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        host_sin_k = ttnn.from_torch(
            sin_k,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_cos_q, self.decode_cos_q_buffer)
        ttnn.copy_host_to_device_tensor(host_sin_q, self.decode_sin_q_buffer)
        ttnn.copy_host_to_device_tensor(host_cos_k, self.decode_cos_k_buffer)
        ttnn.copy_host_to_device_tensor(host_sin_k, self.decode_sin_k_buffer)

    def _forward_decode_device(self, start_pos: int, trace_decode: bool) -> ttnn.Tensor:
        h = ttnn.embedding(self.decode_token_buffer, self.embed, layout=ttnn.TILE_LAYOUT)
        h = ensure_4d_hidden(h)
        for layer in self.layers:
            h = layer(
                h,
                start_pos,
                1,
                self.decode_pos_buffer,
                self.decode_cos_q_buffer,
                self.decode_sin_q_buffer,
                self.decode_cos_k_buffer,
                self.decode_sin_k_buffer,
                trace_decode,
            )
            h = ensure_4d_hidden(h)

        h = self.norm(h)
        h_last = ttnn.slice(
            h,
            (0, 0, 0, 0),
            (h.shape[0], h.shape[1], 1, h.shape[-1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)
        return h_last

    def _forward_decode(self, input_ids: torch.Tensor, start_pos: int) -> ttnn.Tensor:
        self._update_decode_token_buffer(input_ids)
        self._update_decode_pos_buffer(start_pos)
        self._update_decode_rope_buffers(start_pos)
        h_last = self._forward_decode_device(start_pos, False)

        if self.use_decode_trace:
            ttnn.copy(h_last, self.decode_hidden_buffer)
            ttnn.deallocate(h_last)
            if self.decode_trace_id is None:
                warmup_logits = ttnn.linear(self.decode_hidden_buffer, self.lm_head)
                ttnn.deallocate(warmup_logits)
                self.decode_trace_id = ttnn.begin_trace_capture(self.tt_device, cq_id=0)
                self.decode_trace_logits = ttnn.linear(self.decode_hidden_buffer, self.lm_head)
                ttnn.end_trace_capture(self.tt_device, self.decode_trace_id, cq_id=0)
                if not self._decode_trace_capture_logged:
                    print("decode_trace: captured lm_head trace")
                    self._decode_trace_capture_logged = True
            else:
                ttnn.execute_trace(self.tt_device, self.decode_trace_id, cq_id=0, blocking=False)
                if not self._decode_trace_execute_logged:
                    print("decode_trace: executing captured lm_head trace")
                    self._decode_trace_execute_logged = True
            return self.decode_trace_logits

        logits = ttnn.linear(h_last, self.lm_head)
        ttnn.deallocate(h_last)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del attention_mask
        del cache_position
        del kwargs

        batch, seq_len = input_ids.shape
        if batch != 1:
            raise ValueError("Only batch=1 supported")

        if past_key_values is None:
            self.reset()
        elif seq_len != 1:
            raise ValueError("Only 1-token decode supported when using cache")

        start_pos = self._pos
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {start_pos + seq_len} exceeds max_seq_len {self.max_seq_len}")

        if seq_len == 1:
            logits_device = self._forward_decode(input_ids, start_pos)
            padded_seq = 1
        else:
            padded_seq = pad_to_tile(seq_len)
            if seq_len < padded_seq:
                input_ids = F.pad(input_ids, (0, padded_seq - seq_len), value=0)
            h = self._forward_hidden(input_ids, start_pos, seq_len, None)
            logits_device = ttnn.linear(h, self.lm_head)
            ttnn.deallocate(h)

        logits = ttnn.to_torch(logits_device).reshape(batch, padded_seq, -1)[:, :seq_len, :]
        if seq_len > 1 or not self.use_decode_trace:
            ttnn.deallocate(logits_device)

        self._pos = start_pos + seq_len
        return CausalLMOutputWithPast(
            logits=logits.float(),
            past_key_values=(self._tt_past_key_values if use_cache else None),
        )

    def next_token_device(self, input_ids: torch.Tensor, past_key_values=None, use_cache: bool = True) -> tuple[int, object]:
        batch, seq_len = input_ids.shape
        if batch != 1:
            raise ValueError("Only batch=1 supported")

        # Keep decode path simple and reuse the normal forward path.
        if past_key_values is not None or seq_len == 1:
            outputs = self.forward(input_ids, past_key_values=past_key_values, use_cache=use_cache)
            token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            return token, outputs.past_key_values

        self.reset()
        start_pos = self._pos
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {start_pos + seq_len} exceeds max_seq_len {self.max_seq_len}")

        padded_seq = pad_to_tile(seq_len)
        if seq_len < padded_seq:
            input_ids = F.pad(input_ids, (0, padded_seq - seq_len), value=0)

        h = self._forward_hidden(input_ids, start_pos, seq_len, None)
        last_token_idx = seq_len - 1
        h_last = ttnn.slice(
            h,
            (0, 0, last_token_idx, 0),
            (h.shape[0], h.shape[1], last_token_idx + 1, h.shape[-1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)

        logits_device = ttnn.linear(h_last, self.lm_head)
        ttnn.deallocate(h_last)
        token_ids = ttnn.argmax(
            logits_device,
            dim=3,
            keepdim=True,
            use_multicore=False,
            output_tensor=self.argmax_output_buffer,
        )
        token_ids_torch = ttnn.to_torch(token_ids).reshape(-1)
        ttnn.deallocate(logits_device)

        self._pos = start_pos + seq_len
        past = self._tt_past_key_values if use_cache else None
        return int(token_ids_torch[0].item()), past

    def prefill_logits_last_device(self, input_ids: torch.Tensor, use_cache: bool = True) -> tuple:
        batch, seq_len = input_ids.shape
        if batch != 1:
            raise ValueError("Only batch=1 supported")

        self.reset()
        start_pos = self._pos
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {start_pos + seq_len} exceeds max_seq_len {self.max_seq_len}")

        cur_pos_tensor = None
        if seq_len == 1:
            cur_pos = torch.full((TILE_SIZE,), -1, dtype=torch.int32)
            cur_pos[0] = start_pos
            cur_pos_tensor = ttnn.from_torch(cur_pos, dtype=ttnn.int32, device=self.tt_device)

        padded_seq = pad_to_tile(seq_len)
        if seq_len < padded_seq:
            input_ids = F.pad(input_ids, (0, padded_seq - seq_len), value=0)

        h = self._forward_hidden(input_ids, start_pos, seq_len, cur_pos_tensor)
        last_token_idx = seq_len - 1
        h_last = ttnn.slice(
            h,
            (0, 0, last_token_idx, 0),
            (h.shape[0], h.shape[1], last_token_idx + 1, h.shape[-1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)

        logits_device = ttnn.linear(h_last, self.lm_head)
        ttnn.deallocate(h_last)

        if cur_pos_tensor is not None:
            ttnn.deallocate(cur_pos_tensor)

        logits = ttnn.to_torch(logits_device).reshape(batch, 1, -1)[:, 0, :].float()
        ttnn.deallocate(logits_device)

        self._pos = start_pos + seq_len
        past = self._tt_past_key_values if use_cache else None
        return logits, past


def build_model(hf_model, tt_device, max_seq_len: Optional[int] = None) -> TtnnQwen35MoeForCausalLM:
    """Build the TT model from a HuggingFace reference model."""
    return TtnnQwen35MoeForCausalLM(hf_model, tt_device, max_seq_len)
