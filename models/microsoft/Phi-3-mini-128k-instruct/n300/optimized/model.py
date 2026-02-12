# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phi-3 Mini 128k Instruct optimized TTNN model for N300.

Optimizations versus the functional bringup:
- Prefill last-token logits fast path (`prefill_logits_last_device`) to lower TTFT.
- Decode trace with preallocated token/pos/RoPE buffers.
- Fused QKV projection (single matmul) with TP-safe shard ordering.
- Decode path keeps QKV head creation outputs in L1 (`DECODE_MEMORY_CONFIG`) before attention.

Use `demo.py` and `eval.py` at repo root for timing and teacher-forcing accuracy checks.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import ttnn
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


TILE_SIZE = 32
HEAD_DIM_TILE = 64
PAGED_BLOCK_SIZE = 64
MAX_CACHE_SEQ_LEN = 12288
WEIGHT_DTYPE = ttnn.bfloat16
WEIGHT_LAYOUT = ttnn.TILE_LAYOUT
MESH_SHAPE = (1, 2)
MESH_TOPOLOGY = ttnn.Topology.Linear
MESH_NUM_LINKS = 1
USE_DECODE_TRACE = True
DECODE_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG


def pad_to_tile(x: int) -> int:
    """Pad to tile boundary (32)."""
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def pad_head_dim(x: int) -> int:
    """Pad head dimension to the rotary tile requirement (64)."""
    return ((x + HEAD_DIM_TILE - 1) // HEAD_DIM_TILE) * HEAD_DIM_TILE


def mesh_shape_to_axis(mesh_shape: tuple[int, int]) -> Optional[int]:
    """Return the mesh axis used for 1D tensor parallel or None for 2D."""
    if mesh_shape[0] == 1 and mesh_shape[1] > 1:
        return 1
    if mesh_shape[1] == 1 and mesh_shape[0] > 1:
        return 0
    if mesh_shape[0] > 1 and mesh_shape[1] > 1:
        return None
    raise ValueError(f"Expected mesh shape with at least one dimension > 1, got {mesh_shape}")


def num_mesh_devices(mesh_shape: tuple[int, int]) -> int:
    return mesh_shape[0] * mesh_shape[1]


@dataclass
class ModelConfig:
    """Model configuration extracted from HuggingFace."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[dict]
    hidden_act: str
    tie_word_embeddings: bool
    max_position_embeddings: int
    original_max_position_embeddings: int
    partial_rotary_factor: float

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        original_max = getattr(hf_config, "original_max_position_embeddings", hf_config.max_position_embeddings)
        partial_rotary = getattr(hf_config, "partial_rotary_factor", 1.0)
        return cls(
            hf_config.vocab_size,
            hf_config.hidden_size,
            hf_config.intermediate_size,
            hf_config.num_hidden_layers,
            hf_config.num_attention_heads,
            num_kv_heads,
            head_dim,
            hf_config.rms_norm_eps,
            hf_config.rope_theta,
            getattr(hf_config, "rope_scaling", None),
            hf_config.hidden_act,
            hf_config.tie_word_embeddings,
            hf_config.max_position_embeddings,
            original_max,
            partial_rotary,
        )


@dataclass
class PagedAttentionConfig:
    """Paged KV cache configuration."""

    block_size: int
    max_num_blocks: int


@dataclass
class ParallelConfig:
    mesh_device: ttnn.MeshDevice
    mesh_shape: tuple[int, int]
    num_devices: int
    mesh_axis: Optional[int]
    num_links: int
    topology: ttnn.Topology
    replicate_mapper: object
    shard_width_mapper: object
    shard_height_mapper: object
    shard_kv_mapper: object
    vocab_composer: object


def validate_parallel_config(config: ModelConfig, num_devices: int) -> None:
    if num_devices != 2:
        raise ValueError("N300 model expects a 2-device mesh")
    if config.num_attention_heads % num_devices != 0:
        raise ValueError("num_attention_heads must divide evenly across devices")
    if config.num_key_value_heads % num_devices != 0:
        raise ValueError("num_key_value_heads must divide evenly across devices")
    if config.hidden_size % num_devices != 0:
        raise ValueError("hidden_size must divide evenly across devices")
    if config.intermediate_size % num_devices != 0:
        raise ValueError("intermediate_size must divide evenly across devices")


def all_reduce_tensor(x: ttnn.Tensor, parallel: ParallelConfig) -> ttnn.Tensor:
    if parallel.num_devices == 1:
        return x
    if parallel.mesh_axis is None:
        return ttnn.all_reduce(
            x,
            num_links=parallel.num_links,
            topology=parallel.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return ttnn.all_reduce(
        x,
        cluster_axis=parallel.mesh_axis,
        num_links=parallel.num_links,
        topology=parallel.topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def compute_attention_scaling(config: ModelConfig) -> float:
    factor = config.max_position_embeddings / config.original_max_position_embeddings
    if factor <= 1.0:
        return 1.0
    return math.sqrt(1 + math.log(factor) / math.log(config.original_max_position_embeddings))


def compute_rope_cache(
    config: ModelConfig,
    max_seq_len: int,
    use_long: Optional[bool] = None,
) -> tuple:
    """
    Precompute RoPE cos/sin cache in HuggingFace format.
    Returns cos, sin tensors of shape [1, 1, max_seq_len, head_dim].
    """
    if config.partial_rotary_factor != 1.0:
        raise ValueError("partial_rotary_factor != 1.0 is not supported in this bringup")

    head_dim = config.head_dim
    padded_head_dim = pad_head_dim(head_dim)
    attention_scaling = 1.0

    if config.rope_scaling:
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        if rope_type != "longrope":
            raise ValueError(f"rope_scaling {rope_type} is not supported in this bringup")

        long_factor = config.rope_scaling["long_factor"]
        short_factor = config.rope_scaling["short_factor"]
        if use_long is None:
            use_long = max_seq_len > config.original_max_position_embeddings
        ext_factors = torch.tensor(long_factor if use_long else short_factor, dtype=torch.float32)

        inv_freq = 1.0 / (
            ext_factors * (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        )
        attention_scaling = config.rope_scaling.get("attention_factor")
        if attention_scaling is None:
            attention_scaling = compute_attention_scaling(config)
    else:
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    if attention_scaling != 1.0:
        cos = cos * attention_scaling
        sin = sin * attention_scaling
    if padded_head_dim != head_dim:
        half = head_dim // 2
        half_padded = padded_head_dim // 2
        pad = half_padded - half
        cos_left = torch.cat([cos[:, :half], torch.ones((max_seq_len, pad), dtype=cos.dtype)], dim=-1)
        cos_right = torch.cat([cos[:, half:], torch.ones((max_seq_len, pad), dtype=cos.dtype)], dim=-1)
        sin_left = torch.cat([sin[:, :half], torch.zeros((max_seq_len, pad), dtype=sin.dtype)], dim=-1)
        sin_right = torch.cat([sin[:, half:], torch.zeros((max_seq_len, pad), dtype=sin.dtype)], dim=-1)
        cos = torch.cat([cos_left, cos_right], dim=-1)
        sin = torch.cat([sin_left, sin_right], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin = sin.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    return cos, sin


def resolve_max_seq_len(hf_config, max_seq_len: Optional[int]) -> int:
    """Resolve max sequence length and cap to the validated N300 cache budget."""
    config_max = getattr(hf_config, "max_position_embeddings", None)
    if config_max is None:
        config_max = getattr(hf_config, "max_seq_len", None)
    if max_seq_len is None:
        if config_max is None:
            raise ValueError("max_seq_len is required when config has no max_position_embeddings")
        max_seq_len = config_max
    if config_max is not None and max_seq_len > config_max:
        raise ValueError(f"max_seq_len {max_seq_len} exceeds config max {config_max}")
    if max_seq_len > MAX_CACHE_SEQ_LEN:
        max_seq_len = MAX_CACHE_SEQ_LEN
    return max_seq_len


class RMSNorm:
    """RMSNorm layer."""

    def __init__(self, weight: torch.Tensor, eps: float, parallel: ParallelConfig):
        self.eps = eps
        self.weight = ttnn.as_tensor(
            weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parallel.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=parallel.replicate_mapper,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)


class Attention:
    """Multi-head attention with a fused QKV projection, 1D tensor parallel."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache_short: ttnn.Tensor,
        sin_cache_short: ttnn.Tensor,
        cos_cache_long: ttnn.Tensor,
        sin_cache_long: ttnn.Tensor,
        parallel: ParallelConfig,
        paged_attention_config: PagedAttentionConfig,
        page_table: ttnn.Tensor,
    ):
        self.parallel = parallel
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_local_heads = self.n_heads // parallel.num_devices
        self.n_local_kv_heads = self.n_kv_heads // parallel.num_devices
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.head_dim = config.head_dim
        self.head_dim_padded = pad_head_dim(self.head_dim)
        self.head_dim_half = self.head_dim // 2
        self.head_dim_half_padded = self.head_dim_padded // 2
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.paged_attention_config = paged_attention_config
        self.page_table = page_table
        device_grid = self.parallel.mesh_device.core_grid
        grid_x = device_grid.x
        grid_y = device_grid.y
        if grid_x * grid_y > TILE_SIZE:
            grid_y = max(1, TILE_SIZE // grid_x)
        self.decode_heads_grid = ttnn.CoreGrid(x=grid_x, y=grid_y)
        padded_heads = pad_to_tile(self.n_local_heads)
        self.decode_heads_memcfg = ttnn.create_sharded_memory_config(
            (padded_heads, self.head_dim),
            self.decode_heads_grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.decode_q_pad_zeros = None
        self.decode_k_pad_zeros = None
        if self.head_dim_padded != self.head_dim:
            pad = self.head_dim_half_padded - self.head_dim_half
            pad_tile = pad_to_tile(pad)
            self.decode_q_pad_zeros = ttnn.zeros(
                (1, 1, TILE_SIZE * self.n_local_heads, pad_tile),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.parallel.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.decode_k_pad_zeros = ttnn.zeros(
                (1, 1, TILE_SIZE * self.n_local_kv_heads, pad_tile),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.parallel.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.cos_cache_short = cos_cache_short
        self.sin_cache_short = sin_cache_short
        self.cos_cache_long = cos_cache_long
        self.sin_cache_long = sin_cache_long

        p = f"model.layers.{layer_idx}.self_attn."
        qkv_weight = state_dict[f"{p}qkv_proj.weight"]
        q_end = self.n_heads * self.head_dim
        k_end = q_end + self.n_kv_heads * self.head_dim
        v_end = k_end + self.n_kv_heads * self.head_dim
        self.qkv_proj = self._load_qkv_weight(
            qkv_weight[:q_end, :],
            qkv_weight[q_end:k_end, :],
            qkv_weight[k_end:v_end, :],
            parallel.shard_width_mapper,
        )
        self.o_proj = self._load_weight(state_dict[f"{p}o_proj.weight"], parallel.shard_height_mapper)

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
            device=parallel.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=parallel.shard_kv_mapper,
        )
        self.v_cache = ttnn.as_tensor(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parallel.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=parallel.shard_kv_mapper,
        )

    def _load_weight(self, w: torch.Tensor, mesh_mapper) -> ttnn.Tensor:
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=self.parallel.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def _load_qkv_weight(
        self, q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor, mesh_mapper
    ) -> ttnn.Tensor:
        q_chunks = torch.chunk(q_weight, self.parallel.num_devices, dim=0)
        k_chunks = torch.chunk(k_weight, self.parallel.num_devices, dim=0)
        v_chunks = torch.chunk(v_weight, self.parallel.num_devices, dim=0)
        qkv_weight = torch.cat(
            [torch.cat((q_chunks[i], k_chunks[i], v_chunks[i]), dim=0) for i in range(self.parallel.num_devices)],
            dim=0,
        )
        return self._load_weight(qkv_weight, mesh_mapper)

    def _pad_head_dim(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.head_dim_padded == self.head_dim:
            return x
        pad = self.head_dim_half_padded - self.head_dim_half
        pad_tile = pad_to_tile(pad)
        zeros = ttnn.zeros(
            (x.shape[0], x.shape[1], x.shape[2], pad_tile),
            dtype=x.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.parallel.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        zeros = ttnn.slice(
            zeros,
            (0, 0, 0, 0),
            (zeros.shape[0], zeros.shape[1], zeros.shape[2], pad),
        )
        left = ttnn.slice(
            x,
            (0, 0, 0, 0),
            (x.shape[0], x.shape[1], x.shape[2], self.head_dim_half),
        )
        right = ttnn.slice(
            x,
            (0, 0, 0, self.head_dim_half),
            (x.shape[0], x.shape[1], x.shape[2], self.head_dim),
        )
        left = ttnn.concat([left, zeros], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        right = ttnn.concat([right, zeros], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.concat([left, right], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _pad_head_dim_decode(self, x: ttnn.Tensor, zeros_padded: ttnn.Tensor) -> ttnn.Tensor:
        if self.head_dim_padded == self.head_dim:
            return x
        pad = self.head_dim_half_padded - self.head_dim_half
        zeros = ttnn.slice(
            zeros_padded,
            (0, 0, 0, 0),
            (zeros_padded.shape[0], zeros_padded.shape[1], zeros_padded.shape[2], pad),
        )
        left = ttnn.slice(
            x,
            (0, 0, 0, 0),
            (x.shape[0], x.shape[1], x.shape[2], self.head_dim_half),
        )
        right = ttnn.slice(
            x,
            (0, 0, 0, self.head_dim_half),
            (x.shape[0], x.shape[1], x.shape[2], self.head_dim),
        )
        left = ttnn.concat([left, zeros], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        right = ttnn.concat([right, zeros], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.concat([left, right], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _slice_head_dim(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.head_dim_padded == self.head_dim:
            return x
        left = ttnn.slice(
            x,
            (0, 0, 0, 0),
            (x.shape[0], x.shape[1], x.shape[2], self.head_dim_half_padded),
        )
        right = ttnn.slice(
            x,
            (0, 0, 0, self.head_dim_half_padded),
            (x.shape[0], x.shape[1], x.shape[2], self.head_dim_padded),
        )
        left = ttnn.slice(
            left,
            (0, 0, 0, 0),
            (left.shape[0], left.shape[1], left.shape[2], self.head_dim_half),
        )
        right = ttnn.slice(
            right,
            (0, 0, 0, 0),
            (right.shape[0], right.shape[1], right.shape[2], self.head_dim_half),
        )
        return ttnn.concat([left, right], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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
        is_prefill = seq_len > 1
        padded_seq = pad_to_tile(seq_len)
        if is_prefill:
            qkv = ttnn.linear(x, self.qkv_proj)
        else:
            qkv = ttnn.linear(x, self.qkv_proj, memory_config=DECODE_MEMORY_CONFIG)

        num_heads = self.n_local_heads
        num_kv_heads = self.n_local_kv_heads

        if is_prefill:
            use_long = seq_len > self.original_max_position_embeddings
            cos_cache = self.cos_cache_long if use_long else self.cos_cache_short
            sin_cache = self.sin_cache_long if use_long else self.sin_cache_short

            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)

            q_mem = ttnn.get_memory_config(q)
            k_mem = ttnn.get_memory_config(k)
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)

            cos = cos_cache[:, :, :padded_seq, :]
            sin = sin_cache[:, :, :padded_seq, :]
            q = self._slice_head_dim(ttnn.experimental.rotary_embedding(self._pad_head_dim(q), cos, sin))
            k = self._slice_head_dim(ttnn.experimental.rotary_embedding(self._pad_head_dim(k), cos, sin))

            q = ttnn.to_memory_config(q, q_mem)
            k = ttnn.to_memory_config(k, k_mem)

            ttnn.experimental.paged_fill_cache(self.k_cache, k, self.page_table, batch_idx=0)
            ttnn.experimental.paged_fill_cache(self.v_cache, v, self.page_table, batch_idx=0)

            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale
            )
            attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            if cur_pos_tensor is None:
                raise ValueError("cur_pos_tensor is required for decode")

            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                memory_config=DECODE_MEMORY_CONFIG,
            )
            if not trace_decode:
                ttnn.deallocate(qkv)

            q_mem = ttnn.get_memory_config(q)
            k_mem = ttnn.get_memory_config(k)
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)

            q_batch = q.shape[1]
            q_heads = q.shape[2]
            q_bh = q_batch * q_heads
            q_bh_padded = pad_to_tile(q_bh)
            q = ttnn.reshape(q, (1, 1, q_bh, self.head_dim), (1, 1, q_bh_padded, self.head_dim))
            if decode_cos_q is None or decode_sin_q is None:
                use_long = start_pos >= self.original_max_position_embeddings
                cos_cache = self.cos_cache_long if use_long else self.cos_cache_short
                sin_cache = self.sin_cache_long if use_long else self.sin_cache_short
                q = self._slice_head_dim(
                    ttnn.experimental.rotary_embedding(
                        self._pad_head_dim_decode(q, self.decode_q_pad_zeros),
                        cos_cache,
                        sin_cache,
                        start_pos,
                    )
                )
            else:
                q = self._slice_head_dim(
                    ttnn.experimental.rotary_embedding(
                        self._pad_head_dim_decode(q, self.decode_q_pad_zeros),
                        decode_cos_q,
                        decode_sin_q,
                    )
                )
            q = ttnn.reshape(q, (1, q_batch, q_heads, self.head_dim), (1, q_batch, q_heads, self.head_dim))

            k_batch = k.shape[1]
            k_heads = k.shape[2]
            k_bh = k_batch * k_heads
            k_bh_padded = pad_to_tile(k_bh)
            k = ttnn.reshape(k, (1, 1, k_bh, self.head_dim), (1, 1, k_bh_padded, self.head_dim))
            if decode_cos_k is None or decode_sin_k is None:
                use_long = start_pos >= self.original_max_position_embeddings
                cos_cache = self.cos_cache_long if use_long else self.cos_cache_short
                sin_cache = self.sin_cache_long if use_long else self.sin_cache_short
                k = self._slice_head_dim(
                    ttnn.experimental.rotary_embedding(
                        self._pad_head_dim_decode(k, self.decode_k_pad_zeros),
                        cos_cache,
                        sin_cache,
                        start_pos,
                    )
                )
            else:
                k = self._slice_head_dim(
                    ttnn.experimental.rotary_embedding(
                        self._pad_head_dim_decode(k, self.decode_k_pad_zeros),
                        decode_cos_k,
                        decode_sin_k,
                    )
                )
            k = ttnn.reshape(k, (1, k_batch, k_heads, self.head_dim), (1, k_batch, k_heads, self.head_dim))

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
            attn_out = ttnn.to_memory_config(attn_out, self.decode_heads_memcfg)
            attn_out = ttnn.experimental.nlp_concat_heads_decode(
                attn_out,
                num_heads=num_heads,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        expected_width = num_heads * self.head_dim
        if attn_out.shape[-1] != expected_width:
            attn_out = ttnn.slice(
                attn_out,
                (0, 0, 0, 0),
                (attn_out.shape[0], attn_out.shape[1], attn_out.shape[2], expected_width),
            )

        out = ttnn.linear(attn_out, self.o_proj)
        return all_reduce_tensor(out, self.parallel)


class MLP:
    """Gated MLP using a split gate/up projection, 1D tensor parallel."""

    def __init__(self, layer_idx: int, state_dict: dict, parallel: ParallelConfig):
        p = f"model.layers.{layer_idx}.mlp."
        self.parallel = parallel
        gate_up_weight = state_dict[f"{p}gate_up_proj.weight"]
        split = gate_up_weight.shape[0] // 2
        self.gate_proj = self._load_weight(gate_up_weight[:split, :], parallel.shard_width_mapper)
        self.up_proj = self._load_weight(gate_up_weight[split:, :], parallel.shard_width_mapper)
        self.down_proj = self._load_weight(state_dict[f"{p}down_proj.weight"], parallel.shard_height_mapper)

    def _load_weight(self, w: torch.Tensor, mesh_mapper) -> ttnn.Tensor:
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=self.parallel.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.silu(ttnn.linear(x, self.gate_proj))
        up = ttnn.linear(x, self.up_proj)
        out = ttnn.linear(ttnn.mul(gate, up), self.down_proj)
        return all_reduce_tensor(out, self.parallel)


class DecoderLayer:
    """Single transformer layer."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache_short: ttnn.Tensor,
        sin_cache_short: ttnn.Tensor,
        cos_cache_long: ttnn.Tensor,
        sin_cache_long: ttnn.Tensor,
        parallel: ParallelConfig,
        paged_attention_config: PagedAttentionConfig,
        page_table: ttnn.Tensor,
    ):
        p = f"model.layers.{layer_idx}."
        self.attn_norm = RMSNorm(state_dict[f"{p}input_layernorm.weight"], config.rms_norm_eps, parallel)
        self.ffn_norm = RMSNorm(state_dict[f"{p}post_attention_layernorm.weight"], config.rms_norm_eps, parallel)
        self.attn = Attention(
            config,
            layer_idx,
            state_dict,
            cos_cache_short,
            sin_cache_short,
            cos_cache_long,
            sin_cache_long,
            parallel,
            paged_attention_config,
            page_table,
        )
        self.mlp = MLP(layer_idx, state_dict, parallel)

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
        x = ttnn.add(
            x,
            self.attn(
                self.attn_norm(x),
                start_pos,
                seq_len,
                cur_pos_tensor=cur_pos_tensor,
                decode_cos_q=decode_cos_q,
                decode_sin_q=decode_sin_q,
                decode_cos_k=decode_cos_k,
                decode_sin_k=decode_sin_k,
                trace_decode=trace_decode,
            ),
        )
        x = ttnn.add(x, self.mlp(self.ffn_norm(x)))
        return x


class TtnnPhi3ForCausalLM(torch.nn.Module, GenerationMixin):
    """
    Phi-3 model with 100% ttnn execution and 1D tensor parallel on N300.
    HuggingFace `generate()`-compatible via `GenerationMixin`.
    """

    def __init__(self, hf_model, tt_device, max_seq_len: Optional[int] = None):
        super().__init__()

        self.tt_device = tt_device
        self.hf_config = hf_model.config
        self.tt_config = ModelConfig.from_hf(hf_model.config)
        self.max_seq_len = resolve_max_seq_len(self.hf_config, max_seq_len)
        self._pos = 0
        self.paged_attention_config = PagedAttentionConfig(
            PAGED_BLOCK_SIZE,
            math.ceil(self.max_seq_len / PAGED_BLOCK_SIZE),
        )

        if self.tt_config.hidden_act != "silu":
            raise ValueError(f"hidden_act {self.tt_config.hidden_act} is not supported in this bringup")

        self.config = self.hf_config
        self.generation_config = GenerationConfig.from_model_config(self.config)
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = self.generation_config.eos_token_id
        self._supports_cache_class = False
        self.main_input_name = "input_ids"
        self.register_buffer("_torch_dummy", torch.empty(0, dtype=torch.float32), persistent=False)

        mesh_shape = tuple(tt_device.shape)
        num_devices = num_mesh_devices(mesh_shape)
        mesh_axis = mesh_shape_to_axis(mesh_shape)
        validate_parallel_config(self.tt_config, num_devices)

        self.parallel = ParallelConfig(
            mesh_device=tt_device,
            mesh_shape=mesh_shape,
            num_devices=num_devices,
            mesh_axis=mesh_axis,
            num_links=MESH_NUM_LINKS,
            topology=MESH_TOPOLOGY,
            replicate_mapper=ttnn.ReplicateTensorToMesh(tt_device),
            shard_width_mapper=ttnn.ShardTensorToMesh(tt_device, dim=3),
            shard_height_mapper=ttnn.ShardTensorToMesh(tt_device, dim=2),
            shard_kv_mapper=ttnn.ShardTensorToMesh(tt_device, dim=1),
            vocab_composer=ttnn.ConcatMeshToTensor(tt_device, dim=3),
        )

        state_dict = hf_model.state_dict()

        print("  Loading embeddings...")
        self.embed = ttnn.as_tensor(
            state_dict["model.embed_tokens.weight"].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )

        print("  Computing RoPE cache...")
        cos_short, sin_short = compute_rope_cache(self.tt_config, self.max_seq_len, use_long=False)
        cos_long, sin_long = compute_rope_cache(self.tt_config, self.max_seq_len, use_long=True)
        self.cos_cache_short_host = cos_short
        self.sin_cache_short_host = sin_short
        self.cos_cache_long_host = cos_long
        self.sin_cache_long_host = sin_long
        self.cos_cache_short = ttnn.as_tensor(
            cos_short,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.sin_cache_short = ttnn.as_tensor(
            sin_short,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.cos_cache_long = ttnn.as_tensor(
            cos_long,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.sin_cache_long = ttnn.as_tensor(
            sin_long,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )

        page_table = torch.arange(self.paged_attention_config.max_num_blocks, dtype=torch.int32)
        page_table = page_table.repeat(TILE_SIZE, 1)
        self.page_table = ttnn.as_tensor(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )

        head_dim_padded = pad_head_dim(self.tt_config.head_dim)
        self.decode_token_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 1, TILE_SIZE), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.decode_pos_buffer = ttnn.from_torch(
            torch.zeros((TILE_SIZE,), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.decode_q_rope_seq = (self.tt_config.num_attention_heads // self.parallel.num_devices) * TILE_SIZE
        self.decode_k_rope_seq = (self.tt_config.num_key_value_heads // self.parallel.num_devices) * TILE_SIZE
        self.decode_cos_q_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.decode_q_rope_seq, head_dim_padded), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.decode_sin_q_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.decode_q_rope_seq, head_dim_padded), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.decode_cos_k_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.decode_k_rope_seq, head_dim_padded), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.decode_sin_k_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.decode_k_rope_seq, head_dim_padded), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.use_decode_trace = USE_DECODE_TRACE
        self.decode_trace_id = None
        self.decode_trace_logits = None

        print(f"  Loading {self.tt_config.num_hidden_layers} layers...")
        self.layers = [
            DecoderLayer(
                self.tt_config,
                i,
                state_dict,
                self.cos_cache_short,
                self.sin_cache_short,
                self.cos_cache_long,
                self.sin_cache_long,
                self.parallel,
                self.paged_attention_config,
                self.page_table,
            )
            for i in range(self.tt_config.num_hidden_layers)
        ]

        self.norm = RMSNorm(state_dict["model.norm.weight"], self.tt_config.rms_norm_eps, self.parallel)
        lm_head_weight = state_dict.get("lm_head.weight", state_dict["model.embed_tokens.weight"])
        self.lm_head = ttnn.as_tensor(
            lm_head_weight.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.shard_width_mapper,
        )

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

    def reset(self):
        """Reset position counter for new sequence."""
        self._pos = 0
        self._release_decode_trace()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values

    def _logits_to_torch(self, logits: ttnn.Tensor) -> torch.Tensor:
        if self.parallel.num_devices > 1:
            return ttnn.to_torch(logits, mesh_composer=self.parallel.vocab_composer)
        return ttnn.to_torch(logits)

    def _forward_prefill(self, input_ids: torch.Tensor, start_pos: int, seq_len: int) -> ttnn.Tensor:
        tokens = ttnn.from_torch(
            input_ids.reshape(1, 1, 1, -1),
            dtype=ttnn.uint32,
            device=self.tt_device,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        h = ttnn.embedding(tokens, self.embed, layout=ttnn.TILE_LAYOUT)
        for layer in self.layers:
            h = layer(h, start_pos, seq_len)
        h = self.norm(h)
        return ttnn.linear(h, self.lm_head)

    def _forward_prefill_last_logits(self, input_ids: torch.Tensor, start_pos: int, seq_len: int) -> ttnn.Tensor:
        tokens = ttnn.from_torch(
            input_ids.reshape(1, 1, 1, -1),
            dtype=ttnn.uint32,
            device=self.tt_device,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        h = ttnn.embedding(tokens, self.embed, layout=ttnn.TILE_LAYOUT)
        for layer in self.layers:
            h = layer(h, start_pos, seq_len)
        h = self.norm(h)
        last_token_idx = seq_len - 1
        h_last = ttnn.slice(
            h,
            (0, 0, last_token_idx, 0),
            (h.shape[0], h.shape[1], last_token_idx + 1, h.shape[-1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)
        return ttnn.linear(h_last, self.lm_head)

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
        use_long = start_pos >= self.tt_config.original_max_position_embeddings
        cos_cache = self.cos_cache_long_host if use_long else self.cos_cache_short_host
        sin_cache = self.sin_cache_long_host if use_long else self.sin_cache_short_host
        cos_token = cos_cache[:, :, start_pos : start_pos + 1, :]
        sin_token = sin_cache[:, :, start_pos : start_pos + 1, :]
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
        for layer in self.layers:
            h = layer(
                h,
                start_pos,
                1,
                cur_pos_tensor=self.decode_pos_buffer,
                decode_cos_q=self.decode_cos_q_buffer,
                decode_sin_q=self.decode_sin_q_buffer,
                decode_cos_k=self.decode_cos_k_buffer,
                decode_sin_k=self.decode_sin_k_buffer,
                trace_decode=trace_decode,
            )
        h = self.norm(h)
        h = ttnn.slice(
            h,
            (0, 0, 0, 0),
            (h.shape[0], h.shape[1], 1, h.shape[-1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.linear(h, self.lm_head)

    def _forward_decode(self, input_ids: torch.Tensor, start_pos: int) -> ttnn.Tensor:
        self._update_decode_token_buffer(input_ids)
        self._update_decode_pos_buffer(start_pos)
        self._update_decode_rope_buffers(start_pos)

        if self.use_decode_trace:
            if self.decode_trace_id is None:
                warmup_logits = self._forward_decode_device(start_pos, False)
                ttnn.deallocate(warmup_logits)
                self.decode_trace_id = ttnn.begin_trace_capture(self.tt_device)
                self.decode_trace_logits = self._forward_decode_device(start_pos, True)
                ttnn.end_trace_capture(self.tt_device, self.decode_trace_id)
            else:
                ttnn.execute_trace(self.tt_device, self.decode_trace_id, blocking=False)
            return self.decode_trace_logits
        return self._forward_decode_device(start_pos, False)

    def _forward_device_logits(self, input_ids: torch.Tensor, past_key_values, use_cache: bool):
        batch, seq_len = input_ids.shape
        if batch != 1:
            raise ValueError("Only batch=1 supported")

        if past_key_values is None:
            self.reset()
        elif seq_len != 1:
            raise ValueError("Only 1-token decode supported when using cache")

        start_pos = self._pos
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {start_pos + seq_len} exceeds max_seq_len {self.max_seq_len}; "
                "increase --max-seq-len if memory allows"
            )

        if seq_len == 1:
            logits = self._forward_decode(input_ids, start_pos)
            padded_seq = 1
        else:
            padded_seq = pad_to_tile(seq_len)
            if seq_len < padded_seq:
                input_ids = torch.nn.functional.pad(input_ids, (0, padded_seq - seq_len), value=0)
            logits = self._forward_prefill(input_ids, start_pos, seq_len)

        self._pos = start_pos + seq_len
        past = self._tt_past_key_values if use_cache else None
        return logits, seq_len, padded_seq, past

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        batch = input_ids.shape[0]
        logits_device, seq_len, padded_seq, past = self._forward_device_logits(input_ids, past_key_values, use_cache)
        logits = self._logits_to_torch(logits_device).reshape(batch, padded_seq, -1)[:, :seq_len, :]

        if seq_len > 1 or not self.use_decode_trace:
            ttnn.deallocate(logits_device)

        return CausalLMOutputWithPast(
            logits=logits.float(),
            past_key_values=past,
        )

    def prefill_logits_last_device(self, input_ids: torch.Tensor, use_cache: bool = True) -> tuple[torch.Tensor, object]:
        batch, seq_len = input_ids.shape
        if batch != 1:
            raise ValueError("Only batch=1 supported")

        self.reset()
        start_pos = self._pos
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {start_pos + seq_len} exceeds max_seq_len {self.max_seq_len}; "
                "increase --max-seq-len if memory allows"
            )

        padded_seq = pad_to_tile(seq_len)
        if seq_len < padded_seq:
            input_ids = torch.nn.functional.pad(input_ids, (0, padded_seq - seq_len), value=0)

        logits_device = self._forward_prefill_last_logits(input_ids, start_pos, seq_len)
        self._pos = start_pos + seq_len

        logits = self._logits_to_torch(logits_device).reshape(batch, 1, -1)[:, 0, :].float()
        ttnn.deallocate(logits_device)

        past = self._tt_past_key_values if use_cache else None
        return logits, past


def build_model(hf_model, tt_device, max_seq_len: Optional[int] = None) -> TtnnPhi3ForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnPhi3ForCausalLM(hf_model, tt_device, max_seq_len)
