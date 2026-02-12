# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Mistral 7B Instruct v0.3 implementation in ttnn for N300.

This version uses 1D tensor parallel across a 1x2 mesh:
- QKV, gate, and up projections are column-parallel (width sharded).
- Output and down projections are row-parallel (height sharded) followed by all-reduce.

Key optimizations versus functional:
- Fused Q/K/V projection into a single matmul (`self.qkv_proj`)
- `prefill_logits_last_device()` fast path for demo/eval TTFT (compute + transfer last-token logits only)
- Traced decode execution with preallocated token/position/RoPE buffers
- Decode LM head runs on a sliced `[1, 1, 1, hidden]` activation (avoid padded-32 LM head work)

This file defines the ttnn model only. Use `eval.py` at repo root for
teacher-forcing accuracy checks against the HuggingFace reference.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import ttnn
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


TILE_SIZE = 32
MESH_SHAPE = (1, 2)
MESH_TOPOLOGY = ttnn.Topology.Linear
MESH_NUM_LINKS = 1
PAGED_BLOCK_SIZE = 64
WEIGHT_DTYPE = ttnn.bfloat8_b
WEIGHT_LAYOUT = ttnn.TILE_LAYOUT
USE_DECODE_TRACE = os.environ.get("TTNN_USE_DECODE_TRACE", "1").lower() not in ("0", "false", "no", "off")


def pad_to_tile(x: int) -> int:
    """Pad to tile boundary (32)."""
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def mesh_shape_to_axis(mesh_shape: tuple[int, int]) -> int:
    """Return the mesh axis used for 1D tensor parallel."""
    if mesh_shape[0] == 1 and mesh_shape[1] > 1:
        return 1
    if mesh_shape[1] == 1 and mesh_shape[0] > 1:
        return 0
    raise ValueError(f"Expected 1D mesh shape for N300, got {mesh_shape}")


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

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
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
        )


@dataclass
class ParallelConfig:
    mesh_device: ttnn.MeshDevice
    mesh_shape: tuple[int, int]
    num_devices: int
    mesh_axis: int
    num_links: int
    topology: ttnn.Topology
    replicate_mapper: object
    shard_width_mapper: object
    shard_height_mapper: object
    shard_kv_mapper: object
    vocab_composer: object


@dataclass
class PagedAttentionConfig:
    """Paged KV cache configuration."""

    block_size: int
    max_num_blocks: int


def validate_parallel_config(config: ModelConfig, num_devices: int) -> None:
    if num_devices < 2:
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
    return ttnn.all_reduce(
        x,
        cluster_axis=parallel.mesh_axis,
        num_links=parallel.num_links,
        topology=parallel.topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def compute_rope_cache(config: ModelConfig, max_seq_len: int) -> tuple:
    """
    Precompute RoPE cos/sin cache in HuggingFace format.
    Returns cos, sin tensors of shape [1, 1, max_seq_len, head_dim].
    """
    if config.rope_scaling:
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        raise ValueError(f"rope_scaling {rope_type} is not supported in this bringup")

    head_dim = config.head_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin = emb.sin().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    return cos, sin


def resolve_max_seq_len(hf_config, max_seq_len: Optional[int]) -> int:
    """Resolve max sequence length from HF config when not provided."""
    config_max = getattr(hf_config, "max_position_embeddings", None)
    if max_seq_len is None:
        if config_max is None:
            raise ValueError("max_seq_len is required when config has no max_position_embeddings")
        return config_max
    if config_max is not None and max_seq_len > config_max:
        raise ValueError(f"max_seq_len {max_seq_len} exceeds config max {config_max}")
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
    """
    Multi-head attention with GQA support, 1D tensor parallel.

    QKV projections are column-parallel. Output projection is row-parallel with
    an all-reduce to replicate the result.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        parallel: ParallelConfig,
        paged_attention_config: PagedAttentionConfig,
        page_table: ttnn.Tensor,
    ):
        self.parallel = parallel
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_local_heads = self.n_heads // parallel.num_devices
        self.n_local_kv_heads = self.n_kv_heads // parallel.num_devices
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        self.paged_attention_config = paged_attention_config
        self.page_table = page_table

        p = f"model.layers.{layer_idx}.self_attn."
        self.qkv_proj = self._load_qkv_weight(
            state_dict[f"{p}q_proj.weight"],
            state_dict[f"{p}k_proj.weight"],
            state_dict[f"{p}v_proj.weight"],
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
        """Load weight transposed for ttnn.linear: [out, in] -> [1, 1, in, out]."""
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

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_cos: Optional[ttnn.Tensor] = None,
        decode_sin: Optional[ttnn.Tensor] = None,
        trace_decode: bool = False,
    ) -> ttnn.Tensor:
        """Forward pass for prefill (seq_len > 1) or decode (seq_len == 1)."""
        is_prefill = seq_len > 1
        padded_seq = pad_to_tile(seq_len)

        qkv = ttnn.linear(x, self.qkv_proj)

        num_heads = self.n_local_heads
        num_kv_heads = self.n_local_kv_heads

        if is_prefill:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)

            cos = self.cos_cache[:, :, :padded_seq, :]
            sin = self.sin_cache[:, :, :padded_seq, :]
            q = ttnn.experimental.rotary_embedding(q, cos, sin)
            k = ttnn.experimental.rotary_embedding(k, cos, sin)

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
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if not trace_decode:
                ttnn.deallocate(qkv)

            rope_cos = self.cos_cache
            rope_sin = self.sin_cache
            rope_pos = start_pos
            if decode_cos is not None and decode_sin is not None:
                rope_cos = decode_cos
                rope_sin = decode_sin
                rope_pos = 0

            q_batch = q.shape[1]
            q_heads = q.shape[2]
            q_bh = q_batch * q_heads
            q_bh_padded = pad_to_tile(q_bh)
            q = ttnn.reshape(q, (1, 1, q_bh, self.head_dim), (1, 1, q_bh_padded, self.head_dim))
            q = ttnn.experimental.rotary_embedding(q, rope_cos, rope_sin, rope_pos)
            q = ttnn.reshape(q, (1, q_batch, q_heads, self.head_dim), (1, q_batch, q_heads, self.head_dim))

            k_batch = k.shape[1]
            k_heads = k.shape[2]
            k_bh = k_batch * k_heads
            k_bh_padded = pad_to_tile(k_bh)
            k = ttnn.reshape(k, (1, 1, k_bh, self.head_dim), (1, 1, k_bh_padded, self.head_dim))
            k = ttnn.experimental.rotary_embedding(k, rope_cos, rope_sin, rope_pos)
            k = ttnn.reshape(k, (1, k_batch, k_heads, self.head_dim), (1, k_batch, k_heads, self.head_dim))

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
    """SwiGLU MLP, 1D tensor parallel."""

    def __init__(self, layer_idx: int, state_dict: dict, parallel: ParallelConfig):
        p = f"model.layers.{layer_idx}.mlp."
        self.parallel = parallel
        self.gate_proj = self._load_weight(state_dict[f"{p}gate_proj.weight"], parallel.shard_width_mapper)
        self.up_proj = self._load_weight(state_dict[f"{p}up_proj.weight"], parallel.shard_width_mapper)
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
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
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
            cos_cache,
            sin_cache,
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
        decode_cos: Optional[ttnn.Tensor] = None,
        decode_sin: Optional[ttnn.Tensor] = None,
        trace_decode: bool = False,
    ) -> ttnn.Tensor:
        x = ttnn.add(
            x,
            self.attn(self.attn_norm(x), start_pos, seq_len, cur_pos_tensor, decode_cos, decode_sin, trace_decode),
        )
        x = ttnn.add(x, self.mlp(self.ffn_norm(x)))
        return x


class TtnnMistralForCausalLM(torch.nn.Module, GenerationMixin):
    """
    Mistral model with 100% ttnn execution and 1D tensor parallel on N300.
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
        cos, sin = compute_rope_cache(self.tt_config, self.max_seq_len)
        self.cos_cache_host = cos
        self.sin_cache_host = sin
        self.cos_cache = ttnn.as_tensor(
            cos,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.sin_cache = ttnn.as_tensor(
            sin,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )

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
        self.decode_cos_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 1, self.tt_config.head_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.decode_sin_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 1, self.tt_config.head_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.use_decode_trace = USE_DECODE_TRACE
        self.decode_trace_id = None
        self.decode_trace_logits = None

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

        print(f"  Loading {self.tt_config.num_hidden_layers} layers...")
        self.layers = [
            DecoderLayer(
                self.tt_config,
                i,
                state_dict,
                self.cos_cache,
                self.sin_cache,
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

    def reset(self):
        """Reset position counter for new sequence."""
        self._pos = 0
        if self.decode_trace_id is not None:
            ttnn.release_trace(self.tt_device, self.decode_trace_id)
            self.decode_trace_id = None
            self.decode_trace_logits = None

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values

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
        cos_slice = self.cos_cache_host[:, :, start_pos : start_pos + 1, :]
        sin_slice = self.sin_cache_host[:, :, start_pos : start_pos + 1, :]
        host_cos = ttnn.from_torch(
            cos_slice,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        host_sin = ttnn.from_torch(
            sin_slice,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_cos, self.decode_cos_buffer)
        ttnn.copy_host_to_device_tensor(host_sin, self.decode_sin_buffer)

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

        token_idx = seq_len - 1
        h_last = ttnn.slice(
            h,
            (0, 0, token_idx, 0),
            (h.shape[0], h.shape[1], token_idx + 1, h.shape[-1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)
        return ttnn.linear(h_last, self.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def prefill_logits_last_device(self, input_ids: torch.Tensor, use_cache: bool = True) -> tuple[torch.Tensor, object]:
        batch, seq_len = input_ids.shape
        assert batch == 1, "Only batch=1 supported"

        self.reset()
        start_pos = self._pos
        if start_pos != 0:
            raise ValueError("prefill_logits_last_device must be called at start_pos=0")
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {start_pos + seq_len} exceeds max_seq_len {self.max_seq_len}")

        padded_seq = pad_to_tile(seq_len)
        if seq_len < padded_seq:
            input_ids = torch.nn.functional.pad(input_ids, (0, padded_seq - seq_len), value=0)

        logits = self._forward_prefill_last_logits(input_ids, start_pos, seq_len)
        self._pos = start_pos + seq_len

        if self.parallel.num_devices > 1:
            logits_torch = ttnn.to_torch(logits, mesh_composer=self.parallel.vocab_composer)
        else:
            logits_torch = ttnn.to_torch(logits)
        logits_torch = logits_torch.reshape(batch, 1, -1)[:, 0, :].float()
        ttnn.deallocate(logits)
        past = self._tt_past_key_values if use_cache else None
        return logits_torch, past

    def _forward_decode_device(self, start_pos: int, trace_decode: bool) -> ttnn.Tensor:
        h = ttnn.embedding(self.decode_token_buffer, self.embed, layout=ttnn.TILE_LAYOUT)
        for layer in self.layers:
            h = layer(
                h,
                start_pos,
                1,
                self.decode_pos_buffer,
                self.decode_cos_buffer,
                self.decode_sin_buffer,
                trace_decode,
            )
        h = self.norm(h)
        h = ttnn.slice(
            h,
            (0, 0, 0, 0),
            (h.shape[0], h.shape[1], 1, h.shape[-1]),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        logits = ttnn.linear(h, self.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if not trace_decode:
            ttnn.deallocate(h)
        return logits

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
            logits = self.decode_trace_logits
        else:
            logits = self._forward_decode_device(start_pos, False)
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
        """Forward pass compatible with HuggingFace generate()."""
        batch, seq_len = input_ids.shape
        assert batch == 1, "Only batch=1 supported"

        if past_key_values is None:
            self.reset()
        else:
            assert seq_len == 1, "Only 1-token decode supported when using cache"

        start_pos = self._pos
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {start_pos + seq_len} exceeds max_seq_len {self.max_seq_len}; "
                "increase --max_seq_len if memory allows"
            )

        if seq_len == 1:
            logits_device = self._forward_decode(input_ids, start_pos)
            padded_seq = 1
        else:
            padded_seq = pad_to_tile(seq_len)
            if seq_len < padded_seq:
                input_ids = torch.nn.functional.pad(input_ids, (0, padded_seq - seq_len), value=0)

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
            logits_device = ttnn.linear(h, self.lm_head)

        if self.parallel.num_devices > 1:
            logits_torch = ttnn.to_torch(logits_device, mesh_composer=self.parallel.vocab_composer)
        else:
            logits_torch = ttnn.to_torch(logits_device)
        logits_torch = logits_torch.reshape(batch, padded_seq, -1)[:, :seq_len, :]
        if seq_len > 1 or not self.use_decode_trace:
            ttnn.deallocate(logits_device)

        self._pos = start_pos + seq_len

        return CausalLMOutputWithPast(
            logits=logits_torch.float(),
            past_key_values=(self._tt_past_key_values if use_cache else None),
        )


def build_model(hf_model, tt_device, max_seq_len: Optional[int] = None) -> TtnnMistralForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnMistralForCausalLM(hf_model, tt_device, max_seq_len)
