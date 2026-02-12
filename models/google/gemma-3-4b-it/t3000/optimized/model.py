# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Gemma 3 4B Instruct implementation in ttnn with 1D tensor parallel on T3000.
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
WEIGHT_DTYPE = ttnn.bfloat8_b
WEIGHT_LAYOUT = ttnn.TILE_LAYOUT
PAGED_BLOCK_SIZE = 64
MESH_SHAPE = (2, 4)
MESH_TOPOLOGY = ttnn.Topology.Linear
MESH_NUM_LINKS = 1
USE_DECODE_TRACE = True


def pad_to_tile(x: int) -> int:
    """Pad to tile boundary (32)."""
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


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
    rope_local_base_freq: float
    hidden_activation: str
    attention_bias: bool
    query_pre_attn_scalar: float
    sliding_window: int
    sliding_window_pattern: Optional[int]
    layer_types: Optional[list]
    tie_word_embeddings: bool
    final_logit_softcapping: Optional[float]

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        text_config = getattr(hf_config, "text_config", hf_config)
        sliding_window_pattern = getattr(text_config, "sliding_window_pattern", None)
        if sliding_window_pattern is None:
            sliding_window_pattern = getattr(text_config, "_sliding_window_pattern", None)
        return cls(
            text_config.vocab_size,
            text_config.hidden_size,
            text_config.intermediate_size,
            text_config.num_hidden_layers,
            text_config.num_attention_heads,
            text_config.num_key_value_heads,
            text_config.head_dim,
            text_config.rms_norm_eps,
            text_config.rope_theta,
            getattr(text_config, "rope_scaling", None),
            text_config.rope_local_base_freq,
            text_config.hidden_activation,
            text_config.attention_bias,
            text_config.query_pre_attn_scalar,
            text_config.sliding_window,
            sliding_window_pattern,
            getattr(text_config, "layer_types", None),
            text_config.tie_word_embeddings,
            getattr(text_config, "final_logit_softcapping", None),
        )


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


@dataclass
class PagedAttentionConfig:
    """Paged KV cache configuration."""

    block_size: int
    max_num_blocks: int


def validate_parallel_config(config: ModelConfig, num_devices: int) -> None:
    if num_devices != 8:
        raise ValueError("T3000 model expects an 8-device mesh")
    if config.num_attention_heads % num_devices != 0:
        raise ValueError("num_attention_heads must divide evenly across devices")
    if config.num_key_value_heads % num_devices != 0 and num_devices % config.num_key_value_heads != 0:
        raise ValueError("num_key_value_heads must divide evenly across devices or divide num_devices")
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


def compute_rope_cache(
    head_dim: int, max_seq_len: int, rope_theta: float, rope_scaling: Optional[dict]
) -> tuple:
    """
    Precompute RoPE cos/sin cache in HuggingFace format.
    Returns cos, sin tensors of shape [1, 1, max_seq_len, head_dim].
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    if rope_scaling:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type not in (None, "default", "linear"):
            raise ValueError(f"rope_scaling {rope_type} is not supported in this bringup")
        if rope_type == "linear":
            inv_freq = inv_freq / rope_scaling["factor"]

    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin = emb.sin().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    return cos, sin


def resolve_max_seq_len(hf_config, max_seq_len: Optional[int]) -> int:
    """Resolve max sequence length from HF config when not provided."""
    text_config = getattr(hf_config, "text_config", hf_config)
    config_max = getattr(text_config, "max_position_embeddings", None)
    if max_seq_len is None:
        if config_max is None:
            raise ValueError("max_seq_len is required when config has no max_position_embeddings")
        return config_max
    if config_max is not None and max_seq_len > config_max:
        raise ValueError(f"max_seq_len {max_seq_len} exceeds config max {config_max}")
    return max_seq_len


class RMSNorm:
    """Gemma3 RMSNorm layer (scale is 1 + weight)."""

    def __init__(self, weight: torch.Tensor, eps: float, parallel: ParallelConfig):
        self.eps = eps
        scale = weight + 1.0
        self.weight = ttnn.as_tensor(
            scale.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parallel.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=parallel.replicate_mapper,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)


class MLP:
    """Gated MLP (gelu) for Gemma3, with 1D tensor parallel."""

    def __init__(self, layer_idx: int, state_dict: dict, parallel: ParallelConfig):
        p = f"language_model.model.layers.{layer_idx}.mlp."
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
        gate = ttnn.gelu(ttnn.linear(x, self.gate_proj))
        up = ttnn.linear(x, self.up_proj)
        out = ttnn.linear(ttnn.mul(gate, up), self.down_proj)
        return all_reduce_tensor(out, self.parallel)


class Attention:
    """Multi-head attention with Q/K RMSNorm and local/global RoPE."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache_global: ttnn.Tensor,
        sin_cache_global: ttnn.Tensor,
        cos_cache_local: ttnn.Tensor,
        sin_cache_local: ttnn.Tensor,
        parallel: ParallelConfig,
        paged_attention_config: PagedAttentionConfig,
        page_table: ttnn.Tensor,
    ):
        self.parallel = parallel
        self.n_heads = config.num_attention_heads
        self.original_kv_heads = config.num_key_value_heads
        self.n_kv_heads = self.original_kv_heads
        self.kv_repeat = 1
        if parallel.num_devices > self.n_kv_heads:
            if parallel.num_devices % self.n_kv_heads != 0:
                raise ValueError("num_devices must be a multiple of num_key_value_heads for KV padding")
            self.kv_repeat = parallel.num_devices // self.n_kv_heads
            self.n_kv_heads *= self.kv_repeat
        self.n_local_heads = self.n_heads // parallel.num_devices
        self.n_local_kv_heads = self.n_kv_heads // parallel.num_devices
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(config.query_pre_attn_scalar)
        self.paged_attention_config = paged_attention_config
        self.page_table = page_table
        if config.layer_types is not None:
            self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        else:
            pattern = config.sliding_window_pattern
            if isinstance(pattern, (list, tuple)):
                if layer_idx >= len(pattern):
                    raise ValueError("sliding_window_pattern is shorter than num_hidden_layers")
                self.is_sliding = bool(pattern[layer_idx])
            else:
                if not pattern:
                    pattern = 6
                self.is_sliding = bool((layer_idx + 1) % pattern)

        if config.attention_bias:
            raise ValueError("attention_bias=True is not supported in this bringup")

        self.cos_cache = cos_cache_local if self.is_sliding else cos_cache_global
        self.sin_cache = sin_cache_local if self.is_sliding else sin_cache_global

        p = f"language_model.model.layers.{layer_idx}.self_attn."
        self.q_proj = self._load_weight(state_dict[f"{p}q_proj.weight"], parallel.shard_width_mapper)
        self.k_proj = self._load_kv_weight(state_dict[f"{p}k_proj.weight"], parallel.shard_width_mapper)
        self.v_proj = self._load_kv_weight(state_dict[f"{p}v_proj.weight"], parallel.shard_width_mapper)
        self.o_proj = self._load_weight(state_dict[f"{p}o_proj.weight"], parallel.shard_height_mapper)
        self.q_norm = RMSNorm(state_dict[f"{p}q_norm.weight"], config.rms_norm_eps, parallel)
        self.k_norm = RMSNorm(state_dict[f"{p}k_norm.weight"], config.rms_norm_eps, parallel)

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

    def _load_kv_weight(self, w: torch.Tensor, mesh_mapper) -> ttnn.Tensor:
        if self.kv_repeat == 1:
            return self._load_weight(w, mesh_mapper)
        w = w.reshape(self.original_kv_heads, self.head_dim, w.shape[1])
        w = w.repeat_interleave(self.kv_repeat, dim=0)
        w = w.reshape(self.n_kv_heads * self.head_dim, w.shape[2])
        return self._load_weight(w, mesh_mapper)

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
        is_prefill = seq_len > 1
        padded_seq = pad_to_tile(seq_len)

        q = ttnn.linear(x, self.q_proj)
        k = ttnn.linear(x, self.k_proj)
        v = ttnn.linear(x, self.v_proj)
        qkv = ttnn.concat([q, k, v], dim=-1)

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

            q_mem = ttnn.get_memory_config(q)
            k_mem = ttnn.get_memory_config(k)
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = ttnn.to_memory_config(q, q_mem)
            k = ttnn.to_memory_config(k, k_mem)

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

            q_mem = ttnn.get_memory_config(q)
            k_mem = ttnn.get_memory_config(k)
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
            q = self.q_norm(q)
            k = self.k_norm(k)

            q_batch = q.shape[1]
            q_heads = q.shape[2]
            q_bh = q_batch * q_heads
            q_bh_padded = pad_to_tile(q_bh)
            q = ttnn.reshape(q, (1, 1, q_bh, self.head_dim), (1, 1, q_bh_padded, self.head_dim))
            if decode_cos is None or decode_sin is None:
                q = ttnn.experimental.rotary_embedding(q, self.cos_cache, self.sin_cache, start_pos)
            else:
                q = ttnn.experimental.rotary_embedding(q, decode_cos, decode_sin)
            q = ttnn.reshape(q, (1, q_batch, q_heads, self.head_dim), (1, q_batch, q_heads, self.head_dim))

            k_batch = k.shape[1]
            k_heads = k.shape[2]
            k_bh = k_batch * k_heads
            k_bh_padded = pad_to_tile(k_bh)
            k = ttnn.reshape(k, (1, 1, k_bh, self.head_dim), (1, 1, k_bh_padded, self.head_dim))
            if decode_cos is None or decode_sin is None:
                k = ttnn.experimental.rotary_embedding(k, self.cos_cache, self.sin_cache, start_pos)
            else:
                k = ttnn.experimental.rotary_embedding(k, decode_cos, decode_sin)
            k = ttnn.reshape(k, (1, k_batch, k_heads, self.head_dim), (1, k_batch, k_heads, self.head_dim))

            q = ttnn.to_memory_config(q, q_mem)
            k = ttnn.to_memory_config(k, k_mem)

            ttnn.experimental.paged_update_cache(
                self.k_cache, k, update_idxs_tensor=cur_pos_tensor, page_table=self.page_table
            )
            ttnn.experimental.paged_update_cache(
                self.v_cache, v, update_idxs_tensor=cur_pos_tensor, page_table=self.page_table
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


class DecoderLayer:
    """Single transformer layer."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache_global: ttnn.Tensor,
        sin_cache_global: ttnn.Tensor,
        cos_cache_local: ttnn.Tensor,
        sin_cache_local: ttnn.Tensor,
        parallel: ParallelConfig,
        paged_attention_config: PagedAttentionConfig,
        page_table: ttnn.Tensor,
    ):
        p = f"language_model.model.layers.{layer_idx}."
        self.attn_norm = RMSNorm(state_dict[f"{p}input_layernorm.weight"], config.rms_norm_eps, parallel)
        self.post_attn_norm = RMSNorm(state_dict[f"{p}post_attention_layernorm.weight"], config.rms_norm_eps, parallel)
        self.pre_ffn_norm = RMSNorm(state_dict[f"{p}pre_feedforward_layernorm.weight"], config.rms_norm_eps, parallel)
        self.post_ffn_norm = RMSNorm(
            state_dict[f"{p}post_feedforward_layernorm.weight"], config.rms_norm_eps, parallel
        )
        self.attn = Attention(
            config,
            layer_idx,
            state_dict,
            cos_cache_global,
            sin_cache_global,
            cos_cache_local,
            sin_cache_local,
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
        residual = x
        x = self.attn_norm(x)
        x = self.attn(
            x,
            start_pos,
            seq_len,
            cur_pos_tensor=cur_pos_tensor,
            decode_cos=decode_cos,
            decode_sin=decode_sin,
            trace_decode=trace_decode,
        )
        x = self.post_attn_norm(x)
        x = ttnn.add(residual, x)

        residual = x
        x = self.pre_ffn_norm(x)
        x = self.mlp(x)
        x = self.post_ffn_norm(x)
        x = ttnn.add(residual, x)
        return x


class TtnnGemma3ForCausalLM(torch.nn.Module, GenerationMixin):
    """
    Gemma 3 model with 100% ttnn execution and 1D tensor parallel on T3000.
    HuggingFace `generate()`-compatible via `GenerationMixin`.
    """

    def __init__(self, hf_model, tt_device, max_seq_len: Optional[int] = None):
        super().__init__()

        self.tt_device = tt_device
        self.hf_config = hf_model.config
        self.tt_config = ModelConfig.from_hf(hf_model.config)
        self.max_seq_len = resolve_max_seq_len(self.hf_config, max_seq_len)
        self._pos = 0

        if self.tt_config.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(f"hidden_activation {self.tt_config.hidden_activation} is not supported in this bringup")

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
        if "model.language_model.layers.0.self_attn.q_proj.weight" in state_dict:
            remapped = {}
            prefix = "model.language_model."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    remapped[f"language_model.model.{key[len(prefix):]}"] = value
            state_dict = {**state_dict, **remapped}

        print("  Loading embeddings...")
        embed_scale = torch.tensor(self.tt_config.hidden_size**0.5, dtype=torch.bfloat16)
        embed_weight = state_dict["language_model.model.embed_tokens.weight"].to(torch.bfloat16) * embed_scale
        self.embed = ttnn.as_tensor(
            embed_weight.unsqueeze(0).unsqueeze(0).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )

        print("  Computing RoPE cache...")
        cos_global, sin_global = compute_rope_cache(
            self.tt_config.head_dim,
            self.max_seq_len,
            rope_theta=self.tt_config.rope_theta,
            rope_scaling=self.tt_config.rope_scaling,
        )
        cos_local, sin_local = compute_rope_cache(
            self.tt_config.head_dim,
            self.max_seq_len,
            rope_theta=self.tt_config.rope_local_base_freq,
            rope_scaling=None,
        )
        self.cos_cache_global_host = cos_global
        self.sin_cache_global_host = sin_global
        self.cos_cache_local_host = cos_local
        self.sin_cache_local_host = sin_local
        self.cos_cache_global = ttnn.as_tensor(
            cos_global,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.sin_cache_global = ttnn.as_tensor(
            sin_global,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.cos_cache_local = ttnn.as_tensor(
            cos_local,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.sin_cache_local = ttnn.as_tensor(
            sin_local,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
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
        self.decode_rope_seq = (self.tt_config.num_attention_heads // self.parallel.num_devices) * TILE_SIZE
        self.decode_pos_buffer = ttnn.from_torch(
            torch.zeros((TILE_SIZE,), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        decode_rope_shape = (1, 1, self.decode_rope_seq, self.tt_config.head_dim)
        self.decode_cos_global_buffer = ttnn.from_torch(
            torch.zeros(decode_rope_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.decode_sin_global_buffer = ttnn.from_torch(
            torch.zeros(decode_rope_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.decode_cos_local_buffer = ttnn.from_torch(
            torch.zeros(decode_rope_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.decode_sin_local_buffer = ttnn.from_torch(
            torch.zeros(decode_rope_shape, dtype=torch.bfloat16),
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
                self.cos_cache_global,
                self.sin_cache_global,
                self.cos_cache_local,
                self.sin_cache_local,
                self.parallel,
                self.paged_attention_config,
                self.page_table,
            )
            for i in range(self.tt_config.num_hidden_layers)
        ]

        self.norm = RMSNorm(state_dict["language_model.model.norm.weight"], self.tt_config.rms_norm_eps, self.parallel)
        lm_head_weight = state_dict.get("language_model.lm_head.weight")
        if lm_head_weight is None:
            lm_head_weight = state_dict["language_model.model.embed_tokens.weight"]
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
        cos_global_token = self.cos_cache_global_host[:, :, start_pos : start_pos + 1, :]
        sin_global_token = self.sin_cache_global_host[:, :, start_pos : start_pos + 1, :]
        cos_local_token = self.cos_cache_local_host[:, :, start_pos : start_pos + 1, :]
        sin_local_token = self.sin_cache_local_host[:, :, start_pos : start_pos + 1, :]

        cos_global_slice = cos_global_token.repeat(1, 1, self.decode_rope_seq, 1)
        sin_global_slice = sin_global_token.repeat(1, 1, self.decode_rope_seq, 1)
        cos_local_slice = cos_local_token.repeat(1, 1, self.decode_rope_seq, 1)
        sin_local_slice = sin_local_token.repeat(1, 1, self.decode_rope_seq, 1)

        host_cos_global = ttnn.from_torch(
            cos_global_slice,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        host_sin_global = ttnn.from_torch(
            sin_global_slice,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        host_cos_local = ttnn.from_torch(
            cos_local_slice,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        host_sin_local = ttnn.from_torch(
            sin_local_slice,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_cos_global, self.decode_cos_global_buffer)
        ttnn.copy_host_to_device_tensor(host_sin_global, self.decode_sin_global_buffer)
        ttnn.copy_host_to_device_tensor(host_cos_local, self.decode_cos_local_buffer)
        ttnn.copy_host_to_device_tensor(host_sin_local, self.decode_sin_local_buffer)

    def _forward_decode_device(self, start_pos: int, trace_decode: bool) -> ttnn.Tensor:
        h = ttnn.embedding(self.decode_token_buffer, self.embed, layout=ttnn.TILE_LAYOUT)
        for layer in self.layers:
            decode_cos = self.decode_cos_local_buffer if layer.attn.is_sliding else self.decode_cos_global_buffer
            decode_sin = self.decode_sin_local_buffer if layer.attn.is_sliding else self.decode_sin_global_buffer
            h = layer(
                h,
                start_pos,
                1,
                cur_pos_tensor=self.decode_pos_buffer,
                decode_cos=decode_cos,
                decode_sin=decode_sin,
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
            raise ValueError(f"sequence length {start_pos + seq_len} exceeds max_seq_len {self.max_seq_len}")

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
        logits, seq_len, padded_seq, past = self._forward_device_logits(input_ids, past_key_values, use_cache)
        logits_torch = self._logits_to_torch(logits).reshape(batch, padded_seq, -1)[:, :seq_len, :]

        if self.tt_config.final_logit_softcapping is not None:
            softcap = self.tt_config.final_logit_softcapping
            logits_torch = torch.tanh(logits_torch / softcap) * softcap

        if seq_len > 1 or not self.use_decode_trace:
            ttnn.deallocate(logits)

        return CausalLMOutputWithPast(
            logits=logits_torch.float(),
            past_key_values=past,
        )

    def prefill_logits_last_device(self, input_ids: torch.Tensor, use_cache: bool = True) -> tuple[torch.Tensor, object]:
        batch, seq_len = input_ids.shape
        if batch != 1:
            raise ValueError("Only batch=1 supported")

        self.reset()
        start_pos = self._pos
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {start_pos + seq_len} exceeds max_seq_len {self.max_seq_len}")

        padded_seq = pad_to_tile(seq_len)
        if seq_len < padded_seq:
            input_ids = torch.nn.functional.pad(input_ids, (0, padded_seq - seq_len), value=0)

        logits = self._forward_prefill_last_logits(input_ids, start_pos, seq_len)
        self._pos = start_pos + seq_len

        logits_torch = self._logits_to_torch(logits).reshape(batch, 1, -1)[:, 0, :]
        if self.tt_config.final_logit_softcapping is not None:
            softcap = self.tt_config.final_logit_softcapping
            logits_torch = torch.tanh(logits_torch / softcap) * softcap
        logits_torch = logits_torch.float()
        ttnn.deallocate(logits)

        past = self._tt_past_key_values if use_cache else None
        return logits_torch, past


def build_model(hf_model, tt_device, max_seq_len: Optional[int] = None) -> TtnnGemma3ForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnGemma3ForCausalLM(hf_model, tt_device, max_seq_len)
