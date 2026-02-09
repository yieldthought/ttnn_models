# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Arcee AFM-4.5B implementation in ttnn for T3000.

This version uses 1D tensor parallel across a 2x4 mesh (flattened across 8 devices):
- QKV and up projections are column-parallel (width sharded).
- Output and down projections are row-parallel (height sharded) followed by all-reduce.

This file defines the ttnn model only. Use `eval.py` at repo root for
teacher-forcing accuracy checks against the HuggingFace reference.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import ttnn
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


TILE_SIZE = 32
MESH_SHAPE = (2, 4)
MESH_TOPOLOGY = ttnn.Topology.Linear
MESH_NUM_LINKS = 1
WEIGHT_DTYPE = ttnn.bfloat8_b
WEIGHT_LAYOUT = ttnn.TILE_LAYOUT
MAX_CACHE_SEQ_LEN = 256


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
    max_position_embeddings: int
    hidden_act: str

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
            hf_config.max_position_embeddings,
            hf_config.hidden_act,
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


def validate_parallel_config(config: ModelConfig, num_devices: int) -> None:
    if num_devices != 8:
        raise ValueError("T3000 model expects an 8-device mesh")
    if config.hidden_size % num_devices != 0:
        raise ValueError("hidden_size must divide evenly across devices")
    if config.intermediate_size % num_devices != 0:
        raise ValueError("intermediate_size must divide evenly across devices")


def padded_head_counts(config: ModelConfig, num_devices: int) -> tuple[int, int]:
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
    head_ratio = config.num_attention_heads // config.num_key_value_heads
    padded_kv = ((config.num_key_value_heads + num_devices - 1) // num_devices) * num_devices
    padded_heads = head_ratio * padded_kv
    return padded_heads, padded_kv


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


def compute_rope_cache(config: ModelConfig, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin cache using HF rope utils (supports Yarn)."""
    rope_type = "default"
    if config.rope_scaling:
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
    rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
    inv_freq, attention_scaling = rope_init_fn(config, device=torch.device("cpu"), seq_len=max_seq_len)

    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling

    cos = cos.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin = sin.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    return cos, sin


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
        max_seq_len: int,
        padded_num_heads: int,
        padded_num_kv_heads: int,
    ):
        self.parallel = parallel
        self.n_heads = padded_num_heads
        self.n_kv_heads = padded_num_kv_heads
        self.n_heads_real = config.num_attention_heads
        self.n_kv_heads_real = config.num_key_value_heads
        self.n_local_heads = self.n_heads // parallel.num_devices
        self.n_local_kv_heads = self.n_kv_heads // parallel.num_devices
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

        p = f"model.layers.{layer_idx}.self_attn."
        q_weight = self._pad_out_features(state_dict[f"{p}q_proj.weight"], self.n_heads * self.head_dim)
        k_weight = self._pad_out_features(state_dict[f"{p}k_proj.weight"], self.n_kv_heads * self.head_dim)
        v_weight = self._pad_out_features(state_dict[f"{p}v_proj.weight"], self.n_kv_heads * self.head_dim)
        o_weight = self._pad_in_features(state_dict[f"{p}o_proj.weight"], self.n_heads * self.head_dim)

        self.q_proj = self._load_weight(q_weight, parallel.shard_width_mapper)
        self.k_proj = self._load_weight(k_weight, parallel.shard_width_mapper)
        self.v_proj = self._load_weight(v_weight, parallel.shard_width_mapper)
        self.o_proj = self._load_weight(o_weight, parallel.shard_height_mapper)

        cache_shape = (TILE_SIZE, self.n_kv_heads, max_seq_len, self.head_dim)
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

    def _pad_out_features(self, w: torch.Tensor, padded_out: int) -> torch.Tensor:
        if w.shape[0] > padded_out:
            raise ValueError("padded_out must be >= weight out features")
        if w.shape[0] == padded_out:
            return w
        pad_rows = padded_out - w.shape[0]
        return torch.nn.functional.pad(w, (0, 0, 0, pad_rows))

    def _pad_in_features(self, w: torch.Tensor, padded_in: int) -> torch.Tensor:
        if w.shape[1] > padded_in:
            raise ValueError("padded_in must be >= weight in features")
        if w.shape[1] == padded_in:
            return w
        pad_cols = padded_in - w.shape[1]
        return torch.nn.functional.pad(w, (0, pad_cols))

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Forward pass for prefill (seq_len > 1) or decode (seq_len == 1)."""
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

            cos = self.cos_cache[:, :, :padded_seq, :]
            sin = self.sin_cache[:, :, :padded_seq, :]
            q = ttnn.experimental.rotary_embedding(q, cos, sin)
            k = ttnn.experimental.rotary_embedding(k, cos, sin)

            grid = self.parallel.mesh_device.core_grid
            grid_x = min(grid.x, num_kv_heads)
            while grid_x > 1 and num_kv_heads % grid_x != 0:
                grid_x -= 1
            if num_kv_heads % grid_x != 0:
                raise ValueError("num_kv_heads must divide evenly across the shard grid")
            shard_grid = ttnn.CoreGrid(x=grid_x, y=num_kv_heads // grid_x)
            if shard_grid.y > grid.y:
                raise ValueError("shard grid exceeds device core grid")
            shard_mem_config = ttnn.create_sharded_memory_config(
                k.shape,
                shard_grid,
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            k_sharded = ttnn.to_memory_config(k, shard_mem_config)
            v_sharded = ttnn.to_memory_config(v, shard_mem_config)
            ttnn.fill_cache(self.k_cache, k_sharded, batch_idx=0)
            ttnn.fill_cache(self.v_cache, v_sharded, batch_idx=0)
            ttnn.deallocate(k_sharded)
            ttnn.deallocate(v_sharded)

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
            ttnn.deallocate(qkv)

            q = ttnn.reshape(q, (1, 1, q.shape[1] * num_heads, self.head_dim))
            q = ttnn.experimental.rotary_embedding(q, self.cos_cache, self.sin_cache, start_pos)
            q = ttnn.reshape(q, (1, q.shape[2] // num_heads, num_heads, self.head_dim))

            k = ttnn.reshape(k, (1, 1, k.shape[1] * num_kv_heads, self.head_dim))
            k = ttnn.experimental.rotary_embedding(k, self.cos_cache, self.sin_cache, start_pos)
            k = ttnn.reshape(k, (1, k.shape[2] // num_kv_heads, num_kv_heads, self.head_dim))

            ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=cur_pos_tensor)
            ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=cur_pos_tensor)

            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q, self.k_cache, self.v_cache, cur_pos_tensor=cur_pos_tensor, scale=self.scale
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
    """Arcee MLP with relu2 activation, 1D tensor parallel."""

    def __init__(self, layer_idx: int, hidden_act: str, state_dict: dict, parallel: ParallelConfig):
        self.hidden_act = hidden_act
        self.parallel = parallel
        p = f"model.layers.{layer_idx}.mlp."
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

    def _act(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.hidden_act != "relu2":
            raise ValueError(f"Unsupported activation: {self.hidden_act}")
        relu = ttnn.relu(x)
        return ttnn.mul(relu, relu)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        up = ttnn.linear(x, self.up_proj)
        out = ttnn.linear(self._act(up), self.down_proj)
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
        max_seq_len: int,
        padded_num_heads: int,
        padded_num_kv_heads: int,
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
            max_seq_len,
            padded_num_heads,
            padded_num_kv_heads,
        )
        self.mlp = MLP(layer_idx, config.hidden_act, state_dict, parallel)

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        x = ttnn.add(x, self.attn(self.attn_norm(x), start_pos, seq_len, cur_pos_tensor=cur_pos_tensor))
        x = ttnn.add(x, self.mlp(self.ffn_norm(x)))
        return x


class TtnnArceeForCausalLM(torch.nn.Module, GenerationMixin):
    """
    Arcee model with 100% ttnn execution and 1D tensor parallel on T3000.
    HuggingFace `generate()`-compatible via `GenerationMixin`.
    """

    def __init__(self, hf_model, tt_device, max_seq_len: Optional[int] = None):
        super().__init__()

        self.tt_device = tt_device
        self.hf_config = hf_model.config
        self.tt_config = ModelConfig.from_hf(hf_model.config)
        if max_seq_len is None:
            max_seq_len = self.tt_config.max_position_embeddings
        self.max_seq_len = max_seq_len
        self.cache_seq_len = min(max_seq_len, MAX_CACHE_SEQ_LEN)
        self._pos = 0
        self.vocab_size = self.tt_config.vocab_size

        if self.tt_config.hidden_act != "relu2":
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
        validate_parallel_config(self.tt_config, num_devices)

        padded_num_heads, padded_num_kv_heads = padded_head_counts(self.tt_config, num_devices)
        self.padded_num_heads = padded_num_heads
        self.padded_num_kv_heads = padded_num_kv_heads

        mesh_axis = mesh_shape_to_axis(mesh_shape)

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
        self.tt_device = tt_device
        padded_vocab_size = ((self.vocab_size + self.parallel.num_devices - 1) // self.parallel.num_devices) * self.parallel.num_devices
        self.padded_vocab_size = padded_vocab_size

        state_dict = hf_model.state_dict()

        print("  Loading embeddings...")
        self.embed = ttnn.as_tensor(
            state_dict["model.embed_tokens.weight"].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )

        print("  Computing RoPE cache...")
        cos, sin = compute_rope_cache(self.tt_config, self.cache_seq_len)
        self.cos_cache = ttnn.as_tensor(
            cos,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.parallel.replicate_mapper,
        )
        self.sin_cache = ttnn.as_tensor(
            sin,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
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
                self.cache_seq_len,
                self.padded_num_heads,
                self.padded_num_kv_heads,
            )
            for i in range(self.tt_config.num_hidden_layers)
        ]

        self.norm = RMSNorm(state_dict["model.norm.weight"], self.tt_config.rms_norm_eps, self.parallel)
        lm_head_weight = state_dict.get("lm_head.weight", state_dict["model.embed_tokens.weight"])
        if self.padded_vocab_size != self.vocab_size:
            pad_rows = self.padded_vocab_size - self.vocab_size
            lm_head_weight = torch.nn.functional.pad(lm_head_weight, (0, 0, 0, pad_rows))
        self.lm_head = ttnn.as_tensor(
            lm_head_weight.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=self.tt_device,
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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values

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
        if batch != 1:
            raise ValueError("Only batch=1 supported")

        if past_key_values is None:
            self.reset()
        else:
            if seq_len != 1:
                raise ValueError("Only 1-token decode supported when using cache")

        start_pos = self._pos
        if start_pos + seq_len > self.cache_seq_len:
            raise ValueError(
                f"sequence length {start_pos + seq_len} exceeds cache length {self.cache_seq_len}; "
                "increase MAX_CACHE_SEQ_LEN if memory allows"
            )

        cur_pos_tensor = None
        if seq_len == 1:
            cur_pos_tensor = ttnn.from_torch(
                torch.full((TILE_SIZE,), start_pos, dtype=torch.int32),
                dtype=ttnn.int32,
                device=self.tt_device,
                mesh_mapper=self.parallel.replicate_mapper,
            )

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
            h = layer(h, start_pos, seq_len, cur_pos_tensor=cur_pos_tensor)

        h = self.norm(h)
        logits = ttnn.linear(h, self.lm_head)

        if self.parallel.num_devices > 1:
            logits = ttnn.to_torch(logits, mesh_composer=self.parallel.vocab_composer)
        else:
            logits = ttnn.to_torch(logits)
        logits = logits.reshape(batch, padded_seq, -1)[:, :seq_len, :self.vocab_size]

        self._pos = start_pos + seq_len

        return CausalLMOutputWithPast(
            logits=logits.float(),
            past_key_values=(self._tt_past_key_values if use_cache else None),
        )


def build_model(hf_model, tt_device, max_seq_len: Optional[int] = None) -> TtnnArceeForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnArceeForCausalLM(hf_model, tt_device, max_seq_len)
