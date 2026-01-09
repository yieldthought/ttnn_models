# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Simple Phi-3 Mini 128k Instruct implementation in ttnn with device-only compute.

This mirrors the HuggingFace architecture with a fused QKV projection and
gated MLP (gate_up_proj + down_proj).
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
WEIGHT_DTYPE = ttnn.bfloat16
WEIGHT_LAYOUT = ttnn.TILE_LAYOUT
MAX_CACHE_SEQ_LEN = 256


def pad_to_tile(x: int) -> int:
    """Pad to tile boundary (32)."""
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def pad_head_dim(x: int) -> int:
    """Pad head dimension to the rotary tile requirement (64)."""
    return ((x + HEAD_DIM_TILE - 1) // HEAD_DIM_TILE) * HEAD_DIM_TILE


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


def compute_attention_scaling(config: ModelConfig) -> float:
    factor = config.max_position_embeddings / config.original_max_position_embeddings
    if factor <= 1.0:
        return 1.0
    return math.sqrt(1 + math.log(factor) / math.log(config.original_max_position_embeddings))


def compute_rope_cache(config: ModelConfig, max_seq_len: int) -> tuple:
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
        if max_seq_len > config.original_max_position_embeddings:
            ext_factors = torch.tensor(long_factor, dtype=torch.float32)
        else:
            ext_factors = torch.tensor(short_factor, dtype=torch.float32)

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


class RMSNorm:
    """RMSNorm layer."""

    def __init__(self, weight: torch.Tensor, eps: float, tt_device):
        self.eps = eps
        self.weight = ttnn.as_tensor(
            weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)


class Attention:
    """Multi-head attention with a fused QKV projection."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        tt_device,
        max_seq_len: int,
    ):
        self.tt_device = tt_device
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.head_dim_padded = pad_head_dim(self.head_dim)
        self.head_dim_half = self.head_dim // 2
        self.head_dim_half_padded = self.head_dim_padded // 2
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

        p = f"model.layers.{layer_idx}.self_attn."
        self.qkv_proj = self._load_weight(state_dict[f"{p}qkv_proj.weight"])
        self.o_proj = self._load_weight(state_dict[f"{p}o_proj.weight"])

        cache_shape = (TILE_SIZE, self.n_kv_heads, max_seq_len, self.head_dim)
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
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _pad_head_dim(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.head_dim_padded == self.head_dim:
            return x
        pad = self.head_dim_half_padded - self.head_dim_half
        pad_tile = pad_to_tile(pad)
        zeros = ttnn.zeros(
            (x.shape[0], x.shape[1], x.shape[2], pad_tile),
            dtype=x.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
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
    ) -> ttnn.Tensor:
        is_prefill = seq_len > 1
        padded_seq = pad_to_tile(seq_len)

        qkv = ttnn.linear(x, self.qkv_proj)

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

            cos = self.cos_cache[:, :, :padded_seq, :]
            sin = self.sin_cache[:, :, :padded_seq, :]
            q = self._slice_head_dim(ttnn.experimental.rotary_embedding(self._pad_head_dim(q), cos, sin))
            k = self._slice_head_dim(ttnn.experimental.rotary_embedding(self._pad_head_dim(k), cos, sin))

            q = ttnn.to_memory_config(q, q_mem)
            k = ttnn.to_memory_config(k, k_mem)

            # Shard KV for fill_cache to avoid interleaved grid-size limits at long prefill lengths.
            grid = self.tt_device.core_grid
            if self.n_kv_heads % grid.x != 0:
                raise ValueError("n_kv_heads must be divisible by device grid.x for sharded fill_cache")
            shard_grid = ttnn.CoreGrid(x=grid.x, y=self.n_kv_heads // grid.x)
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
                num_heads=self.n_heads,
                num_kv_heads=self.n_kv_heads,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)

            q_mem = ttnn.get_memory_config(q)
            k_mem = ttnn.get_memory_config(k)
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)

            q = ttnn.reshape(q, (1, 1, q.shape[1] * self.n_heads, self.head_dim))
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            q = self._slice_head_dim(
                ttnn.experimental.rotary_embedding(
                    self._pad_head_dim(q), self.cos_cache, self.sin_cache, start_pos
                )
            )
            q = ttnn.reshape(q, (1, q.shape[2] // self.n_heads, self.n_heads, self.head_dim))

            k = ttnn.reshape(k, (1, 1, k.shape[1] * self.n_kv_heads, self.head_dim))
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
            k = self._slice_head_dim(
                ttnn.experimental.rotary_embedding(
                    self._pad_head_dim(k), self.cos_cache, self.sin_cache, start_pos
                )
            )
            k = ttnn.reshape(k, (1, k.shape[2] // self.n_kv_heads, self.n_kv_heads, self.head_dim))

            q = ttnn.to_memory_config(q, q_mem)
            k = ttnn.to_memory_config(k, k_mem)

            ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=cur_pos_tensor)
            ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=cur_pos_tensor)

            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q, self.k_cache, self.v_cache, cur_pos_tensor=cur_pos_tensor, scale=self.scale
            )
            attn_out = ttnn.transpose(attn_out, 1, 2)
            attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        expected_width = self.n_heads * self.head_dim
        if attn_out.shape[-1] != expected_width:
            attn_out = ttnn.slice(
                attn_out,
                (0, 0, 0, 0),
                (attn_out.shape[0], attn_out.shape[1], attn_out.shape[2], expected_width),
            )

        return ttnn.linear(attn_out, self.o_proj)


class MLP:
    """Gated MLP using a single gate_up projection."""

    def __init__(self, layer_idx: int, state_dict: dict, tt_device):
        p = f"model.layers.{layer_idx}.mlp."
        self.gate_up_proj = self._load_weight(state_dict[f"{p}gate_up_proj.weight"], tt_device)
        self.down_proj = self._load_weight(state_dict[f"{p}down_proj.weight"], tt_device)

    def _load_weight(self, w: torch.Tensor, tt_device) -> ttnn.Tensor:
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate_up = ttnn.linear(x, self.gate_up_proj)
        half = gate_up.shape[-1] // 2
        gate = ttnn.slice(
            gate_up,
            (0, 0, 0, 0),
            (gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], half),
        )
        up = ttnn.slice(
            gate_up,
            (0, 0, 0, half),
            (gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], half * 2),
        )
        gate = ttnn.silu(gate)
        return ttnn.linear(ttnn.mul(gate, up), self.down_proj)


class DecoderLayer:
    """Single transformer layer."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        tt_device,
        max_seq_len: int,
    ):
        p = f"model.layers.{layer_idx}."
        self.attn_norm = RMSNorm(state_dict[f"{p}input_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.ffn_norm = RMSNorm(state_dict[f"{p}post_attention_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.attn = Attention(config, layer_idx, state_dict, cos_cache, sin_cache, tt_device, max_seq_len)
        self.mlp = MLP(layer_idx, state_dict, tt_device)

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


class TtnnPhi3ForCausalLM(torch.nn.Module, GenerationMixin):
    """
    Phi-3 model with 100% ttnn execution.
    HuggingFace `generate()`-compatible via `GenerationMixin`.
    """

    def __init__(self, hf_model, tt_device, max_seq_len: int = 2048):
        super().__init__()

        self.tt_device = tt_device
        self.hf_config = hf_model.config
        self.tt_config = ModelConfig.from_hf(hf_model.config)
        self.max_seq_len = max_seq_len
        self.cache_seq_len = min(max_seq_len, MAX_CACHE_SEQ_LEN)
        self._pos = 0

        if self.tt_config.hidden_act != "silu":
            raise ValueError(f"hidden_act {self.tt_config.hidden_act} is not supported in this bringup")

        self.config = self.hf_config
        self.generation_config = GenerationConfig.from_model_config(self.config)
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = self.generation_config.eos_token_id
        self._supports_cache_class = False
        self.main_input_name = "input_ids"
        self.register_buffer("_torch_dummy", torch.empty(0, dtype=torch.float32), persistent=False)

        state_dict = hf_model.state_dict()

        print("  Loading embeddings...")
        self.embed = ttnn.as_tensor(
            state_dict["model.embed_tokens.weight"].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print("  Computing RoPE cache...")
        cos, sin = compute_rope_cache(self.tt_config, max_seq_len)
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

        print(f"  Loading {self.tt_config.num_hidden_layers} layers...")
        self.layers = [
            DecoderLayer(
                self.tt_config, i, state_dict, self.cos_cache, self.sin_cache, tt_device, self.cache_seq_len
            )
            for i in range(self.tt_config.num_hidden_layers)
        ]

        self.norm = RMSNorm(state_dict["model.norm.weight"], self.tt_config.rms_norm_eps, tt_device)
        lm_head_weight = state_dict.get("lm_head.weight", state_dict["model.embed_tokens.weight"])
        self.lm_head = ttnn.as_tensor(
            lm_head_weight.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        batch, seq_len = input_ids.shape
        assert batch == 1, "Only batch=1 supported"

        if past_key_values is None:
            self.reset()
        else:
            assert seq_len == 1, "Only 1-token decode supported when using cache"

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
            )

        padded_seq = pad_to_tile(seq_len)
        if seq_len < padded_seq:
            input_ids = torch.nn.functional.pad(input_ids, (0, padded_seq - seq_len), value=0)

        tokens = ttnn.from_torch(
            input_ids.reshape(1, 1, 1, -1),
            dtype=ttnn.uint32,
            device=self.tt_device,
        )
        h = ttnn.embedding(tokens, self.embed, layout=ttnn.TILE_LAYOUT)

        for layer in self.layers:
            h = layer(h, start_pos, seq_len, cur_pos_tensor=cur_pos_tensor)

        h = self.norm(h)
        logits = ttnn.linear(h, self.lm_head)

        logits = ttnn.to_torch(logits).reshape(batch, padded_seq, -1)[:, :seq_len, :]

        self._pos = start_pos + seq_len

        return CausalLMOutputWithPast(
            logits=logits.float(),
            past_key_values=(self._tt_past_key_values if use_cache else None),
        )


def build_model(hf_model, tt_device, max_seq_len: int = 2048) -> TtnnPhi3ForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnPhi3ForCausalLM(hf_model, tt_device, max_seq_len)
