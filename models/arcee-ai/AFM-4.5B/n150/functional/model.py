# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Minimal Arcee AFM-4.5B implementation in ttnn with full device execution.

This mirrors the HuggingFace Arcee architecture:
- RMSNorm
- GQA attention with RoPE (Yarn scaling)
- MLP with ReLU^2 activation (relu2)

Use eval.py at repo root for teacher-forcing accuracy checks.
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
MAX_CACHE_SEQ_LEN = 256
WEIGHT_DTYPE = ttnn.bfloat8_b
WEIGHT_LAYOUT = ttnn.TILE_LAYOUT


def pad_to_tile(x: int) -> int:
    """Pad to tile boundary (32)."""
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


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
        return cls(
            hf_config.vocab_size,
            hf_config.hidden_size,
            hf_config.intermediate_size,
            hf_config.num_hidden_layers,
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
            hf_config.rms_norm_eps,
            hf_config.rope_theta,
            hf_config.rope_scaling,
            hf_config.max_position_embeddings,
            hf_config.hidden_act,
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


class Attention:
    """Multi-head attention with GQA support, fully on ttnn."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        tt_device,
        cache_seq_len: int,
    ):
        self.tt_device = tt_device
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

        p = f"model.layers.{layer_idx}.self_attn."
        self.q_proj = self._load_weight(state_dict[f"{p}q_proj.weight"])
        self.k_proj = self._load_weight(state_dict[f"{p}k_proj.weight"])
        self.v_proj = self._load_weight(state_dict[f"{p}v_proj.weight"])
        self.o_proj = self._load_weight(state_dict[f"{p}o_proj.weight"])

        cache_shape = (TILE_SIZE, self.n_kv_heads, cache_seq_len, self.head_dim)
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
        """Load weight transposed for ttnn.linear: [out, in] -> [1, 1, in, out]."""
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

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

        if is_prefill:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=self.n_heads,
                num_kv_heads=self.n_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)

            cos = self.cos_cache[:, :, :padded_seq, :]
            sin = self.sin_cache[:, :, :padded_seq, :]
            q = ttnn.experimental.rotary_embedding(q, cos, sin)
            k = ttnn.experimental.rotary_embedding(k, cos, sin)

            ttnn.fill_cache(self.k_cache, k, batch_idx=0)
            ttnn.fill_cache(self.v_cache, v, batch_idx=0)

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

            q = ttnn.reshape(q, (1, 1, q.shape[1] * self.n_heads, self.head_dim))
            q = ttnn.experimental.rotary_embedding(q, self.cos_cache, self.sin_cache, start_pos)
            q = ttnn.reshape(q, (1, q.shape[2] // self.n_heads, self.n_heads, self.head_dim))

            k = ttnn.reshape(k, (1, 1, k.shape[1] * self.n_kv_heads, self.head_dim))
            k = ttnn.experimental.rotary_embedding(k, self.cos_cache, self.sin_cache, start_pos)
            k = ttnn.reshape(k, (1, k.shape[2] // self.n_kv_heads, self.n_kv_heads, self.head_dim))

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
    """Arcee MLP with relu2 activation, fully on ttnn."""

    def __init__(self, layer_idx: int, hidden_act: str, state_dict: dict, tt_device):
        self.hidden_act = hidden_act
        p = f"model.layers.{layer_idx}.mlp."
        self.up_proj = self._load_weight(state_dict[f"{p}up_proj.weight"], tt_device)
        self.down_proj = self._load_weight(state_dict[f"{p}down_proj.weight"], tt_device)

    def _load_weight(self, w: torch.Tensor, tt_device) -> ttnn.Tensor:
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _act(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.hidden_act != "relu2":
            raise ValueError(f"Unsupported activation: {self.hidden_act}")
        relu = ttnn.relu(x)
        return ttnn.mul(relu, relu)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        up = ttnn.linear(x, self.up_proj)
        return ttnn.linear(self._act(up), self.down_proj)


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
        cache_seq_len: int,
    ):
        p = f"model.layers.{layer_idx}."
        self.attn_norm = RMSNorm(state_dict[f"{p}input_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.ffn_norm = RMSNorm(state_dict[f"{p}post_attention_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.attn = Attention(config, layer_idx, state_dict, cos_cache, sin_cache, tt_device, cache_seq_len)
        self.mlp = MLP(layer_idx, config.hidden_act, state_dict, tt_device)

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        x = ttnn.add(x, self.attn(self.attn_norm(x), start_pos, seq_len, cur_pos_tensor))
        x = ttnn.add(x, self.mlp(self.ffn_norm(x)))
        return x


class TtnnArceeForCausalLM(torch.nn.Module, GenerationMixin):
    """Arcee model with 100% ttnn execution."""

    def __init__(self, hf_model, tt_device, max_seq_len: int = 2048):
        super().__init__()

        self.tt_device = tt_device
        self.hf_config = hf_model.config
        self.tt_config = ModelConfig.from_hf(hf_model.config)
        self.max_seq_len = max_seq_len
        self.cache_seq_len = min(max_seq_len, MAX_CACHE_SEQ_LEN)
        self._pos = 0

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
        cos, sin = compute_rope_cache(self.tt_config, self.cache_seq_len)
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
        self.lm_head = ttnn.as_tensor(
            state_dict["lm_head.weight"].T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
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
        """Reset position counter for a new sequence."""
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
        if batch != 1:
            raise ValueError("Only batch=1 supported")

        if past_key_values is None:
            self.reset()
        elif seq_len != 1:
            raise ValueError("Only 1-token decode supported when using cache")

        start_pos = self._pos
        if start_pos + seq_len > self.cache_seq_len:
            raise ValueError(
                f"Sequence length {start_pos + seq_len} exceeds cache size {self.cache_seq_len}; "
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
            h = layer(h, start_pos, seq_len, cur_pos_tensor)

        h = self.norm(h)
        logits = ttnn.linear(h, self.lm_head)

        logits = ttnn.to_torch(logits).reshape(batch, padded_seq, -1)[:, :seq_len, :]
        self._pos = start_pos + seq_len
        return CausalLMOutputWithPast(
            logits=logits.float(),
            past_key_values=(self._tt_past_key_values if use_cache else None),
        )


def build_model(hf_model, tt_device, max_seq_len: int = 2048) -> TtnnArceeForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnArceeForCausalLM(hf_model, tt_device, max_seq_len)
