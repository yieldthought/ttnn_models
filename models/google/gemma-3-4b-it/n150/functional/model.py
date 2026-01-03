# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Minimal Gemma 3 4B Instruct implementation in ttnn with device-only compute.
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
MAX_CACHE_SEQ_LEN = 128


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
    rope_local_base_freq: float
    hidden_activation: str
    attention_bias: bool
    query_pre_attn_scalar: float
    sliding_window: int
    sliding_window_pattern: int
    tie_word_embeddings: bool
    final_logit_softcapping: Optional[float]

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        text_config = getattr(hf_config, "text_config", hf_config)
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
            text_config.sliding_window_pattern,
            text_config.tie_word_embeddings,
            getattr(text_config, "final_logit_softcapping", None),
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


class RMSNorm:
    """Gemma3 RMSNorm layer (scale is 1 + weight)."""

    def __init__(self, weight: torch.Tensor, eps: float, tt_device):
        self.eps = eps
        scale = weight + 1.0
        self.weight = ttnn.as_tensor(
            scale.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)


class MLP:
    """Gated MLP (gelu) for Gemma3."""

    def __init__(self, layer_idx: int, state_dict: dict, tt_device):
        p = f"language_model.model.layers.{layer_idx}.mlp."
        self.gate_proj = self._load_weight(state_dict[f"{p}gate_proj.weight"], tt_device)
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

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.gelu(ttnn.linear(x, self.gate_proj))
        up = ttnn.linear(x, self.up_proj)
        return ttnn.linear(ttnn.mul(gate, up), self.down_proj)


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
        tt_device,
        max_seq_len: int,
    ):
        self.tt_device = tt_device
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(config.query_pre_attn_scalar)
        self.is_sliding = bool((layer_idx + 1) % config.sliding_window_pattern)

        if config.attention_bias:
            raise ValueError("attention_bias=True is not supported in this bringup")

        self.cos_cache = cos_cache_local if self.is_sliding else cos_cache_global
        self.sin_cache = sin_cache_local if self.is_sliding else sin_cache_global

        p = f"language_model.model.layers.{layer_idx}.self_attn."
        self.q_proj = self._load_weight(state_dict[f"{p}q_proj.weight"], tt_device)
        self.k_proj = self._load_weight(state_dict[f"{p}k_proj.weight"], tt_device)
        self.v_proj = self._load_weight(state_dict[f"{p}v_proj.weight"], tt_device)
        self.o_proj = self._load_weight(state_dict[f"{p}o_proj.weight"], tt_device)
        self.q_norm = RMSNorm(state_dict[f"{p}q_norm.weight"], config.rms_norm_eps, tt_device)
        self.k_norm = RMSNorm(state_dict[f"{p}k_norm.weight"], config.rms_norm_eps, tt_device)

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

    def _load_weight(self, w: torch.Tensor, tt_device) -> ttnn.Tensor:
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
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

            q_mem = ttnn.get_memory_config(q)
            k_mem = ttnn.get_memory_config(k)
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
            q = self.q_norm(q)
            k = self.k_norm(k)

            q = ttnn.reshape(q, (1, 1, q.shape[1] * self.n_heads, self.head_dim))
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            q = ttnn.experimental.rotary_embedding(q, self.cos_cache, self.sin_cache, start_pos)
            q = ttnn.reshape(q, (1, q.shape[2] // self.n_heads, self.n_heads, self.head_dim))

            k = ttnn.reshape(k, (1, 1, k.shape[1] * self.n_kv_heads, self.head_dim))
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.experimental.rotary_embedding(k, self.cos_cache, self.sin_cache, start_pos)
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
        tt_device,
        max_seq_len: int,
    ):
        p = f"language_model.model.layers.{layer_idx}."
        self.attn_norm = RMSNorm(state_dict[f"{p}input_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.post_attn_norm = RMSNorm(state_dict[f"{p}post_attention_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.pre_ffn_norm = RMSNorm(state_dict[f"{p}pre_feedforward_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.post_ffn_norm = RMSNorm(
            state_dict[f"{p}post_feedforward_layernorm.weight"], config.rms_norm_eps, tt_device
        )
        self.attn = Attention(
            config,
            layer_idx,
            state_dict,
            cos_cache_global,
            sin_cache_global,
            cos_cache_local,
            sin_cache_local,
            tt_device,
            max_seq_len,
        )
        self.mlp = MLP(layer_idx, state_dict, tt_device)

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, seq_len, cur_pos_tensor=cur_pos_tensor)
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
    Gemma 3 model with 100% ttnn execution.
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

        if self.tt_config.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(f"hidden_activation {self.tt_config.hidden_activation} is not supported in this bringup")

        self.config = self.hf_config
        self.generation_config = GenerationConfig.from_model_config(self.config)
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = self.generation_config.eos_token_id
        self._supports_cache_class = False
        self.main_input_name = "input_ids"
        self.register_buffer("_torch_dummy", torch.empty(0, dtype=torch.float32), persistent=False)

        state_dict = hf_model.state_dict()

        print("  Loading embeddings...")
        embed_scale = torch.tensor(self.tt_config.hidden_size**0.5, dtype=torch.bfloat16)
        embed_weight = state_dict["language_model.model.embed_tokens.weight"].to(torch.bfloat16) * embed_scale
        self.embed = ttnn.as_tensor(
            embed_weight.unsqueeze(0).unsqueeze(0).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print("  Computing RoPE cache...")
        cos_global, sin_global = compute_rope_cache(
            self.tt_config.head_dim,
            self.cache_seq_len,
            rope_theta=self.tt_config.rope_theta,
            rope_scaling=self.tt_config.rope_scaling,
        )
        cos_local, sin_local = compute_rope_cache(
            self.tt_config.head_dim,
            self.cache_seq_len,
            rope_theta=self.tt_config.rope_local_base_freq,
            rope_scaling=None,
        )
        self.cos_cache_global = ttnn.as_tensor(
            cos_global,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_cache_global = ttnn.as_tensor(
            sin_global,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.cos_cache_local = ttnn.as_tensor(
            cos_local,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_cache_local = ttnn.as_tensor(
            sin_local,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

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
                tt_device,
                self.cache_seq_len,
            )
            for i in range(self.tt_config.num_hidden_layers)
        ]

        self.norm = RMSNorm(state_dict["language_model.model.norm.weight"], self.tt_config.rms_norm_eps, tt_device)
        lm_head_weight = state_dict.get("language_model.lm_head.weight")
        if lm_head_weight is None:
            lm_head_weight = state_dict["language_model.model.embed_tokens.weight"]
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

        if self.tt_config.final_logit_softcapping is not None:
            softcap = self.tt_config.final_logit_softcapping
            logits = torch.tanh(logits / softcap) * softcap

        self._pos = start_pos + seq_len

        return CausalLMOutputWithPast(
            logits=logits.float(),
            past_key_values=(self._tt_past_key_values if use_cache else None),
        )


def build_model(hf_model, tt_device, max_seq_len: int = 2048) -> TtnnGemma3ForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnGemma3ForCausalLM(hf_model, tt_device, max_seq_len)
