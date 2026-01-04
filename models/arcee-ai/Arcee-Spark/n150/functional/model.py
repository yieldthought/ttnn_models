# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Simple Arcee-Spark (Qwen2) implementation in ttnn - 100% device execution.

This is a minimal bringup for Qwen2-style attention + SwiGLU MLP.

Key ttnn operations used:
- ttnn.embedding: Token embedding lookup
- ttnn.linear: Linear projections (QKV, output, MLP) with QKV bias
- ttnn.rms_norm: RMSNorm normalization
- ttnn.experimental.rotary_embedding: RoPE (HuggingFace format)
- ttnn.experimental.nlp_create_qkv_heads[_decode]: Reshape for multi-head attention
- ttnn.experimental.nlp_concat_heads: Concatenate heads after attention
- ttnn.transformer.scaled_dot_product_attention[_decode]: Fused attention
- ttnn.fill_cache / ttnn.experimental.paged_update_cache: KV cache management
- ttnn.silu, ttnn.mul: MLP activations

Use `eval.py` at repo root for teacher-forcing accuracy checks against the
HuggingFace reference.
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
ATTN_DTYPE = ttnn.bfloat8_b
EMBED_DTYPE = ttnn.bfloat8_b
MLP_GATE_DTYPE = ttnn.bfloat8_b
MLP_UP_DTYPE = ttnn.bfloat8_b
MLP_DOWN_DTYPE = ttnn.bfloat8_b
LM_HEAD_DTYPE = ttnn.bfloat8_b
WEIGHT_LAYOUT = ttnn.TILE_LAYOUT
MAX_CACHE_SEQ_LEN = 256
DEBUG_TORCH_ATTN_DECODE = False
DEBUG_TORCH_CACHE_DECODE = False
DEBUG_TORCH_ROPE_DECODE = False


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
    hidden_act: str
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
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
            hf_config.rope_scaling,
            hf_config.hidden_act,
            hf_config.tie_word_embeddings,
        )


def compute_rope_cache(config: ModelConfig, max_seq_len: int) -> tuple:
    """
    Precompute RoPE cos/sin cache in HuggingFace format.
    Returns cos, sin tensors of shape [1, 1, max_seq_len, head_dim].
    """
    if config.rope_scaling:
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        if rope_type != "default":
            raise ValueError(f"rope_scaling {rope_type} is not supported in this bringup")

    head_dim = config.head_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin = emb.sin().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
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
    """Multi-head attention with GQA support, fully on ttnn."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        cos_cache_torch: torch.Tensor,
        sin_cache_torch: torch.Tensor,
        tt_device,
        max_seq_len: int,
    ):
        self.tt_device = tt_device
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.kv_repeat = self.n_heads // self.n_kv_heads
        self.cache_kv_heads = self.n_heads if self.kv_repeat > 1 else self.n_kv_heads

        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        self.cos_cache_torch = cos_cache_torch
        self.sin_cache_torch = sin_cache_torch

        p = f"model.layers.{layer_idx}.self_attn."
        self.q_proj = self._load_weight(state_dict[f"{p}q_proj.weight"])
        self.k_proj = self._load_weight(state_dict[f"{p}k_proj.weight"])
        self.v_proj = self._load_weight(state_dict[f"{p}v_proj.weight"])
        self.o_proj = self._load_weight(state_dict[f"{p}o_proj.weight"])
        self.q_bias = self._load_bias(state_dict[f"{p}q_proj.bias"])
        self.k_bias = self._load_bias(state_dict[f"{p}k_proj.bias"])
        self.v_bias = self._load_bias(state_dict[f"{p}v_proj.bias"])

        cache_shape = (TILE_SIZE, self.cache_kv_heads, max_seq_len, self.head_dim)
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
        self._torch_k_cache = []
        self._torch_v_cache = []

    def _load_weight(self, w: torch.Tensor) -> ttnn.Tensor:
        """Load weight transposed for ttnn.linear: [out, in] -> [1, 1, in, out]."""
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=ATTN_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_bias(self, b: torch.Tensor) -> ttnn.Tensor:
        """Load bias as [1, 1, 1, out] for ttnn.linear."""
        return ttnn.as_tensor(
            b.view(1, 1, 1, -1).to(torch.bfloat16),
            dtype=ATTN_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def reset_cache(self):
        self._torch_k_cache = []
        self._torch_v_cache = []

    def _torch_repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def _torch_apply_rope(self, q: torch.Tensor, k: torch.Tensor, start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cache_torch[:, :, start_pos : start_pos + 1, :]
        sin = self.sin_cache_torch[:, :, start_pos : start_pos + 1, :]

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k

    def _repeat_kv_heads(
        self,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        heads_dim: int,
        k_mem: Optional[ttnn.MemoryConfig] = None,
        v_mem: Optional[ttnn.MemoryConfig] = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if self.kv_repeat == 1:
            if k_mem is not None:
                k = ttnn.to_memory_config(k, k_mem)
            if v_mem is not None:
                v = ttnn.to_memory_config(v, v_mem)
            return k, v
        if heads_dim not in (1, 2):
            raise ValueError("heads_dim must be 1 or 2")

        def reshard_after_repeat(tensor: ttnn.Tensor, mem_config: Optional[ttnn.MemoryConfig]) -> ttnn.Tensor:
            if mem_config is None:
                return tensor
            return ttnn.to_memory_config(tensor, mem_config)

        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.repeat_interleave(k, self.kv_repeat, dim=heads_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.repeat_interleave(v, self.kv_repeat, dim=heads_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = reshard_after_repeat(k, k_mem)
        v = reshard_after_repeat(v, v_mem)
        return k, v

    def _torch_attention_decode(self, q: ttnn.Tensor, start_pos: int) -> ttnn.Tensor:
        q_t = ttnn.to_torch(q).squeeze(0).unsqueeze(2)
        if DEBUG_TORCH_CACHE_DECODE:
            if not self._torch_k_cache:
                raise ValueError("torch cache is empty; run with --prefill_decode for DEBUG_TORCH_CACHE_DECODE")
            k_t = torch.cat(self._torch_k_cache, dim=2)
            v_t = torch.cat(self._torch_v_cache, dim=2)
        else:
            cur_len = start_pos + 1
            k_attn = ttnn.slice(
                self.k_cache,
                (0, 0, 0, 0),
                (TILE_SIZE, self.cache_kv_heads, cur_len, self.head_dim),
            )
            v_attn = ttnn.slice(
                self.v_cache,
                (0, 0, 0, 0),
                (TILE_SIZE, self.cache_kv_heads, cur_len, self.head_dim),
            )
            k_t = ttnn.to_torch(k_attn)
            v_t = ttnn.to_torch(v_attn)
            ttnn.deallocate(k_attn)
            ttnn.deallocate(v_attn)

        n_rep = self.n_heads // self.cache_kv_heads
        k_t = self._torch_repeat_kv(k_t, n_rep)
        v_t = self._torch_repeat_kv(v_t, n_rep)

        attn_weights = torch.matmul(q_t.float(), k_t.transpose(-2, -1).float()) * self.scale
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_probs, v_t.float()).to(q_t.dtype)
        attn_out = attn_out.squeeze(2).unsqueeze(0)
        return ttnn.from_torch(
            attn_out.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
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

        if DEBUG_TORCH_CACHE_DECODE and not DEBUG_TORCH_ATTN_DECODE:
            raise ValueError("DEBUG_TORCH_CACHE_DECODE requires DEBUG_TORCH_ATTN_DECODE")
        if DEBUG_TORCH_CACHE_DECODE and is_prefill:
            raise ValueError("DEBUG_TORCH_CACHE_DECODE requires --prefill_decode")

        q = ttnn.linear(x, self.q_proj, bias=self.q_bias)
        k = ttnn.linear(x, self.k_proj, bias=self.k_bias)
        v = ttnn.linear(x, self.v_proj, bias=self.v_bias)
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

            k, v = self._repeat_kv_heads(k, v, heads_dim=1)

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
            v_mem = ttnn.get_memory_config(v)
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)

            if DEBUG_TORCH_ROPE_DECODE:
                q_t = ttnn.to_torch(q).squeeze(0).unsqueeze(2)
                k_t = ttnn.to_torch(k).squeeze(0).unsqueeze(2)
                q_t, k_t = self._torch_apply_rope(q_t, k_t, start_pos)
                q = ttnn.from_torch(
                    q_t.squeeze(2).unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.tt_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                k = ttnn.from_torch(
                    k_t.squeeze(2).unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.tt_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                q = ttnn.reshape(q, (1, 1, q.shape[1] * self.n_heads, self.head_dim))
                q = ttnn.experimental.rotary_embedding(q, self.cos_cache, self.sin_cache, start_pos)
                q = ttnn.reshape(q, (1, q.shape[2] // self.n_heads, self.n_heads, self.head_dim))

                k = ttnn.reshape(k, (1, 1, k.shape[1] * self.n_kv_heads, self.head_dim))
                k = ttnn.experimental.rotary_embedding(k, self.cos_cache, self.sin_cache, start_pos)
                k = ttnn.reshape(k, (1, k.shape[2] // self.n_kv_heads, self.n_kv_heads, self.head_dim))

            q = ttnn.to_memory_config(q, q_mem)
            k = ttnn.to_memory_config(k, k_mem)
            v = ttnn.to_memory_config(v, v_mem)

            repeat_mem = q_mem if self.kv_repeat > 1 else k_mem
            k, v = self._repeat_kv_heads(k, v, heads_dim=2, k_mem=repeat_mem, v_mem=repeat_mem)

            if DEBUG_TORCH_CACHE_DECODE:
                self._torch_k_cache.append(ttnn.to_torch(k).squeeze(0).unsqueeze(2))
                self._torch_v_cache.append(ttnn.to_torch(v).squeeze(0).unsqueeze(2))
            else:
                ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=cur_pos_tensor)
                ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=cur_pos_tensor)

            if DEBUG_TORCH_ATTN_DECODE:
                attn_out = self._torch_attention_decode(q, start_pos)
            else:
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
    """SwiGLU MLP, fully on ttnn."""

    def __init__(self, layer_idx: int, state_dict: dict, tt_device):
        p = f"model.layers.{layer_idx}.mlp."
        self.gate_proj = self._load_weight(state_dict[f"{p}gate_proj.weight"], tt_device, MLP_GATE_DTYPE)
        self.up_proj = self._load_weight(state_dict[f"{p}up_proj.weight"], tt_device, MLP_UP_DTYPE)
        self.down_proj = self._load_weight(state_dict[f"{p}down_proj.weight"], tt_device, MLP_DOWN_DTYPE)

    def _load_weight(self, w: torch.Tensor, tt_device, dtype: ttnn.DataType) -> ttnn.Tensor:
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=dtype,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.silu(ttnn.linear(x, self.gate_proj))
        up = ttnn.linear(x, self.up_proj)
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
        cos_cache_torch: torch.Tensor,
        sin_cache_torch: torch.Tensor,
        tt_device,
        max_seq_len: int,
    ):
        p = f"model.layers.{layer_idx}."
        self.attn_norm = RMSNorm(state_dict[f"{p}input_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.ffn_norm = RMSNorm(state_dict[f"{p}post_attention_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.attn = Attention(
            config,
            layer_idx,
            state_dict,
            cos_cache,
            sin_cache,
            cos_cache_torch,
            sin_cache_torch,
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
        x = ttnn.add(x, self.attn(self.attn_norm(x), start_pos, seq_len, cur_pos_tensor=cur_pos_tensor))
        x = ttnn.add(x, self.mlp(self.ffn_norm(x)))
        return x


class TtnnQwen2ForCausalLM(torch.nn.Module, GenerationMixin):
    """
    Qwen2 model with 100% ttnn execution.
    HuggingFace `generate()`-compatible via `GenerationMixin`.
    """

    def __init__(self, hf_model, tt_device, max_seq_len: int = 2048):
        super().__init__()

        self.tt_device = tt_device
        self.hf_config = hf_model.config
        self.tt_config = ModelConfig.from_hf(hf_model.config)
        self.max_seq_len = max_seq_len
        max_cache_seq_len = max(TILE_SIZE, (MAX_CACHE_SEQ_LEN // TILE_SIZE) * TILE_SIZE)
        self.cache_seq_len = min(max_seq_len, max_cache_seq_len)
        self._pos = 0

        if self.tt_config.hidden_act != "silu":
            raise ValueError(f"hidden_act {self.tt_config.hidden_act} is not supported in this bringup")
        if getattr(self.hf_config, "use_sliding_window", False):
            raise ValueError("sliding_window attention is not supported in this bringup")

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
            dtype=EMBED_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print("  Computing RoPE cache...")
        cos, sin = compute_rope_cache(self.tt_config, max_seq_len)
        self.cos_cache_torch = cos
        self.sin_cache_torch = sin
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
                self.tt_config,
                i,
                state_dict,
                self.cos_cache,
                self.sin_cache,
                self.cos_cache_torch,
                self.sin_cache_torch,
                tt_device,
                self.cache_seq_len,
            )
            for i in range(self.tt_config.num_hidden_layers)
        ]

        self.norm = RMSNorm(state_dict["model.norm.weight"], self.tt_config.rms_norm_eps, tt_device)
        lm_head_weight = state_dict.get("lm_head.weight", state_dict["model.embed_tokens.weight"])
        self.lm_head = ttnn.as_tensor(
            lm_head_weight.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=LM_HEAD_DTYPE,
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
        for layer in self.layers:
            layer.attn.reset_cache()

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


def build_model(hf_model, tt_device, max_seq_len: int = 2048) -> TtnnQwen2ForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnQwen2ForCausalLM(hf_model, tt_device, max_seq_len)
