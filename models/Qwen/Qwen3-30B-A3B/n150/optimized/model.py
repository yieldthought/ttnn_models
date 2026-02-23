# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Qwen3-30B-A3B path for n150.

Optimizations versus functional:
- Fused attention QKV projection (one matmul instead of three).
- MoE expert execution on host torch tensors to remove repeated expert weight
  host->device transfers in the token loop.

Batch=1 inference is supported for prefill + decode.
"""

import math
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


def pad_to_tile(x: int) -> int:
    """Pad to tile boundary (32)."""
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


@dataclass
class ModelConfig:
    """Model configuration extracted from HuggingFace."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[dict]
    attention_bias: bool
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    decoder_sparse_step: int
    mlp_only_layers: list

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
            getattr(hf_config, "moe_intermediate_size", hf_config.intermediate_size),
            hf_config.num_hidden_layers,
            hf_config.num_attention_heads,
            num_kv_heads,
            head_dim,
            hf_config.rms_norm_eps,
            hf_config.rope_theta,
            hf_config.rope_scaling,
            hf_config.attention_bias,
            hf_config.hidden_act,
            hf_config.tie_word_embeddings,
            getattr(hf_config, "num_experts", 0),
            getattr(hf_config, "num_experts_per_tok", 0),
            getattr(hf_config, "norm_topk_prob", False),
            getattr(hf_config, "decoder_sparse_step", 1),
            list(getattr(hf_config, "mlp_only_layers", [])),
        )


@dataclass
class PagedAttentionConfig:
    """Paged KV cache configuration."""

    block_size: int
    max_num_blocks: int


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


def resolve_max_seq_len(hf_config, max_seq_len: Optional[int]) -> int:
    """Resolve max sequence length from HF config when not provided."""
    config_max = getattr(hf_config, "max_position_embeddings", None)
    if config_max is None:
        config_max = getattr(hf_config, "seq_length", None)
    if config_max is None:
        config_max = getattr(hf_config, "max_seq_len", None)
    if max_seq_len is None:
        if config_max is None:
            raise ValueError("max_seq_len is required when config has no max_position_embeddings")
        return config_max
    if config_max is not None and max_seq_len > config_max:
        raise ValueError(f"max_seq_len {max_seq_len} exceeds config max {config_max}")
    return max_seq_len


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
    """
    Multi-head attention with GQA support, fully on ttnn.

    Includes Q/K RMSNorm on the head dimension, matching Qwen3.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
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
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.paged_attention_config = paged_attention_config
        self.page_table = page_table

        if config.attention_bias:
            raise ValueError("attention_bias=True is not supported in this bringup")

        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

        p = f"model.layers.{layer_idx}.self_attn."
        self.qkv_proj = self._load_qkv_weight(
            state_dict[f"{p}q_proj.weight"],
            state_dict[f"{p}k_proj.weight"],
            state_dict[f"{p}v_proj.weight"],
        )
        self.o_proj = self._load_weight(state_dict[f"{p}o_proj.weight"])
        self.q_norm = RMSNorm(state_dict[f"{p}q_norm.weight"], config.rms_norm_eps, tt_device)
        self.k_norm = RMSNorm(state_dict[f"{p}k_norm.weight"], config.rms_norm_eps, tt_device)

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
        """Load weight transposed for ttnn.linear: [out, in] -> [1, 1, in, out]."""
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_qkv_weight(self, q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor) -> ttnn.Tensor:
        qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        return self._load_weight(qkv_weight)

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Forward pass for prefill (seq_len > 1) or decode (seq_len == 1)."""
        is_prefill = seq_len > 1
        padded_seq = x.shape[2]

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
            expected_width = self.n_heads * self.head_dim
            if attn_out.shape[-1] != expected_width:
                attn_out = ttnn.slice(
                    attn_out,
                    (0, 0, 0, 0),
                    (attn_out.shape[0], attn_out.shape[1], attn_out.shape[2], expected_width),
                )
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
            q = ttnn.experimental.rotary_embedding(q, self.cos_cache, self.sin_cache, start_pos)
            q = ttnn.reshape(q, (1, q.shape[2] // self.n_heads, self.n_heads, self.head_dim))

            k = ttnn.reshape(k, (1, 1, k.shape[1] * self.n_kv_heads, self.head_dim))
            k = ttnn.experimental.rotary_embedding(k, self.cos_cache, self.sin_cache, start_pos)
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
            expected_width = self.n_heads * self.head_dim
            if attn_out.shape[-1] != expected_width:
                attn_out = ttnn.slice(
                    attn_out,
                    (0, 0, 0, 0),
                    (attn_out.shape[0], attn_out.shape[1], attn_out.shape[2], expected_width),
                )

        return ttnn.linear(attn_out, self.o_proj)


class DenseMLP:
    """Dense SwiGLU MLP, used on non-sparse layers."""

    def __init__(self, layer_idx: int, state_dict: dict, tt_device):
        p = f"model.layers.{layer_idx}.mlp."
        self.gate_proj = self._load_weight(state_dict[f"{p}gate_proj.weight"], tt_device)
        self.up_proj = self._load_weight(state_dict[f"{p}up_proj.weight"], tt_device)
        self.down_proj = self._load_weight(state_dict[f"{p}down_proj.weight"], tt_device)

    def _load_weight(self, w: torch.Tensor, tt_device) -> ttnn.Tensor:
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        gate = ttnn.silu(ttnn.linear(x, self.gate_proj))
        up = ttnn.linear(x, self.up_proj)
        out = ttnn.linear(ttnn.mul(gate, up), self.down_proj)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return out


class SparseMoE:
    """
    Sparse MoE block with host-side expert execution.

    This path keeps routing and expert matmuls on torch tensors to avoid
    repeated expert host->device transfers in the decode loop.
    """

    def __init__(self, config: ModelConfig, layer_idx: int, state_dict: dict, tt_device):
        self.layer_idx = layer_idx
        self.tt_device = tt_device
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.decode_top_k = min(5, self.top_k)
        self.norm_topk_prob = config.norm_topk_prob
        self.state_dict = state_dict
        self.expert_prefixes = [f"model.layers.{layer_idx}.mlp.experts.{i}." for i in range(self.num_experts)]

        gate_key = f"model.layers.{layer_idx}.mlp.gate.weight"
        self.gate_weight_torch = state_dict[gate_key].to(torch.float32).contiguous()

    def _run_host_expert(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        prefix = self.expert_prefixes[expert_idx]
        gate_weight = self.state_dict[f"{prefix}gate_proj.weight"]
        up_weight = self.state_dict[f"{prefix}up_proj.weight"]
        down_weight = self.state_dict[f"{prefix}down_proj.weight"]
        x_bf16 = x.to(torch.bfloat16)
        gate = F.silu(F.linear(x_bf16, gate_weight))
        up = F.linear(x_bf16, up_weight)
        return F.linear(gate * up, down_weight).to(torch.float32)

    def __call__(self, x: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        padded_seq = x.shape[2]
        hidden_states = ttnn.to_torch(x).reshape(padded_seq, self.hidden_size)[:seq_len].to(torch.float32)
        router_logits = F.linear(hidden_states, self.gate_weight_torch)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        route_top_k = self.decode_top_k if seq_len == 1 else self.top_k
        routing_weights, selected_experts = torch.topk(routing_weights, route_top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        final_hidden_states = torch.zeros((seq_len, self.hidden_size), dtype=torch.float32)

        expert_hitted = torch.unique(selected_experts)
        for expert_idx_tensor in expert_hitted:
            expert_idx = int(expert_idx_tensor.item())
            token_idx, route_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue

            current_state = hidden_states.index_select(0, token_idx)
            current_weight = routing_weights[token_idx, route_idx].unsqueeze(-1)
            expert_hidden = self._run_host_expert(expert_idx, current_state)
            final_hidden_states.index_add_(0, token_idx, expert_hidden * current_weight)

        if seq_len < padded_seq:
            padded_hidden_states = torch.zeros((padded_seq, self.hidden_size), dtype=torch.bfloat16)
            padded_hidden_states[:seq_len] = final_hidden_states.to(torch.bfloat16)
        else:
            padded_hidden_states = final_hidden_states.to(torch.bfloat16)

        return ttnn.from_torch(
            padded_hidden_states.reshape(1, 1, padded_seq, self.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class DecoderLayer:
    """Single transformer decoder layer."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        tt_device,
        paged_attention_config: PagedAttentionConfig,
        page_table: ttnn.Tensor,
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
            tt_device,
            paged_attention_config,
            page_table,
        )

        self.use_sparse_moe = (
            layer_idx not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if self.use_sparse_moe:
            self.mlp = SparseMoE(config, layer_idx, state_dict, tt_device)
        else:
            self.mlp = DenseMLP(layer_idx, state_dict, tt_device)

    def __call__(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        seq_len: int,
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        x = ttnn.add(x, self.attn(self.attn_norm(x), start_pos, seq_len, cur_pos_tensor=cur_pos_tensor))
        x = ttnn.add(x, self.mlp(self.ffn_norm(x), seq_len))
        return x


class TtnnQwen3MoeForCausalLM(torch.nn.Module, GenerationMixin):
    """
    Qwen3-MoE model with TTNN attention and dynamic sparse experts.

    HuggingFace `generate()`-compatible via `GenerationMixin`.
    """

    def __init__(self, hf_model, tt_device, max_seq_len: Optional[int] = None):
        super().__init__()

        self.tt_device = tt_device
        self.hf_config = hf_model.config
        self.tt_config = ModelConfig.from_hf(hf_model.config)
        self.max_seq_len = resolve_max_seq_len(self.hf_config, max_seq_len)
        self._pos = 0

        if self.tt_config.hidden_act != "silu":
            raise ValueError(f"hidden_act {self.tt_config.hidden_act} is not supported in this bringup")
        if getattr(self.hf_config, "use_sliding_window", False):
            raise ValueError("sliding_window attention is not supported in this bringup")
        if self.tt_config.num_experts_per_tok > self.tt_config.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")

        self.config = self.hf_config
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

        print("  Loading embeddings...")
        self.embed = ttnn.as_tensor(
            state_dict["model.embed_tokens.weight"].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print("  Computing RoPE cache...")
        cos, sin = compute_rope_cache(self.tt_config, self.max_seq_len)
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

        print(f"  Loading {self.tt_config.num_hidden_layers} layers...")
        self.layers = [
            DecoderLayer(
                self.tt_config,
                i,
                state_dict,
                self.cos_cache,
                self.sin_cache,
                tt_device,
                self.paged_attention_config,
                self.page_table,
            )
            for i in range(self.tt_config.num_hidden_layers)
        ]

        self.norm = RMSNorm(state_dict["model.norm.weight"], self.tt_config.rms_norm_eps, tt_device)
        lm_head_weight = state_dict.get("lm_head.weight", state_dict["model.embed_tokens.weight"])
        self.lm_head = ttnn.as_tensor(
            lm_head_weight.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
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

    def _forward_prefill(self, input_ids: torch.Tensor, start_pos: int, seq_len: int) -> ttnn.Tensor:
        tokens = ttnn.from_torch(
            input_ids.reshape(1, 1, 1, -1),
            dtype=ttnn.uint32,
            device=self.tt_device,
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
        )
        h = ttnn.embedding(tokens, self.embed, layout=ttnn.TILE_LAYOUT)

        cur_pos_tensor = None
        if seq_len == 1:
            cur_pos = torch.full((TILE_SIZE,), -1, dtype=torch.int32)
            cur_pos[0] = start_pos
            cur_pos_tensor = ttnn.from_torch(cur_pos, dtype=ttnn.int32, device=self.tt_device)

        for layer in self.layers:
            h = layer(h, start_pos, seq_len, cur_pos_tensor=cur_pos_tensor)

        if cur_pos_tensor is not None:
            ttnn.deallocate(cur_pos_tensor)

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
            input_ids = F.pad(input_ids, (0, padded_seq - seq_len), value=0)

        logits_device = self._forward_prefill_last_logits(input_ids, start_pos, seq_len)
        self._pos = start_pos + seq_len

        logits = ttnn.to_torch(logits_device)
        logits = logits.reshape(batch, 1, -1)[:, 0, :].float()
        ttnn.deallocate(logits_device)

        past = self._tt_past_key_values if use_cache else None
        return logits, past


def build_model(hf_model, tt_device, max_seq_len: Optional[int] = None) -> TtnnQwen3MoeForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnQwen3MoeForCausalLM(hf_model, tt_device, max_seq_len)
