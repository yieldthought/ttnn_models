# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Simple Llama 3.2 1B implementation in ttnn - 100% device execution.

This demonstrates how to bring up a HuggingFace LLM in ttnn with all compute on device.

Key ttnn operations used:
- ttnn.embedding: Token embedding lookup
- ttnn.linear: Linear projections (QKV, output, MLP)
- ttnn.rms_norm: RMSNorm normalization
- ttnn.experimental.rotary_embedding: RoPE (HuggingFace format)
- ttnn.experimental.nlp_create_qkv_heads[_decode]: Reshape for multi-head attention
- ttnn.experimental.nlp_concat_heads: Concatenate heads after attention
- ttnn.transformer.scaled_dot_product_attention[_decode]: Fused attention
- ttnn.fill_cache / ttnn.experimental.paged_update_cache: KV cache management
- ttnn.silu, ttnn.mul: MLP activations

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


TILE_SIZE = 32
PAGED_BLOCK_SIZE = 64
WEIGHT_DTYPE = ttnn.bfloat16


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

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            rope_scaling=hf_config.rope_scaling,
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
    head_dim = config.head_dim
    
    # Compute inverse frequencies
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # Apply Llama 3.x scaling if present
    rope_scaling_type = None
    if config.rope_scaling:
        rope_scaling_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))

    if rope_scaling_type == "llama3":
        factor = config.rope_scaling["factor"]
        low_freq_factor = config.rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = config.rope_scaling.get("high_freq_factor", 4.0)
        orig_len = config.rope_scaling["original_max_position_embeddings"]
        
        low_wavelen = orig_len / low_freq_factor
        high_wavelen = orig_len / high_freq_factor
        
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_wavelen:
                new_freqs.append(freq.item())
            elif wavelen > low_wavelen:
                new_freqs.append(freq.item() / factor)
            else:
                smooth = (orig_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq.item() / factor + smooth * freq.item())
        inv_freq = torch.tensor(new_freqs)
    
    # Compute freqs and duplicate for full head_dim: [max_seq_len, head_dim]
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    # Shape for ttnn.experimental.rotary_embedding: [1, 1, max_seq_len, head_dim]
    cos = emb.cos().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin = emb.sin().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    
    return cos, sin


class Attention:
    """
    Multi-head attention with GQA support, fully on ttnn.
    
    Key operations:
    - QKV projections via ttnn.linear
    - Head reshaping via nlp_create_qkv_heads[_decode]  
    - RoPE via ttnn.experimental.rotary_embedding
    - KV cache via fill_cache/paged_update_cache
    - SDPA via ttnn.transformer.scaled_dot_product_attention[_decode]
    """
    
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_dict: dict,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        tt_device,
        max_seq_len: int,
        paged_attention_config: Optional[PagedAttentionConfig] = None,
        page_table: Optional[ttnn.Tensor] = None,
    ):
        self.tt_device = tt_device
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.paged_attention_config = paged_attention_config
        self.page_table = page_table
        if self.paged_attention_config and self.page_table is None:
            raise ValueError("page_table is required for paged KV cache")
        
        # RoPE caches
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        
        # Load weights: HF [out, in] -> ttnn [1, 1, in, out]
        p = f"model.layers.{layer_idx}.self_attn."
        self.q_proj = self._load_weight(state_dict[f"{p}q_proj.weight"])
        self.k_proj = self._load_weight(state_dict[f"{p}k_proj.weight"])
        self.v_proj = self._load_weight(state_dict[f"{p}v_proj.weight"])
        self.o_proj = self._load_weight(state_dict[f"{p}o_proj.weight"])
        
        # KV cache:
        # - Paged: [max_num_blocks, n_kv_heads, block_size, head_dim]
        # - Default: [max_batch, n_kv_heads, max_seq_len, head_dim] (max_batch=32)
        if self.paged_attention_config:
            cache_shape = (
                self.paged_attention_config.max_num_blocks,
                self.n_kv_heads,
                self.paged_attention_config.block_size,
                self.head_dim,
            )
        else:
            cache_shape = (TILE_SIZE, self.n_kv_heads, max_seq_len, self.head_dim)
        self.k_cache = ttnn.as_tensor(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_cache = ttnn.as_tensor(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    
    def _load_weight(self, w: torch.Tensor) -> ttnn.Tensor:
        """Load weight transposed for ttnn.linear: [out, in] -> [1, 1, in, out]"""
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT,
            device=self.tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        padded_seq = x.shape[2]
        
        # QKV projections
        q = ttnn.linear(x, self.q_proj)
        k = ttnn.linear(x, self.k_proj)
        v = ttnn.linear(x, self.v_proj)
        
        # Fuse for head reshaping
        qkv = ttnn.concat([q, k, v], dim=-1)
        
        if is_prefill:
            # Prefill path
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=self.n_heads, num_kv_heads=self.n_kv_heads,
                transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)
            
            # RoPE
            cos = self.cos_cache[:, :, :padded_seq, :]
            sin = self.sin_cache[:, :, :padded_seq, :]
            q = ttnn.experimental.rotary_embedding(q, cos, sin)
            k = ttnn.experimental.rotary_embedding(k, cos, sin)
            
            if self.paged_attention_config:
                ttnn.experimental.paged_fill_cache(self.k_cache, k, self.page_table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(self.v_cache, v, self.page_table, batch_idx=0)
            else:
                # Shard KV for fill_cache to avoid interleaved grid-size limits at long prefill lengths.
                grid = self.tt_device.core_grid
                grid_x = min(grid.x, self.n_kv_heads)
                while grid_x > 1 and self.n_kv_heads % grid_x != 0:
                    grid_x -= 1
                shard_grid = ttnn.CoreGrid(x=grid_x, y=self.n_kv_heads // grid_x)
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
            
            # SDPA prefill (causal)
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale,
            )
            
            # Concatenate heads
            attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            # Decode path
            if cur_pos_tensor is None:
                raise ValueError("cur_pos_tensor is required for decode")

            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv, num_heads=self.n_heads, num_kv_heads=self.n_kv_heads,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)
            
            # RoPE with token index
            q = ttnn.experimental.rotary_embedding(q, self.cos_cache, self.sin_cache, start_pos)
            k = ttnn.experimental.rotary_embedding(k, self.cos_cache, self.sin_cache, start_pos)
            
            # Update KV cache (needs position tensor with batch_size entries)
            if self.paged_attention_config:
                ttnn.experimental.paged_update_cache(
                    self.k_cache, k, update_idxs_tensor=cur_pos_tensor, page_table=self.page_table
                )
                ttnn.experimental.paged_update_cache(
                    self.v_cache, v, update_idxs_tensor=cur_pos_tensor, page_table=self.page_table
                )
            else:
                ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=cur_pos_tensor)
                ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=cur_pos_tensor)
            
            # SDPA decode
            if self.paged_attention_config:
                attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q,
                    self.k_cache,
                    self.v_cache,
                    page_table_tensor=self.page_table,
                    cur_pos_tensor=cur_pos_tensor,
                    scale=self.scale,
                )
            else:
                attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                    q, self.k_cache, self.v_cache, cur_pos_tensor=cur_pos_tensor, scale=self.scale,
                )
            
            # Transpose and concatenate heads
            attn_out = ttnn.transpose(attn_out, 1, 2)
            attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        # Output projection
        return ttnn.linear(attn_out, self.o_proj)


class MLP:
    """SwiGLU MLP, fully on ttnn."""
    
    def __init__(self, layer_idx: int, state_dict: dict, tt_device):
        p = f"model.layers.{layer_idx}.mlp."
        self.gate_proj = self._load_weight(state_dict[f"{p}gate_proj.weight"], tt_device)
        self.up_proj = self._load_weight(state_dict[f"{p}up_proj.weight"], tt_device)
        self.down_proj = self._load_weight(state_dict[f"{p}down_proj.weight"], tt_device)
    
    def _load_weight(self, w: torch.Tensor, tt_device) -> ttnn.Tensor:
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT,
            device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.silu(ttnn.linear(x, self.gate_proj))
        up = ttnn.linear(x, self.up_proj)
        return ttnn.linear(ttnn.mul(gate, up), self.down_proj)


class RMSNorm:
    """RMSNorm layer."""
    
    def __init__(self, weight: torch.Tensor, eps: float, tt_device):
        self.eps = eps
        self.weight = ttnn.as_tensor(
            weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        max_seq_len: int,
        paged_attention_config: Optional[PagedAttentionConfig] = None,
        page_table: Optional[ttnn.Tensor] = None,
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
            max_seq_len,
            paged_attention_config=paged_attention_config,
            page_table=page_table,
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


class TtnnLlamaForCausalLM(torch.nn.Module, GenerationMixin):
    """
    Llama model with 100% ttnn execution.
    HuggingFace `generate()`-compatible via `GenerationMixin`.
    """
    
    def __init__(self, hf_model, tt_device, max_seq_len: int = 2048):
        super().__init__()

        self.tt_device = tt_device
        self.hf_config = hf_model.config
        self.tt_config = ModelConfig.from_hf(hf_model.config)
        self.max_seq_len = max_seq_len
        self._pos = 0
        self.paged_attention_config = PagedAttentionConfig(
            block_size=PAGED_BLOCK_SIZE,
            max_num_blocks=math.ceil(max_seq_len / PAGED_BLOCK_SIZE),
        )

        self.config = self.hf_config
        self.generation_config = GenerationConfig.from_model_config(self.config)
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = self.generation_config.eos_token_id
        # Tell HF generate() not to initialize a torch KV cache; we manage KV cache on-device in ttnn.
        self._supports_cache_class = False
        self.main_input_name = "input_ids"
        self.register_buffer("_torch_dummy", torch.empty(0, dtype=torch.float32), persistent=False)
        
        state_dict = hf_model.state_dict()
        
        # Token embeddings
        print("  Loading embeddings...")
        self.embed = ttnn.as_tensor(
            state_dict["model.embed_tokens.weight"].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # RoPE cache
        print("  Computing RoPE cache...")
        cos, sin = compute_rope_cache(self.tt_config, max_seq_len)
        self.cos_cache = ttnn.as_tensor(
            cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_cache = ttnn.as_tensor(
            sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Page table for paged KV cache (tile-aligned batch dimension)
        page_table = torch.arange(self.paged_attention_config.max_num_blocks, dtype=torch.int32)
        page_table = page_table.repeat(TILE_SIZE, 1)
        self.page_table = ttnn.as_tensor(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Transformer layers
        print(f"  Loading {self.tt_config.num_hidden_layers} layers...")
        self.layers = [
            DecoderLayer(
                self.tt_config,
                i,
                state_dict,
                self.cos_cache,
                self.sin_cache,
                tt_device,
                max_seq_len,
                paged_attention_config=self.paged_attention_config,
                page_table=self.page_table,
            )
            for i in range(self.tt_config.num_hidden_layers)
        ]
        
        # Final norm and LM head
        self.norm = RMSNorm(state_dict["model.norm.weight"], self.tt_config.rms_norm_eps, tt_device)
        self.lm_head = ttnn.as_tensor(
            state_dict["lm_head.weight"].T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT,
            device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Sentinel object to signal "device cache is populated" to HF generate().
        self._tt_past_key_values = object()

    @property
    def device(self) -> torch.device:
        return self._torch_dummy.device
    
    def reset(self):
        """Reset position counter for new sequence."""
        self._pos = 0
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # HF generate passes the full sequence each step; slice to the new token when cache is present.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}

    def _reorder_cache(self, past_key_values, beam_idx):
        # We keep the cache fully on device; beam search isn't supported in this simple demo.
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

        cur_pos_tensor = None
        if seq_len == 1:
            # Decode-only ops expect a [B] int32 tensor of positions, where B is tile-aligned (32).
            # Keep this host->device transfer at the model level (not inside Attention) so tracing can
            # later wrap the pure device compute separately.
            cur_pos = torch.full((TILE_SIZE,), -1, dtype=torch.int32)
            cur_pos[0] = start_pos
            cur_pos_tensor = ttnn.from_torch(
                cur_pos,
                dtype=ttnn.int32,
                device=self.tt_device,
            )
        
        # Pad to tile boundary
        padded_seq = pad_to_tile(seq_len)
        if seq_len < padded_seq:
            input_ids = torch.nn.functional.pad(input_ids, (0, padded_seq - seq_len), value=0)
        
        # Embed tokens
        tokens = ttnn.from_torch(
            input_ids.reshape(1, 1, 1, -1),
            dtype=ttnn.uint32, device=self.tt_device,
        )
        h = ttnn.embedding(tokens, self.embed, layout=ttnn.TILE_LAYOUT)
        
        # Forward through layers
        for layer in self.layers:
            h = layer(h, start_pos, seq_len, cur_pos_tensor=cur_pos_tensor)
        
        # Final norm and LM head
        h = self.norm(h)
        logits = ttnn.linear(h, self.lm_head)
        
        # Back to torch, remove padding
        logits = ttnn.to_torch(logits).reshape(batch, padded_seq, -1)[:, :seq_len, :]
        
        self._pos = start_pos + seq_len

        return CausalLMOutputWithPast(
            logits=logits.float(),
            past_key_values=(self._tt_past_key_values if use_cache else None),
        )

def build_model(hf_model, tt_device, max_seq_len: int = 2048) -> TtnnLlamaForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnLlamaForCausalLM(hf_model, tt_device, max_seq_len)
