# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Llama 3.2 1B implementation in ttnn - 100% device execution.

This version keeps the HF RoPE format but reduces math overhead by:
- Fused QKV and fused gate+up MLP matmuls
- Lower-precision weights (BFP8) where safe

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
WEIGHT_DTYPE = ttnn.bfloat8_b
WEIGHT_LAYOUT = ttnn.TILE_LAYOUT
MLP_GATE_UP_DTYPE = ttnn.bfloat4_b
MLP_DOWN_DTYPE = ttnn.bfloat8_b
QKV_WEIGHT_DTYPE = ttnn.bfloat8_b
LM_HEAD_WEIGHT_DTYPE = ttnn.bfloat4_b
MAX_CACHE_SEQ_LEN = 1024
USE_DECODE_TRACE = True
HIFI2_MATMUL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
LOFI_MATMUL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def pad_to_tile(x: int) -> int:
    """Pad to tile boundary (32)."""
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def matmul_1d_program_config(m: int, k: int, n: int, grid) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    grid_x = grid.x if hasattr(grid, "x") else grid[0]
    grid_y = grid.y if hasattr(grid, "y") else grid[1]
    grid_cores = grid_x * grid_y

    if n // TILE_SIZE // grid_cores < 1:
        grid_y = max(1, n // TILE_SIZE // grid_x)
        grid_cores = grid_x * grid_y

    per_core_m = m // TILE_SIZE
    per_core_k = math.ceil(k / TILE_SIZE / grid_cores)
    per_core_n = math.ceil(n / TILE_SIZE / grid_cores)

    max_subblock = 4
    out_subblock_w = max(i for i in range(1, max_subblock + 1) if per_core_n % i == 0)
    out_subblock_h = max(
        i
        for i in range(1, max_subblock + 1)
        if per_core_m % i == 0 and i * out_subblock_w <= max_subblock
    )

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=per_core_k,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def sdpa_prefill_program_config(seq_len: int) -> ttnn.SDPAProgramConfig:
    chunk = 256 if seq_len >= 2048 else 64
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        exp_approx_mode=False,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
    )


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
    ):
        self.tt_device = tt_device
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.qkv_size = self.hidden_size + 2 * self.n_kv_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.sdpa_decode_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=True,
            q_chunk_size=64,
            k_chunk_size=64,
        )
        device_grid = tt_device.compute_with_storage_grid_size()
        grid_x = device_grid.x if hasattr(device_grid, "x") else device_grid[0]
        grid_y = device_grid.y if hasattr(device_grid, "y") else device_grid[1]
        if grid_x * grid_y > TILE_SIZE:
            grid_y = max(1, TILE_SIZE // grid_x)
        self.decode_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)
        self.o_proj_decode_program_config = matmul_1d_program_config(
            TILE_SIZE, self.hidden_size, self.hidden_size, self.decode_grid
        )
        padded_heads = pad_to_tile(self.n_heads)
        self.decode_heads_memcfg = ttnn.create_sharded_memory_config(
            (padded_heads, self.head_dim),
            self.decode_grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        
        # RoPE caches
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        
        p = f"model.layers.{layer_idx}.self_attn."
        self.qkv_proj = self._load_qkv_weight(
            state_dict[f"{p}q_proj.weight"],
            state_dict[f"{p}k_proj.weight"],
            state_dict[f"{p}v_proj.weight"],
            ttnn.DRAM_MEMORY_CONFIG,
        )
        self.o_proj = self._load_weight(state_dict[f"{p}o_proj.weight"], ttnn.DRAM_MEMORY_CONFIG)
        
        # KV cache: [max_batch, n_kv_heads, max_seq_len, head_dim]
        # max_batch must be tile-aligned (32) for decode mode
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
    
    def _load_weight(
        self,
        w: torch.Tensor,
        memory_config: ttnn.MemoryConfig,
        dtype: ttnn.DataType = WEIGHT_DTYPE,
    ) -> ttnn.Tensor:
        """Load weight transposed for ttnn.linear: [out, in] -> [1, 1, in, out]."""
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=dtype,
            layout=WEIGHT_LAYOUT,
            device=self.tt_device,
            memory_config=memory_config,
        )

    def _load_qkv_weight(
        self, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, memory_config: ttnn.MemoryConfig
    ) -> ttnn.Tensor:
        """Stack QKV weights for a single fused projection."""
        wqkv = torch.cat([wq.T, wk.T, wv.T], dim=-1)
        return ttnn.as_tensor(
            wqkv.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=QKV_WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=self.tt_device,
            memory_config=memory_config,
        )
    
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

        if is_prefill:
            qkv = ttnn.linear(
                x,
                self.qkv_proj,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=HIFI2_MATMUL_CONFIG,
                program_config=None,
            )
        else:
            qkv = ttnn.linear(
                x,
                self.qkv_proj,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=LOFI_MATMUL_CONFIG,
                program_config=None,
            )
        
        if is_prefill:
            # Prefill path
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=self.n_heads, num_kv_heads=self.n_kv_heads,
                transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # RoPE
            cos = self.cos_cache[:, :, :padded_seq, :]
            sin = self.sin_cache[:, :, :padded_seq, :]
            q = ttnn.experimental.rotary_embedding(q, cos, sin)
            k = ttnn.experimental.rotary_embedding(k, cos, sin)
            
            # Fill KV cache
            ttnn.fill_cache(self.k_cache, k, batch_idx=0)
            ttnn.fill_cache(self.v_cache, v, batch_idx=0)
            
            # SDPA prefill (causal)
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=sdpa_prefill_program_config(seq_len),
            )
            
            # Concatenate heads
            attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            # Decode path
            if cur_pos_tensor is None:
                raise ValueError("cur_pos_tensor is required for decode")

            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv, num_heads=self.n_heads, num_kv_heads=self.n_kv_heads,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            # RoPE with token index
            if decode_cos is not None and decode_sin is not None:
                q = ttnn.experimental.rotary_embedding(q, decode_cos, decode_sin, 0)
                k = ttnn.experimental.rotary_embedding(k, decode_cos, decode_sin, 0)
            else:
                q = ttnn.experimental.rotary_embedding(q, self.cos_cache, self.sin_cache, start_pos)
                k = ttnn.experimental.rotary_embedding(k, self.cos_cache, self.sin_cache, start_pos)
            
            # Update KV cache (needs position tensor with batch_size entries)
            ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=cur_pos_tensor)
            ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=cur_pos_tensor)
            
            # SDPA decode
            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                self.k_cache,
                self.v_cache,
                cur_pos_tensor=cur_pos_tensor,
                scale=self.scale,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=self.sdpa_decode_program_config,
            )
            
            # Concatenate heads in decode-friendly layout
            attn_out = ttnn.to_memory_config(attn_out, self.decode_heads_memcfg)
            attn_out = ttnn.experimental.nlp_concat_heads_decode(
                attn_out,
                num_heads=self.n_heads,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
        if not trace_decode:
            ttnn.deallocate(qkv)

        # Output projection
        if is_prefill:
            return ttnn.linear(
                attn_out,
                self.o_proj,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=HIFI2_MATMUL_CONFIG,
                program_config=None,
            )

        attn_out = ttnn.linear(
            attn_out,
            self.o_proj,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=LOFI_MATMUL_CONFIG,
            program_config=self.o_proj_decode_program_config,
        )
        return ttnn.to_memory_config(attn_out, ttnn.L1_MEMORY_CONFIG)


class MLP:
    """SwiGLU MLP, fully on ttnn."""
    
    def __init__(
        self,
        layer_idx: int,
        state_dict: dict,
        tt_device,
        gate_up_dtype: ttnn.DataType,
        down_dtype: ttnn.DataType,
    ):
        p = f"model.layers.{layer_idx}.mlp."
        gate_weight = state_dict[f"{p}gate_proj.weight"]
        self.hidden_size = gate_weight.shape[1]
        self.intermediate_size = gate_weight.shape[0]
        self.gate_up_dtype = gate_up_dtype
        self.down_dtype = down_dtype
        self.gate_up_proj = self._load_gate_up_weight(
            gate_weight,
            state_dict[f"{p}up_proj.weight"],
            tt_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        self.down_proj = self._load_weight(state_dict[f"{p}down_proj.weight"], tt_device, ttnn.DRAM_MEMORY_CONFIG)
    
    def _load_weight(self, w: torch.Tensor, tt_device, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
        return ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=self.down_dtype,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=memory_config,
        )

    def _load_gate_up_weight(
        self, gate: torch.Tensor, up: torch.Tensor, tt_device, memory_config: ttnn.MemoryConfig
    ) -> ttnn.Tensor:
        gate_up = torch.cat([up.T, gate.T], dim=-1)
        return ttnn.as_tensor(
            gate_up.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous(),
            dtype=self.gate_up_dtype,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=memory_config,
        )
    
    def __call__(self, x: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        is_prefill = seq_len > 1
        mem_config = ttnn.DRAM_MEMORY_CONFIG if is_prefill else ttnn.L1_MEMORY_CONFIG
        compute_config = HIFI2_MATMUL_CONFIG if is_prefill else LOFI_MATMUL_CONFIG
        gate_up = ttnn.linear(
            x,
            self.gate_up_proj,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            compute_kernel_config=compute_config,
            program_config=None,
        )
        gate_up = ttnn.swiglu(gate_up, memory_config=mem_config)
        return ttnn.linear(
            gate_up,
            self.down_proj,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            compute_kernel_config=compute_config,
            program_config=None,
        )


class RMSNorm:
    """RMSNorm layer."""
    
    def __init__(self, weight: torch.Tensor, eps: float, tt_device):
        self.eps = eps
        weight = weight.view(1, 1, -1, TILE_SIZE)
        self.weight = ttnn.as_tensor(
            weight.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
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
    ):
        p = f"model.layers.{layer_idx}."
        self.attn_norm = RMSNorm(state_dict[f"{p}input_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.ffn_norm = RMSNorm(state_dict[f"{p}post_attention_layernorm.weight"], config.rms_norm_eps, tt_device)
        self.attn = Attention(config, layer_idx, state_dict, cos_cache, sin_cache, tt_device, max_seq_len)
        self.mlp = MLP(layer_idx, state_dict, tt_device, MLP_GATE_UP_DTYPE, MLP_DOWN_DTYPE)
    
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
            self.attn(
                self.attn_norm(x),
                start_pos,
                seq_len,
                cur_pos_tensor,
                decode_cos,
                decode_sin,
                trace_decode,
            ),
        )
        x = ttnn.add(x, self.mlp(self.ffn_norm(x), seq_len))
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
        self.cache_seq_len = min(max_seq_len, MAX_CACHE_SEQ_LEN)
        self._pos = 0
        decode_grid = tt_device.compute_with_storage_grid_size()
        self.lm_head_decode_program_config = matmul_1d_program_config(
            TILE_SIZE, self.tt_config.hidden_size, self.tt_config.vocab_size, decode_grid
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
        cos, sin = compute_rope_cache(self.tt_config, self.cache_seq_len)
        self.cos_cache_host = cos
        self.sin_cache_host = sin
        self.cos_cache = ttnn.as_tensor(
            cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_cache = ttnn.as_tensor(
            sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.decode_token_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 1, TILE_SIZE), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
        )
        self.decode_pos_buffer = ttnn.from_torch(
            torch.zeros((TILE_SIZE,), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
        )
        self.decode_cos_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 1, self.tt_config.head_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
        )
        self.decode_sin_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 1, self.tt_config.head_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_device,
        )
        self.use_decode_trace = USE_DECODE_TRACE
        self.decode_trace_id = None
        self.decode_trace_logits = None
        self.decode_pos_initialized = False
        self.decode_pos_start = None
        
        # Transformer layers
        print(f"  Loading {self.tt_config.num_hidden_layers} layers...")
        self.layers = [
            DecoderLayer(self.tt_config, i, state_dict, self.cos_cache, self.sin_cache, tt_device, self.cache_seq_len)
            for i in range(self.tt_config.num_hidden_layers)
        ]
        
        # Final norm and LM head
        self.norm = RMSNorm(state_dict["model.norm.weight"], self.tt_config.rms_norm_eps, tt_device)
        lm_head_weight = state_dict["lm_head.weight"].T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16).contiguous()
        self.lm_head = ttnn.as_tensor(
            lm_head_weight,
            dtype=LM_HEAD_WEIGHT_DTYPE,
            layout=WEIGHT_LAYOUT,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Sentinel object to signal "device cache is populated" to HF generate().
        self._tt_past_key_values = object()

    @property
    def device(self) -> torch.device:
        return self._torch_dummy.device
    
    def reset(self):
        """Reset position counter for new sequence."""
        self._pos = 0
        self.decode_pos_initialized = False
        self.decode_pos_start = None
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # HF generate passes the full sequence each step; slice to the new token when cache is present.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}

    def _reorder_cache(self, past_key_values, beam_idx):
        # We keep the cache fully on device; beam search isn't supported in this simple demo.
        return past_key_values

    def _forward_device_logits(self, input_ids: torch.Tensor, past_key_values, use_cache: bool):
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

        if seq_len == 1:
            logits = self._forward_decode(input_ids, start_pos)
            padded_seq = TILE_SIZE
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
        """Forward pass compatible with HuggingFace generate()."""
        batch = input_ids.shape[0]
        logits, seq_len, padded_seq, past = self._forward_device_logits(input_ids, past_key_values, use_cache)
        if seq_len == 1 and padded_seq == TILE_SIZE:
            logits_rm = ttnn.untilize_with_unpadding(
                logits,
                [0, 0, 0, logits.shape[-1] - 1],
                use_multicore=True,
            )
            logits_torch = ttnn.to_torch(logits_rm).reshape(batch, 1, -1)
            ttnn.deallocate(logits_rm)
            return CausalLMOutputWithPast(
                logits=logits_torch.float(),
                past_key_values=past,
            )
        logits = ttnn.to_torch(logits).reshape(batch, padded_seq, -1)[:, :seq_len, :]
        return CausalLMOutputWithPast(
            logits=logits.float(),
            past_key_values=past,
        )

    def next_token_device(self, input_ids: torch.Tensor, past_key_values=None, use_cache: bool = True) -> tuple[int, object]:
        logits, seq_len, padded_seq, past = self._forward_device_logits(input_ids, past_key_values, use_cache)
        if seq_len == 1 and padded_seq == TILE_SIZE:
            logits_rm = ttnn.untilize_with_unpadding(
                logits,
                [0, 0, 0, logits.shape[-1] - 1],
                use_multicore=True,
            )
        else:
            logits_rm = ttnn.untilize(logits, use_multicore=True)
        if not self.use_decode_trace or seq_len > 1:
            ttnn.deallocate(logits)
        token_ids = ttnn.argmax(logits_rm, dim=3, use_multicore=True)
        token_ids_torch = ttnn.to_torch(token_ids).reshape(-1)
        ttnn.deallocate(token_ids)
        ttnn.deallocate(logits_rm)
        token = int(token_ids_torch[seq_len - 1].item())
        return token, past

    def _update_decode_token_buffers(self, input_ids: torch.Tensor, start_pos: int) -> None:
        token_ids = torch.zeros((TILE_SIZE,), dtype=torch.int32)
        token_ids[: input_ids.numel()] = input_ids.view(-1).to(torch.int32)
        token_ids = token_ids.reshape(1, 1, 1, -1)
        host_tokens = ttnn.from_torch(
            token_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_tokens, self.decode_token_buffer)

        if not self.decode_pos_initialized or self.decode_pos_start != start_pos:
            host_pos = ttnn.from_torch(
                torch.full((TILE_SIZE,), start_pos, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_pos, self.decode_pos_buffer)
            self.decode_pos_initialized = True
            self.decode_pos_start = start_pos

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
        return ttnn.linear(
            h,
            self.lm_head,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=LOFI_MATMUL_CONFIG,
            program_config=None,
        )

    def _forward_decode_device(self, start_pos: int, trace_decode: bool, use_rope_buffer: bool) -> ttnn.Tensor:
        decode_cos = self.decode_cos_buffer if use_rope_buffer else None
        decode_sin = self.decode_sin_buffer if use_rope_buffer else None

        h = ttnn.embedding(self.decode_token_buffer, self.embed, layout=ttnn.TILE_LAYOUT)
        for layer in self.layers:
            h = layer(h, start_pos, 1, self.decode_pos_buffer, decode_cos, decode_sin, trace_decode)
        h = self.norm(h)
        return ttnn.linear(
            h,
            self.lm_head,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=LOFI_MATMUL_CONFIG,
            program_config=self.lm_head_decode_program_config,
        )

    def _forward_decode(self, input_ids: torch.Tensor, start_pos: int) -> ttnn.Tensor:
        self._update_decode_token_buffers(input_ids, start_pos)
        if self.use_decode_trace:
            self._update_decode_rope_buffers(start_pos)
            if self.decode_trace_id is None:
                warmup_logits = self._forward_decode_device(start_pos, False, True)
                ttnn.deallocate(warmup_logits)
                self.decode_trace_id = ttnn.begin_trace_capture(self.tt_device)
                self.decode_trace_logits = self._forward_decode_device(start_pos, True, True)
                ttnn.end_trace_capture(self.tt_device, self.decode_trace_id)
            else:
                ttnn.execute_trace(self.tt_device, self.decode_trace_id, blocking=False)
            logits = self.decode_trace_logits
        else:
            logits = self._forward_decode_device(start_pos, False, False)

        ttnn.plus_one(self.decode_pos_buffer)
        if self.decode_pos_start is not None:
            self.decode_pos_start += 1
        return logits

def build_model(hf_model, tt_device, max_seq_len: int = 2048) -> TtnnLlamaForCausalLM:
    """Build the ttnn model from a HuggingFace reference model."""
    return TtnnLlamaForCausalLM(hf_model, tt_device, max_seq_len)
