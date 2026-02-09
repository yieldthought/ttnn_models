---
name: ttnn-multidevice-model-parallel
description: Tensor-parallel and multidevice TTNN/tt-metal model bringup guidance. Use when sharding transformer weights across a mesh, placing CCL ops (all_reduce or all_gather), mapping KV caches, or debugging multi-device prefill or decode mismatches.
---

# TTNN Multidevice Model Parallel

## Overview
Implement 1D tensor-parallel transformer blocks on a TTNN MeshDevice, including weight sharding, cache layout, and CCL placement.

## Workflow
1. Read the HF config and derive sizes: hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size.
2. Choose mesh shape and axis; validate divisibility by num_devices.
3. Define mesh mappers: replicate, shard width (dim=3), shard height (dim=2), shard KV (dim=1).
4. Map weights by pattern:
   - Column-parallel: shard width.
   - Row-parallel: shard height, all_reduce on output.
5. Cache layout:
   - Shard KV cache on dim=1 unless you have a reason to replicate.
   - Use fill_cache for prefill and paged_update_cache for decode.
6. Decode shaping:
   - Use nlp_create_qkv_heads_decode with local head counts.
   - For RoPE with start_pos, reshape to [1, 1, batch*heads, head_dim] (pad batch*heads to tile), apply rotary_embedding, reshape back.
   - If the model has q_norm/k_norm, apply after head split and before RoPE (move to DRAM if needed).
   - After concat, trim padded head width.
7. CCL placement:
   - Use all_reduce after row-parallel matmuls (o_proj, down_proj).
   - Use all_gather only when a downstream op needs full activations on each device.
8. Output composition:
   - Keep logits sharded for device compute; compose on host with ConcatMeshToTensor.

## Quick Reference: 1D TP Mapping
- Replicated: embeddings, RoPE caches, RMSNorm weights, input tokens.
- Column-parallel: q_proj, k_proj, v_proj, gate_proj, up_proj, lm_head.
- Row-parallel: o_proj, down_proj.
- CCL: all_reduce after o_proj and down_proj.
- KV cache: shard by KV heads (dim=1).

## Decode Gotchas
- `nlp_concat_heads_decode` requires a sharded input tensor; if decode attention output is interleaved, use `transpose` + `nlp_concat_heads` or convert to a height-sharded memory config first.
- `nlp_concat_heads`/`nlp_concat_heads_decode` may pad head counts to 32; always slice to `num_heads * head_dim` before a row-parallel matmul.
- `scaled_dot_product_attention_decode` emits `[1, heads, batch, head_dim]`; transpose to `[1, batch, heads, head_dim]` before concatenation.

## Checks and Debug
- Verify prefill first (max_new_tokens 1).
- If decode is wrong, inspect RoPE reshape and cache update positions.
- Confirm output widths: concat heads to num_heads * head_dim, then trim padding.
- Keep one TTNN or TT process at a time on single-card systems.
