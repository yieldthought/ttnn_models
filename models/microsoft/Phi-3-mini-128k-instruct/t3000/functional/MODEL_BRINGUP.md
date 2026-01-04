# MODEL_BRINGUP.md - Phi-3 Mini 128k Instruct (t3000 functional)

## Overview
This is a minimal TTNN bringup of `microsoft/Phi-3-mini-128k-instruct` for T3000 using 1D tensor parallel.
It mirrors the HuggingFace architecture with LongRoPE and a gated MLP.

- Model code: `models/microsoft/Phi-3-mini-128k-instruct/t3000/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Model API contract
- Exposes `build_model(hf_model, tt_device, max_seq_len)`
- Returned class subclasses `torch.nn.Module` and `GenerationMixin`
- Forward returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`

## Parallelism strategy (T3000)
- Mesh shape: 2x4 (eight devices), linear topology.
- Column-parallel: Q, K, V, gate, up, and lm_head projections (weights sharded on dim=3).
- Row-parallel: attention output projection and MLP down projection (weights sharded on dim=2) with `ttnn.all_reduce` across the full mesh.
- KV cache is sharded across devices on dim=1 (KV heads).
- Input tokens, embeddings, and RMSNorm weights are replicated.

## Parallelization summary
- Replicated tensors: token embeddings, RMSNorm weights, RoPE caches, input tokens.
- Column-parallel (weight width sharding, dim=3): Q/K/V splits from `qkv_proj`, `mlp.gate_proj`, `mlp.up_proj`, `lm_head`.
- Row-parallel (weight height sharding, dim=2): `o_proj`, `mlp.down_proj`.
- KV cache: sharded by KV heads (dim=1) across devices.
- CCL ops: `ttnn.all_reduce` after `o_proj` and after `mlp.down_proj` to sum partials.
- Output composition: `ttnn.to_torch(..., mesh_composer=ConcatMeshToTensor)` to gather vocab shards on host.

## Key TTNN ops
- `ttnn.embedding` for token embeddings
- `ttnn.linear` for Q/K/V, output, and MLP projections
- `ttnn.rms_norm` for RMSNorm
- `ttnn.experimental.rotary_embedding` for RoPE
- `ttnn.experimental.nlp_create_qkv_heads[_decode]` and `ttnn.experimental.nlp_concat_heads`
- `ttnn.transformer.scaled_dot_product_attention[_decode]`
- `ttnn.fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## RoPE notes
Phi-3 uses LongRoPE (`rope_scaling.type = longrope`). This bringup:
- Applies the attention scaling factor used by HF.
- Uses the short or long frequency factors based on `max_seq_len` vs
  `original_max_position_embeddings` (4096).
- Pads the RoPE dimension to a multiple of 64 for TT rotary (then slices back).

If you need a prompt that crosses 4096 after prefill, rerun with a `max_seq_len`
large enough for that prompt so the long factors are used consistently.

## KV cache and tiling constraints
- Cache tensors are allocated as `[32, n_kv_heads, cache_seq_len, head_dim]`.
- Cache length is capped to 256 tokens in this bringup (`MAX_CACHE_SEQ_LEN`) to fit on device.
- Batch dimension is tile-aligned to 32 for decode ops.
- Inputs are padded to tile size (32) before embedding and trimmed at the end.
- Prefill uses height-sharded KV inputs for `fill_cache` to avoid interleaved grid-size limits.
- Sharded prefill requires `n_kv_heads` to be divisible by the device grid x-dimension.

## Precision
- Weights and activations use `ttnn.bfloat16`.

## Evaluation
Teacher-forcing accuracy against the HF reference:

```
TT_MESH_GRAPH_DESC_PATH=/home/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python eval.py models/microsoft/Phi-3-mini-128k-instruct/t3000/functional/model.py \
  --model microsoft/Phi-3-mini-128k-instruct \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100
```

If `/home` is full, redirect runtime artifacts to a writable location. On this host,
`/proj_sw/user_dev/moconnor/tt-runtime-root` is a symlinked runtime root (with `tt_metal`,
`ttnn`, and `runtime` from the installed package):

```
TT_MESH_GRAPH_DESC_PATH=/home/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TT_METAL_CACHE=/tmp/tt-metal-cache \
TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root \
TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
python eval.py models/microsoft/Phi-3-mini-128k-instruct/t3000/functional/model.py \
  --model microsoft/Phi-3-mini-128k-instruct \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100
```

Automation wrapper (emits YT_METRICS JSON):

```
TT_MESH_GRAPH_DESC_PATH=/home/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/run_eval.py --mode tt --hf-model microsoft/Phi-3-mini-128k-instruct
```
