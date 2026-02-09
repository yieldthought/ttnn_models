# MODEL_BRINGUP.md - Phi-3 Mini 128k Instruct (n300 functional)

## Overview
This is a minimal TTNN bringup of `microsoft/Phi-3-mini-128k-instruct` for N300 using 1D tensor parallel.
It mirrors the HuggingFace architecture with LongRoPE and a gated MLP.

- Model code: `models/microsoft/Phi-3-mini-128k-instruct/n300/functional/model.py`
- Eval harness: `eval.py` and `scripts/run_eval.py`
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Model API contract
- Exposes `build_model(hf_model, tt_device, max_seq_len)`
- Returned class subclasses `torch.nn.Module` and `GenerationMixin`
- Forward returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`

## Parallelism strategy (N300)
- Mesh shape: 1x2 (two devices), linear topology.
- Column-parallel: Q, K, V, gate, up, and lm_head projections (weights sharded on dim=3).
- Row-parallel: attention output projection and MLP down projection (weights sharded on dim=2) with `ttnn.all_reduce`.
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
- `ttnn.transformer.scaled_dot_product_attention` and `ttnn.transformer.paged_scaled_dot_product_attention_decode`
- `ttnn.experimental.paged_fill_cache` and `ttnn.experimental.paged_update_cache`

## RoPE notes
Phi-3 uses LongRoPE (`rope_scaling.type = longrope`). This bringup:
- Applies the attention scaling factor used by HF.
- Pads the RoPE dimension to a multiple of 64 for TT rotary (then slices back).

This bringup precomputes **both** short and long RoPE caches and selects between
them per call:
- Prefill uses the long cache only when `seq_len > original_max_position_embeddings`.
- Decode switches to the long cache when `start_pos >= original_max_position_embeddings`.
This matches HF’s dynamic LongRoPE behavior so sequences that cross 4096 during
decode use the correct frequency factors.

## KV cache and tiling constraints
- Cache tensors are allocated as `[max_num_blocks, n_kv_heads, block_size, head_dim]`.
- Cache length is capped to 12288 tokens in this bringup (`MAX_CACHE_SEQ_LEN`) with a block size of 64.
- Page table is an identity mapping of shape `[32, max_num_blocks]` (int32, row-major).
- Batch dimension is tile-aligned to 32 for decode ops.
- Set unused decode positions to `-1` in `cur_pos_tensor` so paged ops skip them.
- Inputs are padded to tile size (32) before embedding and trimmed at the end.

## Validated max sequence length
- Target `max_seq_len` is 12288 to match the n150 row and stay within N300 DRAM.
- Demo and eval logs in this directory were run with `--max_seq_len 12288`.

## Precision
- Weights and activations use `ttnn.bfloat16`.

## Evaluation
Teacher-forcing accuracy against the HF reference:

```
python eval.py models/microsoft/Phi-3-mini-128k-instruct/n300/functional/model.py \
  --model microsoft/Phi-3-mini-128k-instruct --max_seq_len 12288
```

On this N300 host, set the mesh to device 0:

```
TT_VISIBLE_DEVICES=0 python eval.py models/microsoft/Phi-3-mini-128k-instruct/n300/functional/model.py \
  --model microsoft/Phi-3-mini-128k-instruct --max_seq_len 12288
```

If `/home` is full, redirect runtime artifacts to a writable location. On this host,
`/proj_sw/user_dev/moconnor/tt-runtime-root` is a symlinked runtime root (with `tt_metal`,
`ttnn`, and `runtime` from the installed package):

```
TT_VISIBLE_DEVICES=0 TT_METAL_CACHE=/tmp/tt-metal-cache TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 python eval.py models/microsoft/Phi-3-mini-128k-instruct/n300/functional/model.py --model microsoft/Phi-3-mini-128k-instruct --max_seq_len 12288
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model microsoft/Phi-3-mini-128k-instruct
```
