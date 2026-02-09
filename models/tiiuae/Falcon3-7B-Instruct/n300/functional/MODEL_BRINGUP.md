# MODEL_BRINGUP.md - Falcon3 7B Instruct (n300 functional)

## Overview
This is a minimal TTNN bringup of `tiiuae/Falcon3-7B-Instruct` for N300 using 1D tensor parallel.
It mirrors the HuggingFace Llama-style architecture with GQA attention and a SwiGLU MLP.

- Model code: `models/tiiuae/Falcon3-7B-Instruct/n300/functional/model.py`
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
- Column-parallel (weight width sharding, dim=3): `q_proj`, `k_proj`, `v_proj`, `mlp.gate_proj`, `mlp.up_proj`, `lm_head`.
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
- `ttnn.transformer.scaled_dot_product_attention` (prefill)
- `ttnn.transformer.paged_scaled_dot_product_attention_decode` (decode)
- `ttnn.experimental.paged_fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## KV cache and tiling constraints
- Cache tensors are allocated as `[max_num_blocks, n_kv_heads, block_size, head_dim]`.
- Cache length is capped to 32768 tokens in this bringup (`MAX_CACHE_SEQ_LEN`).
- `block_size` is 64 and `max_num_blocks = ceil(max_seq_len / 64)`.
- Page table is `[32, max_num_blocks]` in row-major layout with identity mapping.
- Batch dimension is tile-aligned to 32 for decode ops.
- Inputs are padded to tile size (32) before embedding and trimmed at the end.
- Decode uses `cur_pos_tensor` with `-1` entries to skip unused batch slots.

## Precision
- Weights use `ttnn.bfloat8_b`, activations use `ttnn.bfloat16`.

## Evaluation
Teacher-forcing accuracy against the HF reference:

```
python eval.py models/tiiuae/Falcon3-7B-Instruct/n300/functional/model.py \
  --model tiiuae/Falcon3-7B-Instruct \
  --max_seq_len 32768
```

On this N300 host, set the mesh to devices 0 and 2:

```
TT_VISIBLE_DEVICES=0,2 python eval.py models/tiiuae/Falcon3-7B-Instruct/n300/functional/model.py \
  --model tiiuae/Falcon3-7B-Instruct \
  --max_seq_len 32768
```

If `/home` is full, redirect runtime artifacts and HF caches to a writable location:

```
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache TRANSFORMERS_CACHE=/proj_sw/user_dev/moconnor/hf-cache \
  HF_HUB_CACHE=/proj_sw/user_dev/moconnor/hf-cache/hub TT_VISIBLE_DEVICES=0,2 \
  TT_METAL_CACHE=/tmp/tt-metal-cache TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root \
  TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector \
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  python eval.py models/tiiuae/Falcon3-7B-Instruct/n300/functional/model.py \
  --model tiiuae/Falcon3-7B-Instruct \
  --max_seq_len 32768
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model tiiuae/Falcon3-7B-Instruct
```

## Latest validation (2026-02-09)
- Max seq len: 32768 (paged KV cache, block size 64)
- Demo: TTFT 661 ms, decode 5.6 t/s/u
- Eval: Top-1 97%, Top-5 100%
