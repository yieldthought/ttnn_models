# MODEL_BRINGUP.md — Gemma 3 4B IT (t3000 optimized)

## Overview
This is an optimized TTNN bringup of `google/gemma-3-4b-it` for T3000 using 1D tensor parallel.
It keeps the functional code structure but adds high-leverage performance wins:
- `prefill_logits_last_device()` to avoid transferring full prefill logits to host for demo timing.
- Traced decode execution with preallocated device buffers (tokens, positions, RoPE slices).

- Model code: `models/google/gemma-3-4b-it/t3000/optimized/model.py`
- Demo log: `models/google/gemma-3-4b-it/t3000/optimized/demo.log`
- Eval log: `models/google/gemma-3-4b-it/t3000/optimized/eval.log`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/optimized/model.py`

## Model API contract
- The model exposes a `build_model(hf_model, tt_device, max_seq_len)` function.
- The returned class subclasses `torch.nn.Module` and `GenerationMixin` so HF `generate()` works.
- The forward method returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`.

## Parallelism strategy (T3000)
- Mesh shape: 2x4 (eight devices), linear topology.
- Column-parallel: QKV, gate, up, and lm_head projections (weights sharded on dim=3).
- Row-parallel: attention output projection and MLP down projection (weights sharded on dim=2) with `ttnn.all_reduce` across the full mesh.
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
- `ttnn.linear` for QKV, output, and MLP projections
- `ttnn.rms_norm` for RMSNorm and Q/K head norm
- `ttnn.experimental.rotary_embedding` for HuggingFace-format RoPE
- `ttnn.experimental.nlp_create_qkv_heads[_decode]` and `ttnn.experimental.nlp_concat_heads`
- `ttnn.transformer.scaled_dot_product_attention` (prefill) and
  `ttnn.transformer.paged_scaled_dot_product_attention_decode` (decode)
- `ttnn.experimental.paged_fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## Gemma3 specifics
- Q/K RMSNorm uses `(1 + weight)` (Gemma3RMSNorm).
- Embeddings are scaled by `sqrt(hidden_size)` with bfloat16 rounding.
- Global RoPE uses linear scaling (`rope_scaling.factor = 8`).
- Sliding layers use local RoPE with `rope_local_base_freq`. If `layer_types` is present,
  it drives the sliding/global selection; otherwise the `sliding_window_pattern` (default 6)
  is used as a fallback.
- Sliding-window masking is not implemented; it only matters for very long contexts.

## RoPE notes
Decode path detail:
- `rotary_embedding` with `start_pos` expects `[seq_len, 1, B, head_dim]`. For decode,
  reshape Q and K to merge heads into the batch (`[1, 1, B*heads, head_dim]`), apply
  RoPE, then reshape back to `[1, B, heads, head_dim]`.
For traced decode, the model avoids passing `start_pos` into traced ops by copying the
per-token RoPE cos/sin slices into device buffers each step (global + local RoPE),
then calling `rotary_embedding(q, cos_slice, sin_slice)` inside the trace.

## KV cache and tiling constraints
- Paged cache tensors are `[max_num_blocks, n_kv_heads, PAGED_BLOCK_SIZE, head_dim]`.
- `max_num_blocks = ceil(max_seq_len / PAGED_BLOCK_SIZE)` with `PAGED_BLOCK_SIZE = 64`.
- This bringup runs with `max_seq_len = 40960` (matches the n150 row).

## Precision
- Weights use `ttnn.bfloat8_b`.
- Activations use `ttnn.bfloat16`.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after logits are returned.
Decode slices the hidden state down to a single active lane before the LM head projection.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.

```
python eval.py models/google/gemma-3-4b-it/t3000/optimized/model.py --model google/gemma-3-4b-it \
  --max_new_tokens 100 --max_seq_len 40960
```

On this T3000 host, set the mesh graph descriptor and use the PCI devices (the runtime
discovers the remote chips to form the 2x4 mesh):

```
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
TT_VISIBLE_DEVICES=0,1,2,3 \
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
python eval.py models/google/gemma-3-4b-it/t3000/optimized/model.py --model google/gemma-3-4b-it \
  --max_new_tokens 100 --max_seq_len 40960
```

Demo timing (includes one warmup prefill+decode pass in `demo.py`):

```
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
TT_VISIBLE_DEVICES=0,1,2,3 \
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
python demo.py models/google/gemma-3-4b-it/t3000/optimized/model.py --max_seq_len 40960
```

Automation wrapper (emits YT_METRICS JSON):

```
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0,1,2,3 \
python scripts/run_eval.py --mode tt --hf-model google/gemma-3-4b-it
```

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi -r`.

## Baseline (t3000 functional)
Baseline for comparison: `models/google/gemma-3-4b-it/t3000/functional`.

- Top-1: 92%
- Top-5: 100%
- TTFT: 330 ms
- Decode: 4.7 t/s/u
- Seq len: 40960

## Changes (2026-02-12)
- Added traced decode execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`) with preallocated
  device buffers for token ids, positions, and RoPE slices (global + local).
- Added `prefill_logits_last_device()` to avoid transferring the full prefill logits back to host for demo TTFT.
- Re-ran demo + long teacher-forcing eval and recorded logs in this directory.

## Optimization decisions
Kept:
- Traced decode execution with allocation-free decode loop (after trace capture).
- Per-step host copies into persistent device buffers (token ids, cur pos, global/local RoPE slices).
- Decode hidden-state slicing down to a single active lane before `lm_head` projection.
- `prefill_logits_last_device()` for demo TTFT wins.

Rejected / not attempted:
- Decode matmul sharding and program-config tuning: skipped since TTFT and tok/s targets were already met.

### Latest results (t3000 optimized)
- Top-1: 95%
- Top-5: 100%
- TTFT: 94 ms
- Decode: 17.6 t/s/u
- Seq len: 40960
