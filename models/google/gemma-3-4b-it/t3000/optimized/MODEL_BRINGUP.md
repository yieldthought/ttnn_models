# MODEL_BRINGUP.md — Gemma 3 4B IT (t3000 optimized)

## Overview
This is an optimized TTNN implementation of `google/gemma-3-4b-it` for T3000 using 1D tensor parallel.

- Model code: `models/google/gemma-3-4b-it/t3000/optimized/model.py`
- Functional baseline: `models/google/gemma-3-4b-it/t3000/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `demo.py` (TTFT + decode throughput)

## Baseline (t3000 functional)
From `MODELS.md`:

- Top-1: 92%
- Top-5: 100%
- TTFT: 330 ms
- Decode: 4.7 t/s/u
- Seq len: 40960

## Optimizations Kept
- Fused QKV projection in attention (single matmul for Q/K/V) with TP-safe shard ordering.
- Prefill-last-logits fast path (`prefill_logits_last_device`) so demo/eval TTFT does not compute full prefill logits.
- Decode trace path with preallocated token/position/RoPE buffers:
  - `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, `ttnn.execute_trace`
  - RoPE buffers are updated each token so the traced decode graph is position-independent.
- Decode logits computed for a single lane:
  - Slice to `[1 token]` before `lm_head` to avoid padded decode work and reduce host transfer.

## Optimizations Tried (Not Kept)
- None yet.

## Repro Commands
On a T3000 host, set the mesh graph descriptor and use all 8 devices:

Demo (records TTFT + t/s/u):

```
env HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
  TT_VISIBLE_DEVICES=0,1,2,3 \
  TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
  TT_METAL_CACHE=/tmp/tt-metal-cache \
  TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  python demo.py models/google/gemma-3-4b-it/t3000/optimized/model.py --max_seq_len 40960
```

Long teacher-forcing eval (records Top-1/Top-5):

```
env HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
  TT_VISIBLE_DEVICES=0,1,2,3 \
  TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
  TT_METAL_CACHE=/tmp/tt-metal-cache \
  TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  python eval.py models/google/gemma-3-4b-it/t3000/optimized/model.py --model google/gemma-3-4b-it \
    --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100 --max_seq_len 40960
```

If you hit cache/mmap issues, use the repo runbook (`AGENTS.md`) and redirect the metal cache/runtime root.

## Results (2026-02-12)
From `demo.log` and `eval.log` in this directory:

- Top-1: 91%
- Top-5: 100%
- TTFT: 78 ms
- Decode: 19.4 t/s/u
- Seq len: 40960
