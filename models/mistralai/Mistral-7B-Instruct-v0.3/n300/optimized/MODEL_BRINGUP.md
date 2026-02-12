# MODEL_BRINGUP.md - Mistral-7B-Instruct-v0.3 (n300 optimized)

## Overview
This is the optimized N300 path for `mistralai/Mistral-7B-Instruct-v0.3`.

- Model code: `models/mistralai/Mistral-7B-Instruct-v0.3/n300/optimized/model.py`
- Demo log: `models/mistralai/Mistral-7B-Instruct-v0.3/n300/optimized/demo.log`
- Eval log: `models/mistralai/Mistral-7B-Instruct-v0.3/n300/optimized/eval.log`
- Parallelism: 1D tensor parallel across a 1x2 N300 mesh
- Max seq len: `32768` (no capability regression)
- Decode path: traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`)

## Baseline vs Final (same hardware)
| Metric | Functional baseline | Optimized final |
| --- | ---: | ---: |
| Top-1 | 96% | 97% |
| Top-5 | 100% | 100% |
| TTFT | 112 ms | 44 ms |
| t/s/u | 11.1 | 24.8 |
| Seq len | 32768 | 32768 |

## Commands used
Demo:
```bash
env HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
  TRANSFORMERS_CACHE=/proj_sw/user_dev/moconnor/hf-cache \
  HF_HUB_CACHE=/proj_sw/user_dev/moconnor/hf-cache/hub \
  TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
  TT_METAL_CACHE=/tmp/tt-metal-cache \
  TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal \
  PYTHONUNBUFFERED=1 \
  python -u demo.py models/mistralai/Mistral-7B-Instruct-v0.3/n300/optimized/model.py --max_seq_len 32768
```

Eval:
```bash
env HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
  TRANSFORMERS_CACHE=/proj_sw/user_dev/moconnor/hf-cache \
  HF_HUB_CACHE=/proj_sw/user_dev/moconnor/hf-cache/hub \
  TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
  TT_METAL_CACHE=/tmp/tt-metal-cache \
  TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal \
  PYTHONUNBUFFERED=1 \
  python -u eval.py models/mistralai/Mistral-7B-Instruct-v0.3/n300/optimized/model.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --prompt_file prompts/bringup_eval_long.txt \
    --max_new_tokens 100 \
    --max_seq_len 32768
```

## Kept optimizations
1. Fused attention Q/K/V projections into one matmul (`self.qkv_proj`) with per-device Q/K/V chunk ordering that matches width sharding.
2. Added `prefill_logits_last_device()` so demo/eval TTFT uses only final prompt-token logits.
3. Decode trace with preallocated token/position/RoPE buffers and `ttnn.execute_trace` replay.
4. Decode RoPE runs from precomputed per-token cos/sin buffers with separate Q and K buffer widths for GQA.
5. Decode attention intermediates in L1 (`qkv` matmul, `nlp_create_qkv_heads_decode`, paged SDPA decode, decode `nlp_concat_heads`).
6. Decode LM head runs on a sliced `[1, 1, 1, hidden]` activation (avoid padded-32 LM head work).
7. Prefill MLP now uses fused SwiGLU (`ttnn.mul(..., input_tensor_a_activations=[SILU])`) to remove an extra prefill activation op and reduce TTFT.

## Deferred changes
1. Decode-only sharded matmul program-config tuning was deferred because the traced + L1 path already exceeds target TTFT and decode throughput with lower code complexity.
2. Additional structural changes were deferred to keep this optimized path aligned with existing N300 tensor-parallel bringup contracts.
