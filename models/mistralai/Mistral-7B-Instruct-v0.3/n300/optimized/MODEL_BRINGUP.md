# MODEL_BRINGUP.md - Mistral-7B-Instruct-v0.3 (n300 optimized)

## Overview
This is the optimized N300 path for `mistralai/Mistral-7B-Instruct-v0.3` using 1D tensor parallel.

- Model code: `models/mistralai/Mistral-7B-Instruct-v0.3/n300/optimized/model.py`
- Demo log: `models/mistralai/Mistral-7B-Instruct-v0.3/n300/optimized/demo.log`
- Eval log: `models/mistralai/Mistral-7B-Instruct-v0.3/n300/optimized/eval.log`
- Parallelism: 1x2 mesh, linear topology (N300)
- Max seq len: `32768` (no capability regression target)
- Decode path: traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`)

## Baseline vs Final (same hardware)
| Metric | Functional baseline | Optimized final |
| --- | ---: | ---: |
| Top-1 | 96% | TBD |
| Top-5 | 100% | TBD |
| TTFT | 112 ms | TBD |
| t/s/u | 11.1 | TBD |
| Seq len | 32768 | 32768 |

Note: On this host, the N300 2-chip fabric link is not healthy, so the real 1x2 mesh cannot be opened.
`demo.py` / `eval.py` fail during mesh discovery with `phys_deg_hist={0:2}` when using the standard
`n300_mesh_graph_descriptor`. This must be rerun on a healthy N300 host to collect final metrics.

## Commands used
Demo:
```bash
env HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
  TRANSFORMERS_CACHE=/proj_sw/user_dev/moconnor/hf-cache \
  HF_HUB_CACHE=/proj_sw/user_dev/moconnor/hf-cache/hub \
  TT_VISIBLE_DEVICES=5,6 \
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
  TT_VISIBLE_DEVICES=5,6 \
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
1. Fused attention Q/K/V projection into a single matmul (`self.qkv_proj`) with per-device chunk ordering matching width sharding.
2. `prefill_logits_last_device()` prefill fast path for demo/eval TTFT (compute and transfer only last-token logits).
3. Decode trace with preallocated token/position/RoPE buffers and `ttnn.execute_trace` replay.
4. Decode LM head runs on a sliced `[1, 1, 1, hidden]` activation (avoid padded-32 LM head work).

## Mesh health check
To validate whether this host can run a real N300 1x2 mesh, run:

```bash
/proj_sw/user_dev/moconnor/tt-metal/build_Release/tools/umd/system_health
```

If all "internal trace" ETH links report `DOWN/unconnected`, the host cannot run multi-chip N300 workloads.
