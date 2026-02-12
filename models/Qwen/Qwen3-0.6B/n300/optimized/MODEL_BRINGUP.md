# MODEL_BRINGUP.md - Qwen3 0.6B (n300 optimized)

## Overview
This is the optimized TTNN bringup of `Qwen/Qwen3-0.6B` for `n300`.

- Model code: `models/Qwen/Qwen3-0.6B/n300/optimized/model.py`
- Demo log: `models/Qwen/Qwen3-0.6B/n300/optimized/demo.log`
- Eval log: `models/Qwen/Qwen3-0.6B/n300/optimized/eval.log`
- Machine-readable metrics: `models/Qwen/Qwen3-0.6B/n300/optimized/metrics.json`
- Decode path uses traced execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`).

## Baseline vs final
| Metric | Functional baseline (`MODELS.md`) | Final optimized |
| --- | ---: | ---: |
| Top-1 | 99% | 99% |
| Top-5 | 100% | 100% |
| TTFT | 943 ms | 54 ms |
| t/s/u | 2.0 | 55.3 |
| Seq len | 40960 | 40960 |

Note: TTFT and t/s/u depend on prompt and sampling settings. See `demo.log` for exact command and output.

## Kept optimization decisions
1. Decode trace with preallocated decode buffers.
- Preallocates token ids, position tensor, and per-token RoPE cos/sin buffers.
- Captures a stable decode graph and replays it with `ttnn.execute_trace`.

2. Prefill last-token logits fast path (`prefill_logits_last_device`).
- Avoids materializing full prefill logits when only the final prompt token is needed.

3. Paged KV cache (block_size=64).
- Cache layout: `[max_num_blocks, n_kv_heads, block_size, head_dim]`.
- Uses identity page table and paged update/fill ops.

4. Fused QKV projection.
- Replaces 3 matmuls (`q_proj`, `k_proj`, `v_proj`) + concat with one `qkv_proj` matmul.
- Keeps TP-safe shard ordering per device: `[q_i, k_i, v_i]`.

5. BFP8 weights for MLP projections.
- Uses `ttnn.bfloat8_b` for `gate_proj`, `up_proj`, and `down_proj`.
- Kept because long-eval quality remained at 99/100 while decode throughput improved.

## Rejected optimization attempts
1. Single-token decode RoPE buffer (remove per-head RoPE repeats and use `token_index=0` in decode rotary).
- Measured run: TTFT `46.9 ms`, decode `53.3 t/s/u`.
- Decision: rejected. TTFT improved, but decode throughput regressed versus retained optimized behavior.

2. Explicit decode SDPA program/compute config (`ttnn.SDPAProgramConfig` + `WormholeComputeKernelConfig`).
- Measured runs: `43.2 ms / 54.8 t/s/u` and `51.8 ms / 52.3 t/s/u`.
- Decision: rejected. Throughput was not a repeatable improvement and regressed on repeat.

3. `DECODE_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG`.
- Measured runs: `43.8 ms / 57.5 t/s/u` and `75.5 ms / 38.4 t/s/u`.
- Decision: rejected. Large run-to-run instability makes this unsuitable for the optimized default.

### 2026-02-12 sweep summary
- Retained config: unchanged from the existing optimized implementation in this directory.
- Spot re-baselines with retained config were in the same envelope (TTFT `~50-53 ms`, decode `~54-55 t/s/u`) with Top-1 `99%`, Top-5 `100%`.
- Since no candidate delivered a repeatable net win, `MODELS.md` and `metrics.json` remain unchanged.

## Commands used
```bash
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal \
PYTHONUNBUFFERED=1 \
python -u demo.py models/Qwen/Qwen3-0.6B/n300/optimized/model.py \
  --seed 0 \
  --max_seq_len 40960

TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal \
PYTHONUNBUFFERED=1 \
python -u eval.py models/Qwen/Qwen3-0.6B/n300/optimized/model.py \
  --model Qwen/Qwen3-0.6B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 40960 \
  --seed 0
```
