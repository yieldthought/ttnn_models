# MODEL_BRINGUP.md - Phi-3 Mini 128k Instruct (n300 optimized)

## Overview
This is the optimized TTNN bringup of `microsoft/Phi-3-mini-128k-instruct` for `n300`.

- Model code: `models/microsoft/Phi-3-mini-128k-instruct/n300/optimized/model.py`
- Demo log: `models/microsoft/Phi-3-mini-128k-instruct/n300/optimized/demo.log`
- Eval log: `models/microsoft/Phi-3-mini-128k-instruct/n300/optimized/eval.log`
- Machine-readable metrics: `models/microsoft/Phi-3-mini-128k-instruct/n300/optimized/metrics.json`
- Decode path uses traced execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`).

## Baseline vs final
| Metric | Functional baseline (`MODELS.md`) | Final optimized |
| --- | ---: | ---: |
| Top-1 | 90% | 91% |
| Top-5 | 100% | 100% |
| TTFT | 193 ms | 93.78 ms |
| t/s/u | 6.7 | 18.27 |
| Seq len | 12288 | 12288 |

## Kept optimization decisions
1. Decode trace with preallocated decode buffers.
- Preallocates decode token ids, position tensor, and per-token RoPE buffers.
- Captures once and replays decode via `ttnn.execute_trace`.

2. Prefill last-token logits fast path (`prefill_logits_last_device`).
- Avoids materializing full prefill logits when only the final prompt token is needed.
- Reduces prompt-time work in demo/eval flow.

3. Fused QKV projection with TP-safe shard ordering.
- Uses one QKV matmul per attention block.
- Reorders fused QKV chunks as `[q_i, k_i, v_i]` per shard so local head layout matches decode/prefill kernels.

4. Keep decode QKV head creation outputs in L1 (`DECODE_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG`).
- This is a low-complexity decode-path memory placement improvement.
- Long eval accuracy remained unchanged (91/100).

## Rejected optimization attempts
1. Pushing decode `o_proj` and MLP linears to `DECODE_MEMORY_CONFIG`.
- Added decode-only `memory_config` overrides for attention `o_proj` and MLP projections.
- Rejected due throughput regression in measurement loop versus the kept variant.

## Commands used
```bash
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0 \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u demo.py models/microsoft/Phi-3-mini-128k-instruct/n300/optimized/model.py \
  --max_seq_len 12288 \
  --seed 0

TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0 \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u eval.py models/microsoft/Phi-3-mini-128k-instruct/n300/optimized/model.py \
  --model microsoft/Phi-3-mini-128k-instruct \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 12288 \
  --seed 0
```
