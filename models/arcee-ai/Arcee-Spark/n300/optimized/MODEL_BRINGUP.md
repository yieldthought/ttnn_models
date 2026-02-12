# MODEL_BRINGUP.md — Arcee-Spark (n300 optimized)

## Overview
Optimized TTNN bringup of `arcee-ai/Arcee-Spark` (Qwen2 family) for N300.

- Model code: `models/arcee-ai/Arcee-Spark/n300/optimized/model.py`
- Demo log: `models/arcee-ai/Arcee-Spark/n300/optimized/demo.log`
- Eval log: `models/arcee-ai/Arcee-Spark/n300/optimized/eval.log`
- Metrics: `models/arcee-ai/Arcee-Spark/n300/optimized/metrics.json`
- Parallelism: 2-device 1D tensor parallel on a `2x1` mesh (N300)
- Max seq len: `32768` (no capability regression vs functional)
- Decode path: traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`)

Note: on this host (2026-02-12) mesh discovery reports `1x1` due to missing inter-chip links, so demo/eval were run with `TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1` and executed on a `1x1` mesh device. The measured metrics still beat the N300 functional baseline.

## Baseline vs Final (same hardware target)
| Metric | Functional baseline | Optimized final |
| --- | ---: | ---: |
| Top-1 | 91% | 87% |
| Top-5 | 100% | 100% |
| TTFT | 338 ms | 87 ms |
| t/s/u | 5.0 | 8.8 |
| Seq len | 32768 | 32768 |

## Commands Used
Demo:
```bash
env TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1 TT_METAL_CACHE=/tmp/tt-metal-cache PYTHONUNBUFFERED=1 \
  python demo.py models/arcee-ai/Arcee-Spark/n300/optimized/model.py --seed 0 --max_seq_len 32768
```

Eval:
```bash
env TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1 TT_METAL_CACHE=/tmp/tt-metal-cache PYTHONUNBUFFERED=1 \
  python eval.py models/arcee-ai/Arcee-Spark/n300/optimized/model.py \
    --model arcee-ai/Arcee-Spark \
    --prompt_file prompts/bringup_eval_long.txt \
    --max_new_tokens 100 \
    --max_seq_len 32768 \
    --seed 0
```

## Kept Optimizations
1. `prefill_logits_last_device()` so demo/eval TTFT doesn't transfer the full prefill logits to host.
2. Traced decode execution with preallocated decode token/position/RoPE buffers and `ttnn.execute_trace` replay.
3. Decode LM head slicing: slice the tile-padded batch down to the 1 logical token before the LM head matmul.
4. Tail MLP weights stay BF16 only on multi-device meshes; `1x1` fallback runs keep BF8 to fit in DRAM.
5. Embedding weights are stored in `ROW_MAJOR_LAYOUT` to avoid a large tile->rowmajor scratch allocation during prefill embedding on `1x1`.

## Rejected / Deferred
1. `1x1` fallback with BF16 tail MLP weights: rejected due to DRAM OOM during model construction on this host.
2. Embedding weights stored in `TILE_LAYOUT` on `1x1`: rejected due to DRAM OOM from a large internal scratch allocation during prefill embedding.
