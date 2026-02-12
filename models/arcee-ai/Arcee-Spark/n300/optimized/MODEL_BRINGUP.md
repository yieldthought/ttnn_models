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

2026-02-12: refreshed demo/eval artifacts on a healthy discovered `2x1` mesh with `TTNN_ALLOW_SYSTEM_MESH_FALLBACK` unset (no fallback).

## Baseline vs Final (same hardware target)
| Metric | Functional baseline | Optimized final |
| --- | ---: | ---: |
| Top-1 | 91% | 85% |
| Top-5 | 100% | 100% |
| TTFT | 338 ms | 101 ms |
| t/s/u | 5.0 | 16.0 |
| Seq len | 32768 | 32768 |

## Commands Used
Demo:
```bash
TT_METAL_CACHE=/tmp/tt-metal-cache PYTHONUNBUFFERED=1 \
  python demo.py models/arcee-ai/Arcee-Spark/n300/optimized/model.py --seed 0 --max_seq_len 32768
```

Eval:
```bash
TT_METAL_CACHE=/tmp/tt-metal-cache PYTHONUNBUFFERED=1 \
  python eval.py models/arcee-ai/Arcee-Spark/n300/optimized/model.py --model arcee-ai/Arcee-Spark \
    --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100 --max_seq_len 32768 --seed 0
```

## Kept Optimizations
1. `prefill_logits_last_device()` so demo/eval TTFT doesn't transfer the full prefill logits to host.
2. Traced decode execution with preallocated decode token/position/RoPE buffers and `ttnn.execute_trace` replay.
3. Decode LM head slicing: slice the tile-padded batch down to the 1 logical token before the LM head matmul.
4. Keep the final 8 MLP blocks in BF16 weights to preserve long-eval accuracy margin.

## Rejected / Deferred
1. Decode DRAM-sharded matmuls (MLP / `o_proj` / LM head): deferred for now (higher code complexity) since traced decode + last-logits prefill already beats the functional TTFT and tok/s targets on N300.
