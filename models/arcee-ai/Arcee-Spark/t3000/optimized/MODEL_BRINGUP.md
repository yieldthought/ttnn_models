# MODEL_BRINGUP.md — Arcee-Spark (t3000 optimized)

## Overview
This is the optimized T3000 path for `arcee-ai/Arcee-Spark`.

- Model code: `models/arcee-ai/Arcee-Spark/t3000/optimized/model.py`
- Demo log: `models/arcee-ai/Arcee-Spark/t3000/optimized/demo.log`
- Eval log: `models/arcee-ai/Arcee-Spark/t3000/optimized/eval.log`
- Parallelism: 2x4 mesh, 4-way tensor parallel over mesh columns with row replication
- Max seq len: `32768` (no capability regression vs functional)
- Decode path: traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`)
- Retry validation date: `2026-02-11` on healthy 8-chip auto-discovered mesh (`n_log=8`, `n_phys=8`)

## Baseline vs Final (same hardware)
| Metric | Functional baseline | Optimized final |
| --- | ---: | ---: |
| Top-1 | 90% | 90% |
| Top-5 | 100% | 100% |
| TTFT | 343 ms | 72 ms |
| t/s/u | 4.9 | 17.6 |
| Seq len | 32768 | 32768 |

## Commands used
Demo:
```bash
env TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python demo.py models/arcee-ai/Arcee-Spark/t3000/optimized/model.py
```

Eval:
```bash
env TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python eval.py models/arcee-ai/Arcee-Spark/t3000/optimized/model.py --model arcee-ai/Arcee-Spark --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100 --max_seq_len 32768
```

## Kept optimizations
1. Added `prefill_logits_last_device()` so demo/eval prefill timing only computes the final prompt token logits.
2. Added traced decode execution with preallocated decode token/position/RoPE buffers and `ttnn.execute_trace` replay.
3. Added explicit trace lifecycle handling (`release_trace` on reset) to keep repeated runs stable.

## Rejected/Deferred changes (measured this retry)
1. Attention weights all `bfloat8_b` (Q/K/V/O): demo improved to `67 ms`, `18.0 t/s/u` but long eval dropped to `Top-1 89%` (`Top-5 100%`), so rejected for quality regression.
2. Attention `o_proj` only `bfloat8_b` (Q/K/V kept `bfloat16`): demo was `64 ms`, `17.8 t/s/u` but long eval dropped to `Top-1 88%` (`Top-5 100%`), so rejected.
3. No fused-QKV structural rewrite in this pass; current trace + prefill-last path already exceeds release performance deltas without extra maintenance complexity.
