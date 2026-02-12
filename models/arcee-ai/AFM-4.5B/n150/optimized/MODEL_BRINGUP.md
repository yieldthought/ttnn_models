# MODEL_BRINGUP.md — AFM-4.5B (n150 optimized)

## Overview
This is the optimized N150 path for `arcee-ai/AFM-4.5B`.

- Model code: `models/arcee-ai/AFM-4.5B/n150/optimized/model.py`
- Demo log: `models/arcee-ai/AFM-4.5B/n150/optimized/demo.log`
- Eval log: `models/arcee-ai/AFM-4.5B/n150/optimized/eval.log`
- Max seq len: `65536` (no capability regression)
- Decode path: traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`)

## Baseline vs Final (same hardware)
| Metric | Functional baseline | Optimized final |
| --- | ---: | ---: |
| Top-1 | 98% | 98% |
| Top-5 | 100% | 100% |
| TTFT | 72 ms | 57 ms |
| t/s/u | 17.2 | 19.6 |
| Seq len | 65536 | 65536 |

## Commands used
Demo:
```bash
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u demo.py models/arcee-ai/AFM-4.5B/n150/optimized/model.py --seed 0 --max_seq_len 65536
```

Eval:
```bash
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u eval.py models/arcee-ai/AFM-4.5B/n150/optimized/model.py \
  --model arcee-ai/AFM-4.5B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 65536
```

## Kept optimizations
1. Fused attention QKV projection into one matmul (`self.qkv_proj`).
2. Added `prefill_logits_last_device()` so demo/eval TTFT uses only last-token logits extraction (and computes the lm_head only for that token).
3. Added traced decode execution with preallocated decode token/position/RoPE buffers and `ttnn.execute_trace` replay.
4. Moved decode attention intermediates to L1 (`ttnn.linear` decode QKV output, `nlp_create_qkv_heads_decode`, decode SDPA, and decode `nlp_concat_heads`) to reduce decode-path memory traffic.

