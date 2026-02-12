# MODEL_BRINGUP.md - Phi-3 Mini 128k Instruct (n150 optimized)

## Overview
This is the optimized TTNN bringup of `microsoft/Phi-3-mini-128k-instruct` for `n150`.

- Model code: `models/microsoft/Phi-3-mini-128k-instruct/n150/optimized/model.py`
- Demo log: `models/microsoft/Phi-3-mini-128k-instruct/n150/optimized/demo.log`
- Eval log: `models/microsoft/Phi-3-mini-128k-instruct/n150/optimized/eval.log`
- Machine-readable metrics: `models/microsoft/Phi-3-mini-128k-instruct/n150/optimized/metrics.json`
- Decode path uses traced execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`).

## Baseline vs final
| Metric | Functional baseline (`MODELS.md`) | Final optimized |
| --- | ---: | ---: |
| Top-1 | 92% | 94% |
| Top-5 | 99% | 99% |
| TTFT | 80 ms | 69 ms |
| t/s/u | 13.7 | 15.9 |
| Seq len | 12288 | 12288 |

Note: TTFT and t/s/u depend on the demo prompt and sampling settings. See `demo.log` for the exact command and output.

## Kept optimization decisions
1. Decode trace with preallocated decode buffers.
- Preallocates decode token ids, position tensor, and per-token RoPE cos/sin buffers.
- Captures a stable decode graph after prefill and replays it with `ttnn.execute_trace`.
- LongRoPE short/long cache selection is handled by updating per-token RoPE buffers on host based on `start_pos`.

2. Prefill last-token logits fast path (`prefill_logits_last_device`).
- Avoids materializing full prefill logits when only the final prompt token is needed (demo/eval TTFT).

3. BFP8 MLP weights.
- Switched `gate_up_proj` and `down_proj` weights to `ttnn.bfloat8_b`.
- Kept attention and LM head weights in `ttnn.bfloat16`.
- This change provided the throughput gain needed to clear the n150 baseline while keeping long-eval quality above target.

## Rejected optimization attempts
1. Oversized trace region (`TTNN_TRACE_REGION_SIZE=50000000`).
- It fixed trace-capacity errors but caused DRAM OOM during model load.
- Kept `TTNN_TRACE_REGION_SIZE=13000000` instead, which is enough for trace capture and still fits the model.

## Commands used
```bash
TTNN_TRACE_REGION_SIZE=13000000 TT_METAL_CACHE=/tmp/tt-metal-cache PYTHONUNBUFFERED=1 \
python -u demo.py models/microsoft/Phi-3-mini-128k-instruct/n150/optimized/model.py \
  --seed 0 \
  --max_seq_len 12288

TTNN_TRACE_REGION_SIZE=13000000 TT_METAL_CACHE=/tmp/tt-metal-cache PYTHONUNBUFFERED=1 \
python -u eval.py models/microsoft/Phi-3-mini-128k-instruct/n150/optimized/model.py \
  --model microsoft/Phi-3-mini-128k-instruct \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 12288 \
  --seed 0
```
