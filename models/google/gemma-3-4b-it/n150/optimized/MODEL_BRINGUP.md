# MODEL_BRINGUP.md — Gemma 3 4B IT (n150 optimized)

## Overview
This is an optimized TTNN implementation of `google/gemma-3-4b-it` for n150.

- Model code: `models/google/gemma-3-4b-it/n150/optimized/model.py`
- Functional baseline: `models/google/gemma-3-4b-it/n150/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `demo.py` (TTFT + decode throughput)

## Baseline (n150 functional)
From `MODELS.md`:

- Top-1: 92%
- Top-5: 100%
- TTFT: 98 ms
- Decode: 13.9 t/s/u
- Seq len: 40960

## Optimizations Kept
- Fused QKV projection in attention (single matmul per layer).
- Prefill-last-logits fast path (`prefill_logits_last_device`) so demo/eval TTFT does not compute full prefill logits.
- Decode trace path with preallocated token/position/RoPE buffers:
  - `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, `ttnn.execute_trace`
  - RoPE buffers are updated each token so the traced decode graph is position-independent.
- Decode logits computed for a single lane:
  - Slice to `[1 token]` before `lm_head` to avoid padded decode work.

## Optimizations Tried (Not Kept)
- None yet.

## Repro Commands
Demo (records TTFT + t/s/u):

```
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TT_METAL_CACHE=/tmp/tt-metal-cache \
  python demo.py models/google/gemma-3-4b-it/n150/optimized/model.py --max_seq_len 40960
```

Long teacher-forcing eval (records Top-1/Top-5):

```
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TT_METAL_CACHE=/tmp/tt-metal-cache \
  python eval.py models/google/gemma-3-4b-it/n150/optimized/model.py --model google/gemma-3-4b-it \
    --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100 --max_seq_len 40960
```

## Results (2026-02-12)
From `demo.log` and `eval.log` in this directory:

- Top-1: 92%
- Top-5: 100%
- TTFT: 70.1 ms
- Decode: 14.5 t/s/u
- Seq len: 40960
