# MODEL_BRINGUP.md — Gemma 3 4B IT (n300 optimized)

## Target
Optimize `google/gemma-3-4b-it` on N300 for lower TTFT and higher decode throughput while preserving long teacher-forcing quality.

Release requirements:
- Long eval: Top-1 >= 85%, Top-5 >= 95%
- Decode uses traced execution
- TTFT lower than n300 functional baseline
- t/s/u higher than n300 functional baseline
- No seq-len regression vs functional

## Baseline (n300 functional)
From `MODELS.md`:
- Top-1: 94%
- Top-5: 100%
- TTFT: 535 ms
- t/s/u: 3.2
- Seq len: 40960

## Iteration baseline (before this change)
From fresh runs on 2026-02-12 with the prior `n300/optimized` code:
- Top-1: 94%
- Top-5: 100%
- TTFT: 78 ms
- t/s/u: 18.0
- Seq len: 40960

## Optimizations kept
- Fused QKV projection in attention with TP-safe per-device weight packing (`[q_i, k_i, v_i]`):
  - Replaces three matmuls (`q_proj`, `k_proj`, `v_proj`) plus concat with one `qkv_proj` matmul.
  - Keeps local head ordering expected by `nlp_create_qkv_heads(_decode)`.
- Decode traced execution with reusable device buffers:
  - `decode_token_buffer`, `decode_pos_buffer`
  - Global/local RoPE buffers for Q and K
  - `ttnn.begin_trace_capture` + `ttnn.execute_trace`
- `prefill_logits_last_device()` fast path so demo/eval TTFT measures prefill without full prefill-logit host transfer.
- Kept the functional N300 1D TP layout and cache path unchanged (row/column parallel mapping + paged KV cache).

## Optimizations rejected
- None in this iteration.

## Results (2026-02-12)
From `models/google/gemma-3-4b-it/n300/optimized/eval.log` and `models/google/gemma-3-4b-it/n300/optimized/demo.log`:
- Top-1: 94%
- Top-5: 100%
- TTFT: 68 ms
- t/s/u: 18.5
- Seq len: 40960

## Repro commands
Long eval:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TT_VISIBLE_DEVICES=0 TT_METAL_CACHE=/tmp/tt-metal-cache PYTHONUNBUFFERED=1 \
python -u eval.py models/google/gemma-3-4b-it/n300/optimized/model.py --model google/gemma-3-4b-it \
  --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100 --max_seq_len 40960 --seed 0
```

Demo:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TT_VISIBLE_DEVICES=0 TT_METAL_CACHE=/tmp/tt-metal-cache PYTHONUNBUFFERED=1 \
python -u demo.py models/google/gemma-3-4b-it/n300/optimized/model.py --prompt-file prompts/bringup_eval_long.txt \
  --max-new-tokens 100 --temperature 0 --seed 0 --max_seq_len 40960
```
