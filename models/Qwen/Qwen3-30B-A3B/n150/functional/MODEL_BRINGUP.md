# MODEL_BRINGUP.md — Qwen3-30B-A3B (n150 functional)

## Overview
Functional TTNN bringup for `Qwen/Qwen3-30B-A3B` on `n150`.

Key model-specific work:
- Implemented Qwen3-MoE routing (`top_k=8`, `norm_topk_prob` support).
- Added dynamic expert streaming for batch-1 inference:
  - Expert weights are tilized once on host per expert.
  - Selected experts are copied host->device just-in-time with reusable device buffers.
- Kept prefill/decode untraced by design for bringup simplicity.

## Files
- Model: `models/Qwen/Qwen3-30B-A3B/n150/functional/model.py`

## Runtime notes
- Use a large HF cache location on this host:
  - `HF_HOME=/localdev/moconnor/hf-cache`
- Use TT metal cache override to avoid cache-space issues:
  - `TT_METAL_CACHE=/tmp/tt-metal-cache`
- Runtime-root behavior here is noisy but non-fatal with:
  - `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal`

## Validation
### Functional smoke
- Built model end-to-end and ran prefill + decode:
  - prefill logits: `(1, 5, 151936)`
  - decode logits: `(1, 1, 151936)`
- Verified 1-token prefill path (`prefill_logits_last_device`) and follow-on decode:
  - prefill logits: `(1, 151936)`
  - decode logits: `(1, 1, 151936)`
  - This fixes the prior `cur_pos_tensor is required for decode` failure on 1-token prompts.

### Teacher-forcing sample accuracy
- Prompt: `prompts/bringup_eval_long.txt`
- New tokens evaluated: `40`
- Reference dtype: `bfloat16` HF model
- Result:
  - Top-1: `100%` (`1.0`)
  - Top-5: `100%` (`1.0`)

## Implementation notes
- Expert execution uses fixed token chunks (`32`) per expert to avoid recompiling many shape variants.
- This keeps kernel shape churn low while preserving sparse routing semantics.
