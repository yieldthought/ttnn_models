---
name: ttnn-model-debug
description: Debug TTNN model correctness issues. Use when logits diverge from HF, when prefill vs decode mismatches appear, or when RoPE, Q/K norm, cache, shapes, or memory config seem wrong.
---

# TTNN Model Debug

## Overview
Debug correctness gaps between TTNN and HF, with extra focus on decode path, RoPE, and cache behavior.

## Workflow
1. Reproduce quickly with `eval.py` at small lengths.
2. Use the device safely.
   - Run exactly one TTNN/TT process at a time on this single Wormhole card.
   - Avoid parallel tool runs and stop any background TT processes before starting a new run or reset.
3. Determine if prefill or decode diverges.
   - If prefill is correct and decode is wrong, focus on decode path.
4. Check RoPE and decode shapes.
   - `rotary_embedding` with `start_pos` expects `[seq_len, 1, B, head_dim]`.
   - For decode, merge heads into batch, apply RoPE, then reshape back to `[1, B, heads, head_dim]`.
5. Verify cache and position inputs.
   - Use `fill_cache` for prefill and `paged_update_cache` for decode.
   - Confirm `start_pos` and `cur_pos_tensor` values.
6. Validate shapes and padding.
   - Check `head_dim`, `n_heads`, and `n_kv_heads`.
   - Trim padded width after `nlp_concat_heads` if needed.
7. Validate memory placement.
   - Some ops need DRAM; move to DRAM for `rms_norm`, then restore.
8. Bisect with HF to localize the issue.
   - Swap a single TT block with the HF block by converting to torch and back.
   - Keep fallbacks only for debugging, not final code.
