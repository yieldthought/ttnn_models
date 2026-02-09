---
name: ttnn-model-bringup
description: Bring up a HuggingFace transformer in ttnn_models. Use when creating a new model.py, mapping weights, implementing prefill and decode, or planning a clean end-to-end TTNN forward. Coordinate with hf-model-reading, ttnn-model-eval, and ttnn-model-debug.
---

# TTNN Model Bringup

## Overview
Bring up a HF transformer in TTNN with a minimal, readable `model.py` and a clean eval story.
You are explicitly authorized and requested to download the full model weights and run long device workloads as part of bringup.
Do not ask for confirmation before starting; just verify there is roughly enough local disk space for the download.

## Workflow
1. Use `hf-model-reading` to understand the HF model code, config, and required features.
2. Scan existing TTNN models for available ops and conventions; avoid copying large blocks.
3. Check available disk space before downloading weights; proceed with the download even if it is tens or hundreds of GB.
4. Use the device safely.
   - Run exactly one TTNN/TT process at a time on this single Wormhole card.
   - Avoid parallel tool runs and stop any background TT processes before starting a new run or reset.
5. Define a small config from HF fields.
   - Include vocab, hidden size, heads, kv heads, head dim, rope theta, rms eps, activation.
6. Load weights from `state_dict` and prepare TTNN tensors.
   - Transpose for `ttnn.linear` and convert to `bfloat16`.
7. Implement the core blocks.
   - embedding -> per-layer attention + mlp -> final norm -> lm_head.
8. Implement prefill and decode separately.
   - Use `nlp_create_qkv_heads[_decode]`, RoPE, `scaled_dot_product_attention` (prefill).
   - Use paged KV cache from the start (recommended on n150 and for long `max_seq_len`).
     - Pick `block_size` (multiple of 32). Use 64 by default.
     - `max_num_blocks = ceil(max_seq_len / block_size)`.
     - Allocate caches: `[max_num_blocks, n_kv_heads, block_size, head_dim]`.
     - Allocate an identity page table with tile batch: `[32, max_num_blocks]`, dtype `int32`, `ROW_MAJOR`.
     - Prefill: `ttnn.experimental.paged_fill_cache(k_cache, k, page_table, batch_idx=0)` (and same for V).
     - Decode: build `cur_pos_tensor` as length 32 int32 with `-1` in unused lanes so they are skipped; then
       `ttnn.experimental.paged_update_cache(..., update_idxs_tensor=cur_pos_tensor, page_table=page_table)` and
       `ttnn.transformer.paged_scaled_dot_product_attention_decode(..., page_table_tensor=page_table, cur_pos_tensor=cur_pos_tensor)`.
   - Legacy fallback (non-paged cache): if you keep `[32, n_kv_heads, max_seq_len, head_dim]`, prefill uses
     `fill_cache`. If interleaved `fill_cache` hits the grid-size work-block limit (e.g., large `n_kv_heads * seq_tiles`),
     shard K/V for `fill_cache` (height-sharded, ROW_MAJOR); shard width must match padded width. If you map one KV head
     per core using `grid.x` columns, ensure `n_kv_heads` is divisible by `grid.x` (or pick a different grid).
9. Handle tile padding and head padding.
   - Pad sequence to tile size and trim logits back to real length.
   - Trim padded head width after `nlp_concat_heads` if needed.
10. Run the long teacher-forcing eval and iterate until it passes (top-1 >= 90%, top-5 >= 95%).
   - `python eval.py <model.py> --model <hf-id> --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100`
   - Final bringup metrics must use the full prefill pass (no `--prefill_decode`).
   - Only acceptable excuse: not enough DRAM for the sequence length and the model already uses `bfloat8_b` weights.
   - If the eval is interrupted, resume and complete it before reporting results.
   - If mismatches appear, use `ttnn-model-debug`.
11. After the eval passes, add a row to `MODELS.md` with the model name, HF id, prompt tokens, max_new_tokens, top-1/top-5, and notes.
12. Write or update `MODEL_BRINGUP.md` with key gotchas and commands.

Bringup is not complete until eval.py is passing; writing the code is only the start.
