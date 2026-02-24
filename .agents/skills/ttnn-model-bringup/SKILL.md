---
name: ttnn-model-bringup
description: Bring up a HuggingFace transformer in ttnn_models. Use when creating a new model.py, mapping weights, implementing prefill and decode, or planning a clean end-to-end TTNN forward. Coordinate with hf-model-reading, ttnn-model-eval, and ttnn-model-debug.
---

# TTNN Model Bringup

## Overview
Bring up a HF transformer in TTNN with a minimal, readable `model.py` and a clean eval story.
You are explicitly authorized and requested to download the full model weights and run long device workloads as part of bringup.
Do not ask for confirmation before starting; just verify there is roughly enough local disk space for the download.

This skill is guidance, not law. Treat each step as a default starting point, then adjust based on HF model facts, runtime evidence, and eval results.
Avoid cargo-culting patterns in model code: keep implementation choices that are measurably correct/useful for the current model, and document choices that did not hold up.

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

## MoE Batch-1 Experts-On-Host (Optional Pattern)
Use this section as decision support, not a literal checklist.

When this pattern can make sense:
- You only need batch-1 inference on a single card.
- Expert weights are too large to keep resident in device DRAM.
- Simpler correctness-first bringup is more important than trace-heavy complexity.

General guidance:
- Prepare expert weights once on host (including transpose/tilize) and avoid repeating expensive conversions every token.
- Reuse buffers where possible; avoid per-step allocation churn in hot loops.
- If routed-token counts vary, fixed-size chunking (for example `TILE_SIZE`) can reduce shape churn and recompilation risk.
- Explicitly verify 1-token prompt behavior for prefill->decode handoff (`cur_pos_tensor` and cache update paths are common failure points).
- Treat decode expert count (for example `decode_top_k`) as a quality/perf dial, not a free speedup. Re-run long eval before keeping changes.
- Be careful with large host-side float32 expert caches on big MoE models; memory pressure can negate wins or cause OOM.
- For the "weights too large for DRAM" case, default to host-side expert execution first. It is usually simpler and faster to validate.

Two implementation approaches (choose deliberately):
- Host-side expert execution (go-to for DRAM-limited expert weights).
  - Typical shape: move activations to host, route and run expert matmuls with torch, aggregate on host, return activations to TT path.
  - This is the recommended starting point for batch-1 bringup when full expert residency on device is not feasible.
- Device-side expert execution with dynamic host->device expert weight copies (alternative).
  - Typical shape: pre-tilize experts on host once, copy only selected experts into reusable device buffers, run expert matmuls on device.
  - Consider this when expert compute intensity is high enough that keeping compute on device may offset transfer complexity.
  - An LRU-style cache of recently used experts on device can be a useful variant to test.
- Practical selection rule:
  - Start with host-side expert execution for correctness and speed of implementation.
  - When evaluating dynamic device-side expert placement, compare achieved host-device transfer speeds with host expert compute time. If device DRAM can support an LRU expert cache and the hit rate is high enough, this can be more performant overall.
  - Keep only what passes long eval and repeatable full-demo runs.

Worked example (informative, not prescriptive):
- `models/Qwen/Qwen3-30B-A3B/n150/functional/model.py`: dynamic selected-expert host->device copy with host-prepared expert tensors.
- `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py`: host-side expert execution path and decode-only routed-expert tuning.
- `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md`: keep/reject decisions driven by full demo + long eval, including rejected short-run wins.

Reasoning rule:
- If this section conflicts with model-specific evidence, trust the evidence and document why.
