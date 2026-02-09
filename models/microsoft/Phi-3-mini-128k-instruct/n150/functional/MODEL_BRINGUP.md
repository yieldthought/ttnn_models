# MODEL_BRINGUP.md - Phi-3 Mini 128k Instruct (n150 functional)

## Overview
This is a minimal TTNN bringup of `microsoft/Phi-3-mini-128k-instruct` that runs the full forward pass on device.
It mirrors the HuggingFace architecture with a fused QKV projection and a gated MLP.

- Model code: `models/microsoft/Phi-3-mini-128k-instruct/n150/functional/model.py`
- Eval harness: `eval.py` and `scripts/run_eval.py`
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Model API contract
- Exposes `build_model(hf_model, tt_device, max_seq_len)`
- Returned class subclasses `torch.nn.Module` and `GenerationMixin`
- Forward returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`

## Key TTNN ops
- `ttnn.embedding` for token embeddings
- `ttnn.linear` for QKV, output, and MLP projections
- `ttnn.rms_norm` for RMSNorm
- `ttnn.experimental.rotary_embedding` for RoPE
- `ttnn.experimental.nlp_create_qkv_heads[_decode]` and `ttnn.experimental.nlp_concat_heads`
- `ttnn.transformer.scaled_dot_product_attention` (prefill)
- `ttnn.transformer.paged_scaled_dot_product_attention_decode` (decode)
- `ttnn.experimental.paged_fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## RoPE notes
Phi-3 uses LongRoPE (`rope_scaling.type = longrope`). This bringup:
- Applies the attention scaling factor used by HF.
- Uses the short or long frequency factors based on `max_seq_len` vs
  `original_max_position_embeddings` (4096).
- Pads the RoPE dimension to a multiple of 64 for TT rotary (then slices back).

This bringup precomputes **both** short and long RoPE caches and selects between
them per call:
- Prefill uses the long cache only when `seq_len > original_max_position_embeddings`.
- Decode switches to the long cache when `start_pos >= original_max_position_embeddings`.
This matches HF’s dynamic LongRoPE behavior so sequences that cross 4096 during
decode use the correct frequency factors.

## KV cache and tiling constraints
- Cache tensors are paged: `[max_num_blocks, n_kv_heads, block_size, head_dim]` with `block_size=64`.
- Page table is `[32, max_num_blocks]` (tile-aligned batch dimension) with identity block mapping.
- `max_seq_len` resolves from HF config `max_position_embeddings` (131072) or a user override.
- On n150, KV cache + weights fit at `max_seq_len=12288` (single-user DRAM limit).
- Batch dimension is tile-aligned to 32 for decode ops; set unused decode positions to `-1`.
- Inputs are padded to tile size (32) before embedding and trimmed at the end.

## Precision
- Weights and activations use `ttnn.bfloat16`.

## Evaluation
Teacher-forcing accuracy against the HF reference:

```
python eval.py models/microsoft/Phi-3-mini-128k-instruct/n150/functional/model.py \
  --model microsoft/Phi-3-mini-128k-instruct
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model microsoft/Phi-3-mini-128k-instruct --system n150 --max-seq-len 12288
```

## Latest results (2026-02-09)
- Long eval (`prompts/bringup_eval_long.txt`, `--max_new_tokens 100`, `--max_seq_len 12288`): Top-1 92%, Top-5 99%.
- Demo: TTFT 80 ms, decode 13.7 t/s/u.
- Fix: compute both short/long RoPE caches and switch based on `seq_len` (prefill) or
  `start_pos` (decode) to align with HF LongRoPE dynamic selection.
- Note: `eval.py` enforces `--max_seq_len >= 2048`.
