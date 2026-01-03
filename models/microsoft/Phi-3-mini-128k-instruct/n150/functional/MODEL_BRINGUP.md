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
- `ttnn.transformer.scaled_dot_product_attention[_decode]`
- `ttnn.fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## RoPE notes
Phi-3 uses LongRoPE (`rope_scaling.type = longrope`). This bringup:
- Applies the attention scaling factor used by HF.
- Uses the short or long frequency factors based on `max_seq_len` vs
  `original_max_position_embeddings` (4096).
- Pads the RoPE dimension to a multiple of 64 for TT rotary (then slices back).

If you need a prompt that crosses 4096 after prefill, rerun with a `max_seq_len`
large enough for that prompt so the long factors are used consistently.

## KV cache and tiling constraints
- Cache tensors are allocated as `[32, n_kv_heads, cache_seq_len, head_dim]`.
- Cache length is capped to 256 tokens in this bringup (`MAX_CACHE_SEQ_LEN`) to fit on a single device.
- Batch dimension is tile-aligned to 32 for decode ops.
- Inputs are padded to tile size (32) before embedding and trimmed at the end.
- Prefill uses height-sharded KV inputs for `fill_cache` to avoid interleaved grid-size limits.
- Sharded prefill requires `n_kv_heads` to be divisible by the device grid x-dimension.

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
python scripts/run_eval.py --mode tt --hf-model microsoft/Phi-3-mini-128k-instruct
```
