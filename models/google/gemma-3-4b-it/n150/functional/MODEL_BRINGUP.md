# MODEL_BRINGUP.md — Gemma 3 4B IT (n150 functional)

## Overview
This is a minimal TTNN bringup of `google/gemma-3-4b-it` that runs the full forward pass on device.
It is designed to be easy to read and to serve as a template for future bringups.

- Model code: `models/google/gemma-3-4b-it/n150/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Model API contract
- The model exposes a `build_model(hf_model, tt_device, max_seq_len)` function.
- The returned class subclasses `torch.nn.Module` and `GenerationMixin` so HF `generate()` works.
- The forward method returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`.

## Key TTNN ops
- `ttnn.embedding` for token embeddings
- `ttnn.linear` for QKV, output, and MLP projections
- `ttnn.rms_norm` for RMSNorm and Q/K head norm
- `ttnn.experimental.rotary_embedding` for HuggingFace-format RoPE
- `ttnn.experimental.nlp_create_qkv_heads[_decode]` and `ttnn.experimental.nlp_concat_heads`
- `ttnn.transformer.scaled_dot_product_attention[_decode]`
- `ttnn.fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## Gemma3 specifics
- Q/K RMSNorm uses `(1 + weight)` (Gemma3RMSNorm).
- Embeddings are scaled by `sqrt(hidden_size)` with bfloat16 rounding.
- Global RoPE uses linear scaling (`rope_scaling.factor = 8`).
- Sliding layers (pattern = 6) use local RoPE with `rope_local_base_freq`.
- Sliding-window masking is not implemented; it only matters for very long contexts.

## KV cache and tiling constraints
- Cache tensors are `[32, n_kv_heads, max_seq_len, head_dim]`.
- `MAX_CACHE_SEQ_LEN` is set to 128 to cap memory usage; increase if needed.

## Precision
- Weights use `ttnn.bfloat8_b`.
- Activations use `ttnn.bfloat16`.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after logits are returned.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.

```
python eval.py models/google/gemma-3-4b-it/n150/functional/model.py --model google/gemma-3-4b-it
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model google/gemma-3-4b-it
```

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi reset`.
