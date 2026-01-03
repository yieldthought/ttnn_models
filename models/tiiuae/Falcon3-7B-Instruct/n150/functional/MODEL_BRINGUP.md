# MODEL_BRINGUP.md — Falcon3 7B Instruct (n150 functional)

## Overview
This is a minimal TTNN bringup of `tiiuae/Falcon3-7B-Instruct` that runs the full
forward pass on device. It is designed to be easy to read and serve as a
template for Llama-style models.

- Model code: `models/tiiuae/Falcon3-7B-Instruct/n150/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Directory layout
The HF model id is used as the directory path under `models/`.

```
models/tiiuae/Falcon3-7B-Instruct/<system>/functional/model.py
```

## Model API contract
- The model exposes a `build_model(hf_model, tt_device, max_seq_len)` function.
- The returned class subclasses `torch.nn.Module` and `GenerationMixin` so HF `generate()` works.
- The forward method returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`.

## Key TTNN ops
- `ttnn.embedding` for token embeddings
- `ttnn.linear` for QKV, output, and MLP projections
- `ttnn.rms_norm` for RMSNorm
- `ttnn.experimental.rotary_embedding` for HuggingFace-format RoPE
- `ttnn.experimental.nlp_create_qkv_heads[_decode]` and `ttnn.experimental.nlp_concat_heads`
- `ttnn.transformer.scaled_dot_product_attention[_decode]`
- `ttnn.fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## RoPE notes
Falcon3 uses HuggingFace-format RoPE with `rope_theta=1000042` and no scaling.

Decode path detail:
- `rotary_embedding` with `start_pos` expects `[seq_len, 1, B, head_dim]`. For decode,
  reshape Q and K to merge heads into the batch, apply RoPE, then reshape back.

## KV cache and tiling constraints
- Cache tensors are allocated as `[32, n_kv_heads, cache_seq_len, head_dim]`.
- The batch dimension is tile-aligned to 32 for decode ops.
- Prefill uses `ttnn.fill_cache` and decode uses `ttnn.experimental.paged_update_cache`.
- Cache length is capped to 1024 tokens in this bringup (`MAX_CACHE_SEQ_LEN`) to fit on a single device.
  Increase it if you have more DRAM.

If prefill hits a `fill_cache` grid limit, use `--prefill_decode` to debug.
Final bringup metrics must use the full prefill pass (no `--prefill_decode`).

If `nlp_concat_heads` pads the width, slice back to `n_heads * head_dim` before
the output projection.

## Precision
- Weights use `ttnn.bfloat8_b` to fit the 7B model in device DRAM.
- Activations use `ttnn.bfloat16`.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after
logits are returned.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.

```
python eval.py models/tiiuae/Falcon3-7B-Instruct/n150/functional/model.py --model tiiuae/Falcon3-7B-Instruct
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model tiiuae/Falcon3-7B-Instruct
```

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi reset`.
