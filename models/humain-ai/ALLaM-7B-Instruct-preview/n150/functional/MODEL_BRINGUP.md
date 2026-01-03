# MODEL_BRINGUP.md — ALLaM 7B Instruct preview (n150 functional)

## Overview
This is a minimal TTNN bringup of `humain-ai/ALLaM-7B-Instruct-preview` that runs the full forward pass on device.
It is designed to be easy to read and to serve as a template for future bringups.

- Model code: `models/humain-ai/ALLaM-7B-Instruct-preview/n150/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Directory layout
The HF model id is used as the directory path under `models/`.

```
models/humain-ai/ALLaM-7B-Instruct-preview/<system>/functional/model.py
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
ALLaM uses HuggingFace-format RoPE with `rope_theta=1e6`. Use:

- `ttnn.experimental.rotary_embedding`

Avoid `ttnn.experimental.rotary_embedding_llama`, which expects Meta-format RoPE.

## KV cache and tiling constraints
- Cache tensors are allocated as `[32, n_kv_heads, cache_seq_len, head_dim]`.
- The batch dimension is tile-aligned to 32 for decode ops.
- Prefill uses height-sharded KV inputs for `ttnn.fill_cache`, and decode uses `ttnn.experimental.paged_update_cache`.
- Cache length is capped to 256 tokens in this bringup (`MAX_CACHE_SEQ_LEN`) to fit on a single device.
  Increase it if you have more DRAM.

On this device, interleaved `ttnn.fill_cache` hits a grid limit with 32 KV heads, so the model shards KV for prefill.
If prefill still hits a `fill_cache` grid limit, use `--prefill_decode` to debug. Final bringup metrics must use the full prefill pass (no `--prefill_decode`).
`scripts/run_eval.py` enables this automatically for large prefill lengths.

## Precision
- Weights use `ttnn.bfloat8_b` to fit the 7B model in device DRAM.
- Activations use `ttnn.bfloat16`.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after logits are returned.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.

```
python eval.py models/humain-ai/ALLaM-7B-Instruct-preview/n150/functional/model.py --model humain-ai/ALLaM-7B-Instruct-preview
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model humain-ai/ALLaM-7B-Instruct-preview
```

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi reset`.
