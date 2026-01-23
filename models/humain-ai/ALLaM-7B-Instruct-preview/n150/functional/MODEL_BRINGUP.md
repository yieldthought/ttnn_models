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
- `ttnn.experimental.paged_fill_cache` and `ttnn.experimental.paged_update_cache`
- `ttnn.transformer.paged_scaled_dot_product_attention_decode`

## RoPE notes
ALLaM uses HuggingFace-format RoPE with `rope_theta=1e6`. Use:

- `ttnn.experimental.rotary_embedding`

Avoid `ttnn.experimental.rotary_embedding_llama`, which expects Meta-format RoPE.

## KV cache and tiling constraints
- Cache tensors use paged layout: `[max_num_blocks, n_kv_heads, block_size, head_dim]` with `block_size=64`.
- Page table shape is `[32, max_num_blocks]` (tile-aligned batch dimension), identity mapped.
- Prefill uses `ttnn.experimental.paged_fill_cache`, decode uses `ttnn.experimental.paged_update_cache`
  and `ttnn.transformer.paged_scaled_dot_product_attention_decode`.
- `max_seq_len` defaults to the HF `max_position_embeddings` (4096) and can be lowered if DRAM is tight.

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

Max-sequence validation:

```
python scripts/run_eval.py --mode tt --hf-model humain-ai/ALLaM-7B-Instruct-preview --system n150 --prefill-len 128 --decode-len 1 --max-seq-len 4096 --force-prefill
```

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi reset`.
