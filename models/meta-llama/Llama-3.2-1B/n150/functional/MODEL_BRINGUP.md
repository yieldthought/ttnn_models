# MODEL_BRINGUP.md — Llama 3.2 1B (n150 functional)

## Overview
This is a minimal TTNN bringup of `meta-llama/Llama-3.2-1B` that runs the full forward pass on device.
It is designed to be easy to read and to serve as a template for future bringups.

- Model code: `models/meta-llama/Llama-3.2-1B/n150/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Directory layout
The HF model id is used as the directory path under `models/`.

```
models/meta-llama/Llama-3.2-1B/<system>/functional/model.py
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
Llama 3.x on HuggingFace uses the HuggingFace RoPE layout. Use:

- `ttnn.experimental.rotary_embedding`

Avoid `ttnn.experimental.rotary_embedding_llama`, which expects Meta-format RoPE.

## KV cache and tiling constraints
- Cache tensors are allocated as `[32, n_kv_heads, max_seq_len, head_dim]`.
- The batch dimension is tile-aligned to 32 for decode ops.
- Prefill uses `ttnn.fill_cache` and decode uses `ttnn.experimental.paged_update_cache`.

On this device, `ttnn.fill_cache` hits a grid limit for long prefill lengths (around 1024 tokens).
For long prompts, run eval with `--prefill_decode` so the prefill is done token-by-token.
`scripts/run_eval.py` enables this automatically for large prefill lengths.

## Precision
- Weights use `ttnn.bfloat16` in this bringup for simplicity.
- Activations use `ttnn.bfloat16`.

The intent is correctness-first; bfloat8 weights can be tried later if accuracy allows.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after logits are returned.

## Tracing note
For tracing later, keep host-to-device conversions out of the traced region. The model currently
uses `ttnn.from_torch` inside `forward()` for simplicity; when adding traces, split input prep
into a helper and trace only the pure TTNN compute path.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.

```
python eval.py models/meta-llama/Llama-3.2-1B/n150/functional/model.py
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model meta-llama/Llama-3.2-1B
```

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi reset`.
