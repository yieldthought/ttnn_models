# MODEL_BRINGUP.md — AFM-4.5B (n150 functional)

## Overview
This is a minimal TTNN bringup of `arcee-ai/AFM-4.5B` that runs the full forward pass on device.
It is designed to be easy to read and to serve as a template for future Arcee bringups.

- Model code: `models/arcee-ai/AFM-4.5B/n150/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Directory layout
The HF model id is used as the directory path under `models/`.

```
models/arcee-ai/AFM-4.5B/<system>/functional/model.py
```

## Model API contract
- The model exposes a `build_model(hf_model, tt_device, max_seq_len)` function.
- The returned class subclasses `torch.nn.Module` and `GenerationMixin` so HF `generate()` works.
- The forward method returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`.

## Key TTNN ops
- `ttnn.embedding` for token embeddings
- `ttnn.linear` for QKV, output, and MLP projections
- `ttnn.rms_norm` for RMSNorm
- `ttnn.experimental.rotary_embedding` for HF-format RoPE (Yarn scaling)
- `ttnn.experimental.nlp_create_qkv_heads[_decode]` and `ttnn.experimental.nlp_concat_heads`
- `ttnn.transformer.scaled_dot_product_attention[_decode]`
- `ttnn.fill_cache` and `ttnn.experimental.paged_update_cache`

## RoPE notes
Arcee uses Yarn RoPE scaling with the HuggingFace RoPE layout. Use:

- `ttnn.experimental.rotary_embedding`

Avoid `ttnn.experimental.rotary_embedding_llama`, which expects Meta-format RoPE.

## MLP activation
Arcee uses `relu2` (ReLU squared). The TTNN model implements this as:

```
relu = ttnn.relu(x)
relu2 = ttnn.mul(relu, relu)
```

## KV cache and sequence limits
- RoPE cache is precomputed to `MAX_CACHE_SEQ_LEN` (256) to fit device memory.
- Prefill uses `ttnn.fill_cache`, decode uses `ttnn.experimental.paged_update_cache`.
- Decode RoPE merges heads into the batch dimension before applying `rotary_embedding`.
- TT ops may pad head count to 32; trim after `nlp_concat_heads` if needed.

## Precision
- Weights use `ttnn.bfloat8_b` to fit in device memory.
- Activations use `ttnn.bfloat16`.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after logits are returned.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.

```
python eval.py models/arcee-ai/AFM-4.5B/n150/functional/model.py --model arcee-ai/AFM-4.5B --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
```

Latest eval (141 prompt tokens, 100 new tokens): top-1 98.00%, top-5 100.00%.

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi reset`.
