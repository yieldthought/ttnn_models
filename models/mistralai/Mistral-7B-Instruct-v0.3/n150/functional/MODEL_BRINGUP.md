# MODEL_BRINGUP.md — Mistral 7B Instruct v0.3 (n150 functional)

## Overview
This is a minimal TTNN bringup of `mistralai/Mistral-7B-Instruct-v0.3` that runs the full forward pass on device.
It is designed to be easy to read and to serve as a template for future bringups.

- Model code: `models/mistralai/Mistral-7B-Instruct-v0.3/n150/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Directory layout
The HF model id is used as the directory path under `models/`.

```
models/mistralai/Mistral-7B-Instruct-v0.3/<system>/functional/model.py
```

## Model API contract
- The model exposes a `build_model(hf_model, tt_device, max_seq_len)` function.
- The returned class subclasses `torch.nn.Module` and `GenerationMixin` so HF `generate()` works.
- The forward method returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`.

## Sequence length
- `max_seq_len` defaults to `hf_config.max_position_embeddings`.
- Pass a smaller value for debug runs if needed.

## Key TTNN ops
- `ttnn.embedding` for token embeddings
- `ttnn.linear` for QKV, output, and MLP projections
- `ttnn.rms_norm` for RMSNorm
- `ttnn.experimental.rotary_embedding` for HuggingFace-format RoPE
- `ttnn.experimental.nlp_create_qkv_heads[_decode]` and `ttnn.experimental.nlp_concat_heads`
- `ttnn.transformer.scaled_dot_product_attention` (prefill)
- `ttnn.transformer.paged_scaled_dot_product_attention_decode` (decode)
- `ttnn.experimental.paged_fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## RoPE notes
Mistral uses HuggingFace-format RoPE with `rope_theta=1e6`. Use:

- `ttnn.experimental.rotary_embedding`

Decode path detail:
- `rotary_embedding` with `start_pos` expects `[seq_len, 1, B, head_dim]`. For decode,
  reshape Q and K to merge heads into the batch (`[1, 1, B*heads, head_dim]`), apply
  RoPE, then reshape back to `[1, B, heads, head_dim]`.

## KV cache and tiling constraints
- Cache tensors use paged layout: `[max_num_blocks, n_kv_heads, block_size, head_dim]`.
- `block_size=64` and `max_num_blocks=ceil(max_seq_len / block_size)`.
- Page table is an identity mapping with tile-aligned batch: `[32, max_num_blocks]` in ROW_MAJOR.
- Prefill uses `ttnn.experimental.paged_fill_cache`, decode uses `ttnn.experimental.paged_update_cache`.
- Decode positions use `-1` for the padded batch entries so only batch index 0 is active.

## Precision
- Weights use `ttnn.bfloat8_b` to fit the 7B model in device DRAM.
- Activations use `ttnn.bfloat16`.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after logits are returned.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.

```
python eval.py models/mistralai/Mistral-7B-Instruct-v0.3/n150/functional/model.py --model mistralai/Mistral-7B-Instruct-v0.3
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model mistralai/Mistral-7B-Instruct-v0.3 --system n150
python scripts/run_eval.py --mode tt --hf-model mistralai/Mistral-7B-Instruct-v0.3 --system n150 --prefill-len 128 --decode-len 1 --max-seq-len 32768
```

Note: `--force-prefill`/`--prefill_decode` was a hack and has been removed from the eval scripts.

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi reset`.
