# MODEL_BRINGUP.md — Qwen3 0.6B (n150 functional)

## Overview
This is a minimal TTNN bringup of `Qwen/Qwen3-0.6B` that runs the full forward pass on device.
It is designed to be easy to read and to serve as a template for future bringups.

- Model code: `models/Qwen/Qwen3-0.6B/n150/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Directory layout
The HF model id is used as the directory path under `models/`.

```
models/Qwen/Qwen3-0.6B/<system>/functional/model.py
```

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
- `ttnn.transformer.scaled_dot_product_attention` and
  `ttnn.transformer.paged_scaled_dot_product_attention_decode`
- `ttnn.experimental.paged_fill_cache` (prefill) and
  `ttnn.experimental.paged_update_cache` (decode)

## RoPE notes
Qwen3 uses HuggingFace-format RoPE. Use:

- `ttnn.experimental.rotary_embedding`

Decode path detail:
- `rotary_embedding` with `start_pos` expects `[seq_len, 1, B, head_dim]`. For decode,
  reshape Q and K to merge heads into the batch (`[1, 1, B*heads, head_dim]`), apply
  RoPE, then reshape back to `[1, B, heads, head_dim]`.

## KV cache and tiling constraints
- Cache tensors are allocated as `[max_num_blocks, n_kv_heads, block_size, head_dim]`
  with `block_size=64` and `max_num_blocks=ceil(max_seq_len / block_size)`.
- A page table is allocated as `[32, max_num_blocks]` (int32, row-major) with an
  identity block mapping.
- Prefill uses `ttnn.experimental.paged_fill_cache` with the page table.
- Decode uses `ttnn.experimental.paged_update_cache` and
  `ttnn.transformer.paged_scaled_dot_product_attention_decode`.
- `cur_pos_tensor` is filled with `-1` for padded batch slots so decode skips them.

`max_seq_len` defaults to the HF config `max_position_embeddings` (40960 for Qwen3-0.6B).

## Precision
- Weights use `ttnn.bfloat16` in this bringup for simplicity.
- Activations use `ttnn.bfloat16`.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after logits are returned.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.

```
python eval.py models/Qwen/Qwen3-0.6B/n150/functional/model.py --model Qwen/Qwen3-0.6B
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model Qwen/Qwen3-0.6B
```

Max sequence length run (paged KV cache):

```
python scripts/run_eval.py --mode tt --hf-model Qwen/Qwen3-0.6B --system n150 \
  --prefill-len 128 --decode-len 1 --max-seq-len 40960 --force-prefill
```

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi reset`.
