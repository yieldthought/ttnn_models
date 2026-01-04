# MODEL_BRINGUP.md — Arcee-Spark (Qwen2, n150 functional)

## Overview
This is a minimal TTNN bringup of `arcee-ai/Arcee-Spark` (Qwen2 family) that runs the full forward pass on device.
It is designed to be easy to read and to serve as a template for future bringups.

- Model code: `models/arcee-ai/Arcee-Spark/n150/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Directory layout
The HF model id is used as the directory path under `models/`.

```
models/arcee-ai/Arcee-Spark/<system>/functional/model.py
```

## Model API contract
- The model exposes a `build_model(hf_model, tt_device, max_seq_len)` function.
- The returned class subclasses `torch.nn.Module` and `GenerationMixin` so HF `generate()` works.
- The forward method returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`.

## Key TTNN ops
- `ttnn.embedding` for token embeddings
- `ttnn.linear` for QKV (with bias), output, and MLP projections
- `ttnn.rms_norm` for RMSNorm
- `ttnn.experimental.rotary_embedding` for HuggingFace-format RoPE
- `ttnn.experimental.nlp_create_qkv_heads[_decode]` and `ttnn.experimental.nlp_concat_heads`
- `ttnn.transformer.scaled_dot_product_attention[_decode]`
- `ttnn.fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## RoPE notes
Qwen2 uses HuggingFace-format RoPE. Use:

- `ttnn.experimental.rotary_embedding`

Decode path detail:
- `rotary_embedding` with `start_pos` expects `[seq_len, 1, B, head_dim]`. For decode,
  reshape Q and K to merge heads into the batch (`[1, 1, B*heads, head_dim]`), apply
  RoPE, then reshape back to `[1, B, heads, head_dim]`.

## KV cache and tiling constraints
- Cache tensors are allocated as `[32, cache_kv_heads, cache_seq_len, head_dim]`.
- `cache_seq_len = min(max_seq_len, max_cache_seq_len)` where `max_cache_seq_len` scales down when K/V
  are pre-repeated (currently `MAX_CACHE_SEQ_LEN // kv_repeat`, rounded down to a multiple of 32).
  `MAX_CACHE_SEQ_LEN = 512`.
- The batch dimension is tile-aligned to 32 for decode ops.
- Prefill uses `ttnn.fill_cache` and decode uses `ttnn.experimental.paged_update_cache`.
- For Arcee-Spark, `cache_kv_heads == n_heads` because K/V are pre-repeated.

If prefill hits a `fill_cache` grid limit, use `--prefill_decode` to debug. Final bringup
metrics must use the full prefill pass (no `--prefill_decode`).

## Precision
- Most weights use `ttnn.bfloat8_b` to fit the model in device DRAM.
- Attention weights use `ttnn.bfloat16` for accuracy.
- MLP `down_proj` weights use `ttnn.bfloat16`; gate/up remain `ttnn.bfloat8_b`.
- QKV biases use `ttnn.bfloat16`.
- Activations use `ttnn.bfloat16`.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after logits are returned.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.

```
python eval.py models/arcee-ai/Arcee-Spark/n150/functional/model.py --model arcee-ai/Arcee-Spark
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model arcee-ai/Arcee-Spark
```

Current bringup results (full prefill):
- `python eval.py models/arcee-ai/Arcee-Spark/n150/functional/model.py --model arcee-ai/Arcee-Spark --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100`
  - Top-1: 76.00% (0.7600), Top-5: 95.00% (0.9500)
- `python eval.py models/arcee-ai/Arcee-Spark/n150/functional/model.py --model arcee-ai/Arcee-Spark --max_new_tokens 20`
  - Top-1: 85.00% (0.8500), Top-5: 95.00% (0.9500)
- `python eval.py models/arcee-ai/Arcee-Spark/n150/functional/model.py --model arcee-ai/Arcee-Spark --max_new_tokens 1`
  - Top-1: 100.00% (1.0000), Top-5: 100.00% (1.0000)

Top-1 is still below target; prefill-only looks correct while decode remains off.

## Known issues
- `ttnn.transformer.scaled_dot_product_attention_decode` assumes a power-of-two `num_q_heads / num_kv_heads` ratio.
  Arcee-Spark uses 28/4=7, so we pre-repeat K/V heads to full heads before caching and SDPA.
  This is a workaround until the decode kernel supports non-power-of-two GQA.
- Pre-repeating K/V grows cache memory, so `cache_seq_len` is reduced to fit DRAM. Long prompts may
  exceed the cache unless `MAX_CACHE_SEQ_LEN` or cache layout changes.

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi -r 0`.
