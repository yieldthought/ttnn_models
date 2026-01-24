# MODEL_BRINGUP.md — Arcee-Spark (n150 functional)

## Overview
Minimal TTNN bringup of `arcee-ai/Arcee-Spark` (Qwen2 family) with full device execution.

- Model code: `models/arcee-ai/Arcee-Spark/n150/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py`
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

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
- `ttnn.transformer.scaled_dot_product_attention` and `ttnn.transformer.paged_scaled_dot_product_attention_decode`
- `ttnn.experimental.paged_fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## Precision and fidelity
- Attention Q/K/V path stays in BF16 and uses HiFi4 compute kernel config to handle outlier channels.
- MLP weights remain `ttnn.bfloat8_b` to fit DRAM.
- Embedding and LM head weights are `ttnn.bfloat16` for accuracy.

## KV cache and limits
- Paged KV cache uses `[max_num_blocks, n_kv_heads, block_size, head_dim]` with `block_size=64`.
- Page table is identity-mapped with shape `[32, max_num_blocks]`.
- `max_seq_len` defaults to `max_position_embeddings` from the HF config (32768 for Arcee-Spark), but n150 DRAM limits validation to 29952.

## Evaluation
Teacher-forcing accuracy against the HF reference model:

```
python eval.py models/arcee-ai/Arcee-Spark/n150/functional/model.py --model arcee-ai/Arcee-Spark --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
```

Max sequence length validation (prefill only, 1-token decode):

```
python scripts/run_eval.py --mode tt --hf-model arcee-ai/Arcee-Spark --system n150 --prefill-len 128 --decode-len 1 --max-seq-len 29952
```

Note: 30080+ (including the HF max 32768) OOMs during embeddings on n150.

Note: `--force-prefill`/`--prefill_decode` was a hack and has been removed from the eval scripts.
