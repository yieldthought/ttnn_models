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
- `ttnn.transformer.scaled_dot_product_attention[_decode]`
- `ttnn.fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## Precision and fidelity
- Attention Q/K/V path stays in BF16 and uses HiFi4 compute kernel config to handle outlier channels.
- MLP weights remain `ttnn.bfloat8_b` to fit DRAM.
- Embedding and LM head weights are `ttnn.bfloat16` for accuracy.

## KV cache and limits
- Cache tensors are allocated as `[32, n_kv_heads, MAX_CACHE_SEQ_LEN, head_dim]`.
- `MAX_CACHE_SEQ_LEN` is set to 256; evaluation prompts must fit within it.

## Evaluation
Teacher-forcing accuracy against the HF reference model:

```
python eval.py models/arcee-ai/Arcee-Spark/n150/functional/model.py --model arcee-ai/Arcee-Spark --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
```
