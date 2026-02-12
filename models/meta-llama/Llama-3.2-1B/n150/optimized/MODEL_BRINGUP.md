# MODEL_BRINGUP.md - Llama 3.2 1B (n150 optimized)

## Overview
This is the optimized TTNN bringup of `meta-llama/Llama-3.2-1B` for `n150`.

- Model code: `models/meta-llama/Llama-3.2-1B/n150/optimized/model.py`
- Demo log: `models/meta-llama/Llama-3.2-1B/n150/optimized/demo.log`
- Eval log: `models/meta-llama/Llama-3.2-1B/n150/optimized/eval.log`
- Machine-readable metrics: `models/meta-llama/Llama-3.2-1B/n150/optimized/metrics.json`
- Decode path uses traced execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`)

## Model API contract
- Exposes `build_model(hf_model, tt_device, max_seq_len)`.
- Returns a HuggingFace `generate()`-compatible model (`torch.nn.Module` + `GenerationMixin`).
- Returns `CausalLMOutputWithPast(logits=..., past_key_values=...)` with on-device KV cache managed inside the TT model.

## Baseline vs final
| Metric | Functional baseline (`MODELS.md`) | Starting optimized baseline (this pass) | Final optimized |
| --- | ---: | ---: | ---: |
| Top-1 | 92% | 90% | 92% |
| Top-5 | 100% | 100% | 100% |
| TTFT | 34 ms | 27 ms | 22 ms |
| t/s/u | 39.5 | 63.0 | 64.8 |
| Seq len | 131072 | 131072 | 131072 |

Note: TTFT and t/s/u depend on the demo prompt and sampling settings. See `demo.log` for the exact command and output.

## Kept optimization decisions
1. Paged KV cache for long-seq decode.
- K/V layout: `[max_num_blocks, n_kv_heads, block_size, head_dim]`.
- Prefill: `ttnn.experimental.paged_fill_cache`.
- Decode: `ttnn.experimental.paged_update_cache` + `ttnn.transformer.paged_scaled_dot_product_attention_decode`.

2. Decode trace with preallocated decode buffers.
- Avoids per-token allocations in the decode loop.

3. Decode-only DRAM-sharded matmuls for attention `o_proj` and MLP (w1/w3/w2).

4. Prefill-last-logits fast paths.
- `prefill_logits_last_device()` is used by `eval.py`/`demo.py` when host logits are needed.
- `next_token_device()` uses a prefill-last-logits path so greedy TT demo does not materialize full prefill logits.

## Changes in this pass
- Set `LM_HEAD_WEIGHT_DTYPE=ttnn.bfloat8_b` to recover long-eval accuracy while keeping decode throughput high.

## Rejected directions
- LM head weights at `ttnn.bfloat4_b`: regressed teacher-forcing Top-1 too much on this setup.

## Constraints and gotchas
- `eval.py` enforces `max_seq_len >= 2048`.
- Decode uses a tile-padded batch (`B=32`); inactive lanes must use `cur_pos_tensor=-1`.
- `decode_pos_buffer` must stay in DRAM (`paged_update_cache` requires DRAM `update_idxs_tensor`).
- For GQA decode, SDPA output is interleaved and must be resharded before `ttnn.experimental.nlp_concat_heads_decode`.
- Decode MLP sharded path avoids `ttnn.swiglu` and uses separate gate/up matmuls + `ttnn.mul(..., SILU)`.

## Commands used
```bash
python demo.py models/meta-llama/Llama-3.2-1B/n150/optimized/model.py --seed 0 --max_seq_len 131072

python eval.py models/meta-llama/Llama-3.2-1B/n150/optimized/model.py \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --seed 0 \
  --max_seq_len 131072
```

See `doc/ttnn.md` for the general optimization workflow and the paged-KV + decode-trace constraints we rely on here.
