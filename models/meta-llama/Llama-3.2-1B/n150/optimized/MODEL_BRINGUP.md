# MODEL_BRINGUP.md — Llama 3.2 1B (n150 optimized)

## Overview
This is the optimized TTNN bringup of `meta-llama/Llama-3.2-1B` for `n150`.
It is the maintained reference for this directory.

- Model code: `models/meta-llama/Llama-3.2-1B/n150/optimized/model.py`
- Eval harness: `eval.py`
- Demo harness: `demo.py`
- Optimization notes and decisions: `models/meta-llama/Llama-3.2-1B/n150/optimized/TODO.txt`

## Model API contract
- Exposes `build_model(hf_model, tt_device, max_seq_len)`.
- Returns a `torch.nn.Module` + `GenerationMixin` model compatible with HF `generate()`.
- Returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`.

## What this optimized path includes
- Paged KV cache:
  - K/V layout: `[max_num_blocks, n_kv_heads, block_size, head_dim]`
  - `page_table`: `[32, max_num_blocks]` int32, row-major
  - Prefill: `ttnn.experimental.paged_fill_cache`
  - Decode: `ttnn.experimental.paged_update_cache` + `ttnn.transformer.paged_scaled_dot_product_attention_decode`
- Fused QKV projection.
- Decode trace path with allocation-safe decode logits trimming in traced region.
- Decode-only DRAM-sharded matmuls for:
  - MLP (`w1`, `w3`, `w2`)
  - Attention output projection (`o_proj`)
- Tuned LM head decode program config (`8x7` grid on n150) for better K blocking.
- `prefill_logits_last_device()` fast path used by `eval.py` and `demo.py` when available.

## Correctness and runtime constraints
- Keep `max_seq_len >= 2048` for the eval contract.
- Decode uses tile-padded batch (`B=32`), and padded entries must use `cur_pos_tensor=-1`.
- `decode_pos_buffer` must stay in DRAM (`paged_update_cache` requirement for `update_idxs_tensor`).
- For GQA decode:
  - SDPA decode output is interleaved.
  - Reshard before `ttnn.experimental.nlp_concat_heads_decode`.
- `ttnn.swiglu` is not used in decode MLP sharded path; decode MLP uses separate gate/up matmuls + `ttnn.mul(..., SILU)`.

## Performance measurement notes
- `demo.py` performs one prefill+decode warmup pass before timing to compile first-use kernels.
- Decode throughput in recent runs is typically mid/high 60s t/s/u on this box; best recorded run is about 68 t/s/u.

## Validation commands
```bash
python eval.py models/meta-llama/Llama-3.2-1B/n150/optimized/model.py --max_new_tokens 100 --prompt_file prompts/bringup_eval_long.txt --seed 0
python demo.py models/meta-llama/Llama-3.2-1B/n150/optimized/model.py --seed 0
```

Expected current behavior:
- `eval.py` 100-token teacher-forcing around Top-1 90%, Top-5 100%.
- `demo.py` output coherent with decode throughput in the expected range above.

## Deferred/rejected optimization directions
- DRAM-sharded decode QKV: micro-op gain, negligible layer-level win for added complexity.
- DRAM-sharded one-shot LM head: blocked by sharding/L1 static-CB constraints (would need chunking or another path).
- SDPA decode LoFi / `packer_l1_acc=False` variants: slower on this setup.
- Decode core grid below 32 cores: incompatible with decode sharding requirements.

See `doc/ttnn.md` ("Optimizing models") for the generalized optimization workflow and reusable constraints.
