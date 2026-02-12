# MODEL_BRINGUP.md — arcee-ai/Arcee-Spark (n150 optimized)

## Overview
Optimized TTNN bringup of `arcee-ai/Arcee-Spark` (Qwen2 family) on Wormhole n150.

- Model code: `models/arcee-ai/Arcee-Spark/n150/optimized/model.py`
- Logs: `models/arcee-ai/Arcee-Spark/n150/optimized/demo.log`, `models/arcee-ai/Arcee-Spark/n150/optimized/eval.log`

## Functional Baseline (n150)
From `MODELS.md` for `arcee-ai/Arcee-Spark` n150 functional:

- Top-1: 92%
- Top-5: 100%
- TTFT: 99ms
- t/s/u: 13.9
- Seq len: 29952

## Final Results (n150 optimized)
From `demo.log` + `eval.log`:

- Top-1: 91%
- Top-5: 100%
- TTFT: 77ms
- t/s/u: 14.5
- Seq len: 29952

## Kept Optimizations
- Decode uses traced execution with persistent device buffers (token ids, positions, RoPE cos/sin).
- Prefill computes only last-token logits for TTFT.
- Attention: fuse Q/K/V projection into a single matmul.
- MLP: fuse gate+up projection into a single matmul + split.
- Decode attention intermediates in L1 where supported (`nlp_create_qkv_heads_decode(..., memory_config=L1)` and `paged_scaled_dot_product_attention_decode(..., memory_config=L1)`).
- LM head weights in `bfloat8_b` (big decode throughput win versus `bfloat16`).
- Decode trace lifecycle cleanup: release trace on `reset()` so multiple sequences do not leak traces.

## Rejected / Notes
- `decode_pos_buffer` in L1: crashes with paged KV cache update; keep in DRAM.
- Decode concat-heads output in L1: slight decode throughput regression; keep in DRAM.
- `decode_token_buffer` in L1: large decode throughput regression when updated via `copy_host_to_device_tensor`; keep in DRAM.

## Commands
Demo timing (max seq len 29952):

```bash
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
TT_METAL_CACHE=/tmp/tt-metal-cache \
TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
PYTHONUNBUFFERED=1 \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python -u demo.py \
  models/arcee-ai/Arcee-Spark/n150/optimized/model.py \
  --max_seq_len 29952 --seed 0 --device-id 0
```

Teacher-forcing eval (100 tokens):

```bash
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
TT_METAL_CACHE=/tmp/tt-metal-cache \
TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
PYTHONUNBUFFERED=1 \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python -u eval.py \
  models/arcee-ai/Arcee-Spark/n150/optimized/model.py \
  --model arcee-ai/Arcee-Spark --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 --max_seq_len 29952 --seed 0 --device_id 0
```
