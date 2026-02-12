# MODEL_BRINGUP.md - Falcon3 7B Instruct (t3000 optimized)

## Overview
Optimized TTNN implementation of `tiiuae/Falcon3-7B-Instruct` for T3000 using 1D tensor parallel across a 2x4 mesh.

- Model code: `models/tiiuae/Falcon3-7B-Instruct/t3000/optimized/model.py`
- Demo log: `models/tiiuae/Falcon3-7B-Instruct/t3000/optimized/demo.log`
- Eval log: `models/tiiuae/Falcon3-7B-Instruct/t3000/optimized/eval.log`
- Machine-readable metrics: `models/tiiuae/Falcon3-7B-Instruct/t3000/optimized/metrics.json`

## Baseline vs Final Metrics (T3000)
Functional baseline (from `MODELS.md`):
- Top-1: 97%
- Top-5: 100%
- TTFT: 199ms
- Decode: 7.3 t/s/u
- Seq len: 32768

Optimized results (this directory):
- Top-1: 97% (see `eval.log`)
- Top-5: 100% (see `eval.log`)
- TTFT: 58ms (see `demo.log`)
- Decode: 26.3 t/s/u (see `demo.log`)
- Seq len: 32768

## Optimizations Kept
- Fused QKV projection:
  - Pack per-device Q/K/V shards into a single `qkv_proj` matmul per layer.
- Prefill last-token logits fast path:
  - Implement `prefill_logits_last_device()` so demo/eval avoid running `lm_head` on the full prompt sequence.
- Traced decode execution:
  - Use `ttnn.begin_trace_capture` / `ttnn.execute_trace` with persistent decode buffers (`decode_token_buffer`, `decode_pos_buffer`, RoPE buffers).
- Decode L1 path:
  - Decode intermediates use `DECODE_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG` (toggle with `TTNN_USE_DECODE_L1_PATH=0`).

## Experiments Skipped
- Fusing gate+up projections into a single matmul:
  - Not needed to beat the baseline metrics; kept the MLP readable.

## Notes
- Decode trace is enabled by default; disable with `TTNN_USE_DECODE_TRACE=0`.
- Decode RoPE uses per-token cos/sin buffers copied from host so the decode trace is position-independent.
- Max sequence length is capped by `MAX_CACHE_SEQ_LEN=32768`.

## Repro
See `demo.log` and `eval.log` for exact commands and outputs used to produce the final metrics.
