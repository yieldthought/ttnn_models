# MODEL_BRINGUP.md — models/Qwen/Qwen3.5-35B-A3B/n150

## Overview
Optimization pass for `models/Qwen/Qwen3.5-35B-A3B/n150` using `ttnn-model-optimization`.

Retained changes:
1. Decode trace is enabled by default (`QWEN35_USE_DECODE_TRACE=1`) and used in the optimized flow.
2. Trace capture targets the decode head (`hidden -> lm_head`) so capture avoids host-MoE writes.
3. Prefill-only on-device argmax (`next_token_device`) is kept for TTFT.
4. Decode-only MoE route cap is kept at `decode_top_k=6` (env override: `QWEN35_DECODE_TOP_K`).

## Baseline vs Final

| Metric | Baseline (functional) | Final (optimized) | Delta |
|---|---:|---:|---:|
| Top-1 (100-token eval) | 97.00% | 96.00% | -1.00 pt |
| Top-5 (100-token eval) | 100.00% | 100.00% | 0.00 pt |
| TTFT | 5403 ms | 5393 ms | -10 ms |
| Decode throughput | 2.46 t/s/u | 4.04 t/s/u | +1.57 t/s/u (+63.8%) |

## Decode Trace Status
- Optimized default path uses decode trace (`USE_DECODE_TRACE` default is on).
- Successful traced decode evidence from `demo.log`:
  - `decode_trace: captured lm_head trace`
  - `decode_trace: executing captured lm_head trace`
- Final measured run: `ttft_ms=5393.270309781656`, `decode_tps_u=4.037444069807221`.

## Optimization Decisions
1. Kept decode-head trace capture/execute.
   - Why: full decode trace is blocked by host MoE writes during capture; decode-head trace captures cleanly and executes every decode step after capture.
2. Kept decode route cap at 6.
   - Why: improves decode throughput while preserving acceptable eval quality.
3. Kept prefill-only device argmax.
   - Why: avoids full-vocab host transfer in prefill token selection path.
4. Rejected full decode trace capture.
   - Why: runtime raises `TT_FATAL: Writes are not supported during trace capture` when host writes are present in capture region.

## Commands Used
Demo command is logged in `demo.log`.
Eval command is logged in `eval.log`.
