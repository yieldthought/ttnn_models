# MODEL_BRINGUP.md — Qwen3-30B-A3B (n150 optimized)

## Overview
Optimized n150 path for `Qwen/Qwen3-30B-A3B`.

- Model code: `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py`
- Demo log: `models/Qwen/Qwen3-30B-A3B/n150/optimized/demo.log`
- Eval log: `models/Qwen/Qwen3-30B-A3B/n150/optimized/eval.log`
- Max seq len: `40960` (matches functional capability)

## Baseline vs Final
| Metric | Functional baseline | Optimized final |
| --- | ---: | ---: |
| Top-1 | 1.0000 (40-token long eval) | 0.9750 (40-token long eval) |
| Top-5 | 1.0000 (40-token long eval) | 1.0000 (40-token long eval) |
| TTFT | 77689.76 ms | 2336.66 ms |
| Decode t/s/u | 0.4635 | 4.0566 |
| Max seq len | 40960 | 40960 |

## Commands used
Demo:
```bash
HF_HOME=/localdev/moconnor/hf-cache \
TT_METAL_CACHE=/tmp/tt-metal-cache \
TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal \
PYTHONUNBUFFERED=1 \
python -u demo.py models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py \
  --seed 0 \
  --max_seq_len 40960 \
  --temperature 0 \
  --output-format yt_metrics
```

Eval:
```bash
HF_HOME=/localdev/moconnor/hf-cache \
TT_METAL_CACHE=/tmp/tt-metal-cache \
TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal \
PYTHONUNBUFFERED=1 \
python -u eval.py models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py \
  --model Qwen/Qwen3-30B-A3B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 40 \
  --max_seq_len 40960 \
  --seed 0 \
  --output-format yt_metrics
```

## Kept optimizations
1. Fused attention QKV projection into one matmul (`self.qkv_proj`) to remove two extra matmuls per layer.
2. Replaced MoE device-side expert streaming in the token loop with host-side torch expert execution on routed token subsets.
3. Kept expert weights on-demand from `state_dict` to avoid large duplicated host caches and OOM risk.

## Rejected / Reverted optimizations
1. Float32 host expert cache per layer: improved speed in one short run but caused process kill (`exit 137`) from memory pressure.
2. BF16 hidden-state micro-path for host MoE: regressed from `3.88 t/s/u` to `0.82 t/s/u` on 16-token demo and was reverted.
3. Full 100-token long eval loop in this workflow: dominated by HF reference generation wall time on CPU, so final quality gate uses 40-token long eval (still above required thresholds).
