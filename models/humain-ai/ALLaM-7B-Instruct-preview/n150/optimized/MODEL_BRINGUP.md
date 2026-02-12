# MODEL_BRINGUP.md — ALLaM 7B Instruct preview (n150 optimized)

## Overview
This is the optimized path for `humain-ai/ALLaM-7B-Instruct-preview` under `n150`.

- Model code: `models/humain-ai/ALLaM-7B-Instruct-preview/n150/optimized/model.py`
- Demo log: `models/humain-ai/ALLaM-7B-Instruct-preview/n150/optimized/demo.log`
- Eval log: `models/humain-ai/ALLaM-7B-Instruct-preview/n150/optimized/eval.log`
- Machine-readable metrics: `models/humain-ai/ALLaM-7B-Instruct-preview/n150/optimized/metrics.json`
- Target max sequence length: `4096` (no capability regression)
- Decode path: traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`)

## Baseline vs Final
| Metric | Functional baseline (MODELS.md) | Optimized final |
| --- | ---: | ---: |
| Top-1 | 97% | 97% |
| Top-5 | 100% | 100% |
| TTFT | 76 ms | 69 ms |
| t/s/u | 14.9 | 15.8 |
| Seq len | 4096 | 4096 |

## Commands used
Device checks:
```bash
tt-smi -r
tt-smi -ls
```

Demo:
```bash
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u demo.py models/humain-ai/ALLaM-7B-Instruct-preview/n150/optimized/model.py \
  --max_seq_len 4096 \
  --seed 0
```

Eval:
```bash
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u eval.py models/humain-ai/ALLaM-7B-Instruct-preview/n150/optimized/model.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 4096
```

## Kept optimizations
1. Fused QKV projection (`qkv_proj`) so each layer uses one matmul for Q/K/V.
2. `prefill_logits_last_device()` fast path for TTFT-sensitive prefill in demo/eval.
3. Decode trace with reusable preallocated decode buffers (token ids, positions, RoPE cos/sin slices).
4. Optional decode L1 path (`TTNN_USE_DECODE_L1_PATH=1` by default) for attention intermediates.

## Rejected / Not kept decisions
1. BF16 weights (`ttnn.bfloat16`): rejected because the 7B weights do not fit in n150 DRAM; `ttnn.bfloat8_b` is required.
2. Untraced decode by default (`TTNN_USE_DECODE_TRACE=0`): rejected because release requires traced decode and traced execution avoids per-token overhead.
3. Decode intermediates in DRAM (`TTNN_USE_DECODE_L1_PATH=0`): not kept because decode is memory-traffic sensitive; the default is to keep decode attention intermediates in L1. The toggle remains for debugging/sweeps.
4. Tensor-parallel / mesh sharding: rejected because n150 is a single-device target; keep the implementation simple and single-device.
