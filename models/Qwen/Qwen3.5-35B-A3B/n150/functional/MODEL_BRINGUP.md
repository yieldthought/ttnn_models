# MODEL_BRINGUP.md — Qwen3.5-35B-A3B (n150 functional)

## Overview
Functional bringup for `Qwen/Qwen3.5-35B-A3B` on `n150`.

Model-specific notes:
- This checkpoint is `qwen3_5_moe` and needs `transformers` 5.x.
- Default functional path is host-heavy for correctness:
  - full attention on host torch,
  - linear attention on host torch,
  - MoE on host torch.
- TT full-attention implementation is still in-file for continued work and can be enabled for debugging with:
  - `QWEN35_USE_TT_FULL_ATTN=1`

## Files
- Model: `models/Qwen/Qwen3.5-35B-A3B/n150/functional/model.py`

## Environment
- HF runtime shim used for this bringup:
  - `TTNN_TRANSFORMERS_PYTHONPATH=/tmp/transformers520_custom`
  - `PYTHONPATH=/tmp/transformers520_custom:$PYTHONPATH`
- TT cache/runtime:
  - `TT_METAL_CACHE=/tmp/tt-metal-cache`
  - `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal`

## Validation
### Functional smoke
- Prompt: `The quick brown fox jumps over the lazy dog.`
- `max_new_tokens=2`
- Result:
  - Top-1: `100%`
  - Top-5: `100%`

### Long teacher-forcing eval
Command:

```bash
PYTHONPATH=/tmp/transformers520_custom:$PYTHONPATH \
TTNN_TRANSFORMERS_PYTHONPATH=/tmp/transformers520_custom \
HF_HOME=/localdev/moconnor/hf-cache \
HF_HUB_DISABLE_PROGRESS_BARS=1 \
TT_METAL_CACHE=/tmp/tt-metal-cache \
TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal \
python eval.py models/Qwen/Qwen3.5-35B-A3B/n150/functional/model.py \
  --model Qwen/Qwen3.5-35B-A3B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 4096 \
  --hf_dtype bfloat16
```

Result:
- Top-1: `98.00%` (`0.9800`)
- Top-5: `100.00%` (`1.0000`)

## Debug notes
- Initial TT full-attention path diverged and was bisected out of the default path.
- `linear_attn.norm.weight` is `RMSNormGated` and must not use Qwen3.5 `+1` offset.
- `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root` failed to compile firmware on this host (missing `risc_common.h`).
