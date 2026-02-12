# MODEL_BRINGUP.md - Qwen3 0.6B (n150 optimized)

## Overview
This is the optimized TTNN bringup of `Qwen/Qwen3-0.6B` for `n150`.

- Model code: `models/Qwen/Qwen3-0.6B/n150/optimized/model.py`
- Demo log: `models/Qwen/Qwen3-0.6B/n150/optimized/demo.log`
- Eval log: `models/Qwen/Qwen3-0.6B/n150/optimized/eval.log`
- Machine-readable metrics: `models/Qwen/Qwen3-0.6B/n150/optimized/metrics.json`
- Decode path uses traced execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`).

## Baseline vs final
| Metric | Functional baseline (`MODELS.md`) | Final optimized |
| --- | ---: | ---: |
| Top-1 | 99% | 97% |
| Top-5 | 100% | 100% |
| TTFT | 52 ms | 25 ms |
| t/s/u | 28.0 | 46.2 |
| Seq len | 40960 | 40960 |

Note: TTFT and t/s/u depend on the demo prompt and sampling settings. See `demo.log` for the exact command and output.

## Kept optimization decisions
1. Decode trace with preallocated decode buffers.
- Preallocates decode token ids, position tensor, and per-token RoPE cos/sin buffers.
- Captures a stable decode graph and replays it with `ttnn.execute_trace`.

2. Prefill last-token logits fast path (`prefill_logits_last_device`).
- Avoids materializing full prefill logits when only the final prompt token is needed.

3. Paged KV cache (block_size=64).
- Cache layout: `[max_num_blocks, n_kv_heads, block_size, head_dim]`.
- Uses an identity page table and paged update/fill ops.

4. Fused QKV projection.
- Replaces 3 matmuls (`q_proj`, `k_proj`, `v_proj`) + concat with one `qkv_proj` matmul.

5. BFP8 weights for the MLP projections.
- MLP weights use `ttnn.bfloat8_b` to improve decode throughput while keeping long-eval quality high.

## Rejected optimization attempts
- None yet.

## Commands used
```bash
TT_METAL_CACHE=/tmp/tt-metal-cache PYTHONUNBUFFERED=1 \
python -u demo.py models/Qwen/Qwen3-0.6B/n150/optimized/model.py \
  --seed 0 \
  --max_seq_len 40960

TT_METAL_CACHE=/tmp/tt-metal-cache PYTHONUNBUFFERED=1 \
python -u eval.py models/Qwen/Qwen3-0.6B/n150/optimized/model.py \
  --model Qwen/Qwen3-0.6B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 40960 \
  --seed 0
```
