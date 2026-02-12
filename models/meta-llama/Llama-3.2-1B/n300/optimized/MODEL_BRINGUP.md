# MODEL_BRINGUP.md - Llama 3.2 1B (n300 optimized)

## Overview
This is the optimized TTNN bringup of `meta-llama/Llama-3.2-1B` for `n300`.

- Model code: `models/meta-llama/Llama-3.2-1B/n300/optimized/model.py`
- Demo log: `models/meta-llama/Llama-3.2-1B/n300/optimized/demo.log`
- Eval log: `models/meta-llama/Llama-3.2-1B/n300/optimized/eval.log`
- Machine-readable metrics: `models/meta-llama/Llama-3.2-1B/n300/optimized/metrics.json`
- Decode path uses traced execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`)

## Baseline vs final
| Metric | Functional baseline (`MODELS.md`) | Optimized final |
| --- | ---: | ---: |
| Top-1 | 90% | 93% |
| Top-5 | 100% | 100% |
| TTFT | 610 ms | 33 ms |
| t/s/u | 6.7 | 41.9 |
| Seq len | 131072 | 131072 |

## Host topology note
On this host the fabric topology is disconnected (`phys_deg_hist={0:2}`), so a true `1x2`
mesh cannot be mapped via auto-discovery.

For measurement, demo/eval were run with:
- `TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1`
- requested mesh `1x2` falls back to discovered `1x1`

The demo log records the effective runtime mesh (`Mesh shape: 1x1`).

## Kept optimization decisions
1. Paged KV cache (`block_size=64`) for long-seq capability without dense cache overhead.
2. Fused TP-aware QKV projection (`qkv_proj`) so each shard receives `[Q_local, K_local, V_local]`.
3. Prefill-last-logits fast path (`prefill_logits_last_device`) to avoid materializing full prefill logits.
4. Decode trace with reusable preallocated decode buffers.
5. `1x1` compatibility (allow `tp_size==1`) so the model can run under mesh fallback on disconnected hosts.

## Rejected optimization attempts
None in this pass (host topology prevented running a true `1x2` mesh, so decode micro-tuning was deferred).

## Commands used
Device checks:
```bash
tt-smi -r
tt-smi -ls
```

Demo:
```bash
TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1 \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u demo.py models/meta-llama/Llama-3.2-1B/n300/optimized/model.py \
  --prompt-file prompts/bringup_eval_long.txt \
  --max-new-tokens 100 \
  --temperature 0 \
  --seed 0 \
  --max_seq_len 131072
```

Eval:
```bash
TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1 \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u eval.py models/meta-llama/Llama-3.2-1B/n300/optimized/model.py \
  --model meta-llama/Llama-3.2-1B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 131072 \
  --seed 0
```
