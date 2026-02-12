# MODEL_BRINGUP.md - Qwen3 0.6B (t3000 optimized)

## Overview
This is the optimized TTNN bringup of `Qwen/Qwen3-0.6B` for `t3000`.

- Model code: `models/Qwen/Qwen3-0.6B/t3000/optimized/model.py`
- Demo log: `models/Qwen/Qwen3-0.6B/t3000/optimized/demo.log`
- Eval log: `models/Qwen/Qwen3-0.6B/t3000/optimized/eval.log`
- Machine-readable metrics: `models/Qwen/Qwen3-0.6B/t3000/optimized/metrics.json`
- Decode path uses traced execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`).

## Baseline vs final
| Metric | Functional baseline (`MODELS.md`) | Final optimized |
| --- | ---: | ---: |
| Top-1 | 98% | 98% |
| Top-5 | 100% | 100% |
| TTFT | 229 ms | 59 ms |
| t/s/u | 6.2 | 61.9 |
| Seq len | 40960 | 40960 |

## Kept optimization decisions
1. Decode trace with preallocated decode buffers.
- Preallocates token ids, position tensor, and RoPE cos/sin buffers.
- Captures a stable decode graph and replays it with `ttnn.execute_trace`.

2. Prefill last-token logits fast path (`prefill_logits_last_device`).
- Avoids materializing full prefill logits when only the final prompt token is needed.

3. Paged KV cache (block_size=64).
- Cache layout: `[max_num_blocks, n_kv_heads, block_size, head_dim]`.
- Uses identity page table and paged update/fill ops.

4. Fused QKV projection.
- Replaces 3 matmuls (`q_proj`, `k_proj`, `v_proj`) + concat with one `qkv_proj` matmul.
- Builds fused weight with TP-safe shard ordering: per-shard `[q_i, k_i, v_i]`.

## Rejected optimization attempts
- None yet.

## Commands used
```bash
TT_VISIBLE_DEVICES=0,1,2,3 \
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u demo.py models/Qwen/Qwen3-0.6B/t3000/optimized/model.py \
  --seed 0 \
  --max_seq_len 40960

TT_VISIBLE_DEVICES=0,1,2,3 \
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u eval.py models/Qwen/Qwen3-0.6B/t3000/optimized/model.py \
  --model Qwen/Qwen3-0.6B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 40960
```

## Environment note
On this host, `tt-smi` in `/home/moconnor/bin/` is broken (`bad interpreter`).
Use `/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python3 -m tt_smi` for reset/list.
