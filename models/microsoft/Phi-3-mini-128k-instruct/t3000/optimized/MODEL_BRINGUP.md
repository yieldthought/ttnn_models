# MODEL_BRINGUP.md - Phi-3 Mini 128k Instruct (t3000 optimized)

## Overview
This is the optimized TTNN bringup of `microsoft/Phi-3-mini-128k-instruct` for `t3000`.

- Model code: `models/microsoft/Phi-3-mini-128k-instruct/t3000/optimized/model.py`
- Demo log: `models/microsoft/Phi-3-mini-128k-instruct/t3000/optimized/demo.log`
- Eval log: `models/microsoft/Phi-3-mini-128k-instruct/t3000/optimized/eval.log`
- Machine-readable metrics: `models/microsoft/Phi-3-mini-128k-instruct/t3000/optimized/metrics.json`
- Decode path uses traced execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`).

## Baseline vs final
| Metric | Functional baseline (`MODELS.md`) | Final optimized |
| --- | ---: | ---: |
| Top-1 | 90% | 92% |
| Top-5 | 100% | 99% |
| TTFT | 184 ms | 105 ms |
| t/s/u | 6.8 | 23.6 |
| Seq len | 12288 | 12288 |

## Kept optimization decisions
1. Decode trace with preallocated decode buffers.
- Preallocates token ids, position tensor, and per-token RoPE cos/sin buffers.
- Uses decode-only preallocated pad zeros for RoPE head-dim padding (avoids `ttnn.zeros` inside trace capture).

2. Prefill last-token logits fast path (`prefill_logits_last_device`).
- Avoids materializing full prefill logits when only the final prompt token is needed (demo/eval TTFT).

3. Fused QKV projection (single matmul) with TP-safe shard ordering.
- Builds a fused weight with per-shard `[q_i, k_i, v_i]` ordering so width-sharding matches `nlp_create_qkv_heads[_decode]`.

4. Paged KV cache (block_size=64).
- Cache layout: `[max_num_blocks, n_kv_heads, block_size, head_dim]`.

## Rejected optimization attempts
- None yet.

## Commands used
```bash
TT_VISIBLE_DEVICES=0,1,2,3 \
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u demo.py models/microsoft/Phi-3-mini-128k-instruct/t3000/optimized/model.py \
  --prompt "Write a short, vivid paragraph about a sunrise over the ocean." \
  --max-new-tokens 120 \
  --max_seq_len 12288 \
  --seed 0

TT_VISIBLE_DEVICES=0,1,2,3 \
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u eval.py models/microsoft/Phi-3-mini-128k-instruct/t3000/optimized/model.py \
  --model microsoft/Phi-3-mini-128k-instruct \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 12288 \
  --seed 0
```

## Environment note
On this host, `tt-smi` in `/home/moconnor/bin/` is broken (`bad interpreter`).
Use `/home/moconnor/.local/bin/tt-smi` for reset/list.
