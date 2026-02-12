# MODEL_BRINGUP.md - Llama 3.2 1B (t3000 optimized)

## Overview
This is the optimized TTNN bringup of `meta-llama/Llama-3.2-1B` for `t3000`.

- Model code: `models/meta-llama/Llama-3.2-1B/t3000/optimized/model.py`
- Demo log: `models/meta-llama/Llama-3.2-1B/t3000/optimized/demo.log`
- Eval log: `models/meta-llama/Llama-3.2-1B/t3000/optimized/eval.log`
- Machine-readable metrics: `models/meta-llama/Llama-3.2-1B/t3000/optimized/metrics.json`
- Decode path uses traced execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`)

## Baseline vs final
| Metric | Functional baseline (`MODELS.md`) | Starting optimized baseline (this pass) | Final optimized |
| --- | ---: | ---: | ---: |
| Top-1 | 92% | 93% | 94% |
| Top-5 | 100% | 100% | 100% |
| TTFT | 267 ms | 36 ms | 37 ms |
| t/s/u | 6.6 | 52.6 | 58.5 |
| Seq len | 131072 | 131072 | 131072 |

## Kept optimization decisions
1. Decode trace with preallocated decode buffers.
- Keeps decode in a fixed graph and avoids per-token TT allocations.
- Preserved from earlier optimized work; still required for this throughput level.

2. Prefill-last-logits fast path (`prefill_logits_last_device`).
- Avoids materializing full prefill logits when only the final prompt token is needed.
- Keeps TTFT low without changing decode semantics.

3. Fused QKV projection with tensor-parallel-safe shard ordering.
- Replaced three attention projection matmuls (`q_proj`, `k_proj`, `v_proj`) plus concat with one `qkv_proj` matmul.
- Built fused weight by chunking Q/K/V per TP shard and concatenating per-shard (`[q_i, k_i, v_i]`) before global pack.
- Result: decode throughput improved from 52.6 to 58.5 t/s/u with no correctness regression.

## Rejected optimization attempts
1. `lm_head` weights in `bfloat8_b`.
- Demo result: 52.3 t/s/u (worse than 52.6 baseline), TTFT unchanged.

2. Decode `nlp_create_qkv_heads_decode` output in L1.
- Demo result: TTFT regressed to 42 ms, decode regressed to 52.4 t/s/u.

3. Decode paged SDPA output in L1.
- Demo result: TTFT regressed to 39 ms, decode regressed to 52.4 t/s/u.

## Commands used
```bash
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0,1,2,3 \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u demo.py models/meta-llama/Llama-3.2-1B/t3000/optimized/model.py --seed 0

TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0,1,2,3 \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u eval.py models/meta-llama/Llama-3.2-1B/t3000/optimized/model.py \
  --model meta-llama/Llama-3.2-1B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 131072
```

## Environment note
On this host, `/home/moconnor/bin/tt-smi` is broken (`bad interpreter`).
Use `/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/tt-smi` for reset/list.
