# MODEL_BRINGUP.md — AFM-4.5B (t3000 optimized)

## Overview
This is the optimized T3000 path for `arcee-ai/AFM-4.5B`.

- Model code: `models/arcee-ai/AFM-4.5B/t3000/optimized/model.py`
- Demo log: `models/arcee-ai/AFM-4.5B/t3000/optimized/demo.log`
- Eval log: `models/arcee-ai/AFM-4.5B/t3000/optimized/eval.log`
- Parallelism: 1D tensor parallel across the 8-chip T3000 mesh with padded heads/KV heads
- Max seq len: `65536` (no capability regression)
- Decode path: traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`)

## Baseline vs Final (same hardware)
| Metric | Functional baseline | Optimized final |
| --- | ---: | ---: |
| Top-1 | 98% | 98% |
| Top-5 | 100% | 100% |
| TTFT | 181 ms | 69 ms |
| t/s/u | 7.1 | 29.0 |
| Seq len | 65536 | 65536 |

## This optimization pass (same command + seed)
| Metric | Starting optimized baseline | Final optimized |
| --- | ---: | ---: |
| TTFT | 77 ms | 69 ms |
| t/s/u | 24.9 | 29.0 |

## Commands used
Demo:
```bash
env TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
  TT_METAL_CACHE=/tmp/tt-metal-cache \
  PYTHONUNBUFFERED=1 \
  python -u demo.py models/arcee-ai/AFM-4.5B/t3000/optimized/model.py --seed 0
```

Eval:
```bash
env TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
  TT_METAL_CACHE=/tmp/tt-metal-cache \
  PYTHONUNBUFFERED=1 \
  python -u eval.py models/arcee-ai/AFM-4.5B/t3000/optimized/model.py \
    --model arcee-ai/AFM-4.5B \
    --prompt_file prompts/bringup_eval_long.txt \
    --max_new_tokens 100 \
    --max_seq_len 65536
```

## Kept optimizations
1. Fused attention QKV projection into one matmul (`self.qkv_proj`) with per-device Q/K/V chunk ordering that matches width sharding.
2. Added `prefill_logits_last_device()` so demo/eval TTFT uses only final prompt-token logits.
3. Added traced decode execution with preallocated decode token/position/RoPE buffers and `ttnn.execute_trace` replay.
4. Moved decode attention intermediates to L1 where safe (`nlp_create_qkv_heads_decode`, paged decode SDPA, decode `nlp_concat_heads`).

## Deferred changes
1. Decode-specific `nlp_concat_heads_decode` migration was deferred in this pass. Current `nlp_concat_heads` decode path already meets the target and keeps the GQA/padded-head flow simple.

## Environment note
1. The default `tt-smi` wrapper at `/home/moconnor/bin/tt-smi` is broken on this host (`bad interpreter`). Use `/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/tt-smi`.
