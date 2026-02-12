# MODEL_BRINGUP.md — AFM-4.5B (n300 optimized)

## Overview
This is the optimized N300 path for `arcee-ai/AFM-4.5B`.

- Model code: `models/arcee-ai/AFM-4.5B/n300/optimized/model.py`
- Demo log: `models/arcee-ai/AFM-4.5B/n300/optimized/demo.log`
- Eval log: `models/arcee-ai/AFM-4.5B/n300/optimized/eval.log`
- Parallelism: 1x2 tensor parallel mesh (QKV/up width-sharded, output/down height-sharded + all-reduce)
- Max seq len: `65536` (no capability regression)
- Decode path: traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`)

## Baseline vs Final (same hardware)
| Metric | Functional baseline | Optimized final |
| --- | ---: | ---: |
| Top-1 | 97% | 99% |
| Top-5 | 100% | 100% |
| TTFT | 283 ms | 56 ms |
| t/s/u | 5.6 | 23.6 |
| Seq len | 65536 | 65536 |

## Commands used
Demo:
```bash
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u demo.py models/arcee-ai/AFM-4.5B/n300/optimized/model.py --seed 0
```

Eval:
```bash
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u eval.py models/arcee-ai/AFM-4.5B/n300/optimized/model.py \
  --model arcee-ai/AFM-4.5B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 65536
```

## Kept optimizations
1. Added fused QKV projection per attention layer to remove two matmuls, with TP-aware packing so each device shard is `[Q_local, K_local, V_local]`.
2. Added `prefill_logits_last_device()` so demo/eval prefill measures only last-token logits extraction.
3. Added traced decode path with preallocated decode token/position/RoPE buffers and `ttnn.execute_trace` reuse.
4. Moved decode QKV/attention intermediates to L1 (`ttnn.linear` decode QKV output, `nlp_create_qkv_heads_decode`, `paged_scaled_dot_product_attention_decode`, and decode `nlp_concat_heads`) to reduce decode-path memory traffic.

## Optimization loop notes (this pass)
1. Baseline on current optimized code: `TTFT 66 ms`, `23.7 t/s/u`.
2. Decode QKV + decode SDPA moved to L1: `TTFT 49 ms`, `23.6 t/s/u`.
3. Added decode `nlp_concat_heads` L1 output (final config): peak decode observed `24.2 t/s/u` at `58 ms` TTFT; logged reproducible run `56 ms`, `23.6 t/s/u`.
4. Long eval after final config: `Top-1 99%`, `Top-5 100%`.

## Rejected or corrected experiments
1. Initial fused-QKV packing as global `[Q, K, V]` broke TP shard layout and caused `Top-1/Top-5 = 0/0`; corrected to per-device local packing.
2. Initial decode RoPE buffer with single sequence row failed decode rotary shape checks on AFM (`Cos cache dimension` mismatch); corrected by using decode RoPE buffers sized to the flattened decode rotary sequence and filling from current position.
3. `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root` caused firmware compile/header failures on this host; kept only `TT_METAL_CACHE=/tmp/tt-metal-cache` override.
