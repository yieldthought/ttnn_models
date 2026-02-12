# MODEL_BRINGUP.md - Falcon3 7B Instruct (n300 optimized)

## Overview
Optimized TTNN implementation of `tiiuae/Falcon3-7B-Instruct` for n300 using 1D tensor parallel on a 1x2 mesh.

- Model code: `models/tiiuae/Falcon3-7B-Instruct/n300/optimized/model.py`
- Demo log: `models/tiiuae/Falcon3-7B-Instruct/n300/optimized/demo.log`
- Eval log: `models/tiiuae/Falcon3-7B-Instruct/n300/optimized/eval.log`
- Machine-readable metrics: `models/tiiuae/Falcon3-7B-Instruct/n300/optimized/metrics.json`

## Baseline vs final (same hardware, same prompt setup)
| Metric | Functional baseline (`MODELS.md`) | Optimized final |
| --- | ---: | ---: |
| Top-1 | 97% | 97% |
| Top-5 | 100% | 100% |
| TTFT | 661 ms | 72 ms |
| t/s/u | 5.6 | 21.8 |
| Seq len | 32768 | 32768 |

Acceptance checks:
- Long eval quality: pass (`97% / 100%`)
- Decode trace: pass (`ttnn.begin_trace_capture` + `ttnn.execute_trace` in `model.py`)
- TTFT improved: pass (`72.32ms < 661ms`)
- t/s/u improved: pass (`21.77 > 5.6`)
- Seq len non-regression: pass (`32768 >= 32768`)

## Kept optimization decisions
1. Fused attention Q/K/V projections into one matmul (`self.qkv_proj`) with per-device QKV packing for TP correctness.
2. Added `prefill_logits_last_device()` so demo/eval TTFT uses only final prompt-token logits.
3. Kept traced decode with preallocated token/position/RoPE buffers (`ttnn.execute_trace` replay path).
4. Kept decode attention and decode MLP intermediates in L1 (`TTNN_USE_DECODE_L1_PATH=1` default).
5. Kept decode MLP SILU fusion via `ttnn.mul(..., input_tensor_a_activations=[SILU])`.
6. Kept decode LM head on sliced `[1, 1, 1, hidden]` activation to avoid padded-32 LM-head work.

## Rejected optimization experiments
1. Decode `transpose + nlp_concat_heads` replacement with `to_memory_config(...decode_heads_memcfg) + nlp_concat_heads_decode`.
   - Result was not repeatable on this host and sometimes regressed decode throughput.
   - Measured runs during this experiment: `75.54ms / 21.23 t/s/u` and `70.85ms / 19.15 t/s/u`.
   - Reverted to the previous decode head concat path.

## Environment note
- `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root` failed on this host (missing `eth_l1_address_map.h` during erisc compile).
- Used `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal` for the recorded demo/eval runs.

## Commands used
Demo:
```bash
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
TRANSFORMERS_CACHE=/proj_sw/user_dev/moconnor/hf-cache \
HF_HUB_CACHE=/proj_sw/user_dev/moconnor/hf-cache/hub \
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal \
TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
python demo.py models/tiiuae/Falcon3-7B-Instruct/n300/optimized/model.py \
  --prompt-file prompts/bringup_eval_long.txt \
  --max-new-tokens 100 \
  --temperature 0 \
  --device-id 0 \
  --max_seq_len 32768
```

Eval:
```bash
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
TRANSFORMERS_CACHE=/proj_sw/user_dev/moconnor/hf-cache \
HF_HUB_CACHE=/proj_sw/user_dev/moconnor/hf-cache/hub \
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
TT_METAL_CACHE=/tmp/tt-metal-cache \
TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal \
TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
python eval.py models/tiiuae/Falcon3-7B-Instruct/n300/optimized/model.py \
  --model tiiuae/Falcon3-7B-Instruct \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 32768 \
  --device_id 0 \
  --seed 0
```
