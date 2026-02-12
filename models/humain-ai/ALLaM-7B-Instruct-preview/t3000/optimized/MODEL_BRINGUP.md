# MODEL_BRINGUP.md — ALLaM 7B Instruct preview (t3000 optimized)

## Overview
Optimized TTNN path for `humain-ai/ALLaM-7B-Instruct-preview` on `t3000`.

- Model code: `models/humain-ai/ALLaM-7B-Instruct-preview/t3000/optimized/model.py`
- Demo log: `models/humain-ai/ALLaM-7B-Instruct-preview/t3000/optimized/demo.log`
- Eval log: `models/humain-ai/ALLaM-7B-Instruct-preview/t3000/optimized/eval.log`
- Experiment log: `models/humain-ai/ALLaM-7B-Instruct-preview/t3000/optimized/experiments.log`
- Decode path: traced execution (`ttnn.begin_trace_capture` + `ttnn.execute_trace`)
- Max sequence length: `4096`

## Baseline vs Final
| Metric | Functional baseline (MODELS.md) | Optimized final |
| --- | ---: | ---: |
| Top-1 | 95% | 97% |
| Top-5 | 100% | 100% |
| TTFT | 127 ms | 61 ms |
| t/s/u | 9.1 | 24.3 |
| Seq len | 4096 | 4096 |

## Device setup and visibility
Before demo/eval:

```bash
/opt/venv/bin/tt-smi -r
/opt/venv/bin/tt-smi -ls
```

`tt-smi -ls` showed PCI devices `0,1,2,3` (`n300 L`) with paired `n300 R` entries, and both optimized runs opened a full `2x4` mesh (`Mesh shape: 2x4` in `demo.log`).

## Commands used
Demo:

```bash
env TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1 \
  TT_VISIBLE_DEVICES=0,1,2,3 \
  TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
  TT_METAL_CACHE=/tmp/tt-metal-cache \
  PYTHONUNBUFFERED=1 \
  python /localdev/moconnor/ttnn_models/demo.py \
  /localdev/moconnor/ttnn_models/models/humain-ai/ALLaM-7B-Instruct-preview/t3000/optimized/model.py \
  --max_seq_len 4096 --max-new-tokens 8 --seed 0
```

Eval:

```bash
env TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1 \
  TT_VISIBLE_DEVICES=0,1,2,3 \
  TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
  TT_METAL_CACHE=/tmp/tt-metal-cache \
  PYTHONUNBUFFERED=1 \
  python /localdev/moconnor/ttnn_models/eval.py \
  /localdev/moconnor/ttnn_models/models/humain-ai/ALLaM-7B-Instruct-preview/t3000/optimized/model.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 --max_seq_len 4096
```

## Kept optimizations
1. Fused QKV projection per attention layer (`qkv_proj`) to reduce matmul count and decode overhead.
2. `prefill_logits_last_device()` fast path so demo/eval prefill computes only final prompt-token logits.
3. Traced decode path with reusable decode buffers (token IDs, positions, RoPE slices) to avoid per-token trace recapture and allocation churn.
4. Decode attention intermediates kept in L1 by default (`TTNN_USE_DECODE_L1_PATH=1`).

## Rejected candidates
1. `TTNN_USE_DECODE_TRACE=0` (no traced decode):
   - TTFT: `59 ms`
   - Decode: `13.8 t/s/u` (far below traced decode)
   - Decision: rejected; traced decode kept.
2. `TTNN_USE_DECODE_L1_PATH=0`:
   - TTFT: `63 ms`
   - Decode: `23.7 t/s/u` (worse than kept `24.3`)
   - Decision: rejected; keep decode L1 path.
3. `TTNN_DECODE_MATMUL_FIDELITY=lofi` and `TTNN_DECODE_MATMUL_FIDELITY=hifi2`:
   - LoFi: `62 ms`, `24.2 t/s/u`
   - HiFi2: `63 ms`, `24.1 t/s/u`
   - Decision: no clear gain; keep default fidelity.
4. Defaulting `TTNN_DECODE_MATMUL_CORE_GRID=4x8` in model code:
   - Demo improved to `24.9 t/s/u` in sweep, but long eval regressed to `Top-1 96.00%` (from `97.00%` baseline) in repeated checks.
   - Decision: rejected due correctness regression risk.

## Acceptance checklist
1. Long eval quality: `Top-1 97.00%`, `Top-5 100.00%`.
2. Decode uses traced execution: yes (default `TTNN_USE_DECODE_TRACE=1` and trace capture/execute path in model).
3. TTFT improved vs functional baseline: `61 ms < 127 ms`.
4. Throughput improved vs functional baseline: `24.3 > 9.1` t/s/u.
5. No capability regression: optimized `max_seq_len=4096` equals functional.
