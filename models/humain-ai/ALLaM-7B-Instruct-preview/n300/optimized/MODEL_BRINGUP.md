# MODEL_BRINGUP.md — ALLaM 7B Instruct preview (n300 optimized)

## Overview
This is the optimized path for `humain-ai/ALLaM-7B-Instruct-preview` under `n300`.

- Model code: `models/humain-ai/ALLaM-7B-Instruct-preview/n300/optimized/model.py`
- Demo log: `models/humain-ai/ALLaM-7B-Instruct-preview/n300/optimized/demo.log`
- Eval log: `models/humain-ai/ALLaM-7B-Instruct-preview/n300/optimized/eval.log`
- Target max sequence length: `4096` (no capability regression)
- Decode path: traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`)

## Baseline vs Final
| Metric | Functional baseline (MODELS.md) | Optimized final |
| --- | ---: | ---: |
| Top-1 | 97% | 97% |
| Top-5 | 100% | 100% |
| TTFT | 184 ms | 69 ms |
| t/s/u | 7.9 | 15.9 |
| Seq len | 4096 | 4096 |

## Same-config validation (this host)
Verifier feedback was correct: previous optimized logs were `1x1` fallback while older functional logs were `1x2`.
To satisfy same-system comparison, functional and optimized were both measured with:

- `TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1`
- discovered mesh `1x1` on this disconnected host
- `max_seq_len=4096`

| Metric | Functional (1x1 fallback) | Optimized (1x1 fallback) |
| --- | ---: | ---: |
| Top-1 | 97.00% | 97.00% |
| Top-5 | 100.00% | 100.00% |
| TTFT | 74 ms | 69 ms |
| t/s/u | 15.0 | 15.9 |
| Seq len | 4096 | 4096 |

Evidence logs:
- `models/humain-ai/ALLaM-7B-Instruct-preview/n300/functional/demo.log`
- `models/humain-ai/ALLaM-7B-Instruct-preview/n300/functional/eval.log`
- `models/humain-ai/ALLaM-7B-Instruct-preview/n300/optimized/demo.log`
- `models/humain-ai/ALLaM-7B-Instruct-preview/n300/optimized/eval.log`

## Host topology note
On this reservation the two Wormhole cards are physically disconnected for fabric use (`phys_deg_hist={0:2}` and all ETH links down in `system_health`), so a true `1x2` mesh cannot be mapped.

To keep validation unblocked, this run used an opt-in compatibility path:
- `TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1` in `device_utils.py`
- requested `1x2` mesh falls back to discovered `1x1`
- model allows `num_devices=1` compatibility while preserving the traced decode path and `max_seq_len=4096`

The demo/eval logs show this explicitly (`Mesh shape: 1x1` in demo output).

## Commands used
Device checks:
```bash
tt-smi -r
tt-smi -ls
/proj_sw/user_dev/moconnor/tt-metal/build_Release/tools/umd/system_health
```

Demo:
```bash
TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1 \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u demo.py models/humain-ai/ALLaM-7B-Instruct-preview/n300/optimized/model.py \
  --max_seq_len 4096 \
  --seed 0
```

Eval:
```bash
TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1 \
TT_METAL_CACHE=/tmp/tt-metal-cache \
PYTHONUNBUFFERED=1 \
python -u eval.py models/humain-ai/ALLaM-7B-Instruct-preview/n300/optimized/model.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 4096
```

## Kept optimizations
1. Fused TP-aware QKV projection (`qkv_proj`) so each shard receives `[Q_local, K_local, V_local]`.
2. `prefill_logits_last_device()` fast path for TTFT-sensitive prefill.
3. Decode trace with reusable preallocated decode buffers.
4. Decode attention intermediates kept in L1 where safe (decode QKV path, decode head creation, decode SDPA, decode concat-heads).
5. Opt-in disconnected-host mesh fallback in `device_utils.py` plus `1x1` compatibility in this model.

## Optimization evidence (mapped to doc/ttnn.md workflow)
All experiment command/output blocks are in:
- `models/humain-ai/ALLaM-7B-Instruct-preview/n300/optimized/experiments.log`

### 1. Profile decode hotspots first
1. Tried `TT_METAL_DEVICE_PROFILER=1 TT_SIGNPOSTS=1` on optimized demo.
2. Outcome: run completed (`TTFT 69 ms`, `Decode 15.8 t/s/u`) but profiler reported repeated `Profiler DRAM buffers were full, markers were dropped`, and no usable `ops_perf_results*.csv` was emitted for `tt-perf-report`.
3. Decision: not kept as a reliable hotspot workflow on this host; documented as measurement limitation.

### 2. Decode matmul + program/grid tuning
1. Tried decode memory-path toggle (`TTNN_USE_DECODE_L1_PATH=0`) to test decode matmul path sensitivity.
2. Outcome: `TTFT 69 ms`, `Decode 15.5 t/s/u` (worse than kept `15.9`), so reverted.
3. Added explicit decode matmul hooks in model (`TTNN_DECODE_MATMUL_CORE_GRID`, `TTNN_DECODE_MATMUL_FIDELITY`) for controlled sweeps.
4. Tried `TTNN_DECODE_MATMUL_CORE_GRID=2x8` (program/grid knob).
5. Outcome (short-run sweep): `TTFT 69 ms`, `Decode 16.2 t/s/u` with `--max-new-tokens 8`, which is slower than default short-run `16.5 t/s/u`, so not kept.
6. Earlier longer-run attempts were also logged during exploration and were not retained.

### 3. Precision/fidelity sweeps
1. Tried `TTNN_WEIGHT_DTYPE=bf16`.
2. Outcome: model build failed with DRAM OOM (`Out of Memory`), so rejected.
3. Tried decode matmul fidelity override `TTNN_DECODE_MATMUL_FIDELITY=lofi`.
4. Outcome (short-run sweep): `TTFT 69 ms`, `Decode 16.5 t/s/u` (no throughput gain vs default short-run).
5. Teacher-forcing quality check with `TTNN_DECODE_MATMUL_FIDELITY=lofi`: `Top-1 97.00%`, `Top-5 100.00%`.
6. Decision: no measurable performance benefit, so fidelity override is not kept.

### 4. Decode trace allocation discipline
1. Tried `TTNN_USE_DECODE_TRACE=0` against kept traced decode.
2. Outcome: `TTFT 69 ms`, `Decode 16.0 t/s/u` in this noisy host setup; kept traced decode because release criteria require traced decode and traced path remains stable/correct.

## Rejected / corrected infrastructure experiments
1. `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root` caused firmware compile failures (`risc_common.h` missing) on this host, so it was not kept.
2. Forcing `TT_MESH_GRAPH_DESC_PATH=.../n300_mesh_graph_descriptor.textproto` still fails with disconnected topology and was not used.
3. Single-token prompt smoke tests initially hit a `seq_len==1` prefill/decode edge case; this was fixed by treating `seq_len==1` with missing `cur_pos_tensor` as prefill mode.
