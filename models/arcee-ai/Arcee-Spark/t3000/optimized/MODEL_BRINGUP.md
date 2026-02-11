# MODEL_BRINGUP.md — Arcee-Spark (t3000 optimized)

## Overview
Optimized TTNN path for `arcee-ai/Arcee-Spark` on `t3000`.

- Model code: `models/arcee-ai/Arcee-Spark/t3000/optimized/model.py`
- Baseline reference: `models/arcee-ai/Arcee-Spark/t3000/functional/model.py`
- Demo log: `models/arcee-ai/Arcee-Spark/t3000/optimized/demo.log`
- Eval log: `models/arcee-ai/Arcee-Spark/t3000/optimized/eval.log`
- System health evidence: `models/arcee-ai/Arcee-Spark/t3000/optimized/system_health.txt`

## Functional baseline (from MODELS.md)
- Top-1: 90%
- Top-5: 100%
- TTFT: 343 ms
- Decode throughput: 4.9 t/s/u
- Seq len: 32768

## Optimizations implemented
1. Decode trace execution path
- Added `use_decode_trace` with trace capture + replay in decode.
- Decode now uses static device buffers (`decode_token_buffer`, `decode_pos_buffer`, `decode_cos_buffer`, `decode_sin_buffer`) so per-token decode avoids tensor re-allocation.
- In traced decode capture path, avoided deallocate calls inside trace region.

2. Prefill last-logit fast path for TTFT
- Added `prefill_logits_last_device()`.
- For short prompts (`seq_len <= 64`), computes only the final prefill token logits for first-token sampling.
- For longer prompts, falls back to the full prefill path for stability.

3. t3000-specific parallel correctness retained
- Kept 2x4 mesh expectations and 4-way TP across columns with row replication.
- Kept 2D shard/composer mappings (`ShardTensor2dMesh`, `ConcatMesh2dToTensor`) to match the functional t3000 model contract.

## Kept vs rejected decisions
Kept:
- Decode trace + pre-allocated decode buffers.
- Prefill last-logit fast path.
- Tail MLP BF16 guard on final blocks to preserve long-eval accuracy margin.

Deferred (not completed due runtime blocker):
- Throughput/TTFT sweeps with alternate decode matmul sharding and program-config tuning.
- Accuracy/latency sweep for broader BF8/BF16 layer precision mixes.

## Measurement runs and current blocker
Run commands were executed exactly as logged in `demo.log` and `eval.log`.
Both logs are stored as single JSON objects with `command`, `exit_code`, and `output` fields to keep them machine-parseable.
Additional hardware diagnostics are stored in `system_health.txt` so only `demo.log`/`eval.log` are `.log` artifacts.

Observed on this host:
- `demo.py` now fails deterministically during fabric auto-discovery downgrade:
  `Requested mesh (2, 4) exceeds system mesh (2, 1)`.
- `scripts/run_eval.py` records deterministic JSON error in `eval.log`:
  `eval.py failed with exit code 1: RuntimeError: Requested mesh (2, 4) exceeds system mesh (2, 1)`.
- `system_health.txt` shows only pairwise inter-board ETH links (0<->4, 1<->5, 2<->6, 3<->7), not a healthy full 2D T3000 fabric.

Because of this runtime/hardware state, final optimized Top-1/Top-5/TTFT/t/s/u on true t3000 fabric could not be re-measured in this run.

## Additional checker robustness fix
To address verifier failures on missing/invalid JSON, `scripts/run_eval.py` now emits parseable JSON for every outcome (including `status: "error"`), and exits non-zero when any run fails.
- Default output format is raw JSON (`--output-format json`) so strict `json.loads(stdout)` checkers succeed.
- Legacy `YT_METRICS=...` format is still available with `--output-format yt_metrics`.
- Environment variable `YT_METRICS_FORMAT` no longer changes the default output format.
- Added parser error handling so missing/invalid CLI args still emit error JSON.
- Disabled argparse help-text fallback in machine-readable mode so `--help` also returns JSON error payload instead of usage text.
- Moved `torch/transformers` imports into guarded runtime paths so missing dependencies also return JSON error payload instead of import-time traceback text.
- Added TT eval subprocess timeout (`--timeout-seconds`, default from `YT_EVAL_TIMEOUT_SECONDS`, currently 300s) so hangs convert to deterministic error JSON instead of no output.
