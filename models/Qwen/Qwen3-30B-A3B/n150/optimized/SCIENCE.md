# SCIENCE.md — Qwen3-30B-A3B n150 optimized

## Objective
Maximize `demo.py` performance while keeping long-prompt eval quality at:
- Top-1 >= 0.90
- Top-5 >= 0.95

## Baseline
Functional baseline run:
- Command: `python -u demo.py models/Qwen/Qwen3-30B-A3B/n150/functional/model.py --seed 0 --max_seq_len 40960 --temperature 0 --output-format yt_metrics`
- Result:
  - TTFT: `77689.76 ms`
  - Decode: `0.4635 t/s/u`
  - Generated tokens: `128`

Functional long-eval quality reference (from functional bringup doc):
- Top-1: `1.0000` (40-token long eval)
- Top-5: `1.0000` (40-token long eval)

## Experiments
### E1: Initial optimized port
Hypothesis:
- Fusing attention QKV and removing MoE per-expert host->device transfers should give the largest speedup.

Changes:
- Added fused QKV weight (`self.qkv_proj`) in attention.
- Replaced device-side MoE expert streaming path with host-side torch expert execution over routed tokens.
- Kept routed-weight accumulation in float32 and returned BF16 to TT path.

Result:
- 16-token demo (`trial2`):
  - TTFT: `3082.28 ms`
  - Decode: `3.8790 t/s/u`

Decision:
- Keep as primary direction.

### E2: Aggressive host expert cache
Hypothesis:
- Caching float32 expert weights per layer would reduce repeated per-call setup overhead.

Changes:
- Added float32 host expert cache for gate/up/down weights.

Result:
- Process killed by OS (`exit 137`) during demo due host memory pressure.

Decision:
- Reject.

### E3: BF16 hidden-state micro-optimization
Hypothesis:
- Precomputing BF16 hidden states and avoiding per-expert casts would improve host MoE throughput.

Changes:
- Switched host MoE path to BF16 hidden-state routing/expert inputs.

Result:
- 16-token demo (`trial3`):
  - TTFT: `69007.45 ms`
  - Decode: `0.8170 t/s/u`
  - Severe regression from E1.

Decision:
- Revert.

### E4: Final full demo + long eval on reverted E1 path
Commands:
- Demo:
  - `python -u demo.py models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py --seed 0 --max_seq_len 40960 --temperature 0 --output-format yt_metrics`
- Eval:
  - `python -u eval.py models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py --model Qwen/Qwen3-30B-A3B --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 40 --max_seq_len 40960 --seed 0 --output-format yt_metrics`

Results:
- Demo:
  - TTFT: `2336.66 ms`
  - Decode: `4.0566 t/s/u`
- Eval:
  - Top-1: `0.9750`
  - Top-5: `1.0000`

Decision:
- Keep as current best.

### E5: Decode-specialized host MoE micro-optimization (rejected)
Hypothesis:
- Special-casing `seq_len==1` and prebinding expert weights would reduce Python overhead in decode.

Changes:
- Added prebound expert weight lists.
- Added a single-token MoE fast path.
- Switched routing to top-k on logits + softmax(top-k logits) when `norm_topk_prob=True`.

Results:
- 16-token demo baseline:
  - TTFT: `3355.85 ms`
  - Decode: `3.5532 t/s/u`
- 16-token demo with patch:
  - TTFT: `13677.77 ms`
  - Decode: `1.7319 t/s/u`

Decision:
- Revert (large regression).

### E6: Decode-only MoE top-k reduction to 4 (quality-fail)
Hypothesis:
- Reducing decode routed experts from 8 to 4 should reduce host MoE compute enough to materially increase decode throughput.

Changes:
- Added `decode_top_k` in `SparseMoE` and used it only when `seq_len == 1`.
- Set `decode_top_k = 4`.

Results:
- 16-token demo:
  - TTFT: `2248.09 ms`
  - Decode: `6.6002 t/s/u`
- Long eval (40-token):
  - Top-1: `0.8750`
  - Top-5: `1.0000`

Decision:
- Reject for target bar (Top-1 below `0.90`).

### E7: Decode-only MoE top-k reduction to 5 (kept)
Hypothesis:
- `decode_top_k = 5` may preserve enough quality while keeping most of the decode speedup.

Changes:
- Kept decode-only routed expert reduction and set `decode_top_k = 5`.

Results:
- 16-token demo:
  - TTFT: `2244.25 ms`
  - Decode: `5.0543 t/s/u`
- Long eval (40-token):
  - Top-1: `0.9250`
  - Top-5: `1.0000`
- Full demo (128-token comparable run):
  - TTFT: `2299.66 ms`
  - Decode: `4.8162 t/s/u`

Decision:
- Keep as current best (passes quality gate with clear throughput gain).

### E8: Decode adaptive top-5->4 by 5th-weight threshold (rejected)
Hypothesis:
- Keep decode quality close to `top_k=5` while recovering some `top_k=4` speed by dropping to 4 experts only when the 5th routed expert is very low-probability.

Changes:
- Added decode-only adaptive rule in `SparseMoE`:
  - Compute decode top-5 routing.
  - If the 5th softmax weight was below `0.015`, keep only top-4 for expert execution.

Results:
- 16-token demo:
  - TTFT: `2695.61 ms`
  - Decode: `4.8730 t/s/u`

Decision:
- Reject (no clear speed gain over the kept `decode_top_k=5` baseline on short demo, added complexity not justified).

### E9: Layer-aware decode top-k split (top-4 early layers, top-5 late layers) (rejected)
Hypothesis:
- Using decode `top_k=4` for earlier sparse layers and `top_k=5` for later sparse layers may keep quality while improving throughput vs uniform `top_k=5`.

Changes:
- Set decode routing to:
  - `top_k=4` for sparse layers with `layer_idx < num_hidden_layers // 2`
  - `top_k=5` for remaining sparse layers
- Prefill kept unchanged (`top_k=8`).

Results:
- 16-token demo:
  - TTFT: `2859.26 ms`
  - Decode: `5.1012 t/s/u`
- Long eval (40-token):
  - Top-1: `0.9250`
  - Top-5: `1.0000`
- Full demo run A:
  - TTFT: `17479.00 ms`
  - Decode: `5.4416 t/s/u`
- Full demo run B (repeat):
  - TTFT: `30254.98 ms`
  - Decode: `3.9121 t/s/u`

Decision:
- Reject and revert.
- Despite passing long-eval quality, full-demo performance was unstable across repeated runs and introduced severe TTFT regression risk.

### E10: Device-side greedy sampling (`next_token_device`) + decode LM-head token trim (rejected)
Hypothesis:
- Avoiding per-step host logits transfer in `demo.py` and trimming decode LM-head to one token should improve decode throughput.

Changes:
- Added `next_token_device` to enable demo greedy sampling on-device.
- Added decode fast path to slice hidden state to one token before LM head in decode.

Results:
- 16-token demo baseline before patch:
  - TTFT: `2612.58 ms`
  - Decode: `4.8037 t/s/u`
- 16-token demo with patch:
  - TTFT: `2457.40 ms`
  - Decode: `5.3328 t/s/u`
- Full demo run A:
  - TTFT: `5376.87 ms`
  - Decode: `4.2524 t/s/u`
- Full demo run B (repeat):
  - TTFT: `10062.86 ms`
  - Decode: `3.7707 t/s/u`

Decision:
- Reject and revert.
- Short-run gain did not hold on full 128-token demo; full-run throughput and TTFT regressed.

### E11: Decode-only SparseMoE host transfer reduction on top of E10 (rejected)
Hypothesis:
- For `seq_len==1`, slicing only the real token from TT to host and simplifying expert accumulation should reduce host overhead further.

Changes:
- Added a decode-only `SparseMoE` path that sliced `[token=0]` before `ttnn.to_torch`.
- Added a single-token routed-expert accumulation loop.

Results:
- 16-token demo:
  - TTFT: `3507.82 ms`
  - Decode: `4.7711 t/s/u`

Decision:
- Reject (regression versus the E10 short-run result and no gain versus kept baseline).

### E12: Remove `next_token_device`, keep decode LM-head trim path (rejected)
Hypothesis:
- Keep only decode LM-head trimming in `forward` while using prior host-side sampling path in demo.

Changes:
- Removed `next_token_device`.
- Kept decode-lm-head trim path for the `forward` decode call.

Results:
- 16-token demo:
  - TTFT: `6999.00 ms`
  - Decode: `3.4845 t/s/u`

Decision:
- Reject and revert.
- This variant was significantly slower than the existing kept path.

### E13: Decode token-position top-k schedule (top-5 warmup then top-4) (rejected)
Hypothesis:
- Keeping decode `top_k=5` only for the first few decode steps, then switching to `top_k=4`, could keep long-eval quality while recovering most of the `top_k=4` decode speedup.

Changes:
- Added an experiment-only decode schedule in `SparseMoE`:
  - `top_k=5` for the first 8 decode tokens.
  - `top_k=4` for remaining decode tokens.
- Evaluated with env knobs:
  - `QWEN3_DECODE_TOPK_LATE=4`
  - `QWEN3_DECODE_TOPK5_WARMUP_TOKENS=8`

Results:
- Long eval (40-token):
  - Top-1: `0.9250`
  - Top-5: `1.0000`
- Full demo run A:
  - TTFT: `24698.94 ms`
  - Decode: `4.0291 t/s/u`
- Full demo run B (repeat):
  - TTFT: `20155.67 ms`
  - Decode: `4.5771 t/s/u`

Decision:
- Reject and revert.
- Although quality passed, full-demo throughput did not beat the kept baseline and TTFT regressed severely across repeated runs.

### E14: Persistent decode token/position buffers + decode-specialized forward path (rejected)
Hypothesis:
- Reusing device-resident decode input/position buffers should reduce per-step host/device tensor allocation overhead and improve decode throughput.

Changes:
- Added persistent decode token and decode position buffers.
- Added decode path that updates those buffers via `ttnn.copy_host_to_device_tensor`.
- Added decode-specific forward branch using those buffers.

Results:
- 16-token demo:
  - TTFT: `2551.82 ms`
  - Decode: `5.3741 t/s/u`
- Full demo run A:
  - TTFT: `2729.77 ms`
  - Decode: `5.0318 t/s/u`
- Full demo run B (repeat):
  - TTFT: `3777.87 ms`
  - Decode: `4.6601 t/s/u`

Decision:
- Reject and revert.
- Despite one strong full run, repeated full-demo behavior was unstable and did not reliably beat the kept baseline.

### E15: Explicit host-tensor deallocation in decode buffer updates (rejected)
Hypothesis:
- Explicitly deallocating temporary host TT tensors (`host_tokens`, `host_pos`) after buffer copies might reduce memory churn and stabilize the E14 path.

Changes:
- Added `ttnn.deallocate()` calls for temporary host decode update tensors.

Results:
- 16-token demo:
  - TTFT: `3480.60 ms`
  - Decode: `4.0843 t/s/u`

Decision:
- Reject and revert.
- This was a clear regression versus both E14 and the kept baseline.

### E16: Prefill MoE top-k reduction sweep (`prefill_top_k`, decode fixed at 5) (rejected)
Hypothesis:
- Reducing prefill routed experts (while keeping decode `top_k=5`) could lower TTFT materially and possibly improve decode by changing routed-expert workload in cache-building.

Changes:
- Temporarily added experiment-only hooks for independent prefill/decode top-k routing in `SparseMoE`.
- Used an in-process sweep (single model load) to compare:
  - `(prefill=8, decode=5)`
  - `(prefill=7, decode=5)`
  - `(prefill=6, decode=5)`
  - `(prefill=8, decode=4)` as a reference point.

In-process sweep results (32-token generation, order-dependent warm-state run):
- `(8,5)`: TTFT `6987.07 ms`, Decode `4.1115 t/s/u`
- `(7,5)`: TTFT `2277.81 ms`, Decode `5.1098 t/s/u`
- `(6,5)`: TTFT `1662.91 ms`, Decode `5.6022 t/s/u`
- `(8,4)`: TTFT `2255.83 ms`, Decode `6.4763 t/s/u`

Official harness validation for `(prefill=6, decode=5)`:
- Full demo (128-token):
  - TTFT: `1991.34 ms`
  - Decode: `4.5283 t/s/u`
- Long eval (40-token):
  - Top-1: `0.8750`
  - Top-5: `1.0000`

Decision:
- Reject and revert.
- Although TTFT improved versus the kept baseline, long-eval Top-1 failed the target bar (`< 0.90`).

### E17: Less aggressive prefill reduction (`prefill_top_k=7`, decode fixed at 5) (rejected)
Hypothesis:
- `prefill_top_k=7` might retain quality while preserving some TTFT benefit from reduced prefill expert routing.

Changes:
- Reused the experiment-only prefill/decode top-k split and set `(prefill=7, decode=5)` for official demo validation.

Results:
- Full demo (128-token):
  - TTFT: `20477.87 ms`
  - Decode: `4.0517 t/s/u`

Decision:
- Reject and revert.
- This configuration regressed both TTFT and decode throughput badly on full demo, so quality eval was skipped.

## Knowledge (non-obvious)
1. For this MoE model, host->device expert weight transfer in the decode/prefill loop was the dominant bottleneck; removing it dwarfed typical TTNN kernel-side tuning.
2. Per-layer float32 expert caching is not viable at this model scale on this host due memory blowup and OOM kill risk.
3. Small-looking BF16-path changes in host expert execution can regress dramatically; BF16 hidden-state path was much slower than the float32 routing + per-expert BF16 cast path.
4. Long-sequence eval turnaround is heavily dominated by HF reference generation and optional model dtype conversion; this can exceed TT execution time.
5. For this model/harness, decode-only MoE top-k is a high-leverage quality/perf dial: `decode_top_k=4` is very fast but misses Top-1, while `decode_top_k=5` passes and still outperforms `decode_top_k=8`.
6. A short 16-token demo can overestimate gains: changes that look better on short decode may regress or destabilize full 128-token demo behavior, so full-run confirmation is required before keeping a patch.
7. On this model/harness, device-side greedy token selection was not a guaranteed win: removing host logits transfer looked good on short runs but regressed full-run TTFT and decode throughput.
8. After long experiment chains, run-to-run timing variance can grow substantially; repeated full-demo confirmation is required before accepting any optimization.
9. Decode token-position scheduling (`top_k=5` early then `top_k=4`) can preserve long-eval Top-1/Top-5 but still fail the real figure of merit if TTFT and full-run decode throughput regress.
10. In this environment, forcing `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root` caused firmware build failures (`risc_common.h` missing); default runtime root was required for successful runs.
11. For this model, persistent decode token/position buffers with per-step host->device copies did not produce a reproducible full-demo win; short-run gains could disappear or reverse on repeat full runs.
12. Under long experiment chains, late checkpoint shard load time (roughly shards 10-16) can stretch significantly and correlate with much worse end-to-end benchmark outcomes.
13. In-process multi-config sweeps are useful for direction-finding, but the first measured config can be heavily biased by residual warmup/compile state; only official harness reruns should be used for keep/reject.
14. Reducing prefill-only MoE routing to 6 experts can improve TTFT on full demo but still fail long-eval quality (`Top-1 = 0.875`), so prefill routing is a quality-critical dial, not just a latency dial.
15. For this model/harness, `prefill_top_k=7` showed catastrophic full-demo instability/regression (`TTFT ~20.5s`, decode ~`4.05`), so small prefill routing changes can have disproportionate runtime side effects.

## Current best figures of merit
- TTFT: `2299.66 ms` (baseline `77689.76 ms`, `97.04%` lower)
- Decode throughput: `4.8162 t/s/u` (baseline `0.4635 t/s/u`, `+939.09%`)
- Long eval quality: Top-1 `0.9250`, Top-5 `1.0000` (both above target)
