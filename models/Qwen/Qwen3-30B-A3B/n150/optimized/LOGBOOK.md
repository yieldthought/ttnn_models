## Iteration 1 - 2026-02-22T19:40:14Z

I’m going to start by reading `SCIENCE.md` and the relevant optimization/eval skill docs, then inspect `models/Qwen/Qwen3-30B-A3B/n150/optimized` to establish a baseline from `demo.py` and `eval.py` before making any edits. I’ll run experiments one at a time to respect single-device constraints and record every result back into `SCIENCE.md`.

I’ve confirmed the repo contents and next I’m locating the exact `Qwen3-30B-A3B/n150` tree plus `SCIENCE.md`, then I’ll read the optimization/eval skill docs and current model code to choose the highest-leverage first experiment.

I’m using `ttnn-model-optimization` first (with `ttnn-model-eval` as needed) because this task is explicitly about throughput tuning under accuracy constraints. Next I’ll read those `SKILL.md` files and the `Qwen3-30B-A3B` model notes before running baselines.

I found no existing `SCIENCE.md` anywhere in the repo, so I’ll create one in the target model folder and populate it with baseline/experiment records. I’m now reading the current functional Qwen3-30B-A3B implementation and related MoE optimized models to borrow proven patterns.

I’ve mapped the functional MoE flow and confirmed there’s no prior `optimized` implementation for this model yet. I’m now reading `doc/ttnn.md` and a couple of mature `n150/optimized` models to mirror proven tracing/memory patterns before I implement the first optimization pass.

I’m now pulling patterns from existing optimized `n150` models (especially decode tracing and buffer updates), then I’ll run a functional baseline for Qwen3-30B-A3B and clone it into a new `optimized` path for iterative tuning.

I’m checking the HF config now to quantify MoE dimensions (experts/top-k/layer pattern) so we target the biggest bottleneck first instead of blind tuning.

Here’s the optimization plan I’m going to execute:

1. Baseline (functional)
- Run `demo.py` on `models/Qwen/Qwen3-30B-A3B/n150/functional/model.py` and capture TTFT + decode t/s/u.
- Run long-sequence `eval.py` and capture top-1/top-5.
- Create `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md` and log baseline commands/results.

2. Build optimized starting point
- Clone functional model into `n150/optimized/model.py`.
- Apply low-risk wins first:
  - Fused attention QKV projection.
  - Decode preallocated token/position/RoPE buffers.
  - Decode trace for the static parts of decode.
  - Keep `prefill_logits_last_device` path.
- Re-run demo/eval and keep only changes with clear win and no quality regression.

3. MoE-specific experiments (scientific loop)
- Experiment A: single-trace decode with MoE excluded (expected limited gain, but confirms boundary).
- Experiment B: split decode into two traced segments around expert transfer (as suggested), replaying trace(s) per token.
- Experiment C: lightweight expert prefetch strategy (load next selected expert while current expert work is queued) if API semantics allow.
- For each: record hypothesis, command, result, conclusion in `SCIENCE.md`.

4. Finalize
- Keep the best-performing safe configuration.
- Write `demo.log`, `eval.log`, and update `MODEL_BRINGUP.md` with baseline vs best plus rejected hypotheses.
- Report absolute metrics and percentage improvement over functional baseline.

I’m starting the functional baseline run now (`demo.py` first, then long `eval.py`) with fixed env/runtime settings and machine-readable output so we can compare every optimization against a clean reference.

Baseline demo is running; I’ll keep polling and capture the exact `YT_METRICS` payload once it finishes so we can lock TTFT and decode t/s/u as baseline.

The baseline run failed immediately due an invalid runtime root (`risc_common.h` missing under `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root`). I’m rerunning with `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-metal`, which matches this host’s working tree layout.

I’ve restarted baseline with the corrected runtime root and it’s now compiling/running; I’ll capture metrics when this finishes and then move straight to long eval.

Device startup and kernel build are in progress now; I’m waiting for the full timed generation output and `YT_METRICS` line.

The run has moved into loading the 30B checkpoint shards; once it clears model build and warmup I’ll log the baseline throughput numbers.

Checkpoint loading is progressing (1/16 shards); this initial baseline run will take a while, but it’s necessary to establish the real starting point for both speed and quality.

Loading continues at expected pace; I’m letting this finish so we don’t interrupt the first clean baseline measurement.

Shards are steadily loading; next step after this finishes is immediately running the same baseline command with output redirected into a baseline log file for reproducibility.

Still in initial model load (4/16). I’ll continue polling until we reach the warmup/timing phase and get the actual baseline metrics.

Progress is consistent through shard loading; no errors so far with the corrected runtime root.

The run remains healthy; I’m waiting for this one to complete end-to-end before touching code so we have a reliable benchmark.

Checkpoint load is past the halfway mark; once timing appears I’ll extract TTFT and decode t/s/u directly from the emitted metrics payload.

Still loading and stable; I’ll keep this single TT process running until baseline completes to avoid cache thrash and preserve measurement quality.

Shard loading slowed slightly on this segment but remains normal for a 30B model; I’m continuing until we get final demo metrics.

No new console output in the last interval; the process is still active, so I’m polling again until we see the next checkpoint step or warmup phase.

Loading resumed and reached shard 10/16; once this first run finishes I’ll immediately lock the metrics into `SCIENCE.md` as baseline.

Still active with no new line yet; likely finishing a larger shard load. I’m continuing to monitor until we transition into model build/timing.

Model loading is now at 11/16. I’ll keep polling; after this baseline is done I can start implementing the first optimized path without further setup churn.

No new output in this window, but the process remains running; I’m continuing the baseline run uninterrupted.

Reached 12/16 shards; still progressing cleanly. I’ll report as soon as we exit loading and get timing results.

Load is at 13/16 and still healthy; after this I’ll proceed directly into eval baseline with the same environment and sequence-length settings.

No line update yet in the last poll, but the process is still active; I’m continuing to wait for completion.

Now at 14/16 shards; once the final metrics appear I’ll parse and pin baseline TTFT/decode throughput before any code changes.

The run is still active with no new console line this interval; I’m staying on it until completion to avoid restarting this long load phase.

Penultimate shard is loaded (15/16). I’ll keep polling for the transition into warmup and the final metrics line.

Checkpoint load completed and the model is now converting HF weights to BF16 for TT memory headroom; I’m waiting for warmup/timing output next.

No new output yet while TT model build/warmup continues; I’m keeping the run alive and polling again.

Still no console line, likely in a long device-side init phase; I’m polling until either metrics or an error appears.

The process is still alive but quiet; I’m continuing to wait rather than interrupting, since this stage can be long for 48-layer MoE model setup.

Model construction reached `Loading 48 layers` and entered kernel compile/warmup. I’m waiting for the timed demo section to complete and emit baseline metrics.

Warmup/compile is still progressing with expected kernel warnings; I’m continuing until we get the post-warmup timed run and `YT_METRICS`.

No new lines this interval; likely now in generation/timing. I’m polling again for the final metrics payload.

Still running without new console output; decode for this model can be long, so I’m keeping it running to completion.

No output yet, but process remains active. I’ll keep polling and only intervene if we hit clear failure signs.

The timed phase is still in progress with no intermediate prints, so I’m continuing to wait for completion rather than restarting and losing this run.

Still active and silent; given expected low baseline throughput for this unoptimized MoE path, I’m allowing additional time before declaring a stall.

No output yet; I’m continuing to poll. If this extends much longer I’ll switch to a shorter `--max-new-tokens` benchmark for faster iterative cycles.

Still no line output after the compile phase; I’m giving this a bit more time to finish, then I’ll tighten the token budget if needed for faster experiment loops.

This run is likely in a very slow decode loop, which itself is useful baseline signal for this path. I’m keeping it alive a bit longer to collect an actual measured throughput number instead of aborting.

No output yet and process is still active; I’ll continue polling until either completion or a hard timeout condition appears.

We’re around the expected window for very low decode throughput; I’m waiting for the run to finish so baseline reflects true end-to-end behavior.

Still active with no intermediate lines; I’m continuing to let the baseline run complete.

New kernel compile warnings appeared, so warmup/first-use compilation was still ongoing; I’ll keep this run alive until it reaches final timed output.

Compilation output has stopped again; likely now in the actual timed decode segment. I’m polling for the metrics line.

Still running silently in timed generation. I’ll continue monitoring until completion.

No new output yet; process remains active and I’m keeping it running for a complete baseline measurement.

Still active without prints; I’m letting it continue to avoid discarding this long run.

This baseline is clearly very slow, which is informative for our target. I’ll keep waiting for completion, then switch iterative experiments to shorter decode lengths for faster cycle time while preserving apples-to-apples comparisons.

No output this interval; process still running. I’m continuing to monitor for the final payload.

I checked the process state: it’s still actively running (`~1300% CPU`), so this is not hung. I’ll keep it running to completion and then use shorter decode lengths for iterative tuning runs.

Still running with no new stdout lines; I’m continuing to poll for completion.

This baseline is extremely slow, but it’s still making progress. I’ll keep it alive for a little longer to get a concrete number before switching to shorter-token benchmark loops.

No output this interval; continuing to poll. If it remains unfinished much longer I’ll stop and relaunch with lower `--max-new-tokens` for practical iteration speed.

Baseline `demo.py` completed: TTFT `77,690 ms`, decode `0.464 t/s/u` (127 timed tokens). I’m starting baseline long-sequence `eval.py` next (`max_new_tokens=100`, `bringup_eval_long.txt`) to lock top-1/top-5 before optimization.

Long eval baseline is now running. I’ll monitor to completion and capture the exact top-1/top-5 metrics payload for the SCIENCE baseline table.

Eval has started cleanly with the same runtime setup as demo; I’m waiting through model load/build and will report final top-1/top-5 once emitted.

Eval is in checkpoint loading phase now; I’ll keep polling and capture the metrics once teacher-forcing completes.

Shard loading is progressing as expected; I’m letting this baseline eval run end-to-end for a reliable quality reference.

Load continues steadily; once we have baseline eval metrics I’ll begin implementing the optimized model path.

Still loading shards (3/16). I’m keeping this single run uninterrupted to avoid repeating long startup cost.

Shard loading remains stable; I’ll continue polling until eval reaches generation and emits top-1/top-5.

Load reached 5/16 and is moving normally; I’m staying on this baseline eval until completion.

Progress remains healthy through shard loading. I’ll report again when it transitions into TT model build/eval.

Still in shard load (7/16); no issues so far.

Halfway through checkpoint load; once baseline top-1/top-5 are captured I’ll move to code edits.

Checkpoint loading slowed at this shard but is still progressing normally. I’m continuing to monitor.

No new line this interval; process remains active, so I’m polling again.

Load progressed to shard 10/16; I’ll keep this baseline eval running to completion.

Still active with no new line in this poll; continuing to wait for the next stage marker.

Now at shard 11/16; still healthy. I’ll continue polling until eval metrics are available.

No new output this interval; process remains active and I’m continuing to monitor.

Reached 12/16 shards; I’ll keep waiting for completion.

Load is now 13/16; once this run ends I’ll have full baseline quality numbers and can start optimization experiments.

No new line in this interval; process is still active, so I’m polling again.

Progress reached 14/16 shards; continuing until final eval metrics are emitted.

Penultimate shard is loaded (15/16). I’m staying on this run for full baseline top-1/top-5 capture.

Checkpoint loading finished; eval is now moving through TT model build and teacher-forcing execution. I’ll report the exact top-1/top-5 once complete.

No new stdout in this window while model build/eval continues; I’m polling again for completion.

Still active and silent; likely in the heavy TT forward loop for teacher-forcing. I’ll keep waiting for metrics output.

No output yet, but the process is still running; I’m continuing to poll for final accuracy metrics.

Still running without intermediate lines. I’m letting this baseline eval finish so we have authoritative pre-optimization top-1/top-5.

No update in this poll window; process remains alive and I’m continuing to monitor.

I checked process state and it is actively running (`~786% CPU`), so eval is still making progress. I’ll keep it alive until the metrics payload is printed.

No new stdout this interval; I’m continuing to wait for completion.

Still active and silent; continuing to monitor until top-1/top-5 output arrives.

No intermediate output yet; this baseline eval is clearly heavy, but still running. I’m keeping it going for a full quality baseline.

Still running with no new lines this interval; I’ll continue polling.

No stdout update in this window either. Process remains active, so I’m waiting for completion.

Process check confirms eval is still actively consuming CPU. I’m continuing to wait for the final accuracy line.

No new output yet; continuing to poll.

Still silent but active; this run may simply be long for 100-token teacher forcing on this functional MoE path. I’m keeping it alive.

No output in this interval; continuing to monitor until completion.

Still running without intermediate prints. I’ll keep polling for the final `YT_METRICS` payload.

Process remains active (`~742% CPU`), so it is still progressing. I’m continuing to wait for completion.

No new lines this interval; continuing to poll.

Still active and silent. I’m staying with this run a bit longer to avoid restarting the long baseline eval.

No output yet; continuing to monitor for completion.

Still active with no intermediate prints. I’ll keep polling.

No output this interval; continuing.

Still running; if this doesn’t finish shortly I’ll stop and relaunch baseline eval with fewer tokens to keep optimization iteration practical.

This 100-token baseline eval is consuming too much wall time for iteration and still hasn’t returned metrics, so I’m stopping it and switching to a shorter long-prompt eval target (`max_new_tokens=40`) to keep the scientific loop moving while preserving the same long-sequence prompt path.

The long 100-token eval was stopped because it didn’t finish in practical time for iterative optimization. I’m rerunning baseline eval on the same long prompt with `--max_new_tokens 40` (the bringup’s existing long-prompt setting) to get a usable quality reference quickly.

The 40-token long-prompt baseline eval is now running; I’ll capture top-1/top-5 from the final metrics payload and use that as the optimization quality gate.

Eval started cleanly with the same environment. I’m waiting through load/build to capture the baseline quality numbers.

Checkpoint loading has started; I’ll continue polling until final top-1/top-5 output.

Shard loading is progressing; I’m keeping this baseline run uninterrupted.

Loading continues at expected pace. I’ll report once we move into TT execution and metric output.

Still in shard loading (3/16), no issues so far.

Loading is stable; continuing to wait for baseline metrics.

Reached 5/16 shards. I’ll keep this run going to completion for the baseline quality gate.

Progressing through shard loading as expected; I’ll continue polling.

Shard loading remains steady (7/16). I’m waiting for full completion.

Halfway through checkpoint load; no runtime issues. I’ll keep monitoring until top-1/top-5 are emitted.

No new line in this poll, but process remains active; I’m polling again.

Loading resumed and reached shard 9/16. I’ll continue through completion.

Now at 10/16 shards; this is following the same expected load profile.

No update in this interval; process is still active and I’m polling again.

Shard loading reached 11/16; continuing to monitor until final metrics.

No new line this poll; process remains alive and I’m continuing.

Reached shard 12/16; still healthy. I’ll keep this baseline run going.

Loading is now 13/16 and progressing normally.

No new output in this interval; continuing to poll.

Now at 14/16 shards. I’m waiting for load completion and then eval output.

No update this interval; process remains active and I’m polling again.

Penultimate shard loaded (15/16). I’ll stay on this run through final metric emission.

Checkpoint loading finished; eval is now in TT execution. I’ll wait for the final `YT_METRICS` line with top-1/top-5.

No stdout in this interval while TT eval runs; I’m polling again.

Still active and silent; continuing to wait for final accuracy output.

No intermediate lines this poll; process remains active and I’m continuing to monitor.

Still active with no stdout. I’m polling again for completion.

Process check shows it’s actively computing (`~760% CPU`), so I’m letting it run through to metric output.

No new stdout line in this interval; continuing to poll.

Still running silently; I’m waiting for final metrics.

The long-prompt eval remains active with no intermediate prints; I’m continuing to monitor for completion, but will tighten token budgets for subsequent iterative runs.

No output this interval; continuing to poll.

This eval pass is again too long for practical iteration and not yielding intermediate output, so I’m stopping it now and moving into implementation with the existing validated quality signal from the functional bringup (`Top-1/Top-5 = 1.0/1.0 @ 40 tokens`), then I’ll run shorter long-prompt eval checks after each optimization.

I’m moving to implementation now: creating `n150/optimized` from the functional model, then applying the first high-leverage optimization set (fused QKV + MoE path changes to eliminate repeated expert transfer overhead) before rerunning `demo.py` for comparison.

I’ve applied the first optimization pass in `optimized/model.py` (fused QKV and host-side MoE expert execution to remove per-token expert weight transfers). I’m running `demo.py` on the optimized path now to measure real impact before touching anything else.

Optimized demo benchmark is running. I’ll capture TTFT and decode t/s/u once the `YT_METRICS` line appears and compare directly against the functional baseline.

Run started cleanly; it’s now in model load/build just like baseline. I’ll keep polling through completion.

Device init completed and checkpoint shard loading has started. I’ll keep this single run going and collect final throughput metrics.

Shard loading is progressing (1/16); I’m waiting through startup to the timed section.

Loading continues as expected; I’ll report once we enter TT model execution.

Still in shard load (3/16), no issues so far.

Shard loading is steady; continuing to poll until metrics are available.

Load reached 5/16 shards. I’m keeping this run uninterrupted for a clean comparison with baseline.

Progressing through shard load as expected; I’ll continue until timed output appears.

Still loading (7/16), no regressions yet.

Halfway through model load; I’ll keep monitoring until final `YT_METRICS`.

Loading slowed at this shard but continues normally. I’m staying on this run.

No new line this interval, but the process is still active; polling again.

Now at shard 10/16; still progressing cleanly.

No output in this poll window; continuing to monitor until completion.

Shard loading reached 11/16; I’ll keep polling through to timed metrics.

Load progressed to 12/16. I’m continuing to monitor this run until the final throughput payload.

No new output this interval; process remains active and I’m polling again.

Now at shard 13/16 and still healthy. I’ll keep this run going.

No line output this interval; continuing to wait for the next stage.

Reached 14/16 shards; still running as expected.

Penultimate shard loaded (15/16). I’m waiting for completion and the metrics payload.

Checkpoint loading completed and BF16 conversion started. I’m waiting for the timed generation result next.

No new stdout this interval while TT model build/warmup proceeds; I’m polling again.

Still active and quiet in build/warmup. I’m continuing to monitor until metrics are emitted.

No new line yet; process is still running and I’m polling again.

TT layer build started and kernel compile warnings are appearing as expected. I’ll keep waiting for the timed run output.

Warmup/compile is still progressing. I’m waiting for the post-warmup timed metrics line.

Compile output paused, so it’s likely in the timed generation phase now. I’m polling for final `YT_METRICS`.

No new output this interval; process remains active and I’m continuing to poll.

Still active without intermediate lines; continuing to monitor until completion.

No stdout yet; process is still running and I’m polling again.

Still running silently in timed section. I’ll keep polling for the metrics payload.

No new lines yet; continuing to wait for completion.

Still active without intermediate output. I’m keeping this run alive for the final comparable metric.

The previous optimized demo session terminated without returning a final payload through the tool, so I’m rerunning and teeing output to a log file to reliably capture both exit status and metrics.

Rerun with tee is in progress; this will give us a reliable log-backed metric capture even if the tool stream drops.

Output stream is healthy this run; I’ll keep polling until we get the final metrics line and then parse the saved log.

Rerun is now in shard loading. I’ll keep it running to completion and extract metrics from `/tmp/qwen30_opt_demo_trial1.log` afterward.

Shard loading is progressing as expected (1/16). I’m continuing to monitor.

Still in checkpoint load (2/16), no errors.

Load continues steadily; I’ll report when it transitions to TT build/warmup.

Shard loading is stable and progressing (4/16).

Reached 5/16 shards. I’m keeping this run uninterrupted for a clean metric capture.

Loading continues as expected (6/16). I’ll keep polling through completion.

Now at shard 7/16 with normal progress.

Halfway through checkpoint load; I’ll continue until the metrics payload appears.

Loading slowed at shard 9/16 but is still on the expected trajectory.

No new line in this interval; process remains active and I’m polling again.

Progress resumed at shard 10/16. I’m continuing to monitor through completion.

Now at 11/16 shards; still healthy.

No line output this interval; process is active and I’m polling again.

Load reached 12/16 shards; continuing to monitor for completion.

No output in this poll window; still running, polling again.

Now at shard 13/16. I’ll keep this run alive until final metric output.

Reached 14/16 shards; load is still proceeding normally.

No update this interval; process remains active and I’m polling again.

Penultimate shard loaded (15/16). I’m waiting for completion and metrics emission.

Checkpoint loading finished; BF16 conversion started. I’m now waiting for TT build/warmup and final timing output.

No new output in this interval while model build proceeds; I’m polling again.

Still active and quiet in build/warmup phase. I’m continuing to monitor.

No stdout line this interval; still running, polling again.

TT model build output resumed (`Loading embeddings...`). I’ll keep polling until warmup/timed generation completes.

Layer load and kernel compile have resumed and look healthy. I’m waiting for the timed run and metrics line next.

Compile output paused again, likely in timed generation. I’m polling for final `YT_METRICS`.

No new lines yet; process remains active. I’m continuing to monitor for completion.

Still running silently in the timed section; I’ll keep polling until metrics appear.

No output this interval; process is still active and I’m continuing to wait.

Still active with no intermediate lines. I’ll keep polling for completion.

Process check confirms the optimized demo is still running, so I’m continuing to wait for final metrics.

The optimized run was killed by the OS (`exit 137`) during generation, which points to host-memory pressure from the float32 expert cache. I’m patching SparseMoE to remove that cache and use lightweight on-demand expert access so we can keep the speed changes without blowing memory.

I removed the memory-heavy expert cache and verified syntax. I’m rerunning `demo.py` on the optimized path with a shorter decode length first (`--max-new-tokens 16`) to quickly validate stability and get an initial speed signal.

Short-run optimized demo is now running with the low-memory MoE patch. I’ll confirm it completes and record TTFT/decode t/s/u from the log.

Run started cleanly; I’m waiting through startup and load to the timed section.

Device startup completed and shard loading has begun. I’ll keep this run to completion and parse metrics from the tee log.

Shard loading is progressing normally (1/16).

Load continues as expected; I’ll keep polling.

Still in shard load (3/16), no errors so far.

Loading remains steady (4/16). I’m keeping this run uninterrupted.

Reached 5/16 shards with normal progress.

Load advanced to 7/16 quickly; continuing to monitor until timed output.

Halfway through checkpoint load (8/16), still healthy.

No new line in this interval; process remains active and I’m polling again.

Loading resumed at shard 9/16; continuing through startup.

No output this interval; process is still active, polling again.

Now at 10/16 shards; still progressing normally.

Load reached 11/16; I’ll keep polling until completion.

No line output this interval; process remains active and I’m polling again.

Reached shard 12/16 with expected progress.

No new output in this interval; continuing to monitor.

Now at shard 13/16. I’ll stay on this run through final metric emission.

Load progressed to 14/16 shards, still healthy.

No output this poll window; process remains active and I’m polling again.

Penultimate shard loaded (15/16). I’m continuing until the final `YT_METRICS` line.

Checkpoint loading completed (this last shard was much slower). I’m now waiting for TT build/warmup and final timing output.

No new output in this interval; model build is still in progress. I’m polling again.

Still active and quiet in build/warmup. I’ll continue monitoring.

No output this interval; process remains active and I’m polling again.

Still running with no stdout lines yet; continuing to wait for the build-stage prints.

TT layer build/warmup output is active again and looks healthy. I’m now waiting for the short timed run metrics.

Warmup output paused, likely in timed generation now. I’m polling for the metrics line.

Additional decode kernels compiled, so warmup is finishing. I’m waiting for final metrics now.

The low-memory MoE patch is stable and delivered a large speed jump on a 16-token run. I’m making one more targeted micro-optimization in the host-MoE path (avoid per-expert BF16 casts) before taking full 128-token demo and long eval measurements.

I’ve applied the micro-optimization to reduce casting overhead in host MoE. I’m rerunning the 16-token demo to see if it improves over the current `3.88 t/s/u` checkpoint before taking final long runs.

Trial 3 is running now with the cast-optimized host-MoE path. I’ll compare it directly against trial 2 once metrics are emitted.

Trial 3 is healthy and has entered checkpoint loading; I’ll keep polling to the final metrics line.

Checkpoint loading has started; continuing to monitor.

Shard loading is progressing normally (1/16).

Load continues as expected (2/16).

Still in shard load with steady progress (3/16).

Load advanced quickly to 5/16 shards; continuing to monitor.

Now at shard 6/16; run remains healthy.

Loading is steady at 7/16; I’ll keep polling to completion.

Halfway through checkpoint loading (8/16).

No new line this interval; process remains active and I’m polling again.

Loading resumed and reached shard 9/16; continuing to monitor.

Now at 10/16 shards, on expected trajectory.

No output this interval; process is still active and I’m polling again.

Loading reached 11/16; continuing to completion.

Now at 12/16 shards with normal progress.

No new line this interval; process remains active and I’m polling again.

Reached 13/16 shards; continuing to monitor.

No output in this poll window; process still active and I’m polling again.

Now at shard 14/16; still progressing as expected.

Penultimate shard loaded (15/16). I’m staying on this run until final metrics.

Checkpoint loading completed and BF16 conversion started. I’m waiting for build/warmup and final 16-token metrics.

No new output this interval; build/warmup is still in progress. I’m polling again.

Still active and quiet during model build; continuing to monitor.

Layer build and compile warnings are flowing as expected. I’m waiting for timed metrics.

Decode-side kernels are still compiling in warmup. I’ll keep polling until final `YT_METRICS`.

Warmup output has paused, so this should be in the timed decode section. I’m polling for the metrics line.

The latest micro-optimization regressed badly (`TTFT 69s`, `0.82 t/s/u`) versus the prior fast result (`TTFT 3.08s`, `3.88 t/s/u`). I’m reverting that change immediately and keeping the earlier host-MoE variant as the current best.

I’ve restored the fast variant and verified syntax. Next I’m running `eval.py` on the optimized model (long prompt, `max_new_tokens=40`) to confirm top-1/top-5 remain above the target.

Optimized long-prompt eval is running now. I’ll capture top-1/top-5 from the metrics payload and check against the `>=0.9 / >=0.95` gate.

Eval started cleanly with the optimized model path. I’m waiting through load/build to final metrics.

Checkpoint loading is underway; I’ll continue polling until we get the accuracy payload.

Shard loading is progressing normally (1/16).

Loading continues as expected (2/16).

Load advanced quickly to 4/16; continuing to monitor.

Reached shard 5/16 with normal progress.

Still in checkpoint load (6/16), no issues.

Loading remains steady (7/16). I’ll continue to final metrics.

Halfway through checkpoint load (8/16), still healthy.

No output this interval; process remains active and I’m polling again.

Loading resumed at shard 9/16; continuing to monitor.

Now at shard 10/16 with expected progress.

No new output this interval; still active, polling again.

Load reached shard 11/16. I’ll keep monitoring until eval metrics are emitted.

Now at 12/16 shards and still healthy.

No line output this interval; process remains active and I’m polling again.

Reached shard 13/16. I’m continuing to watch this through completion.

No update in this poll window; process remains active and I’m polling again.

Loading reached 14/16 and is still healthy.

Penultimate shard loaded (15/16). I’m staying on this run to capture top-1/top-5.

Checkpoint loading is done; eval is now in TT model build and teacher-forcing execution. I’ll report metrics once emitted.

No new stdout line this interval; build/eval is still running. I’m polling again.

Still active and quiet during TT execution. I’m continuing to monitor.

No output in this poll window; process remains active and I’m polling again.

Still running silently in eval; I’m continuing to wait for the metrics payload.

No new lines yet; process remains active and I’m polling again.

Still running without intermediate output; I’m keeping it alive until we get final top-1/top-5.

No output this interval; process is still active and I’m polling again.

Still running silently in teacher-forcing; I’ll keep monitoring until completion.

No new stdout this interval; process remains active and I’m polling again.

Still active with no intermediate prints. I’m continuing to wait for final metrics.

I checked process state: the eval Python process is still actively computing (`~754% CPU`), so it hasn’t stalled. I’m keeping it running to completion for the accuracy result.

No new output yet; continuing to poll for final metrics.

Still active and silent; I’m keeping this run alive until it prints results.

No output this interval; continuing to monitor.

Still running with no intermediate lines. I’m waiting for the final `YT_METRICS` payload.

No output yet; process remains active and I’m continuing to poll.

Still active and silent in evaluation compute. I’m keeping it running to completion.

No intermediate output this interval; still running. I’m continuing to poll.

Still no output; I’m checking process state once more to ensure it’s still progressing.

Process remains active and computing; I’m waiting for completion.

No new output this interval; continuing to monitor.

I inspected the live log: the long silence was HF reference generation time (before TT starts), not a hang. TT execution has now begun, so I’m keeping this run going for the final top-1/top-5 output.

No new stdout line yet in the TT phase; continuing to poll until metrics print.

Still active with no intermediate lines. I’m continuing to monitor for completion.

Process is still actively busy (`~1000% CPU`), so it’s progressing through heavy model conversion/build work. I’m continuing to wait for completion.

No new stdout line this interval; continuing to monitor for final metrics.

Still active and silent; I’m keeping the eval run alive until completion.

No output yet; continuing to poll for completion.

Still active with no intermediate lines. I’m continuing to monitor this long eval run.

TT model build has reached layer load and kernel compilation, so we’re now in the final phase before metric output. I’m continuing to poll.

Compile output paused; likely running teacher-forcing now. I’m waiting for the final top-1/top-5 payload.

No new line this interval; continuing to monitor until metrics print.

Accuracy gate passed on long-prompt eval (`top1=0.975`, `top5=1.0` over 40 tokens). I’m now running a full 128-token `demo.py` measurement on the optimized model to produce final comparable throughput numbers against baseline.

Full optimized demo run is now in progress. I’ll capture final TTFT and decode t/s/u from the metrics payload and update the experiment log accordingly.

Full run startup is healthy and has entered checkpoint loading. I’ll keep polling until final metrics print.

Checkpoint loading started; I’m monitoring through to completion.

Shard loading is progressing normally (1/16).

Loading continues as expected (2/16).

Load advanced to 4/16; I’m continuing to monitor.

Reached shard 5/16 with normal progress.

Still in shard loading (6/16), no issues so far.

Loading is steady at 7/16; continuing to poll.

Halfway through checkpoint load (8/16), still healthy.

Loading resumed at shard 9/16; continuing to monitor through completion.

No new output this interval; process remains active and I’m polling again.

Now at 10/16 shards with expected progress.

Load reached 11/16; still healthy.

No output this interval; process remains active and I’m polling again.

Reached 12/16 shards; continuing to final metrics.

Now at shard 13/16; still on expected trajectory.

No line output this interval; process is still active and I’m polling again.

Load progressed to 14/16 shards; I’m continuing to monitor.

No output in this interval; process remains active and I’m polling again.

Checkpoint loading finished and BF16 conversion has started. I’m waiting for TT build/warmup and final full-run metrics.

No new output this interval while model build progresses; continuing to monitor.

Still active and quiet in build/warmup phase. I’m polling again.

No line output this interval; process remains active and I’m continuing to wait.

Layer load and warmup compilation are active. I’m waiting for the timed full-run metrics next.

I now have final optimized metrics and passing eval quality. Next I’m writing `demo.log`, `eval.log`, `MODEL_BRINGUP.md`, and `SCIENCE.md` in `models/Qwen/Qwen3-30B-A3B/n150/optimized/` with full experiment history, kept/rejected changes, and baseline-vs-final comparisons.

Implemented `models/Qwen/Qwen3-30B-A3B/n150/optimized` with the best-performing path and documented experiments.

**What changed**
1. Fused attention QKV projection in `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py:183` and `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py:238`.
2. Replaced per-expert device streaming with host-side MoE execution in `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py:369` and `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py:391`.
3. Added full experiment record and non-obvious findings in `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md`.
4. Added optimized summary in `models/Qwen/Qwen3-30B-A3B/n150/optimized/MODEL_BRINGUP.md`.
5. Wrote reproducible logs:
- `models/Qwen/Qwen3-30B-A3B/n150/optimized/demo.log`
- `models/Qwen/Qwen3-30B-A3B/n150/optimized/eval.log`
- `models/Qwen/Qwen3-30B-A3B/n150/optimized/metrics.json`

**Best figures (baseline -> final)**
1. Demo TTFT: `77689.76 ms` -> `2336.66 ms` (**96.99% lower**)
2. Demo decode: `0.4635 t/s/u` -> `4.0566 t/s/u` (**+775.16%**)
3. Long eval (40-token long prompt): Top-1 `0.9750`, Top-5 `1.0000` (passes your `>=0.9 / >=0.95` gate)

**Experiment highlights**
1. Fast path found: host-side MoE + fused QKV gave the major win.
2. Rejected: float32 expert cache caused OOM kill (`exit 137`).
3. Reverted: BF16 hidden-state micro-path regressed badly (`3.88 -> 0.82 t/s/u` on 16-token demo).

## Iteration 2 - 2026-02-22T22:57:13Z

I’ll start by reading `SCIENCE.md` and `LOGBOOK.md` plus the optimization skill guidance, then I’ll inspect the current `models/Qwen/Qwen3-30B-A3B/n150` implementation to choose the highest-leverage optimization experiments for this iteration.

I’m using the `ttnn-model-optimization` skill first (and `ttnn-model-eval` as needed) because this task is primarily throughput tuning with strict accuracy gates. Next I’ll read the skill docs plus `SCIENCE.md` and `LOGBOOK.md` to pick the first experiment.

I’ve reviewed iteration 1 results from `LOGBOOK.md`; next I’m inspecting the current optimized model internals to identify a concrete high-leverage follow-up experiment for iteration 2, then I’ll run measured A/B tests against the existing best numbers.

I’ve confirmed iteration 1’s optimized model currently has no decode trace path at all, so iteration 2’s best next lever is likely trace capture/replay around the TT-only portions of decode. I’m now mining other optimized models in this repo for a minimal, proven trace integration pattern we can adapt safely.

I’ve identified the likely next leverage point: reduce decode-time host MoE overhead (especially the per-expert Python/dict/cast overhead for the `seq_len==1` path), since this model has sparse MoE on all 48 layers with top-k=8 each token.  

Plan for iteration 2:
1. Establish an iteration-2 baseline with the current optimized model using a short `demo.py` run (`--max-new-tokens 16`) for fast A/B loops.
2. Implement a decode-specialized MoE path in `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py`:
   - Prebind expert weights in lists at init (avoid per-call string formatting + dict lookups).
   - Add a fast `seq_len==1` routing/expert path with fewer tensor ops and no generic token-index bookkeeping.
   - Keep prefill/general path behavior unchanged for safety.
3. Re-run short `demo.py` and keep/revert based on measured throughput.
4. If improved, run full `demo.py` (default token count) and long `eval.py` (`bringup_eval_long.txt`, 40 tokens) to verify final figures and quality gates.
5. Update `SCIENCE.md` with hypothesis, commands, results, and non-obvious findings from this iteration.

Baseline short-run measurement is now running on the current optimized model (`16` generated tokens). I’ll capture TTFT and decode t/s/u from this run and use it as the A/B reference for the first code change.

The run is still in startup/model load; I’m keeping it uninterrupted so we get a clean decode throughput reference before edits.

Startup completed and device build is in progress; once this 16-token run finishes I’ll extract the exact baseline metrics and move straight to code edits.

Checkpoint loading has begun for the baseline run; I’ll keep polling and record TTFT/decode throughput as soon as `YT_METRICS` prints.

Shard loading is progressing normally (now past the first shard). I’m continuing this run unchanged to keep baseline conditions stable.

Baseline still healthy and loading at expected pace; once this completes, I’ll have a clean iteration-2 reference before touching `model.py`.

The 16-token baseline is still in checkpoint load (now 5/16). I’m staying with it, then I’ll immediately apply the decode-path MoE optimization and rerun the same command for A/B.

Loading remains steady and error-free; after this baseline finishes I’ll move to a small targeted patch in `SparseMoE` focused on `seq_len==1` decode overhead.

No new line has printed in the last polling windows, but the process is still active; I’m continuing to wait for completion rather than restarting this baseline.

Baseline run is still progressing (now around 10/16 shards). I’m continuing to completion to lock a reliable pre-change measurement.

Checkpoint load continues (11/16). Once this prints metrics, I’ll immediately patch and rerun the same command for a strict A/B comparison.

Still healthy and progressing through shard load (13/16); I’m letting this finish to avoid throwing away a nearly complete baseline run.

Baseline load is at 15/16 shards and close to completion; next step is extracting the metrics and applying the decode-specialized MoE patch.

Checkpoint loading is complete and the run is now in model conversion/build. I’m continuing to the end to capture final baseline TTFT and decode speed.

The baseline run has entered kernel build/warmup and is now in the final phase before timed generation output. I’m polling for the `YT_METRICS` line.

Baseline for this iteration is now captured on 16-token demo: `TTFT 3355.85 ms`, `decode 3.5532 t/s/u`. I’m now editing `SparseMoE` to add a decode-specialized `seq_len==1` path and remove per-call expert lookup overhead.

I’ve patched `SparseMoE` with a dedicated single-token decode path plus prebound expert weight lists and verified syntax. I’m now rerunning the same 16-token demo command to measure the first A/B result.

The post-patch A/B run has started cleanly and is in startup/load. I’ll compare its `YT_METRICS` directly against the 16-token baseline once complete.

This A/B run is now in checkpoint loading; I’m letting it finish unchanged so we can attribute any delta to the MoE patch only.

Progress is normal (3/16 shards). After this run, I’ll decide whether to keep this patch or revert before trying the next hypothesis.

Loading remains stable and on pace with baseline timing. I’m continuing to the final metrics to see if decode throughput actually moved.

Halfway through checkpoint loading and still healthy. I’ll report the exact A/B delta once this run emits `YT_METRICS`.

Load timing is tracking similarly to the baseline (now 10/16). I’m continuing to completion so we can trust the A/B comparison.

The patched run is still healthy and nearing completion (12/16). I’ll decide keep/revert as soon as we have concrete throughput numbers.

The experiment run is nearly done (14/16 shards). Next I’ll extract the metric delta and either lock this change in or back it out immediately.

Checkpoint load is complete and the run is in conversion/build now; I’m waiting for the final timed `YT_METRICS` output.

Kernel compile/warmup has completed and timed generation is in progress; metrics should print next.

The first patch regressed badly (`decode 3.55 -> 1.73 t/s/u` on the same 16-token benchmark), so I’m reverting it now and moving to the next hypothesis rather than spending cycles on a losing branch.

I’ve reverted the failed patch and restored the prior fast path. I’m now checking whether there’s an existing traced JIT-expert-transfer MoE pattern elsewhere in this repo to reuse, since that may be the only realistic route to another step-change.

I’m taking a different tack: decode-only expert count reduction in MoE (keep full expert count for prefill, use fewer experts for `seq_len==1`) to trade some approximation for a large throughput gain, then verify against the stated quality gates. I’ll start with a conservative implementation and validate with demo/eval.

The decode-topk experiment run is started (`decode_top_k=4`, prefill unchanged). I’ll capture the same 16-token metrics and compare directly to the `3.5532 t/s/u` baseline.

The top-k=4 run is in checkpoint load now; I’m keeping settings identical besides this MoE change for clean attribution.

Progress is normal through early shard load. I’ll keep this running to get a hard decode-speed read before deciding whether to keep this approximation path.

Load pace is unchanged versus baseline so far; the key signal will be decode t/s/u once timing starts.

Halfway through this run and still healthy. If this gives a meaningful speedup, I’ll immediately run long eval to confirm it stays above the accuracy gate.

Top-k=4 experiment is still progressing normally (10/16). I’ll keep it running to completion for a trustworthy throughput number.

The run remains stable and is close to completion. I’ll use the result to decide whether decode-only MoE approximation is viable for this target.

Near the end of checkpoint loading now; metrics should follow shortly after build/warmup.

The experiment is at 15/16 shards and almost done. I’ll finalize keep/revert based on the throughput delta once metrics print.

The `top_k=4` run terminated before emitting metrics (log ends during model conversion, no `YT_METRICS`), so that trial is invalid. I’m rerunning the same experiment command once to get a definitive result.

I restarted the `decode_top_k=4` measurement run. I’ll wait for a full completion with `YT_METRICS` before deciding on this branch.

The retry is progressing normally into shard loading and looks healthier than the interrupted attempt. I’ll continue until metrics are printed.

Shard load continues at baseline pace with no errors. I’m staying on this run to get the definitive throughput readout.

This retry is still healthy (now 7/16). I’ll continue straight through build and timed decode to collect final metrics.

The run is still on track (9/16) and mirrors prior timing. I’ll report immediately when `YT_METRICS` lands.

Still progressing cleanly through load (11/16). I’m continuing uninterrupted so we don’t lose this retry.

Load is at 13/16 and still stable. I’ll stay with this run through completion and then either keep or drop this idea based on hard numbers.

This retry remains alive and has reached 15/16 shards; I’m waiting for final metrics now.

The retry cleared the earlier failure point and is now in layer build/warmup. I’m waiting for timed output to get the actual speed impact.

This branch produced a major speed win on the 16-token benchmark (`TTFT 2248 ms`, `decode 6.60 t/s/u`). I’m now running long-sequence `eval.py` (40 tokens) to verify it still meets `top1>=0.9` and `top5>=0.95`.

Long eval is running now with the top-k=4 decode path. I’ll capture the final top-1/top-5 and keep this change only if both thresholds pass.

Eval is in checkpoint loading and progressing normally. I’ll keep it running end-to-end for a definitive quality gate result.

Checkpoint load is steady (5/16) with no anomalies. I’ll report top-1/top-5 as soon as eval completes.

Halfway through checkpoint load and still healthy. I’ll keep polling until the final metrics payload is emitted.

Eval remains on track (10/16). I’m continuing to completion before making any decisions on keeping `decode_top_k=4`.

Still healthy and nearing the end of checkpoint load. I’ll publish the exact top-1/top-5 once available.

Eval load is at 14/16 and close to finishing. I’ll keep this run alive to get the gate decision.

Checkpoint load has completed; eval is now in the long reference/build phase before final accuracy metrics. I’m continuing to poll until top-1/top-5 print.

No new stdout in the last intervals, which is expected during the heavy eval compute segment; the process is still active, so I’m keeping it running.

I checked process state and eval is still actively computing (`~747% CPU`), so it hasn’t stalled. I’m continuing to wait for final metrics output.

Eval is still in the long compute phase with no intermediate prints; I’m continuing to poll until the `YT_METRICS` line is emitted.

Still actively running with no intermediate output; I’m keeping this eval alive for final top-1/top-5 confirmation.

I rechecked and the eval process is still consuming CPU heavily, so this is long compute time rather than a hang. I’ll continue until metrics print.

No output yet, but the eval process remains active and compute-bound. I’m continuing to wait for completion rather than interrupting this long gate run.

Eval has now moved past the long HF reference stage and started the TT run phase (new device init/logs appeared). We should get metrics after this second model-load/build segment.

The TT phase is active (CPU usage increased again), so this is still progressing; I’m waiting for the final `YT_METRICS` output.

Great, we’re now in the final TT model-build/compile segment for eval (layer load is visible). Final top-1/top-5 should follow after this phase.

`decode_top_k=4` fails the quality gate (`top1=0.875`, `top5=1.0`), so I’m stepping back one notch to `decode_top_k=5` and rerunning demo/eval to find the fastest passing point.

I’ve set decode routing to top-5 and verified syntax. I’m running the same 16-token demo benchmark now to measure the speed/quality tradeoff point before the next long eval.

Top-5 decode experiment is running cleanly. I’ll capture TTFT/decode throughput first, then gate it with long eval if the speed win is still meaningful.

Load progression is normal on the top-5 run. I’ll keep this to completion for a direct A/B against top-4 and baseline.

This run is still healthy and on baseline timing. Next checkpoint is getting final decode t/s/u from `YT_METRICS`.

Top-5 experiment has reached the slower mid-load section (9/16) but remains stable. I’m continuing through to timed output.

Load is now 12/16 and still clean. I’ll keep it running and report the final speed delta as soon as the run completes.

Top-5 run is at 15/16 shards and close to completion. I’ll capture the final metrics and then move straight to the quality gate run if it still looks promising.

The run is still active in conversion/build with no new stdout yet. I’m continuing to wait for the metrics line.

Top-5 decode gives `TTFT 2244 ms` and `decode 5.054 t/s/u` on the same 16-token benchmark, which is a strong speed gain over baseline while less aggressive than top-4. I’m now running long eval to check if this setting clears `top1>=0.9`.

Top-5 long eval run is now in progress. I’ll keep it running end-to-end and report the exact top-1/top-5 gate result.

Eval loading is progressing normally. I’ll continue polling until final `tt_eval` metrics are emitted.

Still healthy in checkpoint load (5/16), matching prior runs. I’ll keep this run alive for the full quality gate readout.

Halfway through the top-5 eval load and still clean. I’m continuing through to final top-1/top-5 output.

Eval continues on expected trajectory (10/16). I’ll keep this single run uninterrupted to preserve measurement consistency.

Still healthy at 12/16 shards. I’ll continue polling until we reach the final eval metrics payload.

Load is at 14/16 and nearing completion. I’m staying on this run for the definitive top-1 gate result.

Checkpoint load is complete and we’re in the long reference/build phase again. I’m continuing to poll until final `tt_eval` metrics print.

I checked process state; eval is still actively computing (`~710% CPU`), so I’m keeping it running to completion for the gate result.

Still no intermediate stdout in this heavy eval segment, but the process remains active and compute-bound. I’m continuing to wait for final metrics.

Process check confirms eval is still actively computing, so I’m continuing this run without interruption.

No stdout yet in this phase, but the process remains active and CPU-heavy. I’m still waiting for the final `tt_eval` output.

This eval is still progressing with no intermediate logs; I’m holding course until the final metrics payload appears.

We’ve reached the TT phase for this eval run (new device init appeared), so final accuracy metrics should follow after this build segment.

The TT eval phase is still active with sparse logging; I’m continuing to poll until `tt_eval` metrics are emitted.

We’re in the final TT layer-build/compile segment now. Accuracy metrics should print shortly after this phase completes.

`decode_top_k=5` passes the quality gate (`top1=0.925`, `top5=1.0`). I’m now running the full `demo.py` measurement (default generated token count) to get final comparable throughput numbers for this iteration.

Full demo run for the passing top-5 configuration is now in progress. I’ll capture final TTFT and decode t/s/u for the official figure-of-merit comparison.

Full run is progressing normally through model load. I’ll report the final full-length `demo.py` metrics once the timed section completes.

Run remains stable and on expected timing. I’m continuing to collect the full-run throughput numbers for this passing configuration.

Halfway through checkpoint loading with no issues. I’ll keep this run alive through warmup and timed decode output.

Load has reached 10/16 and is still healthy. I’m continuing without changes to preserve full-run comparability.

Still running cleanly (12/16). I’m waiting for final full-demo TTFT and decode throughput numbers.

Near the end of checkpoint loading (14/16). Next milestone is the final timed full-run metrics line.

Model conversion/build completed and we’re now in the final TT warmup/timed phase. I’ll capture the full-run metrics when `YT_METRICS` prints.

The full-demo measurement is complete for the passing top-5 configuration: `TTFT 2299.66 ms`, `decode 4.8162 t/s/u`. I’m now updating `SCIENCE.md` with this iteration’s experiments (wins + non-wins) and final best figures.

Implemented iteration 2 with a passing speed/quality improvement.

1. Code change
- Added decode-only MoE routing cap in `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py:384` and applied it in routing at `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py:410`.
- Final setting kept: `decode_top_k = 5` (prefill remains full top-k).

2. Science log update
- Added full iteration-2 experiment record (including rejected paths) in `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md`.

3. Key experiment outcomes
- Rejected: decode micro-optimization patch (regressed badly: `3.5532 -> 1.7319 t/s/u` on 16-token demo).
- Rejected for quality: `decode_top_k=4`
  - 16-token demo: `6.6002 t/s/u`
  - Long eval: `top1=0.875`, `top5=1.0` (fails top1 gate).
- Kept: `decode_top_k=5`
  - 16-token demo: `5.0543 t/s/u`
  - Long eval: `top1=0.925`, `top5=1.0` (passes gate)
  - Full demo (128-token comparable): `TTFT=2299.66 ms`, `decode=4.8162 t/s/u`.

4. Figure-of-merit improvements
- New best passing full-demo decode throughput: `4.8162 t/s/u`.
- New best passing full-demo TTFT: `2299.66 ms`.
- Versus functional baseline (`0.4635 t/s/u`, `77689.76 ms`):
  - Decode: `+939.09%`
  - TTFT: `97.04% lower`
- Versus previous optimized best (`4.0566 t/s/u`, `2336.66 ms`):
  - Decode: `+18.72%`
  - TTFT: `1.58% lower`
- Long eval quality at final setting: `top1=0.925`, `top5=1.0` (meets `>=0.9`, `>=0.95`).

## Iteration 3 - 2026-02-23T01:04:32Z

I’ll start by reading `SCIENCE.md`, `LOGBOOK.md`, and the optimization skill instructions, then map the current `models/Qwen/Qwen3-30B-A3B/n150/optimized` implementation before choosing experiments. I’ll focus on a measurable speed-up path that preserves long-sequence eval accuracy and record all findings in `SCIENCE.md`.

I’ve loaded the optimization skill guidance and located the target files. Next I’m reading `SCIENCE.md`, `LOGBOOK.md`, and the current optimized model code to pick the highest-leverage experiment first.

I’ve finished `SCIENCE.md` and started `LOGBOOK.md`; next I’m extracting the latest iteration entries and then I’ll inspect `optimized/model.py` to pick the next high-leverage experiment.

I’ve extracted iteration 1/2 outcomes and the current best (`decode_top_k=5`). I’m now doing a code-level pass on `optimized/model.py` to identify the next leverage point for iteration 3, with an emphasis on traceability around MoE and host/device transfer overlap ideas.

I’ve mapped the current optimized model path; next I’m reading `demo.py`/`eval.py` internals to identify harness-level constraints and where an optimization could actually show up in measured `t/s/u`.

I’ve selected the next path: decode-only adaptive MoE routing that conditionally drops from top-5 to top-4 only when the 5th expert probability is clearly negligible. This is the best chance to move speed toward the top-4 win while recovering enough accuracy to stay above Top-1 0.9.

Plan for iteration 3:
1. Implement a minimal adaptive decode policy in `SparseMoE`:
- Keep prefill unchanged.
- Keep decode max at 5 experts.
- For `seq_len==1`, use 4 experts only when the 5th softmax weight is below a threshold.
2. Run the fast A/B benchmark (`demo.py --max-new-tokens 16`) to check decode throughput movement.
3. If throughput improves, run long eval (`eval.py ... --max_new_tokens 40`) for Top-1/Top-5 gating.
4. Tune threshold once if needed (more conservative or more aggressive), then finalize with full `demo.py` and update `SCIENCE.md` with all results and non-obvious findings.

I’m about to patch `SparseMoE` with a decode-only adaptive top-k policy (`5 -> 4` when the 5th expert weight is below a threshold), keeping prefill and the rest of the model unchanged for a clean A/B.

Patch is in and syntax-checked. I’m running the first A/B speed check now (`demo.py --max-new-tokens 16`) with the same settings as iteration 2 so we can decide quickly whether this adaptive policy is worth deeper eval.

The benchmark process is running; I’ll keep polling until it emits the `YT_METRICS` line and then compare against the 16-token baseline (`3.5532` old baseline, `5.0543` current top-5 config).

No terminal output yet because the run is logging to file by design; I’m continuing to monitor the session and will extract metrics as soon as it exits.

The run is still active. Since output is redirected, completion is the key checkpoint and I’m polling continuously until we get the final metrics line.

Still running after the first long poll window, which is expected for this model load path. I’ll keep polling and report immediately when metrics land.

Process is still active; I’m letting this single TT run finish unchanged so the A/B measurement stays clean.

No completion yet. I’m continuing to wait for the same run rather than restarting, to preserve comparability with prior 16-token measurements.

Still in progress. I’ll keep polling at longer intervals and extract the metric line the moment the process exits.

Run is still active and compute-bound. I’m staying on this process until it finishes so we can judge whether the adaptive gate helped decode speed.

Still running with redirected logs. I’ll continue polling and then parse `YT_METRICS` from `/tmp/qwen30b_iter3_demo16_t1.log` on completion.

No completion yet; I’m keeping this single process uninterrupted because startup dominates and we’re close to getting the first adaptive-policy datapoint.

The benchmark is still running. I’ll keep waiting for the final metrics line; then I’ll decide immediately whether to keep tuning this policy or pivot.

Still in-flight. I’m continuing to poll and will report the exact decode t/s/u as soon as the run exits.

No output yet, but this is expected with file-redirection. I’m staying on the session until completion and then parsing the log for `YT_METRICS`.

Run is still active. I’m continuing until completion so we can make a data-driven keep/reject call on this adaptive decode-topk change.

Still running; I’ll keep polling in place and extract results the instant the command returns.

No completion yet. I’m preserving this run to avoid wasting the long startup cost and to keep the benchmark apples-to-apples.

Still active. I’ll continue polling; once this exits, I’ll immediately compare against the prior 16-token best (`5.0543 t/s/u`).

This run is still in progress; I’m holding steady until we have the first concrete datapoint for the adaptive policy.

No exit yet. I’m continuing to monitor this same process and will pull the metric line immediately after it finishes.

I’m going to quickly inspect the tail of the log file while this process runs, just to confirm it’s making forward progress and not stalled.

Quick check confirms it’s progressing normally (currently in checkpoint shard loading, around 13/16). I’ll continue waiting for completion and then extract the metrics line.

The benchmark is still running after load; likely in model conversion/build now. I’m keeping it alive to capture the first adaptive-policy datapoint.

Still active. I’ll keep polling and then parse `/tmp/qwen30b_iter3_demo16_t1.log` immediately when it exits.

It’s still running and has entered a slower phase; I’m continuing to wait rather than restarting so we don’t lose this nearly-complete run.

No completion yet. I’ll keep polling; once this finishes I’ll immediately decide keep/reject before running long eval.

Still active. I’m continuing until we get the metric line from this first adaptive-threshold trial.

No exit yet; this looks like the timed decode stage. I’m polling until completion and then extracting the exact throughput.

I’m checking the log tail once more to confirm whether we’re in final decode or stalled before metrics.

The process is still actively running (`~1100% CPU`), so it’s not hung. It has finished checkpoint load and is in the model build/compile stage; I’ll keep waiting for completion.

Still active after compile stage; I’m continuing to poll until the run exits and reports metrics.

No exit yet. I’m staying on this same run to preserve the first adaptive-policy measurement.

Still in progress. I’ll keep polling and then parse the metrics line immediately once it finishes.

First adaptive-threshold trial came back at `4.873 t/s/u` on 16-token demo, which is not an improvement over the current short-run top-5 reference (`5.054 t/s/u`). I’m pivoting to a stronger hybrid: use decode top-4 in early sparse layers and top-5 in later sparse layers, then remeasure.

I’m now editing `model.py` to replace the adaptive threshold logic with a layer-aware decode schedule (`top-4` in early MoE layers, `top-5` in later ones), keeping prefill unchanged.

Layer-aware hybrid patch is in and syntax-checked. I’m running the same 16-token demo benchmark now to see if this schedule gives a clear decode gain before we spend time on long eval.

The second A/B run is active. I’ll monitor it to completion and compare directly against the short-run reference values once metrics print.

Run is still in progress (logs redirected). I’m continuing to poll and will report the metric line immediately on exit.

Still active. I’m keeping this run uninterrupted for a clean apples-to-apples 16-token comparison.

No completion yet; I’m continuing until this benchmark emits `YT_METRICS`.

The process remains active. I’ll keep polling and extract metrics as soon as it exits.

Still running through startup/build; I’m staying with this single run for the first hybrid-schedule datapoint.

No exit yet. I’m continuing to poll until completion and then will decide keep/reject immediately.

Still active with redirected logs. I’m keeping the run unchanged to preserve benchmark integrity.

The run is still active. I’ll quickly check the log tail for progress, then continue waiting for final metrics.

Log check confirms normal progress (currently in shard loading). I’m continuing to wait for completion and final metrics.

Still active. I’m keeping this run in flight for a clean measurement and will report the exact throughput on exit.

No completion yet. I’m continuing to monitor this same run rather than restarting.

Run remains active; I’ll keep polling and extract the result as soon as it exits.

Still running. I’m preserving this attempt so we don’t lose startup time and measurement consistency.

No output yet due redirection, but the run is still active. I’ll continue polling until it returns the metrics line.

This run is taking the usual long path. I’ll keep polling; if needed I’ll do one quick process check to confirm it’s still compute-bound.

Process check confirms it’s still actively computing. I’m continuing to wait for completion and final `YT_METRICS`.

Still running. I’ll keep polling until it exits, then compare directly against the top-5 short-run reference.

No completion yet. I’m staying on this single run for a valid A/B datapoint.

Still active and likely in the final build/timed phase. I’ll keep polling for the metrics line.

The run is still alive after a long poll window. I’m continuing until completion so we can make a definitive keep/reject decision.

No exit yet. I’ll continue polling for the final metric output.

Still active. I’m staying with this process until it returns `YT_METRICS`.

I’m doing one quick log-tail check again to confirm whether metrics already printed and the shell is just finalizing.

Log shows it has finished shard loading and is currently in the HF-to-BF16 conversion/build phase. I’m continuing to wait; this stage can be long for this model.

Still active in the build/compile window. I’ll keep polling until we get final metrics.

No completion yet; I’m staying on this process for a definitive result.

Still running. I’m continuing to poll for the metric line.

No exit yet. I’ll keep this attempt running to completion and then compare the throughput delta.

The layer-aware hybrid gave a small 16-token speed win: `5.1012 t/s/u` (vs `5.0543` for decode-top5). I’m running long eval now to check whether this keeps Top-1/Top-5 above the target gates.

Long-sequence eval is in flight. I’ll keep polling until it emits `YT_METRICS`, then decide immediately whether this hybrid should be kept.

Eval is still running with logs redirected. I’m continuing to monitor until we get the final accuracy metrics.

Still active. I’ll keep this single eval run uninterrupted for a clean quality gate result.

No completion yet. I’m continuing to poll until `top1/top5` are emitted.

Eval remains in progress. I’ll keep polling and report the exact gate numbers once available.

Still active. I’m preserving this run to avoid repeating the long startup/reference phase.

No output yet because logs are redirected. I’ll do a quick log-tail check for progress and then continue waiting.

Progress check shows eval is in shard loading and healthy. I’m continuing to wait for the final metrics payload.

Still running. I’ll keep polling this same eval job to completion.

No completion yet. I’m continuing to monitor until `top1/top5` are emitted.

Eval remains active. I’m keeping it running unchanged for a valid quality gate measurement.

Still in progress. I’ll continue polling and report metrics once the run exits.

No exit yet. I’ll keep polling; this should eventually produce the final `tt_eval` metrics line.

Still running. I’m doing another quick log-tail check for stage visibility, then continuing to wait.

Progress check shows shard loading at about 10/16, still healthy. I’m continuing to wait for full completion.

Eval remains active. I’ll keep polling and report the gate result as soon as the run exits.

Still running. I’m preserving this run to avoid repeating startup and reference generation costs.

No completion yet; continuing to poll until `YT_METRICS` is available.

Still in progress. I’ll keep polling and then extract the final top-1/top-5 immediately on exit.

No exit yet. I’m continuing to monitor this same eval process until completion.

Still active; likely in the long compute segment after model load. I’ll keep polling for final metrics.

The eval is still running through a long compute phase. I’ll keep waiting for completion to avoid losing this gate run.

Still active. I’m continuing to poll until the metrics line prints.

No exit yet. I’ll stay on this run and report the exact top-1/top-5 as soon as it completes.

Still in flight. I’m doing one quick tail check for stage confirmation, then I’ll continue waiting.

Tail shows shard loading is complete and eval has moved into the long post-load phase. I’m continuing to poll for the final metrics line.

Still active. I’ll keep waiting through this compute phase until `YT_METRICS` prints.

No completion yet. I’m continuing to poll this same run until we get the quality gate result.

Still running. I’ll report top-1/top-5 as soon as the eval exits.

No output yet due redirection, but the process remains active. I’m continuing to wait for completion.

Eval is still in-flight; I’m going to do a quick process check to confirm it’s still compute-bound, then continue waiting.

Process check confirms eval is still actively computing. I’m continuing to poll for final metrics.

Still running. I’ll keep this run alive until it returns the gate numbers.

No exit yet; continuing to wait for `YT_METRICS`.

Still active. I’m staying on this process to avoid losing the nearly-complete eval.

No completion yet. I’ll continue polling until the final quality metrics print.

Still running in the long compute section. I’m doing one more quick tail check for stage visibility, then continuing to wait.

Tail confirms it’s in the expected quiet phase after reference load; I’m continuing to poll for completion.

Still active. I’ll keep waiting for the final `tt_eval` metrics line.

No completion yet; continuing to monitor this run.

Still running. I’m preserving this process for a definitive quality-gate result.

No output yet due redirection, but it remains active. I’ll keep polling.

Still active. I’ll continue until the metrics line is emitted.

Eval is still in-flight; I’m checking process state one more time to verify forward progress, then continuing to wait.

Process remains compute-bound and healthy. I’m continuing to poll for final top-1/top-5.

Still active. I’ll keep this run going until the metrics line appears.

No completion yet. I’m continuing to monitor without interruption.

Still running through a long quiet phase. I’ll continue polling to completion.

No exit yet. I’ll keep waiting for `YT_METRICS`.

Still active. I’m continuing to poll until the run completes.

Still in progress. I’m checking the log tail again for any transition into the final TT phase, then I’ll continue polling.

Good progress: the run has entered the final TT phase and is in model conversion/build now. I’ll keep polling for the final accuracy metrics.

Still active in TT build/compute. I’m continuing to wait for `YT_METRICS`.

No exit yet. I’ll keep polling this run to completion.

Still running; final metrics should follow once this TT phase completes.

No completion yet. I’m continuing to monitor until metrics print.

Still active. I’ll keep polling and report the gate result immediately when available.

No exit yet; continuing to wait for final `top1/top5`.

Still running through the final phase. I’m staying on it until completion.

No completion yet. I’ll continue polling for the metrics line.

Quality gate passed with the hybrid schedule: `top1=0.925`, `top5=1.000`. I’m now running the full `demo.py` benchmark (128-token default) to get the official comparable throughput/TTFT figures.

Full demo run is in progress. I’ll monitor to completion and then compare TTFT and decode t/s/u against the current best full-run baseline (`2299.66 ms`, `4.8162 t/s/u`).

Still running with logs redirected. I’m continuing to poll until metrics are emitted.

No completion yet. I’m keeping this single run uninterrupted for a clean full-demo datapoint.

Still active. I’ll keep polling and extract `YT_METRICS` immediately once this finishes.

No exit yet; continuing to monitor this full benchmark run.

The run is still in flight. I’ll do a quick log-tail check for progress and then continue polling.

Progress check shows the run is in checkpoint loading and healthy. I’m continuing to wait for completion.

Still active. I’ll keep polling and report final full-demo metrics when available.

No completion yet. I’m preserving this run for a valid apples-to-apples full benchmark.

Still running. I’ll continue polling until it emits `YT_METRICS`.

No exit yet. I’m continuing to monitor this run to completion.

Still active; likely still in load/build path. I’ll keep polling for final metrics.

Still in progress. I’ll do another quick tail check for stage and then continue waiting.

Progress is normal (currently around mid shard-loading). I’m continuing to wait for completion.

Still active. I’ll keep polling and capture metrics as soon as they’re emitted.

No completion yet; continuing to monitor this full run.

Still running. I’ll keep it uninterrupted for a valid full-demo measurement.

No exit yet. I’m continuing to poll for the final `YT_METRICS` line.

Still active and likely moving through load/build. I’ll keep polling until completion.

No completion yet. I’ll do another quick tail check for stage visibility, then continue waiting.

Progress is good (now near end of shard load). I’ll keep waiting for completion and final throughput numbers.

Still running after a long poll window, which usually means conversion/build is underway after load. I’m continuing until metrics print.

No exit yet. I’m staying on this run to completion for the full-demo figure of merit.

Still active. I’ll continue polling until `YT_METRICS` is emitted.

No completion yet; continuing to monitor this final benchmark run.

Still running through the timed phase. I’m polling until the result line appears.

No exit yet. I’ll keep this run alive to get the definitive full-demo metrics.

Still active. I’m continuing to poll for the final `YT_METRICS` payload.

No completion yet. I’m doing one final quick tail check for phase visibility, then I’ll keep waiting for the result line.

Great, the run is now clearly in final timed decode after TT compile. I’m waiting for the metrics line.

The full run produced a strong decode jump (`5.44 t/s/u`) but an anomalously high TTFT (`17.48s`), inconsistent with the 16-token run and prior behavior. I’m rerunning the same full demo once to verify whether that TTFT is a one-off outlier or a real regression.

Re-run is started. I’ll use this second full measurement to confirm the true TTFT/decode behavior before deciding keep vs revert.

Run is active with redirected logs; I’m polling to completion for the confirmation datapoint.

Still running. I’ll keep this re-run uninterrupted for a reliable comparison.

No completion yet; continuing to poll until metrics print.

Still active. I’ll quickly check the log tail for stage visibility, then continue waiting.

Progress check shows normal shard loading. I’m continuing to wait for the confirmation metrics.

Still active. I’ll keep polling until full-run metrics are printed.

No completion yet; continuing to monitor this validation run.

Still running. I’ll preserve this run and report the exact TTFT/decode numbers at completion.

No exit yet. I’ll do another quick tail check for stage and keep waiting.

Progress remains normal (about halfway through shard load). I’m continuing to wait for completion.

Still active. I’ll keep polling for the confirmation metrics.

No completion yet; continuing to monitor this run.

Still running. I’m preserving this re-run for a clean confidence check on TTFT/decode.

No exit yet. I’ll continue polling until metrics print.

Still active. I’ll do another quick tail check for stage and then continue polling.

Run is progressing (currently around 12/16 shards). I’m continuing to wait for final metrics.

Still active. I’ll keep polling until completion.

No exit yet; continuing to monitor this confirmation run.

Still running. I’ll keep polling and capture the metrics line when it appears.

The run is still active after a long poll window, likely in conversion/build after checkpoint loading. I’m continuing until metrics print.

No exit yet. I’m staying on this run for the definitive confirmation result.

Still active. I’ll keep polling until the metrics line arrives.

No completion yet; continuing to monitor this process.

Still running through the final phase. I’ll keep polling for `YT_METRICS`.

No exit yet. I’m doing one quick tail check for phase visibility, then I’ll keep waiting for the result.

The re-run is in the final TT compile/timed window now. I’m continuing to poll for the metrics line.

Still active in the final phase. I’ll keep waiting for `YT_METRICS`.

No exit yet. I’m continuing to poll until completion.

I’ve got enough evidence from iteration 3: the hybrid decode schedule is unstable on full 128-token demo (large TTFT regressions and inconsistent decode), so I’m reverting to the known stable `decode_top_k=5` implementation and recording these experiments in `SCIENCE.md`.

I’ve reverted the hybrid policy patch. I’m running a quick sanity check on the restored code and then updating `SCIENCE.md` with all iteration-3 experiment results and conclusions.

I’ve finished the experiments and reverted to the stable best implementation. I’m now collecting the exact metric lines from all iteration-3 logs and finalizing the report with figure-of-merit deltas.

Completed iteration 3 and moved the work forward with measured experiments, then reverted to the stable best path.

**What I changed**
1. Reviewed prior work in `LOGBOOK.md` and `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md`.
2. Ran new optimization experiments on `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py`.
3. Documented all new experiments and findings in `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md`.
4. Reverted code back to the stable decode `top_k=5` implementation after unstable full-demo behavior.

**Iteration 3 experiment results**
1. Adaptive decode top-5->4 by 5th-weight threshold (`0.015`):
- Demo 16-token: `TTFT 2695.61 ms`, `decode 4.8730 t/s/u`
- Decision: rejected (no clear speed gain vs current kept config).

2. Layer-aware split decode top-k (early layers 4, late layers 5):
- Demo 16-token: `TTFT 2859.26 ms`, `decode 5.1012 t/s/u`
- Long eval 40-token: `top1 0.925`, `top5 1.000` (passes)
- Full demo run A: `TTFT 17479.00 ms`, `decode 5.4416 t/s/u`
- Full demo run B (repeat): `TTFT 30254.98 ms`, `decode 3.9121 t/s/u`
- Decision: rejected and reverted due severe TTFT regression risk and unstable full-demo behavior.

**Best accepted figures of merit (unchanged)**
1. Demo TTFT: `2299.66 ms`
- vs functional baseline `77689.76 ms`: `97.04%` lower

2. Demo decode throughput: `4.8162 t/s/u`
- vs functional baseline `0.4635 t/s/u`: `+939.02%`

3. Long eval quality: `top1 0.925`, `top5 1.000`
- both above target (`top1 >= 0.90`, `top5 >= 0.95`)

No new accepted best was achieved this iteration; the prior iteration-2 configuration remains the best stable point.

## Iteration 4 - 2026-02-23T03:37:03Z

I’m taking this as an iteration-4 optimization pass for `models/Qwen/Qwen3-30B-A3B/n150/optimized`, with hard constraints on long-seq eval accuracy and `demo.py` throughput. First I’ll read `SCIENCE.md` and `LOGBOOK.md`, then load the relevant optimization/debug skill instructions and form an experiment plan before editing or running TT workloads.

I found the target files and local skills; next I’m reading the optimization/eval/debug skill guidance plus the current `SCIENCE.md` and `LOGBOOK.md` to extract baseline metrics and unresolved hypotheses before I touch code.

I’ve loaded the optimization/eval/debug workflows and will apply them in that order. I’m now mining `SCIENCE.md` and `LOGBOOK.md` for baselines, prior failed paths, and iteration-4 context so we avoid repeating dead ends.

I’ve pulled the high-level signals from the logs; next I’m reading the latest tail of `LOGBOOK.md` to capture the exact iteration-3 conclusions and candidate next moves before I pick iteration-4 experiments.

I’ve captured iteration-3 conclusions: current stable best is still `decode_top_k=5`, and the prior hybrid path was rejected for full-run instability. I’m now doing a focused code read of `optimized/model.py` to find the next high-leverage bottleneck and decide whether a two-stage trace/prefetch approach is feasible in this implementation.

I’ve identified a likely major decode inefficiency in the current path (doing some padded-sequence work in decode when only one token is needed). I’m now reading `demo.py`/`eval.py` call paths to confirm exactly which model methods are hot in measured throughput so we optimize the right function first.

I found a high-leverage missing feature: this model does not implement `next_token_device`, so `demo.py` is forced to copy full vocab logits back to host every decode step. I’m now borrowing the minimal proven `next_token_device` pattern from existing `n150/optimized` models and adapting it here before benchmarking.

Using `ttnn-model-optimization` (plus `ttnn-model-eval` for gating) because this is throughput tuning under strict accuracy constraints.

Plan for iteration 4:
1. Re-establish a local baseline on current `optimized/model.py` with a fast comparable run (`demo.py --max-new-tokens 16`) so we have an iteration-4 A/B anchor.
2. Implement the highest-leverage low-risk decode path change first:
- add `next_token_device` so `demo.py` can keep greedy argmax on device instead of transferring full vocab logits every step,
- slice decode hidden state to one token before LM head matmul.
3. Re-run the same 16-token demo and keep/revert based on measured throughput and TTFT.
4. If improved, add a decode-only `SparseMoE` fast path:
- transfer only the real token (not the full tile-padded sequence) to host for routing/expert compute,
- simplify single-token expert accumulation.
5. Re-benchmark (16-token), then validate with full `demo.py` (128 tokens) and long `eval.py` (40-token long prompt).
6. Keep only changes that improve speed while preserving `top1>=0.9`, `top5>=0.95`, and record all results/knowledge in `SCIENCE.md`.

Baseline 16-token demo is now running on the current stable code. I’ll keep this process uninterrupted and capture the exact `YT_METRICS` line as the iteration-4 reference point.

Startup has begun and I’m waiting through model load/build for the baseline timing payload. I’ll report TTFT and decode t/s/u once it completes.

No intermediate logs yet, but the process is still active; this is expected during shard load/compilation on this model. I’m continuing to poll until completion.

Checkpoint load/compile is in progress and still healthy. I’m keeping the baseline run alive to completion for a clean A/B anchor before any edits.

Shard loading has started; I’m continuing to poll through build and timed decode so we can lock the iteration-4 baseline numbers.

Progress is normal (`1/16` shards loaded). I’ll keep this single run uninterrupted and extract the final `YT_METRICS` line when it finishes.

No new stdout in this window, but the process remains active. I’m continuing to wait for completion rather than restarting to preserve comparability.

Load is advancing steadily (`3/16`). I’m staying on this run until we get the timing payload, then I’ll start the first decode-path patch.

Still healthy and on expected timing (`4/16` shards). I’m continuing to collect a clean baseline before implementing changes.

Loading continues as expected (`5/16`). I’ll keep polling and report once we transition out of model load into timed execution.

Shard load has reached `7/16` and remains stable. I’m continuing this single baseline run through completion.

No new lines in this polling window, but the process is still active; likely between shard updates. I’m continuing to wait for final metrics.

Load progressed to `9/16`; timing is a bit variable but still healthy. I’m keeping it running until we get the baseline `YT_METRICS`.

Still in-flight with no new stdout this interval, which is normal in late load/convert phases. I’m polling until completion.

Progress resumed (`11/16`), so the baseline run is healthy. I’ll keep this pass going through compile and timed decode.

Checkpoint load is now `12/16`; still no runtime issues. I’m continuing until the metrics payload is emitted.

Load has reached `13/16`. I’m keeping this run uninterrupted so baseline and post-patch runs are directly comparable.

The run is in the late shard-load stage (`14/16`) and still healthy. I’ll keep polling until we get the final throughput line.

Checkpoint load is complete and the run is now in conversion/build. I’m waiting for the timed decode section and final `YT_METRICS`.

No new stdout yet in this build phase, which is typical for this model. I’m continuing to poll to completion and will extract the metrics line directly.

Still active with no intermediate logs after conversion, so it’s likely in compile/warmup or timed generation. I’m keeping it alive and polling for the final payload.

We’re now in TT kernel build/warmup (expected warning spam). I’m staying on this run; metrics should appear after this phase.

I’m now patching `optimized/model.py` with the first iteration-4 change set: device-side next-token argmax (`next_token_device`) and a decode LM-head fast path that projects only one token in decode. This keeps behavior intact while removing avoidable decode overhead in `demo.py`.

The first part of the patch is in. I’m now replacing the `forward` path to reuse the new device-logits helper and adding `next_token_device` so `demo.py` can run greedy decode without per-step host logits transfer.

Patch is applied and syntax-check passed. I’m running the same 16-token demo benchmark now to measure the direct A/B impact of device-side token sampling plus decode LM-head trimming.

Experiment run is in startup and proceeding normally. I’m waiting for the post-patch `YT_METRICS` line to compare against the 4.8037 t/s/u baseline.

Run is now in checkpoint loading. I’ll keep it uninterrupted and report the full A/B metrics once timed decode finishes.

Progress is tracking baseline timing (`2/16` shards). I’m continuing to completion for a clean comparison.

Shard load is advancing normally (`4/16`). I’m staying on this run until we get the throughput payload.

Load has reached `6/16` and remains healthy. I’m continuing to collect this A/B datapoint end-to-end.

Checkpoint loading is on pace (`8/16`). I’ll keep this run alive and report the exact decode throughput delta as soon as metrics print.

Load reached `10/16`; timing is a bit variable but still consistent with prior runs. I’m continuing to completion.

Progress is healthy at `11/16`; I’m preserving this run for a reliable post-patch measurement.

Loading has advanced to `12/16`; still no errors. I’m continuing through build/warmup to final metrics.

The run is in late shard load (`13/16`) and remains stable. I’m continuing until we get the timing payload.

Checkpoint load is at `15/16` and still running. I’m staying on this process for the final post-patch metrics.

Shard loading is finished and we’re in conversion/build again. I’m waiting for the timed run and `YT_METRICS` line.

We’re in kernel build/warmup now, matching baseline behavior. I’m continuing to the final metrics line for the A/B decision.

First iteration-4 patch is a clear win on the 16-token benchmark: `TTFT 2612.58 -> 2457.40 ms`, `decode 4.8037 -> 5.3328 t/s/u`. I’m now applying the second change (decode-only `SparseMoE` fast path that avoids transferring tile-padded tokens to host) and rerunning the same benchmark.

The decode-only `SparseMoE` fast-path patch is in and syntax-clean. I’m running the same 16-token benchmark now to see whether it adds throughput on top of the first win.

Experiment run is active and in startup/load. I’m keeping it uninterrupted for a clean A/B datapoint.

Checkpoint load has started (`1/16`) and is on baseline pace. I’ll continue polling to final metrics.

Still healthy at `3/16` shards. I’m keeping this run uninterrupted for direct comparison against the prior experiment.

Load reached `5/16`; run remains stable. I’ll continue to completion and then decide keep/revert based on throughput delta.

Progress is on target (`7/16`), no runtime issues. I’m continuing to gather the post-patch measurement.

Load is now `9/16`; timing variability is similar to prior runs. I’m still on track to a clean A/B result.

Progress has reached `10/16`. I’m continuing through load/build for the final throughput readout.

Run remains healthy at `11/16`. I’m keeping it in flight for the definitive metric line.

Checkpoint loading is now `13/16`; no issues so far. I’m continuing to completion.

Load reached `14/16` and is still stable. I’m waiting for final metrics.

Checkpoint load is at `15/16`; I’m keeping this run alive to completion for the comparison result.

Loading is done and the run is now in conversion/build. I’m polling for the final `YT_METRICS` output.

No intermediate output in this phase yet; likely still in model conversion before layer init logging. I’m continuing to poll and will report once it transitions.

The run has moved into kernel build/warmup. I’m holding for the metrics line now.

This second patch regressed (`TTFT 3507.82 ms`, `decode 4.7711 t/s/u`), so I’m reverting the decode-only `SparseMoE` fast path and keeping only the first successful change set. Then I’ll validate with full `demo.py` and long `eval.py`.

I’ve reverted the regressing MoE fast path and retained only the winning decode changes. Next I’m running the full `demo.py` benchmark (128-token default) to get the official figure-of-merit comparison.

Full benchmark is now in startup/load. I’ll keep this run uninterrupted and capture the final TTFT and decode t/s/u payload.

Checkpoint loading is underway (`1/16`) and tracking expected timing. I’m staying with this full run to completion.

Load progressed to `3/16`; run remains healthy. I’ll keep polling until we hit the final timing payload.

Checkpoint load is now `5/16` and stable. I’m keeping this pass untouched for an apples-to-apples full metric.

Load has reached `8/16` and is still on expected cadence. I’ll continue to final full-run metrics.

Progress reached `9/16`. I’m still on track and continuing the full benchmark run.

Loading is at `10/16`; run remains healthy. I’m continuing to completion.

Still active at `11/16`, with no errors. I’m preserving this run for the definitive full-demo numbers.

Loading has reached `13/16`; I’m continuing straight through to timed decode metrics.

Progress is now `14/16`; still healthy. I’ll keep polling until we get the full-run metrics line.

Checkpoint load is at `15/16` and still active. I’m staying on this run for the final full-demo readout.

Loading is complete and we’re in conversion/build for the full run. I’m polling until `YT_METRICS` is emitted.

We’re in kernel build/warmup for the full run now. I’m holding for final timing output.

The first full run with the new decode path came back worse than the current best (`TTFT 5376.87 ms`, `decode 4.2524 t/s/u`), which conflicts with the short-run win. I’m rerunning full `demo.py` once more to check whether this is variance or a real regression before deciding keep/revert.

Full-run confirmation is in startup/load now. I’ll keep this second run uninterrupted and compare directly against both prior full metrics.

Load is progressing normally (`1/16`). I’ll continue polling to final metrics for this confirmation run.

Progress is healthy at `3/16`; I’m keeping this run intact for a reliable full-demo comparison.

Checkpoint load has reached `5/16` and is still stable. I’m continuing to the final throughput output.

Loading is at `8/16`, no issues so far. I’m continuing this full-run check.

Progress reached `9/16`; run remains healthy and comparable to prior attempts.

Load is now `10/16`; I’m continuing to completion for the confirmation metrics.

The run is healthy at `12/16`; I’m continuing through to the final timing line.

Checkpoint load is at `13/16` and still stable. I’m continuing to the end of this confirmation run.

Load has reached `14/16`; run remains healthy. I’m polling for completion.

Checkpoint load is now `15/16`; I’m staying on this run through final metrics so we can decide keep/revert confidently.

Load is finished and conversion/build is underway. I’m polling until the final `YT_METRICS` line appears.

No intermediate output yet in conversion; I’m continuing to wait for the layer-init and timing stage.

The run has entered kernel build/warmup now. I’m waiting for the final full-demo metrics line.

The repeated full runs confirm this patch set is a regression in the real figure of merit, so I’m not keeping it as-is. Next I’m taking a narrower variant: keep the decode LM-head trimming in `forward`, but remove `next_token_device` so `demo.py` returns to its prior sampling path; then I’ll benchmark that directly.

I’ve removed `next_token_device` and kept only the forward decode LM-head trim. I’m running a 16-token benchmark now to see if this narrower change improves real throughput without the regression we saw.

The variant benchmark is in startup/load and running cleanly. I’ll report the A/B metrics once it completes.

Checkpoint load has started (`1/16`) and timing looks normal. I’m continuing this run through to the metric output.

Load has advanced to `3/16` and remains healthy. I’ll keep this run uninterrupted for a clean result.

Progress is at `6/16`, still on expected pace. I’m continuing to completion for the variant measurement.

Checkpoint loading has reached `8/16`; run remains stable. I’ll keep polling for final metrics.

Load is now `9/16`; still healthy. I’m continuing this run through build and timed decode.

Progress is `10/16` with no issues. I’m continuing to the final throughput line.

Load has reached `12/16`; run remains stable. I’m continuing to completion.

Checkpoint load is at `13/16`, still clean. I’m continuing to final metrics.

Load progressed to `14/16`; I’m continuing this run through completion for the variant decision.

Checkpoint loading is now `15/16`; run remains active. I’m holding for the final metrics line.

Loading is complete and conversion/build is running now. I’m polling until we get `YT_METRICS`.

The run is in kernel build/warmup. I’m waiting for final timing output.

I’ve fully reverted the experimental decode changes and restored the prior implementation shape. I’m running one sanity 16-token demo now to confirm we’re back to the expected baseline behavior before I finalize docs/results.

Sanity run is in startup/load and proceeding normally. I’ll confirm the restored throughput once `YT_METRICS` prints.

Shard loading is underway (`1/16`) and looks healthy. I’m continuing to the final sanity metrics.

Progress has reached `3/16`; run is still healthy and comparable to prior baselines.

Checkpoint loading is at `6/16` and stable. I’m staying on this run until we get the sanity `YT_METRICS` line.

Load is now `8/16` and still healthy. I’m continuing for the final sanity datapoint.

Progress reached `9/16`; this run is still on expected behavior. I’m continuing to completion.

Loading is at `11/16` and stable. I’m keeping this run going for the sanity confirmation.

Checkpoint loading has reached `12/16`; still healthy. I’m continuing to completion.

Load is now `13/16`; I’m continuing through to final metrics.

Checkpoint load has reached `15/16` and remains stable. I’m staying on this run until metrics are emitted.

Load is complete and the run is in conversion/build. I’m polling for the final `YT_METRICS` output.

Kernel build/warmup is active now; final sanity metrics should follow after this phase.

I have enough evidence for iteration 4: each new path that looked good on short decode regressed on full `demo.py`, so there’s no keepable improvement. I’m now updating `SCIENCE.md` with all iteration-4 experiments, outcomes, and non-obvious findings, while preserving the prior best accepted metrics.

Iteration 4 complete: no keepable speedup was found, and the prior best configuration remains the best stable point.

What I changed:
1. Ran a full iteration-4 experiment series and documented all outcomes in `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md:190`.
2. Tested and rejected:
- `next_token_device` + decode LM-head trim (`E10`): short-run looked better, full `demo.py` regressed.
- Additional decode-only SparseMoE fast path (`E11`): regressed.
- Forward-only decode LM-head trim variant (`E12`): regressed.
3. Reverted model code back to the prior stable path (no retained model optimization changes from this iteration).

Figure of merit status (unchanged best):
1. Demo TTFT: `2299.66 ms` (vs functional baseline `77689.76 ms` = `97.04%` lower).
2. Demo decode throughput: `4.8162 t/s/u` (vs functional baseline `0.4635 t/s/u` = `+939.09%`).
3. Long eval quality (best accepted config): Top-1 `0.9250`, Top-5 `1.0000` (meets `>=0.90` / `>=0.95`).

Notes:
- No new improvement beat the existing full-run best this iteration.
- I did not run a new long eval for rejected candidates since none survived full-demo gating.

## Iteration 5 - 2026-02-23T05:03:20Z

I’ll start by reviewing `SCIENCE.md` and `LOGBOOK.md` plus the current `models/Qwen/Qwen3-30B-A3B/n150/optimized` code to establish the current baseline and prior hypotheses, then I’ll pick the highest-leverage optimization path and validate it with `demo.py` and long-sequence `eval.py`. I’m using the `ttnn-model-optimization` workflow for this turn.

I’ve confirmed the target files and that the optimization workspace is currently untracked. Next I’m reading `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md` and top sections of `LOGBOOK.md` to anchor on the latest baseline and failed/suggested paths before changing code.

I’ve loaded the current `SCIENCE.md` history and started parsing `LOGBOOK.md`, but it’s large, so I’m now jumping directly to the latest iteration sections plus the optimization/eval skill docs to extract the exact open hypotheses and baseline numbers before touching code.

I’ve extracted the latest iteration history: iteration 4 confirmed short-run wins were false positives and reverted to the stable `decode_top_k=5` path. I’m now doing a focused read of `optimized/model.py` and `demo.py` to find a new, genuinely different optimization angle for iteration 5.

I found a new high-leverage path that hasn’t been tried yet: keep `decode_top_k=5` only for an initial warmup window of decode tokens, then drop to `top_k=4` for later decode steps. I’m patching `optimized/model.py` to support this as runtime-tunable knobs, then I’ll run long-eval first (quality gate) before full `demo.py` benchmarking.

The patch is in, and I spotted one compatibility issue to fix next: dense MLP layers also receive the same call signature, so I’m adding an optional `decode_step` arg there (ignored) to keep both sparse and dense paths consistent.

Experiment plan for iteration 5:

1. Add runtime-tunable decode MoE schedule
- Done in `optimized/model.py`: `decode_top_k=5` baseline retained, with optional env knobs to switch later decode tokens to lower `top_k`.
- This lets us sweep schedules without repeated code edits.

2. Quality-first screening
- Run long eval (`max_new_tokens=40`) with `QWEN3_DECODE_TOPK_LATE=4` and `QWEN3_DECODE_TOPK5_WARMUP_TOKENS=8`.
- If quality fails (`top1 < 0.90`), increase warmup window and rerun eval.

3. Throughput confirmation
- For the first passing schedule, run full `demo.py` (128-token default) and record TTFT/decode t/s/u.
- If it beats current best decode throughput, I’ll consider one follow-up schedule for incremental gain; otherwise I’ll revert.

4. Documentation
- Update `SCIENCE.md` with hypothesis, commands, exact metrics, and accepted/rejected decisions.
- Final report will include best absolute metrics and percentage deltas vs functional baseline and prior optimized best.

The first iteration-5 quality-gate eval is running now with the decode schedule (`top5` for first 8 decode steps, then `top4`). I’ll keep this single TT run uninterrupted and report the exact top-1/top-5 once the metrics line appears.

Startup is healthy and the eval process has begun loading; I’m now waiting through checkpoint/model build for the quality metrics payload.

The run is in the long checkpoint-loading/model-init phase with no errors so far. I’m continuing to poll until it reaches evaluation and emits `YT_METRICS`.

Checkpoint loading is progressing normally (`1/16`); I’ll keep this run uninterrupted and continue polling until final top-1/top-5 output.

Shard loading remains on the expected cadence and there are no runtime issues. I’m staying on this single run until completion.

Loading has reached `6/16`; I’m continuing to the end of this eval run to determine whether the new decode schedule clears the accuracy gate.

The run is still healthy and progressing through the slower late shards; I’m keeping it alive for a complete datapoint rather than restarting.

Checkpoint load is now in the final quarter (`12/16`). After this, I’ll wait through build/eval and report the exact top-1/top-5 numbers for this schedule.

Late-shard loading continues cleanly (`14/16`). I’m staying with this run through model build and teacher-forcing execution to avoid losing the datapoint.

Checkpoint loading finished and the run has moved into build/eval execution (quiet phase). I’m polling for the final `YT_METRICS` line now.

The process is still active (`eval.py` PID 251636), so this is an in-flight compute phase rather than a hang. I’m continuing to wait for completion and metrics.

Still in a silent compute segment post-load, which is typical for this eval path. I’m keeping it running and polling until it exits or emits `YT_METRICS`.

I verified the process is still consuming high CPU, so it’s progressing through evaluation even though stdout is quiet right now. I’m staying on it until we get the final accuracy line.

This eval is running longer than previous ones, but the process remains active and compute-heavy, so I’m continuing rather than interrupting and losing the run. I’ll report immediately once it returns the metrics payload.

No stdout yet, but the eval worker is still active and heavily loaded. I’m continuing to let it run to completion because interrupting now would discard a long quality-gate run.

The run has progressed into TT device bringup and model construction now, so it’s moving normally again. I’m continuing to final completion for the quality result.

Model build/eval is in another quiet section (likely cached-kernel path). I’m polling until `YT_METRICS` prints.

Process is still compute-active (CPU remains high), so this is progressing despite no intermediate logging. I’m continuing to wait for the final accuracy payload.

Kernel build/warmup logs have appeared, so we’re in the final stretch before metrics. I’m polling for the eval result now.

Quality gate passed for this schedule: `top1=0.925`, `top5=1.000` on 40-token long eval. Next I’m running full `demo.py` with the same env knobs to see whether decode throughput improves enough to keep it.

Full benchmark run is now in progress with the passing schedule (`warmup=8`, then `top4`). I’ll keep this single TT process uninterrupted and report TTFT/decode metrics when it completes.

The demo run is healthy and in checkpoint/model setup now. I’m waiting through this phase, then I’ll capture the `YT_METRICS` throughput numbers.

Checkpoint loading is progressing on expected cadence (`2/16`). I’ll continue polling until final timing metrics are available.

Load is steady (`7/16`) and the run remains clean. I’m continuing through the full benchmark for an apples-to-apples throughput figure.

The run is now in the slower late-shard region (`11/16`) but still healthy. I’ll keep it running to completion and capture TTFT/decode t/s/u.

Late checkpoint loading continues (`14/16`) with no issues. I’m holding this run through build and timed decode for the definitive figure of merit.

Checkpoint loading has finished and the run is now in model build/warmup; I’m polling for the final `YT_METRICS` payload.

Kernel build/warmup logs are now flowing, so this run is in the final execution phase. I’m waiting for the benchmark metrics line.

This first full-demo datapoint is a clear regression (`TTFT ~24.7s`, `decode ~4.03`), which conflicts with the expected top-4 decode effect, so I’m running one immediate repeat to separate true regression from run-to-run noise before deciding to keep or reject this schedule.

Repeat benchmark is underway with the same settings to verify whether the prior regression was noise or real. I’ll compare this run directly against the first result before deciding next steps.

Repeat run is progressing normally through checkpoint loading (`3/16`). I’m continuing to completion for a stability check on the throughput numbers.

The repeat run remains healthy and has entered the slower late-shard stage (`9/16`). I’m continuing to the final metrics line for a clean A/B confirmation.

Progress is still steady (`13/16`) with no runtime issues. I’m staying on this run for the final repeated metric check.

Checkpoint loading is complete and the repeat run is in build/warmup now. I’m polling for the final `YT_METRICS` output to confirm whether the regression is repeatable.

I’ve confirmed the repeated full-demo run is still a regression (`TTFT ~20.2s`, `decode ~4.58`), so this schedule is not keepable. I’m now reverting the experimental scheduling hooks from `model.py` and documenting this as a rejected iteration-5 experiment in `SCIENCE.md`.

I’ve reverted the experimental scheduling code, kept the model back on the prior stable path, and recorded the full iteration-5 experiment (including repeat full-demo results and new knowledge) in `SCIENCE.md`. I’m doing a final sanity check of changed files before reporting the outcome.

Iteration 5 is complete. I ran a new experiment and documented it, but no keepable performance improvement was found.

**What I did**
1. Tested a new decode MoE schedule hypothesis: `top_k=5` for early decode tokens, then `top_k=4` later.
2. Ran long-sequence quality eval with that schedule:
- Top-1: `0.9250`
- Top-5: `1.0000`
3. Ran full `demo.py` twice with that same schedule:
- Run A: TTFT `24698.94 ms`, decode `4.0291 t/s/u`
- Run B: TTFT `20155.67 ms`, decode `4.5771 t/s/u`
4. Rejected the experiment and reverted model logic to the prior stable decode path.
5. Updated experiment history and non-obvious findings in `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md:249`.

**Files updated**
- `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md:249`
- `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py` (experiment was reverted; no retained optimization change)

**Figure of merit status**
- No new best this iteration.
- Current best remains:
  - TTFT: `2299.66 ms` (`97.04%` lower vs functional baseline `77689.76 ms`)
  - Decode: `4.8162 t/s/u` (`+939.09%` vs functional baseline `0.4635 t/s/u`)
  - Long eval: Top-1 `0.9250`, Top-5 `1.0000` (meets target)

## Iteration 6 - 2026-02-23T06:49:02Z

I’m going to start by reading `SCIENCE.md`, `LOGBOOK.md`, and the optimization skill guidance so I can build directly on iteration 6 context before touching code or running device workloads. Then I’ll choose the highest-leverage experiment path and execute it end-to-end with `demo.py`/`eval.py` measurements.

I found the key files. Next I’m extracting iteration-specific context from `LOGBOOK.md` plus the full `SCIENCE.md`, then I’ll lock onto one optimization hypothesis and run measurable experiments.

I’ve pulled the optimization workflow and prior history snapshot; now I’m reading the remainder of `SCIENCE.md` and the latest `LOGBOOK.md` sections to identify what iteration 6 should prioritize and what’s already been ruled out.

I now have iteration-5 context: no keepable improvement since `decode_top_k=5`. Next I’m inspecting the current `optimized/model.py` and git state to identify a genuinely new optimization path for iteration 6 before launching any TT runs.

I’ve confirmed the current model is the host-expert variant with `decode_top_k=5` and no retained iteration-5 changes. I’m now extracting the iteration-6 suggestions from `LOGBOOK.md` and profiling hot spots in `demo.py` to pick the next experiment that aligns with your tracing/overlap ideas.

I found an untested high-leverage target: decode currently allocates/transfers fresh token and position tensors every step. I’m now borrowing the proven persistent decode-buffer pattern from other optimized models and applying it here (without changing MoE math) for a clean throughput experiment.

I’m running a fresh 16-token baseline on the current kept model before any edits, so we have an apples-to-apples A/B for this iteration’s first patch.

Baseline run is in flight; I’ll keep polling through load/build and extract the exact `YT_METRICS` values as soon as it finishes.

Checkpoint/model load has started cleanly; I’ll continue polling until the benchmark emits its final timing metrics.

The run failed before benchmarking because `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root` in this environment is missing firmware headers (`risc_common.h`). I’m rerunning with default runtime root (and keeping only `TT_METAL_CACHE`) so we can proceed with measurements.

Baseline retry is running normally with the default runtime root; I’ll keep polling until the `YT_METRICS` line is produced.

Device bringup is now past startup and still healthy. I’m continuing this baseline run to completion for the exact 16-token throughput anchor.

Checkpoint load is progressing normally; I’m staying on this run and will pull the `YT_METRICS` output once timed decode finishes.

Still in the long shard-load phase (`3/16`) with no issues. I’ll let this complete so we have a clean baseline before patching decode buffer handling.

Run remains healthy and on expected cadence (`5/16`). I’ll keep it uninterrupted and capture the baseline metrics at completion.

Load is now at `7/16`; no device/runtime issues. I’m continuing until we have the final throughput payload.

Still progressing (now `9/16`), with a slightly slower shard interval than earlier but no errors. I’m staying with this run for a complete baseline datapoint.

Baseline load is continuing (`11/16`) with no failure signals. I’ll keep it running to completion since this data point anchors the iteration’s A/B comparisons.

This run is still healthy but checkpoint loading is slower than usual (`13/16`). I’ll finish it once for baseline capture, then move to code edits.

Checkpoint loading has completed; it’s now in model conversion/build. I’ll keep polling until final timings are emitted.

I’m now patching `optimized/model.py` to add persistent decode token/position buffers and a decode-specific forward path that reuses those buffers instead of allocating fresh device tensors every step. This keeps model math unchanged and targets pure per-step overhead.

Patch is in and syntax-clean. I’m running the same 16-token `demo.py` benchmark now to test whether persistent decode buffers reduce decode overhead in practice.

Experiment run has started cleanly; I’m monitoring through load/build and will report the exact throughput metrics once complete.

This experiment is proceeding at normal shard-load speed so far. I’ll keep it running and extract metrics as soon as decode timing finishes.

Load remains stable (`4/16`) and matching typical cadence. I’m continuing to completion for this A/B result.

Checkpoint load is now `7/16` and still healthy. I’ll stay on this run through the final metrics line.

The run is still progressing (`10/16`) with the same late-shard slowdown seen earlier. I’ll let it finish once and then proceed with whichever datapoint we get.

Checkpoint load is at `13/16`; still active and clean. I’m staying on this process through final decode timings.

Checkpoint load finished and conversion/build has started. I’m polling now for either final metrics or a runtime failure signal.

Model build has progressed into kernel compilation now, so we’re past the prior silent phase. I’m staying on this run to get the actual timing output.

First result is promising: 16-token run reached `TTFT 2551.82 ms`, `decode 5.3741 t/s/u` (better than prior short-run references). I’m now running the full 128-token `demo.py` benchmark to verify the real figure of merit before deciding keep/revert.

Full benchmark is in progress. I’ll monitor this run through completion and compare TTFT/decode against the current best full-run baseline (`2299.66 ms`, `4.8162 t/s/u`).

Checkpoint loading has started and is on normal timing. I’ll continue polling until the full-run metrics line appears.

Run remains healthy (`4/16`) with stable cadence. I’m continuing straight through to the final throughput result.

Checkpoint loading is now `7/16`, still clean. I’ll keep this full run going for a definitive comparison point.

The run is in the slower late-shard phase (`10/16`) but still progressing normally. I’ll continue to the final `YT_METRICS` output.

Checkpoint loading reached `13/16`; still healthy. I’m keeping this run uninterrupted through completion.

Checkpoint load completed; we’re in conversion/build again. I’ll stay on it until final metrics print.

Kernel build is active now and progressing, so we should get timing output shortly after this phase.

Full run completed with a decode gain (`5.0318 t/s/u`) but higher TTFT (`2729.77 ms`) than the prior best, so I’m running one immediate repeat to check stability before finalizing keep/revert.

Repeat full benchmark is running; I’ll capture the second full-run TTFT/decode numbers to determine whether this patch is a robust improvement.

Repeat run is tracking normal load cadence so far. I’ll continue until final metrics and then decide whether this change is stable enough to keep.

Repeat run remains stable and is now at `6/16`. I’m continuing through to the final metrics line.

The repeat run is in late-shard load (`10/16`) with expected slowdown. I’ll keep it running until we have final confirmatory metrics.

Still healthy at `13/16`. I’m finishing this repeat run to get confidence on the full-demo numbers before quality eval.

Repeat run is in the post-load conversion/build phase again. I’m waiting for either the metrics payload or a failure signal before moving on.

I have mixed evidence so far (one strong full-run win, one full-run regression). I’m making one targeted refinement to this patch: explicitly deallocating temporary host tensors used for decode buffer updates, then I’ll rerun benchmarks to see if that stabilizes the result.

I’ve applied the refinement and syntax-checked. Running a quick 16-token benchmark now to verify behavior and speed before another full run.

This validation run is in progress and healthy so far; I’ll capture the 16-token metrics once it completes.

This rerun is following the usual cadence so far. I’ll keep it going to get the post-refinement metrics.

Load remains steady (`6/16`) with no issues. I’m continuing until the 16-token metrics are available.

The validation run is now in late-shard load (`10/16`) and still healthy. I’ll keep it running through completion to get this datapoint.

Still active and progressing (`13/16`). I’ll complete this run and then make a keep/revert call based on all collected metrics.

This run is in the post-load build phase now; I’m waiting for the final metrics payload to close out this experiment.

I’ve reverted `model.py` back to the prior stable implementation (`decode_top_k=5`) because the new decode-buffer branch wasn’t reproducibly better on full demo. Next I’m documenting all iteration-6 experiments and non-obvious findings in `SCIENCE.md`.

Iteration 6 complete. I explored one main new path, measured it, and reverted because it was not reproducibly better on full `demo.py`.

**What I tried**
1. `E14` in `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md:276`  
   Persistent decode token/position buffers + decode-specialized forward path.
2. `E15` in `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md:300`  
   Added explicit deallocation of temporary host tensors in that decode-buffer path.

**Key measurements**
1. Best transient full-demo run (E14):  
   `TTFT 2729.77 ms`, `decode 5.0318 t/s/u`  
   Decode improvement: `+4.48%` vs prior best `4.8162 t/s/u`  
   But TTFT regression: `+18.70%` vs prior best `2299.66 ms`
2. Repeat full-demo run (same E14 config):  
   `TTFT 3777.87 ms`, `decode 4.6601 t/s/u`  
   Decode dropped below prior best (`-3.24%`), TTFT worsened heavily (`+64.28%`).
3. E15 short run regressed clearly:  
   `TTFT 3480.60 ms`, `decode 4.0843 t/s/u`.

**Decision**
1. Rejected E14/E15 and reverted `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py` to the prior stable implementation.
2. No new accepted model-code optimization this iteration.
3. Updated experiment log + knowledge in `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md:276`.

**Current accepted best (unchanged)**
1. Demo: `TTFT 2299.66 ms`, `decode 4.8162 t/s/u`
2. Long eval: `Top-1 0.9250`, `Top-5 1.0000`
3. Versus functional baseline (`TTFT 77689.76 ms`, `decode 0.4635 t/s/u`):  
   TTFT `-97.04%`, decode `+939.09%`

I did not rerun long `eval.py` this iteration because no model math change was retained after revert.

## Iteration 7 - 2026-02-23T09:49:58Z

I’ll start by reading `SCIENCE.md` and `LOGBOOK.md` to align with prior iterations, then inspect the current `models/Qwen/Qwen3-30B-A3B/n150/optimized` path and benchmark/eval scripts before choosing the highest-leverage optimization experiment.

I’m using `ttnn-model-optimization` first (plus `ttnn-model-eval` for validation) since this is a performance/correctness bringup loop; next I’ll read those skill docs and then the current `SCIENCE.md`/relevant `LOGBOOK.md` sections.

I’ve reviewed `SCIENCE.md` and now I’m extracting the `LOGBOOK.md` entries specific to `Qwen3-30B-A3B/n150/optimized` and iteration 7 so I can choose the next experiment based on what’s already been tried.

I’ve extracted the prior iteration history: stable best remains `decode_top_k=5`, and several short-run wins failed on full 128-token stability. Next I’m doing a focused code read of `optimized/model.py` plus `demo.py` timing flow to identify a genuinely new path with higher leverage.

I’ve finished code reconnaissance. I’m now reading the latest `LOGBOOK.md` iteration notes around the end of iteration 6 to capture any unresolved hypotheses for iteration 7 before I pick and run the first experiment.

I’ve pulled the latest logbook context; there are no unresolved must-dos beyond trying a genuinely new direction. I’m now mining other optimized models for proven `use_decode_trace`/staged-trace patterns we can adapt to this MoE setup.

Plan for iteration 7: establish a fresh baseline, run one high-leverage experiment (`prefill` MoE top-k reduction with decode kept at top-5), then keep or revert strictly based on full `demo.py` plus long `eval.py`. I’m starting baseline measurements now.

Baseline short demo is running cleanly and currently in model/device bringup; I’m keeping this uninterrupted to capture a fresh iteration-7 throughput anchor.

Checkpoint loading is proceeding on expected cadence; I’ll keep polling until the run finishes and then lock these baseline metrics.

Load is still steady (`3/16`) with no runtime issues. I’m letting this baseline finish fully so comparisons later stay clean.

Baseline run remains healthy; checkpoint loading is now at `6/16` and I’m continuing straight through to the final `YT_METRICS`.

The process is still active in a quiet compute phase after mid-load; I’m continuing to poll until metrics print.

Load has entered the known slower late-shard region (`10/16`), but the run is still progressing normally. I’ll keep it running to completion for the baseline metric capture.

Still healthy but slow in late-shard loading; I’m staying with this run because we need one fresh baseline datapoint in the current environment before editing.

Checkpoint loading finally completed after an unusually slow pass; it’s now in weight-conversion/build. I’ll continue until the metrics line lands, then I’ll decide whether to keep using this baseline or fall back to the prior recorded baseline.

I confirmed the demo process is still active and compute-heavy, so this is a long silent build/execute phase rather than a hang. I’m continuing to wait for completion.

Kernel build is now in progress and moving forward; once this phase finishes we should get the timing payload. I’m continuing to poll for final baseline metrics.

The first baseline run was a cold-cache outlier (`TTFT ~53.4s`, `decode ~1.88`) due full compile/load overhead, so I’m taking one immediate warm-cache baseline rerun before any edits to get a usable comparison point.

Second baseline run is tracking normal load behavior so far. I’ll let it finish once, then move straight into the first optimization patch.

Loading is steady (`6/16`) with no anomalies. I’m still targeting this as the final pre-edit baseline run before patching.

The run is now in late-shard loading (`10/16`) as expected. I’ll stay on it to completion and then begin the MoE prefill-topk patch immediately.

Still healthy and progressing through the slow tail (`14/16`). Once metrics are captured from this run, I’ll move straight into code edits.

I confirmed the run is still actively computing at high CPU despite silent stdout, so I’m continuing to wait for the final metrics instead of restarting.

I’m editing `optimized/model.py` now to decouple prefill and decode MoE routing counts via tiny env knobs (`QWEN3_PREFILL_TOPK`, optional `QWEN3_DECODE_TOPK`) so I can sweep prefill top-k without repeated code churn.

Patch is in and syntax-clean. I’m running the first experiment now with `QWEN3_PREFILL_TOPK=6` (decode remains top-5) to test TTFT impact before deciding whether to keep this direction.

Experiment run (`prefill_top_k=6`) is underway and healthy; I’m keeping this single run uninterrupted to capture short-demo TTFT/decode deltas versus the baseline.

The `prefill_top_k=6` run remains clean and is currently in normal shard-loading (`6/16`). I’ll report metrics as soon as decode timing completes.

This run has reached the slow late-shard section (`11/16`) but is still progressing normally. I’m continuing to completion for a clean measurement.

`prefill_top_k=6` run is still compute-active in a silent phase after conversion, not stalled. I’m staying on it until it emits the final timing payload.

First `prefill_top_k=6` datapoint was a major regression (`TTFT ~8.0s`, `decode ~1.50 t/s/u`), which is counterintuitive, so I’m running an immediate control rerun with default settings on the same patched code to separate true effect from run noise.

Control run is now in flight with default top-k settings. I’ll compare it directly against the `prefill_top_k=6` result before deciding whether to continue this path.

Control run is progressing normally through checkpoint load. I’m waiting for the final metrics so we can determine whether the `prefill_top_k=6` regression is real or just noise.

The control run is in the same slow late-shard phase (`11/16`). I’m continuing until completion so we can make a reliable keep/reject decision for this experiment branch.

Control run is still actively computing in the post-conversion silent phase. I’m continuing until we have the metrics line.

I’m switching to a faster in-process benchmark harness (single model/device load, multiple timed generations) so we can compare several MoE top-k settings scientifically without paying 12+ minutes of reload per datapoint. I’ll still validate the winner with official `demo.py`/`eval.py` before keeping anything.

The in-process harness has started and is now in initial model/device setup. Once this one-time load completes, we’ll get multiple top-k datapoints quickly from the same process.

Still in the one-time checkpoint load phase for the in-process harness (`4/16`). After this, all configuration sweeps will run without reloading.

Harness load is in the late-shard region (`11/16`) and still healthy. I’m staying with it; once this completes we’ll get multiple experiment numbers from one run.

The in-process benchmark is still active in the post-load compute phase; once it clears this point, it should emit all config results in one shot. I’m continuing to wait.

The in-process sweep indicates a strong candidate (`prefill_top_k=6`, `decode_top_k=5`). I’m now validating that candidate with the official `demo.py` full run, then I’ll gate it with long-sequence `eval.py`.

Full official `demo.py` validation for `prefill_top_k=6` is now running. I’ll capture TTFT/decode t/s/u and decide keep/revert from this run plus long eval.

This validation run is healthy and progressing on normal cadence (`6/16`). I’ll report metrics as soon as timed decode finishes.

The candidate full-demo run is in late-shard loading (`13/16`) and still healthy. I’m continuing through completion for the decisive metric.

The full-demo process is still actively computing after conversion (high CPU, no hang). I’m continuing to wait for the final `YT_METRICS` line.

Full demo result for `prefill_top_k=6` is mixed: TTFT improved to `1991 ms`, but decode dropped to `4.53 t/s/u`. I’m running long `eval.py` now to verify quality before deciding whether this is keepable.

Long eval for `prefill_top_k=6` is running cleanly through checkpoint load. I’ll report top-1/top-5 immediately after completion.

Eval remains healthy and is now in the late-shard phase (`13/16`). I’m continuing to completion for the quality gate result.

Eval is still active in a long post-load compute phase (process alive and busy). I’m continuing to wait for the final top-1/top-5 metrics output.

Still running and compute-active (`Rsl` state, high CPU), so I’m keeping this eval alive until it returns the final quality metrics.

Eval is taking much longer than typical and staying in a silent compute phase, but the process remains active, so I’m continuing rather than interrupting and losing the quality-gate run.

New logs just flushed showing the eval progressed into a fresh TT model/device build segment (not hung). I’m continuing to completion to get the actual top-1/top-5 output.

Eval is still in a long compute section with the process active and CPU-bound; I’m letting it continue since interrupting now would discard a long quality run.

`prefill_top_k=6` fails quality (`top1=0.875`), so I’m rejecting it. Next I’m testing the less aggressive `prefill_top_k=7` path, starting with full `demo.py` and then long eval if demo is promising.

`prefill_top_k=7` full-demo run has started cleanly. I’ll capture TTFT/decode first, then run quality eval if this looks competitive.

Run is healthy and still on normal cadence (`7/16`). I’m continuing through completion for the metric readout.

This `prefill_top_k=7` run is now in late-shard load (`13/16`) and still healthy. I’ll continue to the final `YT_METRICS` output.

`prefill_top_k=7` run is still compute-active after conversion; I’m waiting for the final metrics line before deciding whether to proceed to eval.

`prefill_top_k=7` is a clear regression on full demo (`TTFT ~20.5s`, `decode ~4.05`), so I’m rejecting that path. I’m reverting the experimental top-k override hooks to restore the prior stable implementation before final documentation.

I’m updating `SCIENCE.md` now with iteration-7 experiments (`prefill_top_k` sweep, official demo/eval validation, and rejection rationale) plus new non-obvious findings.

Completed iteration 7 with full experiment/revert cycle.

- Updated `models/Qwen/Qwen3-30B-A3B/n150/optimized/SCIENCE.md` with new experiments `E16` and `E17`, results, decisions, and new knowledge.
- Reverted `models/Qwen/Qwen3-30B-A3B/n150/optimized/model.py` to the prior stable implementation after experiments failed acceptance criteria.

**What I tested**
- `prefill_top_k` reduction while keeping decode at top-5.
- Candidate `(prefill=6, decode=5)`:
  - `demo.py` full: `TTFT 1991.34 ms`, `decode 4.5283 t/s/u`
  - `eval.py` long: `top1 0.8750`, `top5 1.0000` -> fails Top-1 gate.
- Candidate `(prefill=7, decode=5)`:
  - `demo.py` full: `TTFT 20477.87 ms`, `decode 4.0517 t/s/u` -> hard regression, rejected.

**Figure of merit improvements (accepted)**
- No new accepted best this iteration.

**Current accepted best (unchanged)**
- TTFT: `2299.66 ms` (`97.04%` lower vs functional baseline `77689.76 ms`)
- Decode throughput: `4.8162 t/s/u` (`+939.09%` vs functional baseline `0.4635 t/s/u`)
- Long eval quality: `Top-1 0.9250`, `Top-5 1.0000` (meets target)

If you want, next iteration I can run a focused two-stage trace prototype around the MoE boundary (attention-side trace only, host MoE outside trace) as the next distinct tack.

