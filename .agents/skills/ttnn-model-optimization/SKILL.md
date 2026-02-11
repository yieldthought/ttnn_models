---
name: ttnn-model-optimization
description: Optimize TTNN model performance while preserving correctness. Use when profiling prefill/decode, selecting high-leverage changes, evaluating complexity vs gain, and documenting outcomes.
---

# TTNN Model Optimization

## Overview
Use this skill to improve TTNN model performance with the smallest clean change that gives a measurable win.
This is guidance, not a rigid checklist. Use judgment for the model, hardware limits, and evidence in front of you.
If this file conflicts with measured behavior or repo docs, trust measurements and primary docs.

## Goals
1. Preserve correctness first.
   - Keep eval parity and coherent generation while optimizing.
2. Prioritize likely big wins before low-impact tuning.
3. Prefer simpler implementations when gains are similar.
4. Leave an evidence trail for both wins and non-wins.

## Sources of truth
1. `doc/ttnn.md` is the primary reference for optimization techniques, commands, and current best practices.
2. Use model-local notes (`MODEL_BRINGUP.md`, `TODO.txt`, `LOGBOOK.md`) for hypotheses and experiment history.
3. Use related skills as needed:
   - `ttnn-model-eval` for measurement and reporting.
   - `ttnn-model-debug` for correctness regressions.
   - `ttnn-model-bringup` when implementation structure must change.

## Workflow
1. Define target metrics and constraints.
   - Identify what matters: prefill latency, decode tok/s, TTFT, memory headroom, compile time, or end-to-end user latency.
   - Fix the measurement setup (same prompt lengths, batch, runtime flags, warmup approach, and command).
2. Establish a clean baseline.
   - Run baseline commands and capture exact outputs before touching code.
   - Confirm baseline correctness with eval before performance work.
3. Pick the next candidate by expected leverage.
   - Start with changes likely to move major kernels or memory traffic.
   - Defer micro-tuning until major bottlenecks are addressed.
4. Apply one change at a time.
   - Keep patches small and reversible.
   - Re-measure immediately with the same command and setup.
5. Score value vs complexity.
   - Keep changes that show clear, repeatable benefit for acceptable code complexity.
   - Revert or skip changes that add substantial complexity for marginal gain.
6. Re-run correctness checks after each kept change.
   - If performance improves but correctness regresses, treat it as a failure and debug first.
7. Document outcomes.
   - Record exact command, before/after metrics, confidence, and caveats.
   - Preserve failed/neutral experiments briefly so future optimization avoids repeated dead ends.

## Decision guidance (use judgment)
1. A change is usually worth it when it has a clear measurable gain, low maintenance burden, and no correctness risk.
2. A change is usually not worth it when it materially complicates model code but does not move key metrics.
3. For borderline cases, prefer the simpler path unless there is a strong reason to optimize for a specific deployment target.

## Exit criteria
1. Key target metric(s) improved versus baseline with reproducible measurement.
2. Eval quality remains acceptable for the model bringup bar.
3. The retained optimization set is coherent and maintainable.
4. Results and rationale are written down in repo docs/logs so others can reproduce them.
5. Optimization work continues until all currently worthwhile-looking candidates are evaluated, not just the first successful step.
