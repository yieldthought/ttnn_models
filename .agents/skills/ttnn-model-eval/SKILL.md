---
name: ttnn-model-eval
description: Evaluate TTNN bringup models in this repo. Use when running or interpreting eval.py or scripts/run_eval.py, reporting top1 or top5 accuracy, comparing prefill vs decode, or using --prefill_decode and YT_METRICS output.
---

# TTNN Model Eval

## Overview
Validate TTNN model outputs against the HF reference using the repo eval harnesses.

## Workflow
1. Pick the harness.
   - Use `eval.py` for quick teacher-forcing accuracy.
   - Use `scripts/run_eval.py` for CI-style YT_METRICS output.
2. Use the device safely.
   - Run exactly one TTNN/TT process at a time on this single Wormhole card.
   - Avoid parallel tool runs and stop any background TT processes before starting a new eval or reset.
3. Run a fast sanity check.
   - `python eval.py <model.py> --model <hf-id> --max_new_tokens 20`
4. Exercise decode behavior.
   - Use `--prefill_decode` only to debug when `fill_cache` fails; final bringup metrics must use the full prefill pass.
5. Keep runs small when iterating.
   - Reduce `--max_new_tokens` or prompt length first.
6. Capture results clearly.
   - Report top1 and top5 with the exact command only after the run completes; if interrupted, resume before updating MODELS.md.
7. If accuracy drops, switch to `ttnn-model-debug`.
