# MODEL_BRINGUP.md — Arcee-Spark (n300 optimized)

## Overview
Optimized TTNN bringup of `arcee-ai/Arcee-Spark` (Qwen2 family) for N300.

- Model code: `models/arcee-ai/Arcee-Spark/n300/optimized/model.py`
- Eval harness: `eval.py` (teacher forcing) and `demo.py`

## Key optimizations vs functional
- Decode uses trace capture + replay (`use_decode_trace=True`) with preallocated device buffers for tokens, positions, and 1-token RoPE cos/sin.
- Decode slices the tile-padded batch down to a single logical token before the LM head.
- Adds `prefill_logits_last_device()` so demo/eval avoid transferring full prefill logits to host.

## Baseline (functional)
From `MODELS.md` (same prompt, same system):
- Top-1: 91%
- Top-5: 100%
- TTFT: 338ms
- t/s/u: 5.0
- Seq len: 32768

## Commands

Demo:
```
python demo.py models/arcee-ai/Arcee-Spark/n300/optimized/model.py
```

Long eval:
```
python eval.py models/arcee-ai/Arcee-Spark/n300/optimized/model.py \
  --model arcee-ai/Arcee-Spark \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 32768 \
  --seed 0
```

## Status (2026-02-12)
Runs are currently blocked on this host because fabric mesh discovery reports a 1x1 system mesh (no inter-chip links), so `open_tt_device((1, 2), ...)` fails.

- See `models/arcee-ai/Arcee-Spark/n300/optimized/demo.log` and `models/arcee-ai/Arcee-Spark/n300/optimized/eval.log` for the exact error.
- Once the system mesh reports 1x2 again, rerun demo/eval and update `MODELS.md` with the optimized row.
