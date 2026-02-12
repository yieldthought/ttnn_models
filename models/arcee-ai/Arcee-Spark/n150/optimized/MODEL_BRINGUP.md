# MODEL_BRINGUP.md — Arcee-Spark (n150 optimized)

## Overview
Optimized TTNN bringup of `arcee-ai/Arcee-Spark` (Qwen2 family) on Wormhole n150.

- Model code: `models/arcee-ai/Arcee-Spark/n150/optimized/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py`
- Directory convention: `models/<org>/<model_name>/<system>/optimized/model.py`

## Functional Baseline (n150)
From `MODELS.md` for `arcee-ai/Arcee-Spark` n150 functional:

- Top-1: 92%
- Top-5: 100%
- TTFT: 99ms
- t/s/u: 13.9
- Seq len: 29952

## Optimization Goals
- Decode uses traced execution.
- TTFT strictly lower than functional baseline.
- Decode throughput (t/s/u) strictly higher than functional baseline.
- No capability regression: max sequence length >= 29952.

## Kept Changes
- Decode trace + persistent device buffers for token ids, positions, and RoPE cos/sin.
- Prefill computes only last-token logits (`prefill_logits_last_device`).
- Fuse Q/K/V projections into a single matmul.
- Slice decode hidden-state down to 1 token before LM head to avoid a 32-lane padded LM head.

## Rejected Changes
- None yet.

## Commands
Teacher-forcing eval (100 tokens):

```
python eval.py models/arcee-ai/Arcee-Spark/n150/optimized/model.py --model arcee-ai/Arcee-Spark --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
```

Demo timing:

```
python demo.py models/arcee-ai/Arcee-Spark/n150/optimized/model.py
```

Max sequence length validation (prefill only, 1-token decode):

```
python scripts/run_eval.py --mode tt --hf-model arcee-ai/Arcee-Spark --system n150 --prefill-len 128 --decode-len 1 --max-seq-len 29952
```

## Results
TODO: Fill in final Top-1/Top-5/TTFT/t/s/u and confirm seq-len validation after runs.
