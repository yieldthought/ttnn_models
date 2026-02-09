# MODEL_BRINGUP.md — Arcee-Spark (n300 functional)

## Overview
Minimal TTNN bringup of `arcee-ai/Arcee-Spark` (Qwen2 family) with 1D tensor parallel on N300.

## Deviations from n150
- QKV, gate, and up projections are column-parallel (width sharded) across the mesh.
- Output and down projections are row-parallel (height sharded) followed by all-reduce.
- KV cache is sharded across KV heads; LM head is sharded across vocab and gathered for logits.
- Mesh shape uses `1x2` for 1D tensor parallel.

## Max sequence length
- `max_seq_len` uses the HF `max_position_embeddings` value (32768). No reduction needed for demo/eval.

## Changes
- 2026-02-09: Switched the mesh shape to `1x2` to align with the default N300 mesh and other n300 models.

## Commands
```
python demo.py models/arcee-ai/Arcee-Spark/n300/functional/model.py
python eval.py models/arcee-ai/Arcee-Spark/n300/functional/model.py --model arcee-ai/Arcee-Spark --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100 --max_seq_len 32768
```
