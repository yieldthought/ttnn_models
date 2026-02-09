# MODEL_BRINGUP.md — AFM-4.5B (n300 functional)

## Overview
This is the N300 1D tensor-parallel bringup of `arcee-ai/AFM-4.5B`.
It mirrors the n150 architecture (Yarn RoPE + relu2) with sharded weights and
paged KV cache for long context.

- Model code: `models/arcee-ai/AFM-4.5B/n300/functional/model.py`
- Eval harness: `eval.py` (teacher forcing)
- Parallelism: 1x2 mesh, QKV/MLP-up column-parallel; output/MLP-down row-parallel + all-reduce

## Deviations from n150
- Added 1D tensor parallel across a 1x2 mesh (weight sharding + all-reduce).
- KV cache is paged as in n150, but KV heads are sharded across the mesh.
- LM head vocab is padded to a multiple of the mesh size (128005 -> 128006),
  and logits are trimmed back to the original vocab size.
- On this system, mesh auto-discovery reported a 2x1 shape; use the N300
  mesh graph descriptor to open a 1x2 mesh.

## Max sequence length
- Uses the HF `max_position_embeddings`: 65536.

## Commands
Demo (with mesh graph descriptor):
```
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
  python demo.py models/arcee-ai/AFM-4.5B/n300/functional/model.py
```

Eval (teacher-forcing, 100 new tokens):
```
TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
  python eval.py models/arcee-ai/AFM-4.5B/n300/functional/model.py \
  --model arcee-ai/AFM-4.5B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 65536
```
