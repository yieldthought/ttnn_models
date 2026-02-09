# MODEL_BRINGUP.md — AFM-4.5B (t3000 functional)

## Overview
This is a TTNN bringup of `arcee-ai/AFM-4.5B` on T3000 using 1D tensor parallel on a 2x4 mesh.

- Model code: `models/arcee-ai/AFM-4.5B/t3000/functional/model.py`
- Eval harness: `eval.py` (teacher forcing)

## Deviations from n150
- 1D tensor parallel: QKV and up projections are column-parallel; output and down projections are row-parallel with all-reduce.
- Head padding for 8-way TP: the attention projections pad to 40 heads / 8 KV heads (ratio 5) so heads shard evenly across the 8-device 2x4 mesh. The extra heads are zero-padded and do not affect outputs.
- Paged attention + paged KV cache are kept in line with the n150 reference.
- Logits are gathered across devices with a vocab concat mesh composer.

## Max sequence length
- HF `max_position_embeddings` is 65536 and the bringup uses `max_seq_len=65536`.

## Demo and eval commands
```
env TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto PYTHONUNBUFFERED=1 python -u demo.py models/arcee-ai/AFM-4.5B/t3000/functional/model.py
env TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto PYTHONUNBUFFERED=1 python -u eval.py models/arcee-ai/AFM-4.5B/t3000/functional/model.py --model arcee-ai/AFM-4.5B --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100 --max_seq_len 65536
```
