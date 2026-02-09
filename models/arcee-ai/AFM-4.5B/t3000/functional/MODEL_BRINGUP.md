# MODEL_BRINGUP.md — AFM-4.5B (t3000 functional)

## Overview
This is a TTNN bringup of `arcee-ai/AFM-4.5B` on T3000 using 1D tensor parallel on a 2x4 mesh.

- Model code: `models/arcee-ai/AFM-4.5B/t3000/functional/model.py`
- Eval harness: `eval.py` (teacher forcing)

## Deviations from n150
- 1D tensor parallel: QKV and up projections are column-parallel; output and down projections are row-parallel with all-reduce.
- Head padding for 8-way TP: the attention projections pad to 40 heads / 8 KV heads (ratio 5) so heads shard evenly across the 8-device 2x4 mesh. The extra heads are zero-padded and do not affect outputs.
- KV cache is a standard (non-paged) cache sized to a bringup-friendly limit.
- Logits are gathered across devices with a vocab concat mesh composer.

## Max sequence length
- HF `max_position_embeddings` is 65536, but the T3000 bringup uses a capped cache length to fit memory.
- `MAX_CACHE_SEQ_LEN` is set to 256 and `cache_seq_len = min(max_seq_len, MAX_CACHE_SEQ_LEN)`.
- The eval harness enforces `--max_seq_len >= 2048`, so run eval with 2048 while keeping the internal cache capped at 256.
- If memory allows, increase `MAX_CACHE_SEQ_LEN` and rerun demo/eval.

## Demo and eval commands
```
python demo.py models/arcee-ai/AFM-4.5B/t3000/functional/model.py
python eval.py models/arcee-ai/AFM-4.5B/t3000/functional/model.py --model arcee-ai/AFM-4.5B --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100 --max_seq_len 2048
```
