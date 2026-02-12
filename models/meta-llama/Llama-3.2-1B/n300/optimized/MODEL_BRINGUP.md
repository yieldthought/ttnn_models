# MODEL_BRINGUP.md — Llama 3.2 1B (n300 optimized)

## Target
Optimize `meta-llama/Llama-3.2-1B` on N300 for better end-to-end demo performance while preserving teacher-forcing accuracy.

Release requirements (from issue):
- Long eval: Top-1 >= 85%, Top-5 >= 95%
- Decode uses traced execution
- TTFT is strictly lower than the functional baseline on the same hardware
- t/s/u is strictly higher than the functional baseline on the same hardware
- No capability regression: optimized Seq len >= functional Seq len

## Baseline (functional)
From `MODELS.md` functional row for `meta-llama/Llama-3.2-1B` on `n300`:
- Top-1: 90%
- Top-5: 100%
- TTFT: 610ms
- t/s/u: 6.7
- Seq len: 131072

## Optimized design
Implemented in `models/meta-llama/Llama-3.2-1B/n300/optimized/model.py`.

Changes kept:
- Paged KV cache (`block_size=64`) + paged SDPA decode.
- Fused QKV projection (single matmul per layer).
- Decode uses traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`) with preallocated decode buffers:
  token ids, `cur_pos_tensor`, and RoPE cos/sin slices.
- Prefill-last-logits fast path (`prefill_logits_last_device`) to avoid full-sequence logits transfers during TTFT measurement.
- BFP8 weights for matmuls (`ttnn.bfloat8_b`) where supported.

## Results
Final results (this optimized model):
- Top-1: 91%
- Top-5: 100%
- TTFT: 32ms
- t/s/u: 41.8
- Seq len: 131072

Decode notes:
- Decode uses traced execution (`use_decode_trace=True`, trace captured after prefill and replayed during the decode loop).

## Rejected changes
- Fused QKV projection: correctness regression in long eval (Top-1 dropped to ~83% in local runs), so we kept the separate `q_proj`/`k_proj`/`v_proj` path.

## Repro commands
Eval (teacher forcing, 100 tokens):
```
TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
python eval.py models/meta-llama/Llama-3.2-1B/n300/optimized/model.py --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100 --seed 0
```

Demo (timed):
```
TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
python demo.py models/meta-llama/Llama-3.2-1B/n300/optimized/model.py --prompt-file prompts/bringup_eval_long.txt --max-new-tokens 100 --seed 0
```
