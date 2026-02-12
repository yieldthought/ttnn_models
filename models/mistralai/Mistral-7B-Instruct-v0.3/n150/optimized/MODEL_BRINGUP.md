# MODEL_BRINGUP.md - Mistral-7B-Instruct-v0.3 (n150 optimized)

## Overview
This is the optimized n150 path for `mistralai/Mistral-7B-Instruct-v0.3`.

- Model code: `models/mistralai/Mistral-7B-Instruct-v0.3/n150/optimized/model.py`
- Demo log: `models/mistralai/Mistral-7B-Instruct-v0.3/n150/optimized/demo.log`
- Eval log: `models/mistralai/Mistral-7B-Instruct-v0.3/n150/optimized/eval.log`
- Max seq len: `32768` (no capability regression)
- Decode path: traced execution (`ttnn.begin_trace_capture` / `ttnn.execute_trace`)

## Baseline vs Final (same hardware)
| Metric | Functional baseline | Optimized final |
| --- | ---: | ---: |
| Top-1 | 93% | 96% |
| Top-5 | 100% | 100% |
| TTFT | 105 ms | 90 ms |
| t/s/u | 16.5 | 17.9 |
| Seq len | 32768 | 32768 |

## Commands used
Demo:
```bash
python -u demo.py models/mistralai/Mistral-7B-Instruct-v0.3/n150/optimized/model.py \
  --prompt-file prompts/bringup_eval_long.txt \
  --max-new-tokens 100
```

Eval:
```bash
python -u eval.py models/mistralai/Mistral-7B-Instruct-v0.3/n150/optimized/model.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 32768
```

## Kept optimizations
1. Fused attention Q/K/V projections into one matmul (`self.qkv_proj`).
2. Added `prefill_logits_last_device()` so demo/eval TTFT uses only final prompt-token logits.
3. Decode trace with preallocated token/position/RoPE buffers and `ttnn.execute_trace` replay.
4. Decode RoPE runs from precomputed per-token cos/sin buffers (no `start_pos` argument inside the traced path).
   - For GQA, Q and K use separate RoPE buffers (`decode_cos_q_*` vs `decode_cos_k_*`) to match their different `(batch * heads)` widths.
5. Decode attention intermediates in L1 (`qkv` matmul, `nlp_create_qkv_heads_decode`, paged SDPA decode, decode `nlp_concat_heads`).
6. Decode LM head runs on a sliced `[1, 1, 1, hidden]` activation (avoid padded-32 LM head work).

## Deferred changes
1. Decode-only sharded matmul program-config tuning (dram-sharded decode MLP / o-proj) was deferred; the current traced + L1 path is the smallest change that gives a measurable win.
