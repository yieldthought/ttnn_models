# MODEL_BRINGUP.md - Falcon3 7B Instruct (n150 optimized)

## Overview
Optimized TTNN implementation of `tiiuae/Falcon3-7B-Instruct` for n150.

- Model code: `models/tiiuae/Falcon3-7B-Instruct/n150/optimized/model.py`
- Demo log: `models/tiiuae/Falcon3-7B-Instruct/n150/optimized/demo.log`
- Eval log: `models/tiiuae/Falcon3-7B-Instruct/n150/optimized/eval.log`
- Machine-readable metrics: `models/tiiuae/Falcon3-7B-Instruct/n150/optimized/metrics.json`

## Baseline vs final (same hardware, same prompt setup)
| Metric | Functional baseline (`MODELS.md`) | Optimized final |
| --- | ---: | ---: |
| Top-1 | 97% | 97% |
| Top-5 | 100% | 100% |
| TTFT | 144 ms | 102 ms |
| t/s/u | 13.4 | 15.2 |
| Seq len | 32768 | 32768 |

Acceptance checks:
- Long eval quality: pass (`97% / 100%`)
- Decode trace: pass (`ttnn.begin_trace_capture` + `ttnn.execute_trace`)
- TTFT improved: pass (`101.75ms < 144ms`)
- t/s/u improved: pass (`15.17 > 13.4`)
- Seq len non-regression: pass (`32768 >= 32768`)

## Kept optimization decisions
1. Fused attention Q/K/V projections into one matmul (`self.qkv_proj`).
2. Added `prefill_logits_last_device()` so demo/eval TTFT uses only final prompt-token logits.
3. Kept traced decode with preallocated token/position/RoPE buffers (`ttnn.execute_trace` replay path).
4. Kept decode attention intermediates in L1 (`TTNN_USE_DECODE_L1_PATH=1` default).
5. Kept decode MLP SILU fusion via `ttnn.mul(..., input_tensor_a_activations=[SILU])`.
6. Kept decode LM head on sliced `[1, 1, 1, hidden]` activation to avoid padded-32 LM-head work.

## Tested alternatives (not kept)
1. `TTNN_USE_DECODE_L1_PATH=0`:
   - TTFT: `102.03ms`
   - t/s/u: `14.68`
   - Decision: rejected (lower decode throughput than default L1 path).
2. `TTNN_USE_DECODE_TRACE=0`:
   - TTFT: `101.12ms`
   - t/s/u: `15.11`
   - Decision: rejected (slightly lower throughput, and release requirement requires traced decode).

## Commands used
Demo:
```bash
python demo.py models/tiiuae/Falcon3-7B-Instruct/n150/optimized/model.py \
  --prompt-file prompts/bringup_eval_long.txt \
  --max-new-tokens 100 \
  --temperature 0 \
  --device-id 0 \
  --max_seq_len 32768
```

Eval:
```bash
python eval.py models/tiiuae/Falcon3-7B-Instruct/n150/optimized/model.py \
  --model tiiuae/Falcon3-7B-Instruct \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100 \
  --max_seq_len 32768 \
  --device_id 0 \
  --seed 0
```
