# Sequence Length Notes (n150 functional models)

This doc records sequence-length limits and a repeatable method to raise them on n150.
The usual failure is `fill_cache` hitting a grid-size limit when inputs are interleaved.
Fix that first, then push max length until DRAM runs out.

## What changed for Llama-3.2-1B (n150 functional)

- Root cause: interleaved `fill_cache` uses one work block per KV-head tile.
  With 8 KV heads and 32x32 tiles, `seq_len=512` -> 16 tiles -> 8 * 16 = 128 blocks,
  which exceeds the 8x8 grid (64) and crashes.
- Fix: shard K/V before `fill_cache` (height-sharded, ROW_MAJOR).
- Resulting max `max_seq_len` on n150 (DRAM-limited): **9344**.
  - `max_seq_len=9344` builds and runs short prefill (128 tokens).
  - `max_seq_len=9376` builds but OOMs during prefill.
  - `max_seq_len=9408+` OOMs during model build or lm_head upload.

Summary sweep (Llama-3.2-1B, n150 functional):

| max_seq_len | Build | Prefill(128) | Result |
| --- | --- | --- | --- |
| 8192 | ok | ok | pass |
| 9216 | ok | ok | pass |
| 9344 | ok | ok | **max** |
| 9376 | ok | OOM | fail |
| 9408 | OOM | - | fail |
| 9472 | OOM | - | fail |
| 9600 | OOM | - | fail |
| 10240 | OOM | - | fail |
| 12288 | OOM | - | fail |
| 16384 | OOM | - | fail |

## Fixing `fill_cache` grid limits

Use sharded K/V for prefill:

```python
# Shard KV for fill_cache to avoid interleaved grid-size limits at long prefill lengths.
grid = tt_device.core_grid
grid_x = min(grid.x, n_kv_heads)
while grid_x > 1 and n_kv_heads % grid_x != 0:
    grid_x -= 1
shard_grid = ttnn.CoreGrid(x=grid_x, y=n_kv_heads // grid_x)
if shard_grid.y > grid.y:
    raise ValueError("shard grid exceeds device core grid")
shard_mem_config = ttnn.create_sharded_memory_config(
    k.shape,
    shard_grid,
    ttnn.ShardStrategy.HEIGHT,
    ttnn.ShardOrientation.ROW_MAJOR,
)
k_sharded = ttnn.to_memory_config(k, shard_mem_config)
v_sharded = ttnn.to_memory_config(v, shard_mem_config)

ttnn.fill_cache(k_cache, k_sharded, batch_idx=0)
ttnn.fill_cache(v_cache, v_sharded, batch_idx=0)
ttnn.deallocate(k_sharded)
ttnn.deallocate(v_sharded)
```

Notes:
- Height-sharded + ROW_MAJOR is required by `fill_cache`.
- Pick a shard grid that evenly divides `n_kv_heads` and fits within the device grid.

## How to measure max seq length for another model

1. Identify KV cache shape and head config.
   - Cache shape is usually `[32, n_kv_heads, max_seq_len, head_dim]` on n150.
2. Fix `fill_cache` if interleaved prefill hits the grid limit (see snippet above).
3. Find the DRAM limit with short prefill (fast, avoids O(S^2) prefill compute):
   - Build the TT model with a large `max_seq_len`.
   - Run a short prefill (e.g., 128 tokens) to ensure runtime buffers can still allocate.
   - If you see OOM during model build or prefill, lower `max_seq_len` and retry.
4. Optionally run a longer prefill (e.g., 4096) to validate long-seq behavior.

Tip: cache memory dominates. For Llama-3.2-1B (16 layers, 8 KV heads, head_dim 64),
KV cache bytes per token ~ 32 (batch tiles) * 8 * 64 * 2 (BF16) * 2 (K+V) * 16 layers
~ 0.5 MB per token. That makes 2048 tokens ~ 1 GB of cache.

## Eval tooling for sweeps

`scripts/run_eval.py` now supports prefill sweeps:

- Range sweep: `--prefill-len-range start:end[:step]`
- Force full prefill (avoid auto prefill_decode): `--force-prefill`

Example:

```bash
python scripts/run_eval.py --mode tt --hf-model meta-llama/Llama-3.2-1B \
  --prefill-len-range 512:4096:512 --decode-len 4 --force-prefill
```

## Runtime gotchas

- Run exactly one TT process at a time on n150.
- After a hang or OOM, reset the device: `tt-smi -r`.
- If you see `cannot map elf file into memory: No space left on device`, rerun with:
  - `TT_METAL_CACHE=/tmp/tt-metal-cache`
  - `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root`
