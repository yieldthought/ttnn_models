# Sequence Length Notes (n150 functional models)

This doc records sequence-length limits and a repeatable method to raise them on n150.
The biggest limiter is KV cache layout: the default cache `[32, n_kv_heads, max_seq_len, head_dim]`
incurs a 32x batch tile tax. Paged KV cache removes that and enables full model lengths.

## What changed for Llama-3.2-1B (n150 functional)

- **Non-paged cache (legacy):**
  - Sharded `fill_cache` avoids interleaved grid-size limits.
  - DRAM limit hit at `max_seq_len=9344` (OOM at 9376+).
- **Paged cache (current):**
  - Cache shape `[max_num_blocks, n_kv_heads, block_size, head_dim]` with `block_size=64`.
  - Page table `[32, max_num_blocks]` (tile-aligned batch dim) with identity block mapping.
  - Supports full HF `max_position_embeddings=131072` on n150.
  - Validation: 1-token teacher-forcing succeeds at `--max_seq_len 131072`.

## Paged KV cache recipe (recommended)

1. Pick `block_size` (multiple of 32). Use 64 by default.
2. `max_num_blocks = ceil(max_seq_len / block_size)`.
3. Allocate K/V cache as `[max_num_blocks, n_kv_heads, block_size, head_dim]`.
4. Create a page table (identity mapping) with tile batch:
   - Shape `[32, max_num_blocks]`, dtype `int32`, layout `ROW_MAJOR`.
5. Prefill: use paged fill.
6. Decode: use paged update + paged SDPA decode.
7. Decode positions: set `cur_pos_tensor` to `-1` for padded batch entries so unused slots are skipped.

Code sketch:

```python
block_size = 64
max_num_blocks = math.ceil(max_seq_len / block_size)

page_table = torch.arange(max_num_blocks, dtype=torch.int32).repeat(32, 1)
page_table = ttnn.as_tensor(
    page_table,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=tt_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Prefill
ttnn.experimental.paged_fill_cache(k_cache, k, page_table, batch_idx=0)
ttnn.experimental.paged_fill_cache(v_cache, v, page_table, batch_idx=0)

# Decode
ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=cur_pos_tensor, page_table=page_table)
ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=cur_pos_tensor, page_table=page_table)
attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q, k_cache, v_cache, page_table_tensor=page_table, cur_pos_tensor=cur_pos_tensor
)
```

### Memory math (paged cache)

Per-token bytes (batch=1):
`n_kv_heads * head_dim * 2 (K+V) * 2 bytes * num_layers`

Llama-3.2-1B:
`8 * 64 * 2 * 2 * 16 = 32768 bytes` (~32 KB/token)  
`131072 tokens ~ 4.0 GiB` of KV cache.

## Non-paged cache grid limits (legacy)

If you keep the default cache layout, you still need sharded K/V for `fill_cache`:

```python
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
```

## How to sweep max sequence length

`scripts/run_eval.py` supports max-seq sweeps:

```bash
python scripts/run_eval.py --mode tt --hf-model meta-llama/Llama-3.2-1B \
  --prefill-len 128 --decode-len 1 --max-seq-len 131072 --force-prefill
```

```bash
python scripts/run_eval.py --mode tt --hf-model meta-llama/Llama-3.2-1B \
  --prefill-len 128 --decode-len 1 --max-seq-len-range 8192:131072:8192 --force-prefill
```

Prefill sweeps still work via `--prefill-len-range`.

## Runtime gotchas

- Run exactly one TT process at a time on n150.
- After a hang or OOM, reset the device: `tt-smi -r`.
- If you see `cannot map elf file into memory: No space left on device`, rerun with:
  - `TT_METAL_CACHE=/tmp/tt-metal-cache`
  - `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root`
