# TTNN Bringup Notes (tt-metal/ttnn)

Practical guidance for bringing up and optimizing HuggingFace LLMs on TTNN. This is not
a full API reference. It focuses on the ops and constraints we use in the Llama 3.2 1B
n150 optimized path and the failure modes we keep hitting.

## Quick bringup checklist

- Read the HuggingFace modeling file and config. Capture: hidden_size, num_attention_heads,
  num_key_value_heads, head_dim, intermediate_size, rope_scaling, and vocab_size.
- Inspect weight shapes (safetensors or state_dict) before wiring TTNN ops. Q/K/V shapes
  drive head_dim and GQA logic.
- Decide prefill vs decode flow early. The cache layout and decode shapes are strict.
- For long sequence lengths (and generally on n150), use a paged KV cache from day 1
  (block_size=64, page_table shape `[32, max_num_blocks]`).
- Keep `ttnn.from_torch` and `ttnn.as_tensor` outside trace capture. Avoid allocate/deallocate
  inside traces.
- Log both `shape` and `padded_shape` at each stage, plus `dtype`, `layout`, and memory config.
- Keep activations in TILE layout for matmul/attention; use ROW_MAJOR only where required.

## Shape conventions and tiling

- B: batch (decode batch is tile aligned, typically 32)
- S: sequence length (padded to tile)
- H: hidden size
- n_qh, n_kh: num_query_heads, num_kv_heads
- d: head_dim = H / n_qh
- `cur_pos_tensor`: `[32]` int32 for decode; use `-1` for inactive lanes so kernels skip them.

TTNN uses padded shapes due to 32x32 tiles. Use a `pad_to_tile()` helper and check
`tensor.padded_shape()` whenever an op refuses to run.

## Prefill / decode skeleton

```python
# Paged KV cache (recommended)
block_size = 64  # multiple of 32
max_num_blocks = math.ceil(max_seq_len / block_size)

page_table = torch.arange(max_num_blocks, dtype=torch.int32).repeat(32, 1)
page_table = ttnn.as_tensor(
    page_table,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Prefill (seq_len > 1)
qkv = ttnn.concat([q, k, v], dim=-1)
q, k, v = ttnn.experimental.nlp_create_qkv_heads(
    qkv, num_heads=n_qh, num_kv_heads=n_kh, transpose_k_heads=False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
q = ttnn.experimental.rotary_embedding(q, cos, sin)
k = ttnn.experimental.rotary_embedding(k, cos, sin)
ttnn.experimental.paged_fill_cache(k_cache, k, page_table, batch_idx=0)
ttnn.experimental.paged_fill_cache(v_cache, v, page_table, batch_idx=0)
attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Decode (seq_len == 1)
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    qkv, num_heads=n_qh, num_kv_heads=n_kh, memory_config=ttnn.L1_MEMORY_CONFIG,
)
q = ttnn.experimental.rotary_embedding(q, cos_cache, sin_cache, start_pos)
k = ttnn.experimental.rotary_embedding(k, cos_cache, sin_cache, start_pos)
ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=cur_pos_tensor, page_table=page_table)
ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=cur_pos_tensor, page_table=page_table)
attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q,
    k_cache,
    v_cache,
    page_table_tensor=page_table,
    cur_pos_tensor=cur_pos_tensor,
    scale=scale,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
# For GQA decode, SDPA output is interleaved; reshard before concat-heads-decode.
attn = ttnn.to_memory_config(attn, decode_heads_memcfg)
attn = ttnn.experimental.nlp_concat_heads_decode(
    attn, num_heads=n_qh, memory_config=decode_output_memcfg,
)
```

## Sequence length and paged KV cache

The legacy cache layout `[32, n_kv_heads, max_seq_len, head_dim]` effectively pays a 32x
batch tile tax even for batch=1 decode. A paged KV cache removes that and is the easiest
path to large HuggingFace `max_position_embeddings`.

Paged cache recipe:
- Choose `block_size` (multiple of 32). Use 64 by default.
- `max_num_blocks = ceil(max_seq_len / block_size)`.
- Allocate K/V cache: `[max_num_blocks, n_kv_heads, block_size, head_dim]`.
- Allocate an identity page table with tile batch: `[32, max_num_blocks]`, int32, ROW_MAJOR.
- Prefill: `ttnn.experimental.paged_fill_cache`.
- Decode: `ttnn.experimental.paged_update_cache(..., page_table=page_table)` and
  `ttnn.transformer.paged_scaled_dot_product_attention_decode(..., page_table_tensor=page_table, cur_pos_tensor=cur_pos_tensor)`.
- Decode positions: set `cur_pos_tensor` to `-1` for padded batch entries so unused slots are skipped.

Rough KV memory math (batch=1):
`bytes_per_token = n_kv_heads * head_dim * 2 (K+V) * dtype_bytes * num_layers`.
For BF16, `dtype_bytes=2`.

Sweeping max sequence length (teacher forcing, 1-token decode):
```bash
python scripts/run_eval.py --mode tt --hf-model <hf-id> \
  --prefill-len 128 --decode-len 1 --max-seq-len <max_seq_len>
python scripts/run_eval.py --mode tt --hf-model <hf-id> \
  --prefill-len 128 --decode-len 1 --max-seq-len-range 8192:131072:8192
```

## Debug playbook

- Check HuggingFace source: `transformers/models/<arch>/modeling_<arch>.py` for QKV layout,
  RoPE format, and cache semantics.
- Validate weight shapes with `safetensors` or `state_dict` and match them to TTNN linear
  and head-splitting expectations. Prefer `AutoConfig` + `safetensors.safe_open` for shape
  inspection so you don't have to load the full model.
- If outputs diverge, swap a submodule (attention or MLP) to torch to isolate the issue.
- For each TTNN tensor, log `shape`, `padded_shape`, `dtype`, `layout`, and memory config.
- Use `ttnn.to_torch` only for inspection or output; keep it outside trace capture.

## Optimizing models

This section captures the optimization workflow and lessons from the Llama-3.2-1B n150
investigation so they are not lost when `SCIENCE.md` is removed. Treat all numbers as
model-specific, but the ordering and constraints are broadly reusable.

### Optimization order (highest leverage first)

1. Lock correctness + evaluation contract first. Set a fixed eval target (for example: teacher-forcing Top-1/Top-5, coherent demo output), and keep warm-run and cold-run numbers separate.
2. Use paged KV cache as the default decode path. It avoids the legacy `[32, n_kv_heads, seq, head_dim]` cache tax and scales to long sequence lengths.
3. Profile decode hotspots before changing kernels. Use trace signposts + `tt-perf-report` to identify dominant device-time ops.
4. Optimize memory-bound decode matmuls first. Start with decode MLP matmuls, then decode `o_proj`, then LM head if feasible.
5. Tune lightweight program-config/grid knobs after big kernel/layout changes, and keep these changes isolated and easy to revert.
6. Sweep precision and compute fidelity per submodule, not globally. Re-run full teacher-forcing after each sweep because decode kernel config can shift Top-1.
7. Make decode traces allocation-free. Capture trace after prefill and avoid post-trace allocations in the decode loop.

### What was worth it (Llama-3.2-1B n150)

- DRAM-sharded decode MLP matmuls were the biggest decode win.
- This required decode-only sharded weights and a decode MLP path that avoids `ttnn.swiglu` on width-sharded inputs.
- DRAM-sharded decode `o_proj` gave a smaller but real win.
- LM head decode grid tuning (avoid full `8x8` when it hurts K blocking) gave another small win.
- Keeping decode token/RoPE buffers in L1 (while keeping position indices in DRAM) gave a small gain.
- `prefill_logits_last_device()` reduced unnecessary prefill logits work and improved TTFT in this setup.

### What was not worth it (or was blocked)

- DRAM-sharded decode QKV improved the QKV op in microbenchmarks but barely moved layer-level time.
- Device-side top-k sampling prototype was slower end-to-end in this setup.
- DRAM-sharded one-shot LM head decode hit L1/static-CB limits and needs vocab chunking or a different path.
- Reducing decode core grid below 32 cores failed for `nlp_concat_heads_decode` due to padded decode batch sharding requirements.
- Some SDPA decode fidelity/accumulation settings looked attractive but were slower on measured runs.
- Several decode program-config tweaks moved accuracy by ~1 Top-1 point or regressed throughput; treat as fragile and revalidate.

### Hard constraints to keep in mind

- Paged decode with tile-padded decode batch needs inactive `cur_pos_tensor` lanes set to `-1`.
- `ttnn.experimental.paged_update_cache(..., update_idxs_tensor=...)` requires DRAM buffer type for `update_idxs_tensor`.
- For GQA models, `ttnn.transformer.paged_scaled_dot_product_attention_decode` does not support sharded output memory config.
- DRAM-sharded matmul program configs require input activation A to be sharded.
- DRAM-sharded matmul program configs require weight B memory layout to be `WIDTH_SHARDED`.
- DRAM-sharded matmul program configs require output memory config to be sharded.
- `ttnn.experimental.nlp_concat_heads_decode` decode path effectively requires one core per padded decode user lane.
- `ttnn.slice` on sharded tensors has been unreliable in this bringup; interleaved tensors are safer for slicing.

### Measurement discipline

- `demo.py` performs one prefill+decode warmup pass before timing to compile first-use kernels.
- Keep evaluation prompt/seed fixed when comparing optimizations.
- Measure accuracy (`eval.py` teacher-forcing).
- Measure end-to-end demo throughput (`demo.py`) plus microbench regions for root-cause attribution.
- If an optimization adds significant code complexity but only yields noise-level gains, prefer reverting it.

### Exact command examples

Teacher-forcing eval (100 tokens):
```bash
python eval.py models/meta-llama/Llama-3.2-1B/n150/optimized/model.py \
  --max_new_tokens 100 \
  --prompt_file prompts/bringup_eval_long.txt \
  --seed 0
```

Demo timing (includes built-in one-pass warmup):
```bash
python demo.py models/meta-llama/Llama-3.2-1B/n150/optimized/model.py --seed 0
```

Layer/LM-head profiling with signposts:
```bash
python -m tracy -r -p -v -o /tmp/llama32-prof-tracy scripts/profile_llama32_1b_optimized.py
tt-perf-report /tmp/llama32-prof-tracy/reports/*/ops_perf_results_*.csv \
  --start-signpost LAYER0_START --end-signpost LAYER0_END
tt-perf-report /tmp/llama32-prof-tracy/reports/*/ops_perf_results_*.csv \
  --start-signpost LM_HEAD_START --end-signpost LM_HEAD_END
```

## Llama bringup ops reference

Each entry includes a minimal call pattern, typical shapes, and constraints/gotchas
observed in tt-metal. Paths refer to the Llama 3.2 1B n150 optimized bringup in
`models/meta-llama/Llama-3.2-1B/n150/optimized/model.py`.

### `ttnn.as_tensor`

Purpose: move host data (weights, constants) into a TTNN tensor.

Call:
```python
tt_weight = ttnn.as_tensor(
    torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

Typical shapes:
- Weights: `[1, 1, in, out]` (pretransposed for `ttnn.linear`)

Gotchas:
- Docstring mismatch: implementation raises if `device` is provided without `memory_config`
  (`ttnn/operations/core.py`).
- Uses `from_torch` under the hood; see `ttnn.from_torch` constraints below.
- `cache_file_name` uses `dtype`/`layout` in the cache key; for replicated tensors, it caches
  an unsharded copy first.
- BFP8/BFP4 behavior differs on Wormhole vs Grayskull (tilizer may RTE on WH for BFP8).

### `ttnn.from_torch`

Purpose: move runtime inputs to TTNN (often outside traces).

Call:
```python
tokens = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device)
```

Typical shapes:
- Input tokens: `[1, 1, 1, S]` (ROW_MAJOR input to embedding)

Gotchas:
- If `spec` is provided, `dtype`, `layout`, `memory_config`, and `tile` must be `None`, and
  `spec.shape` must match tensor shape.
- Sharded `memory_config` requires a shard spec (or ND shard spec).
- BFP8/BFP4 conversions may internally do a two-step conversion (BF16 + tilize) as described
  in the docstring.

### `ttnn.to_torch`

Purpose: pull a TTNN tensor back to torch for inspection or output.

Call:
```python
logits = ttnn.to_torch(tt_logits)
```

Typical shapes:
- Output logits: `[B, S, V]` after reshaping

Gotchas:
- If the tensor is on device, it always pulls to host (`from_device`) first.
- `torch_rank` squeezes leading dimensions only if they are `1`; otherwise it raises.

### `ttnn.embedding`

Purpose: token embedding lookup.

Call:
```python
h = ttnn.embedding(tokens, embed_weight, layout=ttnn.TILE_LAYOUT)
```

Typical shapes:
- `tokens`: `[1, 1, 1, S]` (uint32 or bfloat16)
- `embed_weight`: `[1, 1, V, H]` (ROW_MAJOR, BF16)
- `h`: `[1, 1, S, H]` (TILE)

Gotchas:
- C++ implementation requires:
  - `weight` layout must be `ROW_MAJOR` and `BFLOAT16` (`embedding_device_operation.cpp`).
  - `input` dtype must be `UINT32` or `BFLOAT16`.
  - Input/weights must be interleaved (no sharding); weights must be shaped `[1, 1, *, *]`.
  - If output is TILE, `input` width must be multiple of `TILE_HEIGHT` and `weight` columns
    multiple of `TILE_WIDTH`.
  - For sharded outputs, only height-sharded ROW_MAJOR output is supported.
- If `dtype` is requested but output layout is ROW_MAJOR, typecast is not performed
  (typecast only happens for TILE output).

### `ttnn.linear`

Purpose: matrix multiply (optionally with bias).

Call:
```python
y = ttnn.linear(x, weight, bias=None)
```

Typical shapes:
- `x`: `[1, 1, S, in]` (TILE)
- `weight`: `[1, 1, in, out]` (TILE, pretransposed)
- `y`: `[1, 1, S, out]`

Gotchas:
- Inputs must be on device and in TILE layout; supported dtypes are BF16/BFP8/BFP4/FP32.
- Implementation forbids batched `input_tensor_b` when `bias` is provided (`matmul.cpp`).

### `ttnn.rms_norm`

Purpose: RMS normalization.

Call:
```python
y = ttnn.rms_norm(x, epsilon=eps, weight=weight)
```

Typical shapes:
- `x`: `[1, 1, S, H]` (TILE)
- `weight`: `[1, 1, 1, H]` (TILE)

Gotchas:
- Docstring already captures the major constraints (tile layout, on-device, sharding rules,
  weight/bias layouts).
- No additional mismatches found in the Llama bringup path.

### `ttnn.add` / `ttnn.mul`

Purpose: elementwise add or multiply.

Call:
```python
y = ttnn.add(x, residual)
y = ttnn.mul(gate, up)
```

Typical shapes:
- Both inputs: `[1, 1, S, H]` (TILE)

Gotchas:
- For non-sharded inputs, the op converts ROW_MAJOR inputs to TILE internally; preallocated
  output is not supported when both inputs are ROW_MAJOR.
- Broadcast support is limited; some broadcast cases require the broadcasted dimension to be
  `1` and will use `repeat`.
- Certain broadcast patterns or block formats force the legacy path (performance hit); docstring
  does not mention this.

### `ttnn.silu`

Purpose: SiLU activation (MLP gating).

Call:
```python
y = ttnn.silu(x)
```

Typical shapes:
- `x`: `[1, 1, S, H]` (TILE)

Gotchas:
- Unary ops require device tensors; for non-sharded inputs the layout must be TILE and memory
  layout INTERLEAVED.
- Output memory layout must match input memory layout; sharded outputs require sharded inputs.

### `ttnn.concat`

Purpose: concatenate tensors along a dimension (e.g., QKV fusion).

Call:
```python
qkv = ttnn.concat([q, k, v], dim=-1)
```

Typical shapes:
- `q`, `k`, `v`: `[1, 1, S, H_q]`, `[1, 1, S, H_k]`, `[1, 1, S, H_v]`
- `qkv`: `[1, 1, S, H_q + H_k + H_v]`

Gotchas:
- All tensors must be on the same device, same layout, same dtype, and same rank; non-concat
  dims must match.
- Either all tensors are sharded or all are interleaved.
- Sharded concat constraints:
  - Output must be sharded, same grid and memory layout.
  - Only width concat on height-sharded or height concat on width-sharded inputs.
  - Block-sharded inputs unsupported; two-tensor width-sharded concat unsupported.
  - `groups > 1` only supported for height-sharded.
- If concatenating along a TILE dimension with padding (logical != padded), it converts to
  ROW_MAJOR and retilizes (performance hit).

### `ttnn.transpose`

Purpose: swap tensor dimensions.

Call:
```python
y = ttnn.transpose(x, 1, 2)
```

Typical shapes:
- Legacy decode/head path example: `[1, B, n_qh, d]` -> `[1, n_qh, B, d]`

Note:
- This op is usually not needed in the recommended paged decode path that uses
  `ttnn.experimental.nlp_concat_heads_decode`.

Gotchas:
- For rank > 4, transpose uses `permute`; for rank <= 4 it is constrained to N/C/H/W dims.
- Only HC/WH/CN transposes are implemented in the dedicated kernel; others rely on permute.
- Non-zero `pad_value` is only supported for HC; other transpose dims require `pad_value=0`.
- Tile inputs must have H/W multiples of tile sizes; row-major WH requires row size alignment.
- Sharded constraints are strict: HC transpose does not support sharded+tilized; output sharding
  generally requires input sharding.
- BFLOAT8_B is only supported for CN/WH; other dims will typecast to BF16.

### `ttnn.deallocate`

Purpose: free device buffers for intermediates.

Call:
```python
ttnn.deallocate(qkv)
```

Gotchas:
- No functional constraints beyond releasing device buffers; avoid deallocate inside trace capture.

### `ttnn.experimental.paged_fill_cache`

Purpose: copy prefill K/V into a paged KV cache.

Call:
```python
ttnn.experimental.paged_fill_cache(k_cache, k, page_table, batch_idx=0)
```

Typical shapes:
- `k_cache`/`v_cache`: `[max_num_blocks, n_kv, block_size, d]` (TILE, DRAM)
- `k`/`v`: `[1, n_kv, S, d]` (TILE)
- `page_table`: `[32, max_num_blocks]` int32 (ROW_MAJOR, DRAM)

Gotchas:
- `block_size` must be a multiple of 32 (64 is a good default).
- Use an identity page table for bringup: `page_table[b, i] = i`.
- For batch=1 bringup, use `batch_idx=0` and keep the other 31 lanes padded/unused.

### `ttnn.experimental.paged_update_cache`

Purpose: update cache positions during decode.

Call:
```python
ttnn.experimental.paged_update_cache(
    k_cache,
    k,
    update_idxs_tensor=cur_pos_tensor,
    page_table=page_table,
)
```

Typical shapes:
- `k_cache`/`v_cache`: `[max_num_blocks, n_kv, block_size, d]` (TILE, DRAM)
- `k`/`v`: `[1, B, n_kv, d]` (decode, B tile-aligned)
- `cur_pos_tensor`: `[B]` int32 (ROW_MAJOR, DRAM); use `-1` for inactive lanes
- `page_table`: `[32, max_num_blocks]` int32 (ROW_MAJOR, DRAM)

Gotchas:
- Inputs/cache must be on device and TILE layout.
- `cur_pos_tensor` must be tile-aligned batch (usually 32) and int32 row-major in DRAM.
- `page_table` must be int32 row-major; for bringup use identity mapping and keep it in DRAM.

### `ttnn.experimental.rotary_embedding`

Purpose: apply RoPE in HuggingFace format (not meta format).

Call:
```python
q = ttnn.experimental.rotary_embedding(q, cos, sin)
q = ttnn.experimental.rotary_embedding(q, cos_cache, sin_cache, token_index)
```

Typical shapes:
- `q`/`k`: `[1, n_heads, S, d]` (prefill) or `[1, B, n_heads, d]` (decode)
- `cos`/`sin`: `[1, 1, S_max, d]`

Gotchas:
- Inputs (x/cos/sin) must be on device, TILE layout, and share device/dtype/shape.
- `input_tensor.padded_shape()[-1]` must be divisible by `2 * TILE_WIDTH`.
- `cos`/`sin` shapes must be `[1, 1, *, X]` with `X` matching input last dim; lengths must
  cover `seq_len` or `token_index`.
- `token_index` is only valid when `seq_len == 1` (decode-style input).
- If unsharded, input/output memory layout must be interleaved; sharded inputs must not be
  WIDTH_SHARDED.

### `ttnn.experimental.nlp_create_qkv_heads`

Purpose: split fused QKV into per-head tensors for prefill.

Call:
```python
q, k, v = ttnn.experimental.nlp_create_qkv_heads(
    qkv, num_heads=n_qh, num_kv_heads=n_kh, transpose_k_heads=False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

Typical shapes:
- `qkv`: `[1, 1, S, (n_qh + 2*n_kh) * d]`
- `q`: `[1, n_qh, S, d]`
- `k`/`v`: `[1, n_kh, S, d]`

Gotchas:
- If `input_kv` is provided, Q and KV head_dim must match; otherwise it raises.
- If `input_kv` is not provided, input last dim must be divisible by
  `(num_q_heads + 2 * num_kv_heads)`.
- In this bringup path we use `transpose_k_heads=False`.

### `ttnn.experimental.nlp_create_qkv_heads_decode`

Purpose: split fused QKV for decode (batch is tile-aligned).

Call:
```python
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    qkv, num_heads=n_qh, num_kv_heads=n_kh, memory_config=ttnn.L1_MEMORY_CONFIG,
)
```

Typical shapes:
- `qkv`: `[1, 1, B, (n_qh + 2*n_kh) * d]`, `B <= 32`
- `q`: `[1, B, n_qh, d]`
- `k`/`v`: `[1, B, n_kh, d]`

Gotchas:
- Input must be on device and TILE layout.
- Input shape must be `[1, 1, B, head_dim * (num_q_heads + 2*num_kv_heads)]` with `B <= 32`
  and head_dim multiple of TILE_WIDTH.
- `num_q_heads <= 32` and `num_q_heads >= num_kv_heads`.
- Output is HEIGHT_SHARDED; non-sharded input forces `overlap_qk_coregrid=True`.
- If `batch_offset` is provided, `slice_size` must also be provided (and vice-versa).

### `ttnn.experimental.nlp_concat_heads`

Purpose: collapse heads back to hidden dimension.

Call:
```python
y = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

Typical shapes:
- Prefill `attn_out`: `[1, n_qh, S, d]` -> `[1, 1, S, H]`

Note:
- For decode, prefer `ttnn.experimental.nlp_concat_heads_decode`.

Gotchas:
- Input must be on device, TILE layout, dtype BF16/BFP8/FP32.
- Sharded inputs must not be WIDTH_SHARDED and shard width must match padded width.
- For sharded input, output must not be HEIGHT_SHARDED; for interleaved input, output must be
  interleaved.

### `ttnn.experimental.nlp_concat_heads_decode`

Purpose: collapse decode heads back to hidden dimension in the decode-optimized path.

Call:
```python
y = ttnn.experimental.nlp_concat_heads_decode(
    attn_out, num_heads=n_qh, memory_config=decode_output_memcfg,
)
```

Typical shapes:
- `attn_out`: `[1, B, n_qh, d]`
- `y`: `[1, 1, B, H]`

Gotchas:
- Decode path expects a HEIGHT_SHARDED input whose core count matches padded decode batch.
- For the common padded decode batch `B=32`, this effectively requires 32 decode cores.
- For GQA decode, `paged_scaled_dot_product_attention_decode` output is interleaved, so do an
  explicit `ttnn.to_memory_config(..., decode_heads_memcfg)` before this op.

### `ttnn.transformer.scaled_dot_product_attention`

Purpose: fused prefill attention.

Call:
```python
attn_out = ttnn.transformer.scaled_dot_product_attention(
    q, k, v, is_causal=True, scale=scale,
)
```

Typical shapes:
- `q`: `[B, n_qh, S_q, d]`
- `k`/`v`: `[B, n_kh, S_k, d]`
- `attn_out`: `[B, n_qh, S_q, d]`

Gotchas:
- Inputs must be on device, TILE layout, and not sharded; dtypes limited to BF16/BFP8/BFP4.
- No padding allowed on batch/num_heads/head_dim.
- For causal mode, Q and K sequence lengths must match.
- GQA constraint: `num_q_heads >= num_kv_heads` and divisible.
- If `attn_mask` is provided: must be TILE, DRAM, dtype BF16/BFP8/BFP4, shape `[B, 1, Sq, Sk]`,
  and `Sq/Sk` divisible by `q_chunk_size/k_chunk_size` (default 32).

### `ttnn.transformer.paged_scaled_dot_product_attention_decode`

Purpose: fused decode attention using a paged KV cache.

Call:
```python
attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q,
    k_cache,
    v_cache,
    page_table_tensor=page_table,
    cur_pos_tensor=cur_pos_tensor,
    scale=scale,
)
```

Typical shapes:
- `q`: `[1, B, n_qh, d]`
- `k_cache`/`v_cache`: `[max_num_blocks, n_kh, block_size, d]`
- `page_table`: `[32, max_num_blocks]` int32 (ROW_MAJOR)
- `cur_pos_tensor`: `[B]` int32 (ROW_MAJOR)
- `attn_out`: `[1, B, n_qh, d]` (typically interleaved for GQA models)

Gotchas:
- Decode uses tile-aligned batch B (usually 32) even for batch=1; set `cur_pos_tensor` to `-1`
  for inactive lanes so they are skipped.
- `page_table` must be ROW_MAJOR int32 and match how cache blocks are mapped (identity mapping is fine).
- For GQA models, sharded output memory config is not supported. Keep SDPA decode output interleaved,
  then reshard before `ttnn.experimental.nlp_concat_heads_decode` if needed.
