# TTNN Bringup Notes (tt-metal/ttnn)

Practical guidance for bringing up HuggingFace LLMs on TTNN. This is not a full API
reference. It focuses on the ops used by the Llama 3.2 1B functional bringup and the
failure modes we keep hitting.

## Quick bringup checklist

- Read the HuggingFace modeling file and config. Capture: hidden_size, num_attention_heads,
  num_key_value_heads, head_dim, intermediate_size, rope_scaling, and vocab_size.
- Inspect weight shapes (safetensors or state_dict) before wiring TTNN ops. Q/K/V shapes
  drive head_dim and GQA logic.
- Decide prefill vs decode flow early. The cache layout and decode shapes are strict.
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

TTNN uses padded shapes due to 32x32 tiles. Use a `pad_to_tile()` helper and check
`tensor.padded_shape()` whenever an op refuses to run.

## Prefill / decode skeleton

```python
# Prefill (seq_len > 1)
qkv = ttnn.concat([q, k, v], dim=-1)
q, k, v = ttnn.experimental.nlp_create_qkv_heads(
    qkv, num_heads=n_qh, num_kv_heads=n_kh, transpose_k_heads=False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
q = ttnn.experimental.rotary_embedding(q, cos, sin)
k = ttnn.experimental.rotary_embedding(k, cos, sin)
ttnn.fill_cache(k_cache, k, batch_idx=0)
ttnn.fill_cache(v_cache, v, batch_idx=0)
attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Decode (seq_len == 1)
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    qkv, num_heads=n_qh, num_kv_heads=n_kh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
q = ttnn.experimental.rotary_embedding(q, cos_cache, sin_cache, start_pos)
k = ttnn.experimental.rotary_embedding(k, cos_cache, sin_cache, start_pos)
ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=cur_pos_tensor)
ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=cur_pos_tensor)
attn = ttnn.transformer.scaled_dot_product_attention_decode(
    q, k_cache, v_cache, cur_pos_tensor=cur_pos_tensor, scale=scale,
)
attn = ttnn.transpose(attn, 1, 2)
attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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

## Llama bringup ops reference

Each entry includes a minimal call pattern, typical shapes, and constraints/gotchas
observed in tt-metal. Paths refer to the Llama 3.2 1B bringup in
`models/meta-llama/Llama-3.2-1B/n150/functional/model.py`.

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

Purpose: swap tensor dimensions (decode attention output path).

Call:
```python
y = ttnn.transpose(x, 1, 2)
```

Typical shapes:
- Decode attention output: `[1, B, n_qh, d]` -> `[1, n_qh, B, d]`

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

### `ttnn.fill_cache`

Purpose: copy prefill K/V into cache at batch index.

Call:
```python
ttnn.fill_cache(cache, k, batch_idx=0)
```

Typical shapes:
- `cache`: `[B, n_kv, S_max, d]` (TILE, DRAM)
- `k`/`v`: `[B, n_kv, S, d]` (TILE)

Gotchas:
- Requires device tensors, TILE layout, matching width/height, and interleaved cache memory
  layout.
- Input batch size must be 1; `batch_idx` must be in range; input seq_len <= cache seq_len.
- For FILL, input and cache dtypes must match.
- Interleaved inputs with seq_len > 1 must fit within grid size; sharded inputs must not be
  WIDTH_SHARDED and shard width must match padded width.

### `ttnn.experimental.paged_update_cache`

Purpose: update cache positions during decode.

Call:
```python
ttnn.experimental.paged_update_cache(cache, k, update_idxs_tensor=cur_pos_tensor)
```

Typical shapes:
- `cache`: `[B, n_kv, S_max, d]` (TILE, DRAM)
- `k`/`v`: `[B, n_kv, 1, d]` (decode)
- `cur_pos_tensor`: `[B]` int32, B tile-aligned

Gotchas:
- Input/cache must be on device and TILE layout; cache must be interleaved.
- Input must be sharded (not WIDTH_SHARDED) with ROW_MAJOR orientation.
- Input dtype must be BF16/FP32; cache supports BF16/BFP8/BFP4/FP32.
- `update_idxs_tensor` must be INT32 ROW_MAJOR; if sharded it must be HEIGHT_SHARDED in L1,
  if interleaved it must be DRAM.
- `page_table` (if provided) must be ROW_MAJOR; dtype INT32 (interleaved) or UINT16 (sharded).
  `share_cache` is not supported with paged cache.
- `batch_offset` must be 0.

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
- `transpose_k_heads` defaults to true (K is transposed); code relies on this for SDPA layout
  expectations.

### `ttnn.experimental.nlp_create_qkv_heads_decode`

Purpose: split fused QKV for decode (batch is tile-aligned).

Call:
```python
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    qkv, num_heads=n_qh, num_kv_heads=n_kh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

Typical shapes:
- `qkv`: `[1, 1, B, (n_qh + 2*n_kh) * d]`, `B <= 32`
- `q`: `[1, B, n_qh, d]`
- `k`/`v`: `[1, B, n_kh, d]`

Gotchas:
- Input must be on device, TILE layout, and typically WIDTH_SHARDED with ROW_MAJOR sharding
  orientation.
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
- Decode `attn_out`: `[1, n_qh, B, d]` -> `[1, 1, B, H]`

Gotchas:
- Input must be on device, TILE layout, dtype BF16/BFP8/FP32.
- Sharded inputs must not be WIDTH_SHARDED and shard width must match padded width.
- For sharded input, output must not be HEIGHT_SHARDED; for interleaved input, output must be
  interleaved.

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

### `ttnn.transformer.scaled_dot_product_attention_decode`

Purpose: fused decode attention using KV cache.

Call:
```python
attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
    q, k_cache, v_cache, cur_pos_tensor=cur_pos_tensor, scale=scale,
)
```

Typical shapes:
- `q`: `[1, B, n_qh, d]`
- `k_cache`/`v_cache`: `[B, n_kh, S_max, d]`
- `cur_pos_tensor`: `[B]` int32
- `attn_out`: `[1, B, n_qh, d]` (transpose before concat)

Gotchas:
- Decode mode requires `Q` batch size == 1 (shape `[1, B, nqh, dh]`), K/V in DRAM and TILE
  layout.
- `K` and `V` must match shapes; `cur_pos` entries must be < K sequence length.
- `k_chunk_size` must be provided and be a multiple of 32 in unpaged mode.
- GQA is partially supported: output cannot be sharded, `Q` dtype must be BF16, and Q heads
  must be multiple of K heads.
- Paged mode requires page_table (ROW_MAJOR INT32 or UINT16 for sharded) and `cur_pos_tensor`
  for causal mode.
