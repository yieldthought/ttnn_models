# TTNN Bringup Notes (tt-metal/ttnn)

Curated constraints and gotchas for TTNN ops used during model bringup.
Scope: Llama 3.2 1B reference implementation; extend as new models are audited.

## Llama Bringup Ops Audit (`models/meta-llama/Llama-3.2-1B/n150/functional/model.py`)

This section audits the TTNN ops used by the Llama bringup implementation and captures gotchas or doc/impl mismatches.

### `ttnn.as_tensor`

- Docstring mismatch: implementation raises if `device` is provided without `memory_config` (`ttnn/operations/core.py`).
- Uses `from_torch` under the hood; see `ttnn.from_torch` constraints below.
- `cache_file_name` uses `dtype`/`layout` in the cache key; for replicated tensors, it caches an unsharded copy first.
- BFP8/BFP4 behavior differs on Wormhole vs Grayskull (tilizer may RTE on WH for BFP8).

### `ttnn.from_torch`

- If `spec` is provided, `dtype`, `layout`, `memory_config`, and `tile` must be `None`, and `spec.shape` must match tensor shape; otherwise errors.
- Sharded `memory_config` requires a shard spec (or ND shard spec).
- BFP8/BFP4 conversions may internally do a two-step conversion (BF16 + tilize) as described in the docstring.

### `ttnn.to_torch`

- If the tensor is on device, it always pulls to host (`from_device`) first.
- `torch_rank` squeezes leading dimensions only if they are `1`; otherwise it raises.

### `ttnn.embedding`

- C++ implementation requires:
  - `weight` layout must be `ROW_MAJOR` and `BFLOAT16` (`embedding_device_operation.cpp`).
  - `input` dtype must be `UINT32` or `BFLOAT16`.
  - Input/weights must be interleaved (no sharding); weights must be shaped `[1, 1, *, *]`.
  - If output is TILE, `input` width must be multiple of `TILE_HEIGHT` and `weight` columns multiple of `TILE_WIDTH`.
  - For sharded outputs, only height-sharded ROW_MAJOR output is supported.
- If `dtype` is requested but output layout is ROW_MAJOR, typecast is not performed (typecast only happens for TILE output).

### `ttnn.linear`

- Inputs must be on device and in TILE layout; supported dtypes are BF16/BFP8/BFP4/FP32 (docstring).
- Implementation forbids batched `input_tensor_b` when `bias` is provided (`matmul.cpp`).

### `ttnn.rms_norm`

- Docstring already captures the major constraints (tile layout, on-device, sharding rules, weight/bias layouts).
- No additional mismatches found in the Llama bringup path.

### `ttnn.add` / `ttnn.mul`

- For non-sharded inputs, the op converts ROW_MAJOR inputs to TILE internally; preallocated output is not supported when both inputs are ROW_MAJOR.
- Broadcast support is limited; some broadcast cases require the broadcasted dimension to be `1` and will use `repeat`.
- Certain broadcast patterns or block formats force the legacy path (performance hit); docstring does not mention this.

### `ttnn.silu`

- Unary ops require device tensors; for non-sharded inputs the layout must be TILE and memory layout INTERLEAVED.
- Output memory layout must match input memory layout; sharded outputs require sharded inputs.

### `ttnn.concat`

- All tensors must be on the same device, same layout, same dtype, and same rank; non-concat dims must match.
- Either all tensors are sharded or all are interleaved.
- Sharded concat constraints:
  - Output must be sharded, same grid and memory layout.
  - Only width concat on height-sharded or height concat on width-sharded inputs.
  - Block-sharded inputs unsupported; two-tensor width-sharded concat unsupported.
  - `groups > 1` only supported for height-sharded.
- If concatenating along a TILE dimension with padding (logical != padded), it converts to ROW_MAJOR and retilizes (performance hit).

### `ttnn.transpose`

- For rank > 4, transpose uses `permute`; for rank <= 4 it is constrained to N/C/H/W dims.
- Only HC/WH/CN transposes are implemented in the dedicated kernel; others rely on permute.
- Non-zero `pad_value` is only supported for HC; other transpose dims require `pad_value=0`.
- Tile inputs must have H/W multiples of tile sizes; row-major WH requires row size alignment.
- Sharded constraints are strict: HC transpose does not support sharded+tilized; output sharding generally requires input sharding.
- BFLOAT8_B is only supported for CN/WH; other dims will typecast to BF16.

### `ttnn.deallocate`

- No functional constraints beyond releasing device buffers; safe for cleaning up intermediates.

### `ttnn.fill_cache`

- Requires device tensors, TILE layout, matching width/height, and interleaved cache memory layout.
- Input batch size must be 1; `batch_idx` must be in range; input seq_len <= cache seq_len.
- For FILL, input and cache dtypes must match.
- Interleaved inputs with seq_len > 1 must fit within grid size; sharded inputs must not be WIDTH_SHARDED and shard width must match padded width.

### `ttnn.experimental.paged_update_cache`

- Input/cache must be on device and TILE layout; cache must be interleaved.
- Input must be sharded (not WIDTH_SHARDED) with ROW_MAJOR orientation.
- Input dtype must be BF16/FP32; cache supports BF16/BFP8/BFP4/FP32.
- `update_idxs_tensor` must be INT32 ROW_MAJOR; if sharded it must be HEIGHT_SHARDED in L1, if interleaved it must be DRAM.
- `page_table` (if provided) must be ROW_MAJOR; dtype INT32 (interleaved) or UINT16 (sharded). `share_cache` is not supported with paged cache.
- `batch_offset` must be 0.

### `ttnn.experimental.rotary_embedding`

- Inputs (x/cos/sin) must be on device, TILE layout, and share device/dtype/shape.
- `input_tensor.padded_shape()[-1]` must be divisible by `2 * TILE_WIDTH`.
- `cos`/`sin` shapes must be `[1, 1, *, X]` with `X` matching input last dim; lengths must cover `seq_len` or `token_index`.
- `token_index` is only valid when `seq_len == 1` (decode-style input).
- If unsharded, input/output memory layout must be interleaved; sharded inputs must not be WIDTH_SHARDED.

### `ttnn.experimental.nlp_create_qkv_heads`

- If `input_kv` is provided, Q and KV head_dim must match; otherwise it raises.
- If `input_kv` is not provided, input last dim must be divisible by `(num_q_heads + 2 * num_kv_heads)`.
- `transpose_k_heads` defaults to true (K is transposed); code relies on this for SDPA layout expectations.

### `ttnn.experimental.nlp_create_qkv_heads_decode`

- Input must be on device, TILE layout, and typically WIDTH_SHARDED with ROW_MAJOR sharding orientation.
- Input shape must be `[1, 1, B, head_dim * (num_q_heads + 2*num_kv_heads)]` with `B <= 32` and head_dim multiple of TILE_WIDTH.
- `num_q_heads <= 32` and `num_q_heads >= num_kv_heads`.
- Output is HEIGHT_SHARDED; non-sharded input forces `overlap_qk_coregrid=True`.
- If `batch_offset` is provided, `slice_size` must also be provided (and vice-versa).

### `ttnn.experimental.nlp_concat_heads`

- Input must be on device, TILE layout, dtype BF16/BFP8/FP32.
- Sharded inputs must not be WIDTH_SHARDED and shard width must match padded width.
- For sharded input, output must not be HEIGHT_SHARDED; for interleaved input, output must be interleaved.

### `ttnn.transformer.scaled_dot_product_attention`

- Inputs must be on device, TILE layout, and not sharded; dtypes limited to BF16/BFP8/BFP4.
- No padding allowed on batch/num_heads/head_dim.
- For causal mode, Q and K sequence lengths must match.
- GQA constraint: `num_q_heads >= num_kv_heads` and divisible.
- If `attn_mask` is provided: must be TILE, DRAM, dtype BF16/BFP8/BFP4, shape `[B, 1, Sq, Sk]`, and `Sq/Sk` divisible by `q_chunk_size/k_chunk_size` (default 32).

### `ttnn.transformer.scaled_dot_product_attention_decode`

- Decode mode requires `Q` batch size == 1 (shape `[1, B, nqh, dh]`), K/V in DRAM and TILE layout.
- `K` and `V` must match shapes; `cur_pos` entries must be < K sequence length.
- `k_chunk_size` must be provided and be a multiple of 32 in unpaged mode.
- GQA is partially supported: output cannot be sharded, `Q` dtype must be BF16, and Q heads must be multiple of K heads.
- Paged mode requires page_table (ROW_MAJOR INT32 or UINT16 for sharded) and `cur_pos_tensor` for causal mode.
