# MODEL_BRINGUP.md — Llama 3.2 1B (t3000 functional)

## Overview
This is a minimal TTNN bringup of `meta-llama/Llama-3.2-1B` for T3000 using 1D tensor parallel.
The code is intentionally simple and mirrors the n150 functional model while sharding key matmuls.

- Model code: `models/meta-llama/Llama-3.2-1B/t3000/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Parallelism strategy (T3000)
- Mesh shape: 2x4 (eight devices), linear topology.
- Column-parallel: QKV, gate, and up projections (weights sharded on dim=3).
- Row-parallel: attention output projection and MLP down projection (weights sharded on dim=2) with `ttnn.all_reduce` across the full mesh.
- KV cache is sharded across devices on dim=1 (KV heads).
- LM head is column-parallel (vocab sharded on dim=3) and concatenated to host.

## Parallelization summary
- Replicated tensors: token embeddings, RMSNorm weights, RoPE caches, input tokens.
- Column-parallel (weight width sharding, dim=3): `q_proj`, `k_proj`, `v_proj`, `mlp.gate_proj`, `mlp.up_proj`, `lm_head`.
- Row-parallel (weight height sharding, dim=2): `o_proj`, `mlp.down_proj`.
- KV cache: sharded by KV heads (dim=1) across devices.
- CCL ops: `ttnn.all_reduce` after `o_proj` and after `mlp.down_proj` to sum partials.
- Output composition: `ttnn.to_torch(..., mesh_composer=ConcatMeshToTensor)` to gather vocab shards on host.

## Model API contract
- The model exposes a `build_model(hf_model, tt_device, max_seq_len)` function.
- The returned class subclasses `torch.nn.Module` and `GenerationMixin` so HF `generate()` works.
- The forward method returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`.

## Key TTNN ops
- `ttnn.embedding` for token embeddings
- `ttnn.linear` for QKV, output, and MLP projections
- `ttnn.rms_norm` for RMSNorm
- `ttnn.experimental.rotary_embedding` for HuggingFace-format RoPE
- `ttnn.experimental.nlp_create_qkv_heads[_decode]` and `ttnn.experimental.nlp_concat_heads`
- `ttnn.transformer.scaled_dot_product_attention[_decode]`
- `ttnn.fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)
- `ttnn.all_reduce` for row-parallel attention/MLP accumulation

## RoPE notes
Llama 3.x on HuggingFace uses the HuggingFace RoPE layout. Use:

- `ttnn.experimental.rotary_embedding`

Avoid `ttnn.experimental.rotary_embedding_llama`, which expects Meta-format RoPE.

Decode reshapes Q/K to `[1, 1, batch*heads, head_dim]` before RoPE and reshapes back.

## KV cache and tiling constraints
- Cache tensors are allocated as `[32, n_kv_heads, max_seq_len, head_dim]` and sharded on dim=1.
- The batch dimension is tile-aligned to 32 for decode ops.
- Prefill uses `ttnn.fill_cache` and decode uses `ttnn.experimental.paged_update_cache`.

If prefill hits a `fill_cache` grid limit, use `--prefill_decode` to debug.
Final bringup metrics must use the full prefill pass (no `--prefill_decode`).
`scripts/run_eval.py` enables this automatically for large prefill lengths.

## Precision
- Weights use `ttnn.bfloat16` in this bringup for simplicity.
- Activations use `ttnn.bfloat16`.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after logits are returned.
Decode trims padded head width after concatenating heads.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.
On this T3000 host, set the mesh graph descriptor and use all 8 devices:

```
TT_MESH_GRAPH_DESC_PATH=/home/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python eval.py models/meta-llama/Llama-3.2-1B/t3000/functional/model.py
```

Automation wrapper (emits YT_METRICS JSON):

```
TT_MESH_GRAPH_DESC_PATH=/home/moconnor/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/run_eval.py --mode tt --hf-model meta-llama/Llama-3.2-1B
```

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi -r`.
