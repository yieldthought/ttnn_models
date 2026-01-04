# MODEL_BRINGUP.md — ALLaM 7B Instruct preview (n300 functional)

## Overview
This is a minimal TTNN bringup of `humain-ai/ALLaM-7B-Instruct-preview` for N300 using 1D tensor parallel.
It is designed to be easy to read and to serve as a template for future bringups.

- Model code: `models/humain-ai/ALLaM-7B-Instruct-preview/n300/functional/model.py`
- Eval harness: `eval.py` (teacher forcing) and `scripts/run_eval.py` (automation wrapper)
- Directory convention: `models/<org>/<model_name>/<system>/functional/model.py`

## Model API contract
- The model exposes a `build_model(hf_model, tt_device, max_seq_len)` function.
- The returned class subclasses `torch.nn.Module` and `GenerationMixin` so HF `generate()` works.
- The forward method returns `CausalLMOutputWithPast(logits=..., past_key_values=...)`.

## Parallelism strategy (N300)
- Mesh shape: 1x2 (two devices), linear topology.
- Column-parallel: QKV, gate, up, and lm_head projections (weights sharded on dim=3).
- Row-parallel: attention output projection and MLP down projection (weights sharded on dim=2) with `ttnn.all_reduce`.
- KV cache is sharded across devices on dim=1 (KV heads).
- Input tokens, embeddings, and RMSNorm weights are replicated.

## Parallelization summary
- Replicated tensors: token embeddings, RMSNorm weights, RoPE caches, input tokens.
- Column-parallel (weight width sharding, dim=3): `q_proj`, `k_proj`, `v_proj`, `mlp.gate_proj`, `mlp.up_proj`, `lm_head`.
- Row-parallel (weight height sharding, dim=2): `o_proj`, `mlp.down_proj`.
- KV cache: sharded by KV heads (dim=1) across devices.
- CCL ops: `ttnn.all_reduce` after `o_proj` and after `mlp.down_proj` to sum partials.
- Output composition: `ttnn.to_torch(..., mesh_composer=ConcatMeshToTensor)` to gather vocab shards on host.

## Key TTNN ops
- `ttnn.embedding` for token embeddings
- `ttnn.linear` for QKV, output, and MLP projections
- `ttnn.rms_norm` for RMSNorm
- `ttnn.experimental.rotary_embedding` for HuggingFace-format RoPE
- `ttnn.experimental.nlp_create_qkv_heads[_decode]` and `ttnn.experimental.nlp_concat_heads`
- `ttnn.transformer.scaled_dot_product_attention[_decode]`
- `ttnn.fill_cache` (prefill) and `ttnn.experimental.paged_update_cache` (decode)

## RoPE notes
Decode path detail:
- `rotary_embedding` with `start_pos` expects `[seq_len, 1, B, head_dim]`. For decode,
  reshape Q and K to merge heads into the batch (`[1, 1, B*heads, head_dim]`), apply
  RoPE, then reshape back to `[1, B, heads, head_dim]`.

## KV cache and tiling constraints
- Cache tensors are `[32, n_kv_heads, max_seq_len, head_dim]`.
- `MAX_CACHE_SEQ_LEN` is set to 256 to cap memory usage; increase if needed.
- Prefill uses a height-sharded K/V path to avoid `fill_cache` grid limits when sequence length grows.

## Precision
- Weights use `ttnn.bfloat8_b`.
- Activations use `ttnn.bfloat16`.

## Padding
Inputs are padded to the TTNN tile size (32) before embedding and trimmed after logits are returned.
Decode trims padded head width after concatenating heads.

## Evaluation
Teacher-forcing accuracy is computed against the HF reference model.

```
python eval.py models/humain-ai/ALLaM-7B-Instruct-preview/n300/functional/model.py --model humain-ai/ALLaM-7B-Instruct-preview
```

On this N300 host, set the mesh to devices 0 and 2:

```
TT_VISIBLE_DEVICES=0,2 python eval.py models/humain-ai/ALLaM-7B-Instruct-preview/n300/functional/model.py --model humain-ai/ALLaM-7B-Instruct-preview
```

If `/home` is full, redirect runtime artifacts to a writable location. On this host,
`/proj_sw/user_dev/moconnor/tt-runtime-root` is a symlinked runtime root (with `tt_metal`,
`ttnn`, and `runtime` from the installed package):

```
TT_VISIBLE_DEVICES=0,2 TT_METAL_CACHE=/tmp/tt-metal-cache TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 python eval.py models/humain-ai/ALLaM-7B-Instruct-preview/n300/functional/model.py --model humain-ai/ALLaM-7B-Instruct-preview
```

Automation wrapper (emits YT_METRICS JSON):

```
python scripts/run_eval.py --mode tt --hf-model humain-ai/ALLaM-7B-Instruct-preview
```

## Debugging tips
- Start with small prefill/decode lengths (e.g. 16/8).
- Compare TT outputs to HF outputs layer-by-layer if needed.
- Reset hardware if needed: `tt-smi -r`.
