---
name: hf-model-reading
description: Read and understand HuggingFace model implementations. Use when locating HF model code, identifying attention and MLP structure, decoding shapes, RoPE conventions, cache behavior, and weight names before TTNN bringup.
---

# HF Model Reading

## Overview
Locate and read HF model code to capture shapes, conventions, and weight names needed for TTNN bringup.

## Workflow
1. Identify the model family and config.
   - Inspect `hf_model.config` or `AutoConfig` fields.
2. Locate the HF implementation.
   - Look in `transformers/models/<name>/configuration_*.py` and `modeling_*.py`.
   - Use `inspect` in Python to print the module path when needed.
3. Read the config and note key fields.
   - heads, kv heads, head dim, rope theta or scaling, rms eps, activation, attention bias.
4. Read the attention forward path.
   - Track input shapes, qkv projections, head split and transpose order.
   - Find where RoPE is applied and how cache is updated.
5. Read the MLP path.
   - Identify activation, gating, and projection order.
6. Map weight names.
   - Use `state_dict().keys()` to confirm tensor names and shapes.
7. Decide what maps directly to TTNN ops and what needs special handling.
   - Flag sliding window, attention bias, or custom norm behavior.
