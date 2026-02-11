#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Profiler helper for the Llama 3.2 1B optimized TTNN model.

This runs:
- One prefill to populate paged KV caches
- One decode step through a single layer (layer 0)
- One LM head matmul (using the layer 0 output as input)

It emits TT signposts so tt-perf-report can focus on the region of interest.

Example:
  rm -rf /tmp/llama32-prof-tracy
  python -m tracy -r -p -v -o /tmp/llama32-prof-tracy scripts/profile_llama32_1b_optimized.py

  tt-perf-report /tmp/llama32-prof-tracy/reports/*/ops_perf_results_*.csv \\
    --start-signpost LAYER0_START --end-signpost LAYER0_END

  tt-perf-report /tmp/llama32-prof-tracy/reports/*/ops_perf_results_*.csv \\
    --start-signpost LM_HEAD_START --end-signpost LM_HEAD_END

Notes:
- If the CSV has signposts and you omit `--start-signpost/--end-signpost`, `tt-perf-report` analyzes the region
  after the last signpost (often empty if the last signpost is an end marker).
- `TT_METAL_DEVICE_PROFILER=1` + `process_ops_logs.py --device-only` can still be useful for raw device timing, but
  the resulting CSV may not be compatible with `tt-perf-report` (missing `DEVICE ID` column).
"""

import os
import pathlib
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from device_utils import build_tt_model, close_tt_device, load_model_module, open_tt_device, pick_mesh_shape, resolve_tt_metadata


DEFAULT_MODEL = pathlib.Path("models/meta-llama/Llama-3.2-1B/n150/optimized/model.py")
DEFAULT_PROMPT = (
    "Journal entry, 1957: Tonight a tiny sphere called Sputnik 1 crossed the sky, "
    "beeping like a metronome for a new era. The neighbors gathered on the roof, "
    "listening and arguing about what comes next. I wrote in my notebook that"
)


def signpost(name: str) -> None:
    import ttnn

    ttnn.tracy_message(f"TT_SIGNPOST: {name}")


def main() -> None:
    model_path = pathlib.Path(os.environ.get("TT_MODEL_PATH", str(DEFAULT_MODEL)))
    prompt = os.environ.get("TT_PROMPT", DEFAULT_PROMPT)
    max_seq_len = int(os.environ.get("TT_MAX_SEQ_LEN", "2048"))
    device_id = int(os.environ.get("TT_DEVICE_ID", "0"))

    hf_id, system = resolve_tt_metadata(model_path)
    module = load_model_module(model_path)

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="cpu")

    tokens = tokenizer(prompt, return_tensors="pt").input_ids

    mesh_shape = pick_mesh_shape(system, module)
    tt_device, is_mesh, fabric_config = open_tt_device(mesh_shape, device_id)
    try:
        model = build_tt_model(module, hf_model, tt_device, max_seq_len)
        if getattr(model, "use_decode_trace", None) is not None:
            model.use_decode_trace = False

        import ttnn

        with torch.no_grad():
            _ = model(tokens, use_cache=True)
            ttnn.synchronize_device(tt_device)

            start_pos = int(getattr(model, "_pos", tokens.shape[1]))
            model._update_decode_token_buffers(tokens[:, -1:], start_pos)
            model._update_decode_rope_buffers(start_pos)
            ttnn.synchronize_device(tt_device)

            h = ttnn.embedding(model.decode_token_buffer, model.embed, layout=ttnn.TILE_LAYOUT)

            ttnn.synchronize_device(tt_device)
            signpost("LAYER0_START")
            h = model.layers[0](
                h,
                start_pos,
                1,
                model.decode_pos_buffer,
                model.decode_cos_buffer,
                model.decode_sin_buffer,
                trace_decode=False,
            )
            ttnn.synchronize_device(tt_device)
            signpost("LAYER0_END")

            h_norm = model.norm(h, ttnn.L1_MEMORY_CONFIG)
            h_tok = ttnn.slice(
                h_norm,
                (0, 0, 0, 0),
                (h_norm.shape[0], h_norm.shape[1], 1, h_norm.shape[-1]),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            ttnn.synchronize_device(tt_device)
            signpost("LM_HEAD_START")
            logits = ttnn.linear(
                h_tok,
                model.lm_head,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=module.LOFI_MATMUL_CONFIG,
                program_config=model.lm_head_decode_program_config,
            )
            ttnn.synchronize_device(tt_device)
            signpost("LM_HEAD_END")

            ttnn.deallocate(logits)
            ttnn.deallocate(h_tok)
            ttnn.deallocate(h_norm)
            ttnn.deallocate(h)
    finally:
        close_tt_device(tt_device, is_mesh, fabric_config)


if __name__ == "__main__":
    main()
