# TTNN Bringup Demo and Eval

This repo contains TTNN bringup models plus utilities for demo runs and teacher-forcing eval.

## demo.py

Run a HuggingFace model on CPU or a TTNN bringup model on device. The script:
- Chooses the mesh device shape from the model path (`n150`, `n300`, `t3000`).
- Warms up with one prefill and one decode token before timing.
- Reports TTFT (ms) and decode throughput (t/s/u).
- Prints prompt/output with simple ANSI coloring.

### Usage

CPU (HuggingFace):
```bash
python demo.py meta-llama/Llama-3.2-1B
```

TTNN (model.py path):
```bash
python demo.py models/meta-llama/Llama-3.2-1B/n150/functional/model.py
```

Key flags:
- `--prompt` or `--prompt-file` to override the default prompt.
- `--max-new-tokens` to cap generation.
- `--temperature` / `--top-k` to control sampling (`--temperature 0` for greedy).
- `--seed` for reproducible sampling.
- `--device-id` for single-device runs.
- `--cache-dir` for HF downloads.

### Mesh selection

`demo.py` inspects the path to pick a mesh:
- `n150` -> 1x1 (single device)
- `n300` -> 1x2
- `t3000` -> 2x4

If `TT_MESH_GRAPH_DESC_PATH` is not set, `demo.py` uses:
`../tt-metal/tt_metal/fabric/mesh_graph_descriptors/{n300_mesh_graph_descriptor.textproto|t3k_mesh_graph_descriptor.textproto}`.

## eval.py

Teacher-forcing accuracy check against a HuggingFace reference model.

Example:
```bash
python eval.py models/meta-llama/Llama-3.2-1B/n150/functional/model.py \
  --model meta-llama/Llama-3.2-1B \
  --prompt_file prompts/bringup_eval_long.txt \
  --max_new_tokens 100
```

## scripts/run_eval.py

Wrapper around `eval.py` (or HF-only eval) that emits a `YT_METRICS=` JSON line.

Example:
```bash
python scripts/run_eval.py --mode tt --hf-model meta-llama/Llama-3.2-1B --system n150
```

## scripts/run_demo_all.py

Run `demo.py` across all bringup models in `models/` and save outputs.

Example:
```bash
python scripts/run_demo_all.py --output-dir demo_outputs --max-new-tokens 48
```

The script writes two files per model:
- `*.log`: full stdout/stderr from `demo.py`
- `*.txt`: trimmed block starting at `TT demo` / `HF demo`

## Runtime notes

- If you see `cannot map elf file into memory: No space left on device`, rerun with:
  `TT_METAL_CACHE=/tmp/tt-metal-cache` and
  `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root`.
