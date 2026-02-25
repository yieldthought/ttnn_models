# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Teacher-forcing evaluation for ttnn model bring-up.

Usage:
    python eval.py models/meta-llama/Llama-3.2-1B/n150/functional/model.py
"""

import argparse
import json
import pathlib
import os
import sys

import torch
import ttnn


def maybe_prepend_transformers_path():
    """Prepend an optional external transformers runtime to sys.path."""
    extra_path = os.environ.get("TTNN_TRANSFORMERS_PYTHONPATH", "")
    if not extra_path:
        return
    for path in reversed(extra_path.split(":")):
        if path and path not in sys.path:
            sys.path.insert(0, path)


maybe_prepend_transformers_path()

from transformers import AutoModelForCausalLM, AutoTokenizer

from device_utils import build_tt_model, close_tt_device, load_model_module, open_tt_device, pick_mesh_shape, resolve_tt_metadata

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"


def resolve_hf_dtype(dtype_name: str) -> torch.dtype:
    """Map CLI dtype string to torch dtype."""
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported --hf_dtype value: {dtype_name}")


def load_hf_reference_model(model_id: str, torch_dtype: torch.dtype, cache_dir):
    """
    Load a reference HF model.

    Tries AutoModelForCausalLM first, then falls back to
    AutoModelForImageTextToText for checkpoints like qwen3_5_moe.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, cache_dir=cache_dir)
    except Exception as causal_error:
        try:
            from transformers import AutoModelForImageTextToText
        except Exception as image_text_import_error:
            raise causal_error from image_text_import_error
        try:
            return AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch_dtype, cache_dir=cache_dir)
        except Exception as image_text_error:
            raise causal_error from image_text_error

def score_step(logits: torch.Tensor, target_id: int) -> tuple[int, int]:
    top5 = torch.topk(logits, k=5).indices
    top1 = int(top5[0].item() == target_id)
    top5_hit = int((top5 == target_id).any().item())
    return top1, top5_hit


def evaluate(
    tt_model,
    reference_tokens: torch.Tensor,
    prompt_len: int,
    max_new_tokens: int,
) -> tuple[float, float, int]:
    if max_new_tokens < 1:
        return 0.0, 0.0, 0

    top1 = 0
    top5 = 0
    total = 0

    past = None
    prompt_ids = reference_tokens[:prompt_len].unsqueeze(0)
    if hasattr(tt_model, "prefill_logits_last_device"):
        logits, past = tt_model.prefill_logits_last_device(prompt_ids, use_cache=True)
        logits = logits[0]
    else:
        outputs = tt_model(prompt_ids, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[0, -1, :]
    target_id = int(reference_tokens[prompt_len].item())
    step_top1, step_top5 = score_step(logits, target_id)
    top1 += step_top1
    top5 += step_top5
    total += 1

    for i in range(max_new_tokens - 1):
        input_id = reference_tokens[prompt_len + i].view(1, 1)
        outputs = tt_model(input_id, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[0, -1, :]
        target_id = int(reference_tokens[prompt_len + i + 1].item())
        step_top1, step_top5 = score_step(logits, target_id)
        top1 += step_top1
        top5 += step_top5
        total += 1

    return top1 / total, top5 / total, total


def main():
    parser = argparse.ArgumentParser(description="Teacher-forcing eval for ttnn models")
    parser.add_argument("model_path", type=pathlib.Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="1 2 3 4 5 6 7 8 9 10 11 12")
    parser.add_argument("--prompt_file", type=pathlib.Path, default=None)
    parser.add_argument("--prompt_ids_file", type=pathlib.Path, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--cache_dir", default=None, help="Cache directory for HuggingFace downloads")
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "yt_metrics"],
        default=os.environ.get("YT_OUTPUT_FORMAT", "text"),
    )
    args = parser.parse_args()

    structured_output = args.output_format != "text"

    def log(line: str) -> None:
        if not structured_output:
            print(line)

    if args.max_seq_len < 2048:
        raise ValueError(
            f"--max_seq_len={args.max_seq_len} is too small; "
            "ttnn rotary embedding + decode attention assume a 2048-length cache."
        )

    torch.manual_seed(args.seed)
    ttnn.CONFIG.throw_exception_on_fallback = True

    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    log(f"Loading model module: {model_path}")
    model_module = load_model_module(model_path)

    if args.prompt_ids_file is not None and args.prompt_file is not None:
        raise ValueError("Only one of --prompt_file or --prompt_ids_file may be set")

    model_id, system = resolve_tt_metadata(model_path)
    if args.model == DEFAULT_MODEL and model_id != DEFAULT_MODEL:
        print(f"Using HuggingFace model inferred from path: {model_id}")
        args.model = model_id

    log("Loading HuggingFace tokenizer...")
    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log("Loading HuggingFace reference model on CPU...")
    hf_dtype = resolve_hf_dtype(args.hf_dtype)
    hf_model = load_hf_reference_model(args.model, hf_dtype, cache_dir)
    hf_model.eval()

    log("Generating reference tokens...")
    with torch.no_grad():
        attention_mask = None
        if args.prompt_ids_file is not None:
            prompt_ids = json.loads(args.prompt_ids_file.read_text())
            if not isinstance(prompt_ids, list) or not prompt_ids:
                raise ValueError("--prompt_ids_file must contain a JSON list of token ids")
            prompt_ids = [int(token_id) for token_id in prompt_ids]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
        else:
            if args.prompt_file is not None:
                args.prompt = args.prompt_file.read_text()
            encoded = tokenizer(args.prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if args.min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = args.min_new_tokens
        output_ids = hf_model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
    reference_tokens = output_ids[0].cpu()
    prompt_len = input_ids.shape[1]
    actual_new_tokens = reference_tokens.shape[0] - prompt_len
    if actual_new_tokens < args.max_new_tokens:
        log(
            f"Reference generation stopped early (requested {args.max_new_tokens}, got {actual_new_tokens}). "
            "Evaluating available tokens."
        )
    max_new_tokens = actual_new_tokens
    if prompt_len + max_new_tokens > args.max_seq_len:
        raise ValueError(
            f"prompt tokens ({prompt_len}) + new tokens ({max_new_tokens}) exceeds "
            f"--max_seq_len ({args.max_seq_len}); increase --max_seq_len"
        )

    mesh_shape = pick_mesh_shape(system, model_module)
    fabric_config = None
    is_mesh = False
    tt_device = None
    try:
        log(f"Opening device ({mesh_shape[0]}x{mesh_shape[1]})...")
        tt_device, is_mesh, fabric_config = open_tt_device(mesh_shape, args.device_id)

        log("Loading ttnn model...")
        tt_model = build_tt_model(model_module, hf_model, tt_device, args.max_seq_len)
        tt_model.eval()

        log(f"Running teacher-forcing eval ({max_new_tokens} tokens)...")
        with torch.no_grad():
            top1, top5, total = evaluate(
                tt_model,
                reference_tokens,
                prompt_len,
                max_new_tokens,
            )

        if total == 0:
            log("No tokens to evaluate.")
            return

        top1_pct = top1 * 100
        top5_pct = top5 * 100
        metrics = {
            "mode": "tt_eval",
            "model": args.model,
            "top1": top1,
            "top5": top5,
            "top1_pct": top1_pct,
            "top5_pct": top5_pct,
            "total_tokens": total,
            "max_new_tokens": max_new_tokens,
            "max_seq_len": args.max_seq_len,
        }
        payload = json.dumps(metrics)
        if args.output_format == "text":
            print(f"Top-1 accuracy: {top1_pct:.2f}% ({top1:.4f})")
            print(f"Top-5 accuracy: {top5_pct:.2f}% ({top5:.4f})")
            print(f"YT_METRICS={payload}")
        else:
            if args.output_format == "json":
                print(payload)
            else:
                print(f"YT_METRICS={payload}")

    finally:
        close_tt_device(tt_device, is_mesh, fabric_config)


if __name__ == "__main__":
    main()
