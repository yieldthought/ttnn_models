# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Teacher-forcing evaluation for ttnn model bring-up.

Usage:
    python eval.py models/meta-llama/Llama-3.2-1B/n150/functional/model.py
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import torch
import ttnn
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_module(model_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("ttnn_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_tt_model(module, hf_model, tt_device, max_seq_len: int):
    if hasattr(module, "build_model"):
        return module.build_model(hf_model, tt_device, max_seq_len)
    if hasattr(module, "TtnnLlamaForCausalLM"):
        return module.TtnnLlamaForCausalLM(hf_model, tt_device, max_seq_len)
    raise AttributeError("Model module must define build_model or TtnnLlamaForCausalLM")


def score_step(logits: torch.Tensor, target_id: int) -> tuple[int, int]:
    top5 = torch.topk(logits, k=5).indices
    top1 = int(top5[0].item() == target_id)
    top5_hit = int((top5 == target_id).any().item())
    return top1, top5_hit


def prefill_with_decode(tt_model, prompt_tokens: torch.Tensor):
    past = None
    logits = None
    for token_id in prompt_tokens:
        outputs = tt_model(token_id.view(1, 1), past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[0, -1, :]
    return past, logits


def evaluate(
    tt_model,
    reference_tokens: torch.Tensor,
    prompt_len: int,
    max_new_tokens: int,
    prefill_decode: bool,
) -> tuple[float, float, int]:
    if max_new_tokens < 1:
        return 0.0, 0.0, 0

    top1 = 0
    top5 = 0
    total = 0

    if prefill_decode:
        past, logits = prefill_with_decode(tt_model, reference_tokens[:prompt_len])
    else:
        past = None
        prompt_ids = reference_tokens[:prompt_len].unsqueeze(0)
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
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt", default="1 2 3 4 5 6 7 8 9 10 11 12")
    parser.add_argument("--prompt_file", type=pathlib.Path, default=None)
    parser.add_argument("--prompt_ids_file", type=pathlib.Path, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--prefill_decode", action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default=None, help="Cache directory for HuggingFace downloads")
    args = parser.parse_args()

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

    print(f"Loading model module: {model_path}")
    model_module = load_model_module(model_path)

    if args.prompt_ids_file is not None and args.prompt_file is not None:
        raise ValueError("Only one of --prompt_file or --prompt_ids_file may be set")

    print("Loading HuggingFace tokenizer...")
    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading HuggingFace reference model on CPU...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, cache_dir=cache_dir)
    hf_model.eval()

    print("Generating reference tokens...")
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
        print(
            f"Reference generation stopped early (requested {args.max_new_tokens}, got {actual_new_tokens}). "
            "Evaluating available tokens."
        )
    max_new_tokens = actual_new_tokens
    if prompt_len + max_new_tokens > args.max_seq_len:
        raise ValueError(
            f"prompt tokens ({prompt_len}) + new tokens ({max_new_tokens}) exceeds "
            f"--max_seq_len ({args.max_seq_len}); increase --max_seq_len"
        )

    print("Opening device...")
    tt_device = ttnn.open_device(device_id=args.device_id)

    try:
        print("Loading ttnn model...")
        tt_model = build_tt_model(model_module, hf_model, tt_device, args.max_seq_len)
        tt_model.eval()

        print(f"Running teacher-forcing eval ({max_new_tokens} tokens)...")
        with torch.no_grad():
            top1, top5, total = evaluate(
                tt_model,
                reference_tokens,
                prompt_len,
                max_new_tokens,
                args.prefill_decode,
            )

        if total == 0:
            print("No tokens to evaluate.")
            return

        top1_pct = top1 * 100
        top5_pct = top5 * 100
        print(f"Top-1 accuracy: {top1_pct:.2f}% ({top1:.4f})")
        print(f"Top-5 accuracy: {top5_pct:.2f}% ({top5:.4f})")

    finally:
        ttnn.close_device(tt_device)


if __name__ == "__main__":
    main()
