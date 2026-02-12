# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Demo runner for HuggingFace CPU models or TTNN bringup models.

Usage:
    python demo.py meta-llama/Llama-3.2-1B
    python demo.py models/meta-llama/Llama-3.2-1B/n150/functional/model.py
"""

import argparse
import pathlib
import os
import sys
import time
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from device_utils import build_tt_model, close_tt_device, load_model_module, open_tt_device, pick_mesh_shape, resolve_tt_metadata

DEFAULT_PROMPT = (
    "Journal entry, 1957: Tonight a tiny sphere called Sputnik 1 crossed the sky, "
    "beeping like a metronome for a new era. The neighbors gathered on the roof, "
    "listening and arguing about what comes next. I wrote in my notebook that"
)



def read_prompt(prompt: Optional[str], prompt_file: Optional[pathlib.Path]) -> str:
    """Pick the prompt string from CLI args or default."""
    if prompt is not None and prompt_file is not None:
        raise ValueError("Only one of --prompt or --prompt-file may be set")
    if prompt_file is not None:
        return prompt_file.read_text()
    if prompt is not None:
        return prompt
    return DEFAULT_PROMPT


def sync_if_needed(tt_device, is_tt: bool):
    """Synchronize the TT device after a run for timing accuracy."""
    if not is_tt:
        return
    import ttnn

    ttnn.synchronize_device(tt_device)


def warmup_model(model, input_ids: torch.Tensor, is_tt: bool, tt_device, warmup_device_sampling: bool = False):
    """Warm up kernels with one prefill pass and one decode step."""
    trace_enabled = getattr(model, "use_decode_trace", None)
    if trace_enabled is not None:
        model.use_decode_trace = False
    try:
        with torch.no_grad():
            use_device_sampling = is_tt and warmup_device_sampling and hasattr(model, "next_token_device")
            if use_device_sampling:
                next_token, past = model.next_token_device(input_ids, past_key_values=None, use_cache=True)
                next_input = torch.tensor([[next_token]], dtype=torch.long)
                _ = model.next_token_device(next_input, past_key_values=past, use_cache=True)
            else:
                past = None
                if is_tt and hasattr(model, "prefill_logits_last_device"):
                    logits, past = model.prefill_logits_last_device(input_ids, use_cache=True)
                else:
                    outputs = model(input_ids, use_cache=True)
                    past = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                next_token = int(torch.argmax(logits, dim=-1).item())
                next_input = torch.tensor([[next_token]], dtype=torch.long)
                _ = model(next_input, past_key_values=past, use_cache=True)
            sync_if_needed(tt_device, is_tt)
    finally:
        if trace_enabled is not None:
            model.use_decode_trace = trace_enabled

    if hasattr(model, "reset"):
        model.reset()


def maybe_tt_signpost(is_tt: bool, name: str, attributes: Optional[str] = None) -> None:
    if not is_tt:
        return
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1" and os.environ.get("TT_SIGNPOSTS") != "1":
        return
    import ttnn

    message = f"TT_SIGNPOST: {name}"
    if attributes:
        message = f"{message}\n{attributes}"
    ttnn.tracy_message(message)


def pick_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """Pick the next token using temperature + top-k sampling or greedy."""
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    scaled = logits / temperature
    vocab_size = scaled.shape[-1]
    if top_k is not None and top_k > 0:
        k = min(top_k, vocab_size)
        values, indices = torch.topk(scaled, k=k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        sample = torch.multinomial(probs, num_samples=1)
        return int(indices.gather(-1, sample).item())

    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_with_timing(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    is_tt: bool,
    tt_device,
) -> Tuple[str, int, float, float, int]:
    """Generate text with separate prefill/decode timing."""
    if max_new_tokens < 1:
        return "", 0, 0.0, 0.0, 0

    use_device_sampling = is_tt and hasattr(model, "next_token_device") and temperature <= 0.0

    with torch.no_grad():
        maybe_tt_signpost(is_tt, "PREFILL_START")
        start = time.perf_counter()
        if use_device_sampling:
            next_token, past = model.next_token_device(input_ids, past_key_values=None, use_cache=True)
        else:
            if is_tt and hasattr(model, "prefill_logits_last_device"):
                logits, past = model.prefill_logits_last_device(input_ids, use_cache=True)
                next_token = pick_next_token(logits, temperature, top_k)
            else:
                if attention_mask is None:
                    outputs = model(input_ids, use_cache=True)
                else:
                    outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
                next_token = pick_next_token(outputs.logits[:, -1, :], temperature, top_k)
                past = outputs.past_key_values
        if not use_device_sampling:
            sync_if_needed(tt_device, is_tt)
        prefill_time = time.perf_counter() - start
        maybe_tt_signpost(is_tt, "PREFILL_END")
        generated = [next_token]

        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None and next_token == eos_token_id:
            text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return text, len(generated), prefill_time, 0.0, 0

        input_token = torch.empty((1, 1), dtype=torch.long)
        timed_decode_tokens = 0
        remaining_steps = max_new_tokens - 1

        trace_needs_capture = (
            is_tt
            and hasattr(model, "use_decode_trace")
            and bool(getattr(model, "use_decode_trace"))
            and hasattr(model, "decode_trace_id")
            and getattr(model, "decode_trace_id") is None
            and remaining_steps > 0
        )

        if trace_needs_capture and remaining_steps > 0:
            # Capture the decode trace after prefill, so we don't run a full prefill while a trace exists.
            # Exclude this priming step from the timed decode throughput.
            input_token[0, 0] = generated[-1]
            if use_device_sampling:
                next_token, past = model.next_token_device(input_token, past_key_values=past, use_cache=True)
            else:
                outputs = model(input_token, past_key_values=past, use_cache=True)
                past = outputs.past_key_values
                next_token = pick_next_token(outputs.logits[:, -1, :], temperature, top_k)
            sync_if_needed(tt_device, is_tt)
            generated.append(next_token)
            remaining_steps -= 1
            if eos_token_id is not None and next_token == eos_token_id:
                text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return text, len(generated), prefill_time, 0.0, 0

        maybe_tt_signpost(is_tt, "DECODE_TIMING_START")
        decode_start = time.perf_counter()
        for _ in range(remaining_steps):
            input_token[0, 0] = generated[-1]
            if use_device_sampling:
                next_token, past = model.next_token_device(input_token, past_key_values=past, use_cache=True)
            else:
                outputs = model(input_token, past_key_values=past, use_cache=True)
                past = outputs.past_key_values
                next_token = pick_next_token(outputs.logits[:, -1, :], temperature, top_k)
            if not use_device_sampling:
                sync_if_needed(tt_device, is_tt)
            generated.append(next_token)
            timed_decode_tokens += 1
            if eos_token_id is not None and next_token == eos_token_id:
                break
        decode_time = time.perf_counter() - decode_start
        maybe_tt_signpost(is_tt, "DECODE_TIMING_END")

    text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text, len(generated), prefill_time, decode_time, timed_decode_tokens


def use_color() -> bool:
    return sys.stdout.isatty()


def colorize(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def print_report(
    mode: str,
    model_name: str,
    system: Optional[str],
    mesh_shape: Optional[Tuple[int, int]],
    prompt: str,
    output: str,
    prompt_tokens: int,
    generated_tokens: int,
    ttft_ms: float,
    decode_tps: float,
    decode_tokens: int,
):
    """Pretty-print demo output and timing."""
    enabled = use_color()
    header = f"{mode.upper()} demo"
    if system is not None:
        header = f"{header} ({system})"
    print(colorize(header, "1;37", enabled))
    print(f"Model: {model_name}")
    if mesh_shape is not None:
        print(f"Mesh shape: {mesh_shape[0]}x{mesh_shape[1]}")
    print(f"Prompt tokens: {prompt_tokens} | Generated tokens: {generated_tokens}")
    print(f"TTFT: {ttft_ms:.0f} ms | Decode: {decode_tps:.1f} t/s/u ({decode_tokens} tokens)")
    print()
    print(colorize("Prompt:", "1;34", enabled))
    print(colorize(prompt, "34", enabled))
    print()
    print(colorize("Output:", "1;32", enabled))
    print(colorize(output, "32", enabled))


def run_hf_demo(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    cache_dir: Optional[str],
):
    """Run HF CPU model generation with timing."""
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    print(f"Loading HuggingFace model on CPU: {model_id}")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, cache_dir=cache_dir)
    hf_model.eval()

    warmup_model(hf_model, input_ids, False, None)

    output, generated_tokens, prefill_time, decode_time, decode_tokens = generate_with_timing(
        hf_model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        temperature,
        top_k,
        False,
        None,
    )
    ttft_ms = prefill_time * 1000
    decode_tps = 0.0 if decode_time <= 0.0 else decode_tokens / decode_time
    print_report(
        "hf",
        model_id,
        None,
        None,
        prompt,
        output,
        input_ids.shape[1],
        generated_tokens,
        ttft_ms,
        decode_tps,
        decode_tokens,
    )


def run_tt_demo(
    model_path: pathlib.Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    cache_dir: Optional[str],
    device_id: int,
    max_seq_len: Optional[int],
):
    """Run TT model generation with timing."""
    import ttnn

    model_path = model_path.resolve()
    model_module = load_model_module(model_path)
    model_id, system = resolve_tt_metadata(model_path)
    mesh_shape = pick_mesh_shape(system, model_module)

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    max_cache = getattr(model_module, "MAX_CACHE_SEQ_LEN", None)
    max_total = max_seq_len
    limit_name = "max_seq_len"
    if max_total is None:
        max_total = max_cache
        limit_name = "MAX_CACHE_SEQ_LEN"
    elif max_cache is not None and max_cache < max_total:
        max_total = max_cache
        limit_name = "MAX_CACHE_SEQ_LEN"
    if max_total is not None and input_ids.shape[1] + max_new_tokens > max_total:
        max_new_tokens = max(0, max_total - input_ids.shape[1])
        print(f"Adjusting max_new_tokens to {max_new_tokens} to fit {limit_name}={max_total}")

    if max_seq_len is None:
        max_seq_len = max(2048, input_ids.shape[1] + max_new_tokens)
    elif max_seq_len < input_ids.shape[1] + max_new_tokens:
        print(
            "Warning: max_seq_len is smaller than prompt + max_new_tokens; "
            "generation may fail if cache limits are exceeded."
        )

    print("Opening TT device...")
    ttnn.CONFIG.throw_exception_on_fallback = True
    tt_device = None
    is_mesh = False
    fabric_config = None

    try:
        tt_device, is_mesh, fabric_config = open_tt_device(mesh_shape, device_id)
        runtime_mesh_shape = tuple(tt_device.shape) if hasattr(tt_device, "shape") else (1, 1)

        print(f"Loading HuggingFace reference model on CPU: {model_id}")
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, cache_dir=cache_dir)
        hf_model.eval()

        print("Building TT model...")
        tt_model = build_tt_model(model_module, hf_model, tt_device, max_seq_len)
        tt_model.eval()

        print("Running one warmup prefill+decode pass (kernel compile warmup)...")
        warmup_model(tt_model, input_ids, True, tt_device, warmup_device_sampling=temperature <= 0.0)

        output, generated_tokens, prefill_time, decode_time, decode_tokens = generate_with_timing(
            tt_model,
            tokenizer,
            input_ids,
            attention_mask,
            max_new_tokens,
            temperature,
            top_k,
            True,
            tt_device,
        )
        ttft_ms = prefill_time * 1000
        decode_tps = 0.0 if decode_time <= 0.0 else decode_tokens / decode_time
        print_report(
            "tt",
            model_id,
            system,
            runtime_mesh_shape,
            prompt,
            output,
            input_ids.shape[1],
            generated_tokens,
            ttft_ms,
            decode_tps,
            decode_tokens,
        )
    finally:
        if tt_device is not None:
            close_tt_device(tt_device, is_mesh, fabric_config)


def main():
    parser = argparse.ArgumentParser(description="Demo runner for HF or TT models")
    parser.add_argument("model", help="HF model id or TT model.py path")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", type=pathlib.Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    prompt = read_prompt(args.prompt, args.prompt_file)
    model_path = pathlib.Path(args.model)

    if model_path.exists():
        run_tt_demo(
            model_path,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.cache_dir,
            args.device_id,
            args.max_seq_len,
        )
    else:
        run_hf_demo(
            args.model,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.cache_dir,
        )


if __name__ == "__main__":
    main()
