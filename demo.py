# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Demo runner for HuggingFace CPU models or TTNN bringup models.

Usage:
    python demo.py meta-llama/Llama-3.2-1B
    python demo.py models/meta-llama/Llama-3.2-1B/n150/functional/model.py
"""

import argparse
import importlib.util
import os
import pathlib
import sys
import time
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = (
    "Journal entry, 1957: Tonight a tiny sphere called Sputnik 1 crossed the sky, "
    "beeping like a metronome for a new era. The neighbors gathered on the roof, "
    "listening and arguing about what comes next. I wrote in my notebook that"
)

SYSTEMS = ("n150", "n300", "t3000")
SYSTEM_MESH_SHAPES = {
    "n150": (1, 1),
    "n300": (1, 2),
    "t3000": (2, 4),
}
SYSTEM_MESH_DESCRIPTORS = {
    "n150": "n150_mesh_graph_descriptor.textproto",
    "n300": "n300_mesh_graph_descriptor.textproto",
    "t3000": "t3k_mesh_graph_descriptor.textproto",
}


def load_model_module(model_path: pathlib.Path):
    """Load a TT model module from a Python file."""
    spec = importlib.util.spec_from_file_location("ttnn_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_tt_model(module, hf_model, tt_device, max_seq_len: int):
    """Build a TT model using the module's build helpers."""
    if hasattr(module, "build_model"):
        return module.build_model(hf_model, tt_device, max_seq_len)
    if hasattr(module, "TtnnLlamaForCausalLM"):
        return module.TtnnLlamaForCausalLM(hf_model, tt_device, max_seq_len)
    raise AttributeError("Model module must define build_model or TtnnLlamaForCausalLM")


def resolve_tt_metadata(model_path: pathlib.Path) -> Tuple[str, str]:
    """Parse HF model id and system name from a TT model path."""
    parts = model_path.parts
    system = None
    system_index = None
    for idx, part in enumerate(parts):
        if part in SYSTEMS:
            system = part
            system_index = idx
            break
    if system is None or system_index is None:
        raise ValueError(f"Failed to infer system (n150/n300/t3000) from {model_path}")

    try:
        models_index = parts.index("models")
    except ValueError as exc:
        raise ValueError(f"Expected model path under a models/ directory: {model_path}") from exc

    hf_parts = parts[models_index + 1 : system_index]
    if not hf_parts:
        raise ValueError(f"Failed to infer HuggingFace model id from {model_path}")

    return "/".join(hf_parts), system


def pick_mesh_shape(system: str, model_module) -> Tuple[int, int]:
    """Pick the mesh shape for a TT model, preferring module hints."""
    module_shape = getattr(model_module, "MESH_SHAPE", None)
    if module_shape is not None:
        return tuple(module_shape)
    return SYSTEM_MESH_SHAPES[system]


def set_mesh_descriptor(system: str, repo_root: pathlib.Path) -> Optional[str]:
    """Set TT_MESH_GRAPH_DESC_PATH if needed and return the path used."""
    if "TT_MESH_GRAPH_DESC_PATH" in os.environ:
        return os.environ["TT_MESH_GRAPH_DESC_PATH"]

    descriptor_name = SYSTEM_MESH_DESCRIPTORS.get(system)
    if descriptor_name is None:
        return None

    descriptor_path = repo_root / ".." / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / descriptor_name
    descriptor_path = descriptor_path.resolve()
    if not descriptor_path.exists():
        return None

    os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(descriptor_path)
    return os.environ["TT_MESH_GRAPH_DESC_PATH"]


def open_tt_device(mesh_shape: Tuple[int, int], system: str, device_id: int):
    """Open a TT device or mesh device based on mesh shape."""
    import ttnn

    is_mesh = mesh_shape != (1, 1)
    fabric_config = None

    if not is_mesh:
        return ttnn.open_device(device_id=device_id), False, None

    descriptor = set_mesh_descriptor(system, pathlib.Path(__file__).resolve().parent)
    if descriptor is None:
        raise FileNotFoundError(
            "Missing TT_MESH_GRAPH_DESC_PATH and no default mesh descriptor found. "
            "Set TT_MESH_GRAPH_DESC_PATH or place tt-metal at ../tt-metal."
        )
    print(f"Using TT_MESH_GRAPH_DESC_PATH={descriptor}")

    fabric_config = ttnn.FabricConfig.FABRIC_2D if mesh_shape[0] > 1 and mesh_shape[1] > 1 else ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(fabric_config)

    system_mesh_desc = ttnn._ttnn.multi_device.SystemMeshDescriptor()
    system_shape = tuple(system_mesh_desc.shape())
    if mesh_shape[0] > system_shape[0] or mesh_shape[1] > system_shape[1]:
        raise RuntimeError(f"Requested mesh {mesh_shape} exceeds system mesh {system_shape}")

    physical_device_ids = []
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            if not system_mesh_desc.is_local(coord):
                raise RuntimeError(f"Mesh coord {(row, col)} is not local to this host")
            physical_device_ids.append(system_mesh_desc.get_device_id(coord))

    tt_device = ttnn.open_mesh_device(
        ttnn.MeshShape(*mesh_shape),
        physical_device_ids=physical_device_ids,
    )
    return tt_device, True, fabric_config


def close_tt_device(tt_device, is_mesh: bool, fabric_config):
    """Close TT device and reset fabric config."""
    import ttnn

    if is_mesh:
        ttnn.close_mesh_device(tt_device)
    else:
        ttnn.close_device(tt_device)

    if fabric_config is not None:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


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


def warmup_model(model, input_ids: torch.Tensor, is_tt: bool, tt_device):
    """Warm up with one prefill and one decode step."""
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        sync_if_needed(tt_device, is_tt)
        logits = outputs.logits[:, -1, :]
        next_token = int(torch.argmax(logits, dim=-1).item())
        next_input = torch.tensor([[next_token]], dtype=torch.long)
        _ = model(next_input, past_key_values=outputs.past_key_values, use_cache=True)
        sync_if_needed(tt_device, is_tt)

    if hasattr(model, "reset"):
        model.reset()


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

    with torch.no_grad():
        start = time.perf_counter()
        if attention_mask is None:
            outputs = model(input_ids, use_cache=True)
        else:
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
        sync_if_needed(tt_device, is_tt)
        prefill_time = time.perf_counter() - start

        logits = outputs.logits[:, -1, :]
        next_token = pick_next_token(logits, temperature, top_k)
        generated = [next_token]
        past = outputs.past_key_values

        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None and next_token == eos_token_id:
            text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return text, len(generated), prefill_time, 0.0, 0

        decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            input_token = torch.tensor([[generated[-1]]], dtype=torch.long)
            outputs = model(input_token, past_key_values=past, use_cache=True)
            sync_if_needed(tt_device, is_tt)
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = pick_next_token(logits, temperature, top_k)
            generated.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
        decode_time = time.perf_counter() - decode_start

    text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    decode_tokens = max(len(generated) - 1, 0)
    return text, len(generated), prefill_time, decode_time, decode_tokens


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
    if max_cache is not None:
        max_total = max_cache
        if input_ids.shape[1] + max_new_tokens > max_total:
            max_new_tokens = max(0, max_total - input_ids.shape[1])
            print(f"Adjusting max_new_tokens to {max_new_tokens} to fit MAX_CACHE_SEQ_LEN={max_cache}")

    max_seq_len = max(2048, input_ids.shape[1] + max_new_tokens)

    print("Opening TT device...")
    ttnn.CONFIG.throw_exception_on_fallback = True
    tt_device = None
    is_mesh = False
    fabric_config = None

    try:
        tt_device, is_mesh, fabric_config = open_tt_device(mesh_shape, system, device_id)

        print(f"Loading HuggingFace reference model on CPU: {model_id}")
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, cache_dir=cache_dir)
        hf_model.eval()

        print("Building TT model...")
        tt_model = build_tt_model(model_module, hf_model, tt_device, max_seq_len)
        tt_model.eval()

        warmup_model(tt_model, input_ids, True, tt_device)

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
            mesh_shape,
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
