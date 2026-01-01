#!/usr/bin/env python
"""Run eval.py in HF or TT mode and emit YT_METRICS JSON."""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PREFILL_LEN = 20
DEFAULT_DECODE_LEN = 20
PREFILL_DECODE_THRESHOLD = 512


def load_registry(registry_path: pathlib.Path) -> dict:
    """Load the HF model id -> model directory registry."""
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing registry: {registry_path}")
    registry = json.loads(registry_path.read_text())
    if not isinstance(registry, dict):
        raise ValueError("models/registry.json must contain a JSON object")
    return registry


def resolve_model_path(repo_root: pathlib.Path, hf_model_id: str, system: str, registry: dict) -> pathlib.Path:
    """Resolve the model.py path for a given HF model id and system."""
    model_dir = registry.get(hf_model_id)
    if not model_dir:
        raise KeyError(f"HF model id not found in registry: {hf_model_id}")
    model_path = repo_root / "models" / model_dir / system / "functional" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return model_path


def pick_seed_tokens(tokenizer) -> list:
    """Pick a stable set of non-special token ids to form prompts."""
    seed_tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog.", add_special_tokens=False)
    seed_tokens = [token_id for token_id in seed_tokens if token_id not in tokenizer.all_special_ids]
    if seed_tokens:
        return seed_tokens
    for token_id in range(tokenizer.vocab_size):
        if token_id not in tokenizer.all_special_ids:
            return [token_id]
    raise ValueError("Tokenizer has no non-special tokens")


def build_prompt_ids(tokenizer, prefill_len: int) -> list:
    """Build a prompt id list of exactly prefill_len tokens including special tokens."""
    if prefill_len < 1:
        raise ValueError("prefill_len must be >= 1")

    special_ids = tokenizer.build_inputs_with_special_tokens([])
    special_len = len(special_ids)
    if prefill_len <= special_len:
        raise ValueError("prefill_len must exceed the number of special tokens")

    seed_tokens = pick_seed_tokens(tokenizer)
    base_len = prefill_len - special_len
    tokens = (seed_tokens * (base_len // len(seed_tokens) + 1))[:base_len]
    prompt_ids = tokenizer.build_inputs_with_special_tokens(tokens)

    if len(prompt_ids) != prefill_len:
        raise ValueError("Failed to build prompt ids with requested length")

    return prompt_ids


def score_step(logits: torch.Tensor, target_id: int) -> tuple[int, int]:
    top5 = torch.topk(logits, k=5).indices
    top1 = int(top5[0].item() == target_id)
    top5_hit = int((top5 == target_id).any().item())
    return top1, top5_hit


def run_hf_eval(hf_model_id: str, tokenizer, prompt_ids: list, decode_len: int, cache_dir):
    """Compute teacher-forcing accuracy using the HF reference model only."""
    if decode_len < 1:
        return 0.0, 0.0, 0

    model = AutoModelForCausalLM.from_pretrained(hf_model_id, torch_dtype=torch.float32, cache_dir=cache_dir)
    model.eval()

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=decode_len,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    reference_tokens = output_ids[0]
    prompt_len = input_ids.shape[1]
    actual_new_tokens = reference_tokens.shape[0] - prompt_len
    if actual_new_tokens < 1:
        return 0.0, 0.0, 0

    top1 = 0
    top5 = 0
    total = 0

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[0, -1, :]

        target_id = int(reference_tokens[prompt_len].item())
        step_top1, step_top5 = score_step(logits, target_id)
        top1 += step_top1
        top5 += step_top5
        total += 1

        for i in range(actual_new_tokens - 1):
            input_id = reference_tokens[prompt_len + i].view(1, 1)
            outputs = model(input_id, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            target_id = int(reference_tokens[prompt_len + i + 1].item())
            step_top1, step_top5 = score_step(logits, target_id)
            top1 += step_top1
            top5 += step_top5
            total += 1

    return top1 / total, top5 / total, total


def parse_eval_output(output: str) -> tuple[float, float]:
    """Extract top1/top5 values from eval.py output."""
    top1 = None
    top5 = None
    for line in output.splitlines():
        if line.startswith("Top-1 accuracy:"):
            start = line.rfind("(")
            end = line.rfind(")")
            if start != -1 and end != -1:
                top1 = float(line[start + 1 : end])
        if line.startswith("Top-5 accuracy:"):
            start = line.rfind("(")
            end = line.rfind(")")
            if start != -1 and end != -1:
                top5 = float(line[start + 1 : end])
    if top1 is None or top5 is None:
        raise ValueError("Failed to parse eval.py output")
    return top1, top5


def write_prompt_ids(prompt_ids: list, directory: pathlib.Path) -> pathlib.Path:
    """Write prompt ids to a JSON file inside directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "prompt_ids.json"
    path.write_text(json.dumps(prompt_ids))
    return path


def run_tt_eval(
    repo_root: pathlib.Path,
    hf_model_id: str,
    model_path: pathlib.Path,
    prompt_ids: list,
    decode_len: int,
    cache_dir,
    prefill_decode: bool,
    max_seq_len: int,
) -> tuple[float, float]:
    """Run eval.py and parse top1/top5."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        prompt_ids_file = write_prompt_ids(prompt_ids, pathlib.Path(tmp_dir))
        cmd = [
            sys.executable,
            "eval.py",
            str(model_path),
            "--model",
            hf_model_id,
            "--prompt_ids_file",
            str(prompt_ids_file),
            "--max_new_tokens",
            str(decode_len),
            "--max_seq_len",
            str(max_seq_len),
        ]
        if cache_dir:
            cmd.extend(["--cache_dir", cache_dir])
        if prefill_decode:
            cmd.append("--prefill_decode")
        result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError("eval.py failed")

    return parse_eval_output(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Wrapper for eval.py with YT_METRICS output")
    parser.add_argument("--mode", choices=["hf", "tt"], required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--system", default=os.environ.get("YT_SYSTEM", "n150"))
    parser.add_argument("--prefill-len", type=int, default=DEFAULT_PREFILL_LEN)
    parser.add_argument("--decode-len", type=int, default=DEFAULT_DECODE_LEN)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--trace", type=int, default=0)
    parser.add_argument("--prefill-decode", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    if args.batch != 1:
        raise ValueError("Only batch=1 is supported by the bringup eval")
    if args.prefill_len < 1:
        raise ValueError("--prefill-len must be >= 1")
    if args.decode_len < 0:
        raise ValueError("--decode-len must be >= 0")
    if args.trace not in (0, 1):
        raise ValueError("--trace must be 0 or 1")

    repo_root = pathlib.Path(__file__).resolve().parents[1]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt_ids = build_prompt_ids(tokenizer, args.prefill_len)

    top1 = 0.0
    top5 = 0.0
    total = 0

    if args.mode == "hf":
        top1, top5, total = run_hf_eval(args.hf_model, tokenizer, prompt_ids, args.decode_len, args.cache_dir)
    else:
        registry = load_registry(repo_root / "models" / "registry.json")
        model_path = resolve_model_path(repo_root, args.hf_model, args.system, registry)
        max_seq_len = max(2048, args.prefill_len + args.decode_len)
        prefill_decode = args.prefill_decode or args.prefill_len > PREFILL_DECODE_THRESHOLD
        top1, top5 = run_tt_eval(
            repo_root,
            args.hf_model,
            model_path,
            prompt_ids,
            args.decode_len,
            args.cache_dir,
            prefill_decode,
            max_seq_len,
        )
        total = max(args.decode_len, 0)

    metrics = {
        "mode": args.mode,
        "trace": bool(args.trace),
        "top1": float(top1),
        "top5": float(top5),
        "prefill_len": args.prefill_len,
        "decode_len": args.decode_len,
        "batch": args.batch,
        "total": int(total),
    }
    print(f"YT_METRICS={json.dumps(metrics)}")


if __name__ == "__main__":
    main()
