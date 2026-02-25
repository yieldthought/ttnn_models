#!/usr/bin/env python
"""Run eval.py in HF or TT mode and emit YT_METRICS JSON."""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings(
    "ignore",
    message="Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Passing a tuple of `past_key_values` is deprecated",
)


def maybe_prepend_transformers_path() -> None:
    """Prepend an optional external transformers runtime to sys.path."""
    extra_path = os.environ.get("TTNN_TRANSFORMERS_PYTHONPATH", "")
    if not extra_path:
        return
    for path in reversed(extra_path.split(":")):
        if path and path not in sys.path:
            sys.path.insert(0, path)


maybe_prepend_transformers_path()

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()


DEFAULT_PREFILL_LEN = 20
DEFAULT_DECODE_LEN = 20


def emit_metrics(metrics: dict, output_format: str) -> None:
    """Emit metrics in the requested output format."""
    payload = json.dumps(metrics)
    if output_format == "json":
        print(payload)
        return
    print(f"YT_METRICS={payload}")


def resolve_model_path(repo_root: pathlib.Path, hf_model_id: str, system: str) -> pathlib.Path:
    """Resolve model.py path using the HF model id as the directory convention."""
    model_path = repo_root / "models" / hf_model_id / system / "functional" / "model.py"
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


def parse_prefill_len_range(spec: str) -> list:
    """Parse a start:end[:step] range (end inclusive) into a list of lengths."""
    parts = [int(part) for part in spec.split(":") if part]
    if len(parts) not in (2, 3):
        raise ValueError("--prefill-len-range must be start:end or start:end:step")
    start = parts[0]
    end = parts[1]
    step = parts[2] if len(parts) == 3 else 1
    if start < 1 or end < start or step < 1:
        raise ValueError("--prefill-len-range must use positive lengths with start <= end")
    return list(range(start, end + 1, step))


def parse_max_seq_len_range(spec: str) -> list:
    """Parse a start:end[:step] range (end inclusive) into a list of lengths."""
    parts = [int(part) for part in spec.split(":") if part]
    if len(parts) not in (2, 3):
        raise ValueError("--max-seq-len-range must be start:end or start:end:step")
    start = parts[0]
    end = parts[1]
    step = parts[2] if len(parts) == 3 else 1
    if start < 1 or end < start or step < 1:
        raise ValueError("--max-seq-len-range must use positive lengths with start <= end")
    return list(range(start, end + 1, step))


def score_step(logits: torch.Tensor, target_id: int) -> tuple[int, int]:
    top5 = torch.topk(logits, k=5).indices
    top1 = int(top5[0].item() == target_id)
    top5_hit = int((top5 == target_id).any().item())
    return top1, top5_hit


def run_hf_eval(hf_model_id: str, tokenizer, prompt_ids: list, decode_len: int, cache_dir):
    """Compute teacher-forcing accuracy using the HF reference model only."""
    if decode_len < 1:
        return 0.0, 0.0, 0

    try:
        model = AutoModelForCausalLM.from_pretrained(hf_model_id, torch_dtype=torch.float32, cache_dir=cache_dir)
    except Exception as causal_error:
        try:
            from transformers import AutoModelForImageTextToText
        except Exception as image_text_import_error:
            raise causal_error from image_text_import_error
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                hf_model_id, torch_dtype=torch.float32, cache_dir=cache_dir
            )
        except Exception as image_text_error:
            raise causal_error from image_text_error
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
    parser.add_argument("--prefill-len-range", default=None)
    parser.add_argument("--decode-len", type=int, default=DEFAULT_DECODE_LEN)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-seq-len-range", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--trace", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output-format",
        choices=["yt_metrics", "json"],
        default=os.environ.get("YT_OUTPUT_FORMAT", "json"),
    )
    args = parser.parse_args()

    if args.batch != 1:
        raise ValueError("Only batch=1 is supported by the bringup eval")
    if args.prefill_len < 1 and args.prefill_len_range is None:
        raise ValueError("--prefill-len must be >= 1")
    if args.decode_len < 0:
        raise ValueError("--decode-len must be >= 0")
    if args.trace not in (0, 1):
        raise ValueError("--trace must be 0 or 1")
    if args.max_seq_len is not None and args.max_seq_len < 1:
        raise ValueError("--max-seq-len must be >= 1")
    if args.prefill_len_range is not None and args.max_seq_len_range is not None:
        raise ValueError("Use --prefill-len-range or --max-seq-len-range, not both")

    repo_root = pathlib.Path(__file__).resolve().parents[1]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prefill_lens = [args.prefill_len]
    if args.prefill_len_range is not None:
        prefill_lens = parse_prefill_len_range(args.prefill_len_range)

    max_seq_lens = None
    if args.max_seq_len_range is not None:
        max_seq_lens = parse_max_seq_len_range(args.max_seq_len_range)
    elif args.max_seq_len is not None:
        max_seq_lens = [args.max_seq_len]

    for prefill_len in prefill_lens:
        prompt_ids = build_prompt_ids(tokenizer, prefill_len)

        top1 = 0.0
        top5 = 0.0
        total = 0

        if args.mode == "hf":
            top1, top5, total = run_hf_eval(args.hf_model, tokenizer, prompt_ids, args.decode_len, args.cache_dir)
            metrics = {
                "mode": args.mode,
                "trace": bool(args.trace),
                "top1": float(top1),
                "top5": float(top5),
                "prefill_len": prefill_len,
                "decode_len": args.decode_len,
                "batch": args.batch,
                "total": int(total),
            }
            emit_metrics(metrics, args.output_format)
            continue

        model_path = resolve_model_path(repo_root, args.hf_model, args.system)
        run_max_seq_lens = max_seq_lens
        if run_max_seq_lens is None:
            run_max_seq_lens = [max(2048, prefill_len + args.decode_len)]

        for max_seq_len in run_max_seq_lens:
            min_seq_len = prefill_len + args.decode_len
            if max_seq_len < min_seq_len:
                raise ValueError("--max-seq-len must be >= prefill_len + decode_len")

            top1, top5 = run_tt_eval(
                repo_root,
                args.hf_model,
                model_path,
                prompt_ids,
                args.decode_len,
                args.cache_dir,
                max_seq_len,
            )
            total = max(args.decode_len, 0)

            metrics = {
                "mode": args.mode,
                "trace": bool(args.trace),
                "top1": float(top1),
                "top5": float(top5),
                "prefill_len": prefill_len,
                "decode_len": args.decode_len,
                "max_seq_len": max_seq_len,
                "batch": args.batch,
                "total": int(total),
            }
            emit_metrics(metrics, args.output_format)


if __name__ == "__main__":
    main()
