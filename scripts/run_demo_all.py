#!/usr/bin/env python
"""Run demo.py for a set of TTNN models and save outputs."""

import argparse
import pathlib
import subprocess
import sys


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def load_model_paths(models_file: pathlib.Path | None, filter_text: str | None) -> list[pathlib.Path]:
    root = repo_root()
    if models_file is None:
        paths = sorted((root / "models").rglob("functional/model.py"))
    else:
        lines = [line.strip() for line in models_file.read_text().splitlines()]
        lines = [line for line in lines if line and not line.startswith("#")]
        paths = []
        for line in lines:
            path = pathlib.Path(line)
            if not path.is_absolute():
                path = root / path
            paths.append(path)

    if filter_text:
        paths = [path for path in paths if filter_text in str(path)]

    return paths


def slug_from_model_path(model_path: pathlib.Path) -> str:
    parts = model_path.parts
    try:
        models_index = parts.index("models")
    except ValueError:
        return model_path.stem

    if len(parts) - models_index >= 5:
        system = parts[-3]
        hf_parts = parts[models_index + 1 : -3]
        if hf_parts:
            return "_".join(list(hf_parts) + [system])
    return model_path.stem


def write_trimmed_output(log_path: pathlib.Path, summary_path: pathlib.Path) -> bool:
    lines = log_path.read_text().splitlines()
    start_index = None
    for idx, line in enumerate(lines):
        if line.startswith("TT demo") or line.startswith("HF demo"):
            start_index = idx
            break
    if start_index is None:
        return False
    summary_path.write_text("\n".join(lines[start_index:]) + "\n")
    return True


def run_demo(model_path: pathlib.Path, output_dir: pathlib.Path, args) -> tuple[bool, str]:
    slug = slug_from_model_path(model_path)
    log_path = output_dir / f"{slug}.log"
    summary_path = output_dir / f"{slug}.txt"

    cmd = [
        sys.executable,
        "demo.py",
        str(model_path),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
    ]
    if args.prompt is not None:
        cmd.extend(["--prompt", args.prompt])
    if args.prompt_file is not None:
        cmd.extend(["--prompt-file", str(args.prompt_file)])
    if args.cache_dir is not None:
        cmd.extend(["--cache-dir", args.cache_dir])
    cmd.extend(["--device-id", str(args.device_id)])

    with log_path.open("w") as log_file:
        log_file.write(f"Command: {' '.join(cmd)}\n\n")
        result = subprocess.run(cmd, cwd=repo_root(), stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        return False, f"demo.py failed (exit {result.returncode})"

    if not write_trimmed_output(log_path, summary_path):
        return False, "failed to locate demo output block"

    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Run demo.py across TTNN models")
    parser.add_argument("--models-file", type=pathlib.Path, default=None)
    parser.add_argument("--filter", dest="filter_text", default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("demo_outputs"))
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", type=pathlib.Path, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = load_model_paths(args.models_file, args.filter_text)
    if not model_paths:
        raise SystemExit("No model paths found")

    failures = []
    for model_path in model_paths:
        print(f"Running {model_path}")
        ok, message = run_demo(model_path, output_dir, args)
        if not ok:
            failures.append((model_path, message))
            if not args.keep_going:
                break

    if failures:
        print("Failures:")
        for model_path, message in failures:
            print(f"- {model_path}: {message}")
        raise SystemExit(1)

    print("All demos completed.")


if __name__ == "__main__":
    main()
