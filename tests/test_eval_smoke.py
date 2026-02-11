import json
import pathlib
import subprocess
import sys


def parse_metrics(output: str):
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("YT_METRICS="):
            payload = stripped.split("=", 1)[1].strip()
            return json.loads(payload)
        if stripped.startswith("{"):
            return json.loads(stripped)
    return None


def test_run_eval_hf_smoke(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cache_dir = tmp_path / "hf_cache"
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--mode",
        "hf",
        "--hf-model",
        "sshleifer/tiny-gpt2",
        "--prefill-len",
        "8",
        "--decode-len",
        "4",
        "--cache-dir",
        str(cache_dir),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert output_lines
    assert output_lines[0].startswith("YT_METRICS=")
    metrics = parse_metrics(result.stdout)
    assert metrics is not None
    assert metrics["mode"] == "hf"
    assert metrics["top1"] >= 0.99
    assert metrics["top5"] >= 0.99


def test_run_eval_hf_smoke_json_output(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cache_dir = tmp_path / "hf_cache"
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--mode",
        "hf",
        "--hf-model",
        "sshleifer/tiny-gpt2",
        "--prefill-len",
        "8",
        "--decode-len",
        "4",
        "--output-format",
        "json",
        "--cache-dir",
        str(cache_dir),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert output_lines
    assert output_lines[0].strip().startswith("{")
    metrics = parse_metrics(result.stdout)
    assert metrics is not None
    assert metrics["mode"] == "hf"
    assert metrics["top1"] >= 0.99
    assert metrics["top5"] >= 0.99
