import json
import os
import pathlib
import subprocess
import sys


def parse_metrics(output: str):
    text = output.strip()
    if text.startswith("{"):
        return json.loads(text)
    for line in text.splitlines():
        if line.startswith("YT_METRICS="):
            payload = line.split("=", 1)[1].strip()
            return json.loads(payload)
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
    metrics = parse_metrics(result.stdout)
    assert metrics is not None
    assert metrics["mode"] == "hf"
    assert metrics["top1"] >= 0.99
    assert metrics["top5"] >= 0.99


def test_run_eval_tt_failure_emits_json(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cache_dir = tmp_path / "hf_cache"
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--mode",
        "tt",
        "--hf-model",
        "sshleifer/tiny-gpt2",
        "--system",
        "missing-system",
        "--prefill-len",
        "8",
        "--decode-len",
        "1",
        "--cache-dir",
        str(cache_dir),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    metrics = parse_metrics(result.stdout)
    assert result.returncode != 0
    assert metrics is not None
    assert metrics["mode"] == "tt"
    assert metrics["status"] == "error"


def test_run_eval_parse_failure_emits_json():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--mode",
        "tt",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    metrics = parse_metrics(result.stdout)
    assert result.returncode != 0
    assert metrics is not None
    assert metrics["status"] == "error"


def test_run_eval_yt_metrics_output_format():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--mode",
        "tt",
        "--hf-model",
        "sshleifer/tiny-gpt2",
        "--system",
        "missing-system",
        "--prefill-len",
        "8",
        "--decode-len",
        "1",
        "--output-format",
        "yt_metrics",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    metrics = parse_metrics(result.stdout)
    assert result.returncode != 0
    assert result.stdout.startswith("YT_METRICS=")
    assert metrics is not None
    assert metrics["status"] == "error"


def test_run_eval_env_override_does_not_change_default_output_format():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = dict(**os.environ)
    env["YT_METRICS_FORMAT"] = "yt_metrics"
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--mode",
        "tt",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, env=env)
    metrics = parse_metrics(result.stdout)
    assert result.returncode != 0
    assert result.stdout.strip().startswith("{")
    assert metrics is not None
    assert metrics["status"] == "error"


def test_run_eval_help_emits_json():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--help",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    metrics = parse_metrics(result.stdout)
    assert result.returncode != 0
    assert metrics is not None
    assert metrics["status"] == "error"


def test_run_eval_missing_deps_emits_json():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-S",
        "scripts/run_eval.py",
        "--mode",
        "tt",
        "--hf-model",
        "arcee-ai/Arcee-Spark",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    metrics = parse_metrics(result.stdout)
    assert result.returncode != 0
    assert metrics is not None
    assert metrics["status"] == "error"
