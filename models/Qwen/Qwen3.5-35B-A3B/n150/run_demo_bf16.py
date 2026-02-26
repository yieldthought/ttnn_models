#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run demo.py while forcing HuggingFace model load in bfloat16."""

import runpy
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_orig_from_pretrained = AutoModelForCausalLM.from_pretrained


def _from_pretrained_bf16(*args, **kwargs):
    kwargs["torch_dtype"] = torch.bfloat16
    return _orig_from_pretrained(*args, **kwargs)


AutoModelForCausalLM.from_pretrained = _from_pretrained_bf16
sys.argv = ["demo.py", *sys.argv[1:]]
runpy.run_path(str(REPO_ROOT / "demo.py"), run_name="__main__")
