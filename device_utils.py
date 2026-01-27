# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import pathlib
import sys
from typing import Tuple

TRACE_REGION_SIZE = int(os.environ.get("TTNN_TRACE_REGION_SIZE", "10000000"))

SYSTEMS = ("n150", "n300", "t3000")
SYSTEM_MESH_SHAPES = {
    "n150": (1, 1),
    "n300": (1, 2),
    "t3000": (2, 4),
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


def open_tt_device(mesh_shape: Tuple[int, int], device_id: int):
    """Open a TT device or mesh device based on mesh shape."""
    import ttnn

    is_mesh = mesh_shape != (1, 1)
    fabric_config = None

    if not is_mesh:
        if TRACE_REGION_SIZE <= 0:
            return ttnn.open_device(device_id=device_id), False, None
        device = ttnn.CreateDevice(device_id, trace_region_size=TRACE_REGION_SIZE)
        ttnn.SetDefaultDevice(device)
        return device, False, None

    fabric_config = ttnn.FabricConfig.FABRIC_2D if mesh_shape[0] > 1 and mesh_shape[1] > 1 else ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(fabric_config)

    system_mesh_desc = ttnn._ttnn.multi_device.SystemMeshDescriptor()
    system_shape = tuple(system_mesh_desc.shape())
    if mesh_shape[0] > system_shape[0] or mesh_shape[1] > system_shape[1]:
        raise RuntimeError(f"Requested mesh {mesh_shape} exceeds system mesh {system_shape}")

    tt_device = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape), trace_region_size=TRACE_REGION_SIZE)
    return tt_device, True, fabric_config


def close_tt_device(tt_device, is_mesh: bool, fabric_config):
    """Close TT device and reset fabric config."""
    import ttnn

    if tt_device is None:
        return

    if is_mesh:
        ttnn.close_mesh_device(tt_device)
    else:
        if TRACE_REGION_SIZE <= 0:
            ttnn.close_device(tt_device)
        else:
            ttnn.CloseDevice(tt_device)

    if fabric_config is not None:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
