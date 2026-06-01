"""Device + dtype helpers — resolve ``"auto"`` / ``"bfloat16"`` strings to real
torch objects, and report what hardware a run actually landed on.

Configs carry strings (``dtype: bfloat16``) because YAML cannot hold a
``torch.dtype``. These helpers are the single place those strings become real
torch objects, so the mapping lives in one spot instead of being re-implemented
in every backbone wrapper.
"""

from __future__ import annotations

from typing import Any

import torch

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """``"bf16"`` → ``torch.bfloat16``. Raises on an unknown name.

    >>> resolve_dtype("bfloat16")
    torch.bfloat16
    """
    key = name.lower()
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}. Known: {sorted(_DTYPES)}")
    return _DTYPES[key]


def best_device() -> str:
    """``"cuda"`` if a GPU is visible, else ``"mps"`` on Apple Silicon, else ``"cpu"``.

    Lets local laptop smoke-tests (CPU/MPS) and Colab A100 runs share one code
    path — the config says ``device_map: auto`` and this resolves the rest.
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def move_to_device(batch: dict[str, Any], device, dtype) -> dict[str, Any]:
    """Move a collated batch to ``device``; cast only *floating* tensors to ``dtype``.

    Mirrors HF's ``BatchEncoding.to(device, dtype)``: ``pixel_values`` (float) →
    bf16, while ``input_ids`` / ``labels`` / ``span_ids`` (int) stay integer. One
    shared mover for both the trainer and the evaluator's scoring.
    """
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = (
                v.to(device=device, dtype=dtype)
                if torch.is_floating_point(v)
                else v.to(device)
            )
        else:
            out[k] = v
    return out


def describe_device() -> str:
    """One-line human summary of the active accelerator, for run logs.

    >>> describe_device()
    'cuda:0 NVIDIA A100-SXM4-40GB (40.0 GB)'
    """
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1024**3
        return f"cuda:{i} {props.name} ({gb:.1f} GB)"
    return best_device()
