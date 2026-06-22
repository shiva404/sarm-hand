"""Convert Genesis backend tensors to host numpy."""

from __future__ import annotations

import numpy as np


def to_numpy(value) -> np.ndarray:
    """Return a host ``float64`` array from Genesis, torch, or plain sequences."""
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return value.detach().cpu().numpy().astype(np.float64, copy=False)
        except (TypeError, RuntimeError):
            pass

    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy(), dtype=np.float64)
        except (TypeError, RuntimeError):
            pass

    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, RuntimeError):
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            return np.asarray([to_numpy(v) for v in value], dtype=np.float64)
        raise
