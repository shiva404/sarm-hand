"""Tests for dataset fps subsampling helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sarm_hand.data import (
    frame_image_to_hwc_uint8,
    subsample_stride,
)


def test_subsample_stride_30_to_10() -> None:
    assert subsample_stride(30, 10) == 3


def test_subsample_stride_30_to_15() -> None:
    assert subsample_stride(15, 5) == 3


def test_subsample_stride_rejects_upsample() -> None:
    with pytest.raises(ValueError, match="target_fps"):
        subsample_stride(10, 30)


def test_subsample_stride_rejects_non_divisor() -> None:
    with pytest.raises(ValueError, match="divisible"):
        subsample_stride(30, 7)


def test_frame_image_to_hwc_uint8_from_chw_float() -> None:
    chw = torch.zeros(3, 4, 5)
    chw[0, 1, 2] = 1.0
    hwc = frame_image_to_hwc_uint8(chw)
    assert hwc.shape == (4, 5, 3)
    assert hwc.dtype == np.uint8
    assert hwc[1, 2, 0] == 255


def test_frame_image_to_hwc_uint8_passthrough_hwc() -> None:
    hwc = np.zeros((4, 5, 3), dtype=np.uint8)
    hwc[0, 0, 1] = 200
    out = frame_image_to_hwc_uint8(hwc)
    assert out.shape == (4, 5, 3)
    assert out[0, 0, 1] == 200
