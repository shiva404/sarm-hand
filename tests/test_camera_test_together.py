"""Tests for camera-test concurrent/sequential mode selection."""

from __future__ import annotations

from sarm_hand.cameras import _expected_output_size, _validate_output_frame
from sarm_hand.config import CameraSettings, ProjectConfig


def test_expected_output_size_from_config():
    cam = CameraSettings(width=640, height=480)
    assert _expected_output_size(cam) == (640, 480)


def test_validate_output_frame_matches():
    import numpy as np

    cam = CameraSettings(width=640, height=480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert _validate_output_frame(cam, frame) == (640, 480)


def test_validate_output_frame_mismatch_raises():
    import numpy as np
    import pytest

    cam = CameraSettings(width=640, height=480)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="expected 640x480"):
        _validate_output_frame(cam, frame)


def test_multi_camera_config_has_three_entries():
    cfg = ProjectConfig.load()
    assert len(cfg.cameras) >= 2
