"""Tests for black-frame camera fallback."""

from __future__ import annotations

from sarm_hand.cameras import (
    _enable_black_fallback,
    black_frame,
    install_resilient_camera_patch,
)


def test_black_frame_shape():
    frame = black_frame(480, 640)
    assert frame.shape == (480, 640, 3)
    assert frame.max() == 0


def test_enable_black_fallback_marks_camera():
    class Cam:
        config = type("C", (), {"width": 640, "height": 480, "fps": 30})()

    cam = Cam()
    _enable_black_fallback(cam, RuntimeError("test"), label="top")
    assert cam._sarm_black_fallback is True
    assert cam.width == 640
    assert cam.height == 480


def test_resilient_patch_idempotent():
    install_resilient_camera_patch()
    install_resilient_camera_patch()
