"""Tests for USB camera resolution probe helpers."""

from __future__ import annotations

from sarm_hand.cameras import (
    _format_probe_yaml_snippet,
    _recommend_capture_mode,
    _wants_downscale,
)
from sarm_hand.config import CameraSettings


def test_wants_downscale_explicit_capture_smaller_than_output():
    cam = CameraSettings(
        type="opencv",
        index_or_path=0,
        capture_width=1280,
        capture_height=720,
        width=640,
        height=480,
        auto_resolution=False,
    )
    assert _wants_downscale(cam)


def test_no_downscale_when_capture_matches_output():
    cam = CameraSettings(
        type="opencv",
        index_or_path=0,
        capture_width=640,
        capture_height=480,
        width=640,
        height=480,
        auto_resolution=False,
    )
    assert not _wants_downscale(cam)


def test_recommend_prefers_720p_when_available():
    probe = {
        "working": [
            {"frame": (1920, 1080)},
            {"frame": (1280, 720)},
            {"frame": (640, 480)},
        ]
    }
    rec = _recommend_capture_mode(probe, output_width=640, output_height=480)
    assert rec["frame"] == (1280, 720)


def test_recommend_falls_back_to_smallest():
    probe = {"working": [{"frame": (1920, 1080)}, {"frame": (1600, 900)}]}
    rec = _recommend_capture_mode(
        probe,
        output_width=640,
        output_height=480,
        prefer_below=(640, 480),
    )
    assert rec["frame"] == (1600, 900)


def test_format_probe_yaml_snippet():
    cam = CameraSettings(type="opencv", index_or_path=0, fps=5)
    text = _format_probe_yaml_snippet(
        "front",
        cam,
        {"frame": (1280, 720)},
        output_width=640,
        output_height=480,
    )
    assert "capture_width: 1280" in text
    assert "width: 640" in text
    assert "auto_resolution: false" in text
