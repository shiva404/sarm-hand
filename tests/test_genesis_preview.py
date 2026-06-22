"""Tests for Genesis OpenCV camera preview layout."""

from __future__ import annotations

from sarm_hand.genesis.preview import DEFAULT_WINDOW_LAYOUT, CameraPreview


def test_default_window_layout_spreads_cameras():
    assert set(DEFAULT_WINDOW_LAYOUT) == {"front", "top", "arm"}
    positions = list(DEFAULT_WINDOW_LAYOUT.values())
    assert len(positions) == len(set(positions))


def test_preview_accepts_custom_layout():
    layout = {"front": (0, 0), "top": (100, 0)}
    preview = CameraPreview(enabled=False, layout=layout)
    assert preview._layout == layout
