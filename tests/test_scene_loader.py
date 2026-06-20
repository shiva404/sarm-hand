"""Tests for Genesis scene YAML loading (no genesis-world required)."""

from __future__ import annotations

import pytest

from sarm_hand.config import GenesisSettings
from sarm_hand.genesis.scene_loader import _parse_color, _parse_object, load_scene_definition


def test_parse_hex_color():
    assert _parse_color("#ff0000") == (1.0, 0.0, 0.0)


def test_parse_euler_on_cylinder():
    spec = _parse_object(
        "pen",
        {
            "shape": "cylinder",
            "radius": 0.007,
            "height": 0.08,
            "pos": [0.25, 0.08, 0.027],
            "euler": [0, 90, 0],
            "color": "#ffffff",
        },
    )
    assert spec.euler == (0.0, 90.0, 0.0)


def test_load_pick_place_scene():
    definition = load_scene_definition(GenesisSettings(scene="pick_place_desk"))
    names = {obj.name for obj in definition.objects}
    assert "desk" in names
    assert "pen_white" in names
    assert "pen_cap" in names
    assert "pen_black" not in names
    assert "holder_shell" in names
    assert "holder_inner" in names
    assert "red_cube" not in names  # enabled: false

    white = next(obj for obj in definition.objects if obj.name == "pen_white")
    assert white.euler == (0.0, 90.0, 0.0)
    assert white.height == pytest.approx(0.081)
    assert white.color == (0xF4 / 255, 0xF4 / 255, 0xF4 / 255)

    cap = next(obj for obj in definition.objects if obj.name == "pen_cap")
    assert cap.height == pytest.approx(0.054)
    assert cap.color == (0x14 / 255, 0x14 / 255, 0x14 / 255)

    inner = next(obj for obj in definition.objects if obj.name == "holder_inner")
    assert inner.shape == "cylinder"
    assert inner.surface == "aluminum"


def test_parse_sphere_object():
    spec = _parse_object("ball", {"shape": "sphere", "radius": 0.02, "pos": [0, 0, 1], "color": [1, 0, 0]})
    assert spec.shape == "sphere"
    assert spec.radius == 0.02
    assert spec.color == (1.0, 0.0, 0.0)
