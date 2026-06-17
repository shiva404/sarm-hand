"""Tests for Genesis scene YAML loading (no genesis-world required)."""

from __future__ import annotations

from sarm_hand.config import GenesisSettings
from sarm_hand.genesis.scene_loader import _parse_color, _parse_object, load_scene_definition


def test_parse_hex_color():
    assert _parse_color("#ff0000") == (1.0, 0.0, 0.0)


def test_load_pick_place_scene():
    definition = load_scene_definition(GenesisSettings(scene="pick_place_desk"))
    names = {obj.name for obj in definition.objects}
    assert "desk" in names
    assert "pen" in names
    assert "red_cube" not in names  # enabled: false


def test_parse_sphere_object():
    spec = _parse_object("ball", {"shape": "sphere", "radius": 0.02, "pos": [0, 0, 1], "color": [1, 0, 0]})
    assert spec.shape == "sphere"
    assert spec.radius == 0.02
    assert spec.color == (1.0, 0.0, 0.0)
