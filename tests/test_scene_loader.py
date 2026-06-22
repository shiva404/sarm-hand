"""Tests for Genesis scene YAML loading (no genesis-world required)."""

from __future__ import annotations

import numpy as np
import pytest

from sarm_hand.config import GenesisSettings
from sarm_hand.genesis.scene_loader import _parse_color, _parse_object, load_scene_definition, spawn_quat_wxyz


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
    assert "pen" in names
    assert "pen_white" not in names
    assert "pen_cap" not in names
    assert "holder_bottom" in names
    assert "holder_wall_xp" in names
    assert "holder_liner" in names
    assert "holder_shell" not in names
    assert "holder_inner" not in names
    assert "red_cube" not in names  # enabled: false

    pen = next(obj for obj in definition.objects if obj.name == "pen")
    assert pen.shape == "box"
    assert pen.size == pytest.approx((0.135, 0.018, 0.018))
    assert pen.friction == pytest.approx(2.5)
    assert pen.coup_friction == pytest.approx(0.4)
    assert pen.euler == (0.0, 0.0, 90.0)
    assert pen.density == pytest.approx(400)
    assert pen.contact_resistance == pytest.approx(5.0e3)
    assert pen.coup_restitution == pytest.approx(0.0)
    assert pen.color == (0xF4 / 255, 0xF4 / 255, 0xF4 / 255)

    liner = next(obj for obj in definition.objects if obj.name == "holder_liner")
    assert liner.collision is False
    assert liner.surface == "aluminum"


def test_spawn_quat_z90():
    from sarm_hand.genesis.scene_loader import SceneObjectSpec

    q = spawn_quat_wxyz(SceneObjectSpec("pen", "box", (0, 0, 0), euler=(0, 0, 90)))
    assert q is not None
    np.testing.assert_allclose(q, [0.70710678, 0, 0, 0.70710678], atol=1e-6)


def test_parse_collision_flag():
    spec = _parse_object(
        "liner",
        {"shape": "box", "size": [0.03, 0.03, 0.05], "pos": [0, 0, 0], "collision": False},
    )
    assert spec.collision is False

    spec = _parse_object("ball", {"shape": "sphere", "radius": 0.02, "pos": [0, 0, 1], "color": [1, 0, 0]})
    assert spec.shape == "sphere"
    assert spec.radius == 0.02
    assert spec.color == (1.0, 0.0, 0.0)
