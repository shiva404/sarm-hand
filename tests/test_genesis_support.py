"""Tests for desk support clamping."""

from __future__ import annotations

import numpy as np
import pytest

from sarm_hand.genesis.scene_loader import SceneObjectSpec, SceneProp
from sarm_hand.genesis.support import (
    clamp_prop_above_desk,
    desk_top_z,
    resting_pen_center_z,
)


def test_resting_pen_center_z_on_desk_top():
    desk = SceneObjectSpec("desk", "box", (0.20, 0.0, 0.01), size=(0.32, 0.28, 0.02), fixed=True)
    pen = SceneObjectSpec("pen", "box", (0.17, 0.04, 0.028), size=(0.135, 0.018, 0.018))
    assert desk_top_z(desk) == 0.02
    assert resting_pen_center_z(desk, pen) == pytest.approx(0.029)


def test_clamp_prop_above_desk_raises_pen():
    desk = SceneObjectSpec("desk", "box", (0.20, 0.0, 0.01), size=(0.32, 0.28, 0.02), fixed=True)

    class FakeEntity:
        def __init__(self, z: float):
            self.pos = np.array([0.17, 0.04, z])
            self.zero_velocity = None

        def get_pos(self, relative=False):
            return self.pos

        def get_quat(self, relative=False):
            return np.array([1.0, 0.0, 0.0, 0.0])

        def set_pos(self, pos, relative=False, zero_velocity=True):
            self.pos = np.asarray(pos, dtype=np.float64)
            self.zero_velocity = zero_velocity

    entity = FakeEntity(0.015)
    prop = SceneProp(
        spec=SceneObjectSpec("pen", "box", (0.17, 0.04, 0.015), size=(0.135, 0.018, 0.018)),
        entity=entity,
    )
    assert clamp_prop_above_desk(prop, desk)
    assert entity.pos[2] == pytest.approx(0.029, abs=1e-3)
    assert entity.zero_velocity is False


def test_clamp_skips_pen_resting_on_desk():
    desk = SceneObjectSpec("desk", "box", (0.20, 0.0, 0.01), size=(0.32, 0.28, 0.02), fixed=True)

    class FakeEntity:
        def __init__(self):
            self.pos = np.array([0.17, 0.04, 0.029])
            self.moved = False

        def get_pos(self, relative=False):
            return self.pos

        def get_quat(self, relative=False):
            return np.array([1.0, 0.0, 0.0, 0.0])

        def set_pos(self, pos, relative=False, zero_velocity=True):
            self.moved = True

    prop = SceneProp(
        spec=SceneObjectSpec("pen", "box", (0.17, 0.04, 0.029), size=(0.135, 0.018, 0.018)),
        entity=FakeEntity(),
    )
    assert not clamp_prop_above_desk(prop, desk)
    assert not prop.entity.moved
