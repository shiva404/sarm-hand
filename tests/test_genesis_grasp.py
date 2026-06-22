"""Tests for sim grasp tuning helpers."""

from __future__ import annotations

import numpy as np
import pytest

from sarm_hand.config import ProjectConfig
from sarm_hand.genesis.scene import SO101GenesisScene


def test_gripper_target_adds_extra_close_when_squeezing():
    cfg = ProjectConfig.load()
    scene = SO101GenesisScene.__new__(SO101GenesisScene)
    scene.genesis_cfg = cfg.genesis
    scene.robot = type("R", (), {})()
    scene.dof_indices = list(range(6))
    scene._gripper_limit_hi = 1.8

    open_rad = np.deg2rad(42.0)
    closed_rad = np.deg2rad(70.0)
    assert scene._gripper_target_rad(open_rad) == open_rad
    tightened = scene._gripper_target_rad(closed_rad)
    assert tightened > closed_rad
    assert tightened <= 1.8 + 1e-6


def test_latch_releases_on_leader_open_not_sim_extra():
    from sarm_hand.genesis.grasp import should_latch

    close = np.deg2rad(48.0)
    open_ = np.deg2rad(42.0)
    # Sim cmd with +34° extra would stay "closed"; leader at rest must stay latched.
    leader_rest = np.deg2rad(45.7)
    assert should_latch(leader_rest, close_rad=close, open_rad=open_, latched=True)
    leader_open = np.deg2rad(40.0)
    assert not should_latch(leader_open, close_rad=close, open_rad=open_, latched=True)


def test_can_acquire_latch_sim_assist():
    from sarm_hand.genesis.grasp import can_acquire_latch

    close = np.deg2rad(48.0)
    leader = np.deg2rad(46.0)
    sim = np.deg2rad(78.0)
    assert not can_acquire_latch(
        leader,
        sim,
        close_rad=close,
        prop_dist=0.09,
        tight_radius_m=0.08,
        sim_squeeze_rad=np.deg2rad(61.0),
    )
    assert can_acquire_latch(
        leader,
        sim,
        close_rad=close,
        prop_dist=0.05,
        tight_radius_m=0.08,
        sim_squeeze_rad=np.deg2rad(61.0),
    )


def test_quat_rotate_roundtrip():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    v = np.array([0.1, 0.2, 0.3])
    from sarm_hand.genesis.grasp import quat_inv_rotate, quat_rotate

    np.testing.assert_allclose(quat_rotate(q, v), v, atol=1e-9)
    np.testing.assert_allclose(quat_inv_rotate(q, v), v, atol=1e-9)


def test_should_latch_hysteresis():
    from sarm_hand.genesis.grasp import should_latch

    close = np.deg2rad(55.0)
    open_ = np.deg2rad(48.0)
    assert should_latch(np.deg2rad(60), close_rad=close, open_rad=open_, latched=False)
    assert not should_latch(np.deg2rad(50), close_rad=close, open_rad=open_, latched=False)
    assert should_latch(np.deg2rad(50), close_rad=close, open_rad=open_, latched=True)
    assert not should_latch(np.deg2rad(45), close_rad=close, open_rad=open_, latched=True)


def test_anchor_distance_uses_box_surface_not_center():
    from sarm_hand.genesis.grasp import anchor_distance_to_prop
    from sarm_hand.genesis.scene_loader import SceneObjectSpec, SceneProp

    class FakeEntity:
        def get_pos(self, relative=False):
            return np.array([0.17, 0.04, 0.029])

        def get_quat(self, relative=False):
            return np.array([1.0, 0.0, 0.0, 0.0])

    spec = SceneObjectSpec(
        name="pen",
        shape="box",
        pos=(0.17, 0.04, 0.029),
        size=(0.135, 0.018, 0.018),
    )
    prop = SceneProp(spec=spec, entity=FakeEntity())
    jaw = np.array([0.10, 0.04, 0.028])
    dist = anchor_distance_to_prop(prop, jaw)
    assert dist == pytest.approx(0.0025, abs=1e-4)
    assert dist < 0.05
    assert float(np.linalg.norm(np.array([0.17, 0.04, 0.029]) - jaw)) > 0.05


def test_carry_kinematic_sets_pos_and_quat():
    from sarm_hand.genesis.grasp import GraspLatch, carry_kinematic
    from sarm_hand.genesis.scene_loader import SceneObjectSpec, SceneProp

    calls: list[tuple[str, tuple]] = []

    class FakeEntity:
        def set_pos(self, pos, relative=False, zero_velocity=True):
            calls.append(("pos", np.asarray(pos), relative, zero_velocity))

        def set_quat(self, quat, relative=False, zero_velocity=True):
            calls.append(("quat", np.asarray(quat), relative, zero_velocity))

    class FakeLink:
        def get_pos(self, relative=False):
            return np.array([0.0, 0.0, 0.1])

        def get_quat(self, relative=False):
            return np.array([1.0, 0.0, 0.0, 0.0])

    prop = SceneProp(
        spec=SceneObjectSpec("pen", "box", (0.17, 0.04, 0.029), size=(0.135, 0.018, 0.018)),
        entity=FakeEntity(),
    )
    latch = GraspLatch(
        prop_name="pen",
        mode="kinematic",
        anchor_link_name="jaw",
        jaw_link_idx=7,
        prop_link_idx=8,
        offset_local=np.array([0.02, 0.0, -0.05]),
        quat_local=np.array([1.0, 0.0, 0.0, 0.0]),
        world_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    carry_kinematic(latch, prop, FakeLink())
    assert len(calls) == 2
    assert calls[0][0] == "pos"
    assert calls[1][0] == "quat"
    np.testing.assert_allclose(calls[0][1], [0.02, 0.0, 0.05], atol=1e-9)
    np.testing.assert_allclose(calls[1][1], [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_carry_kinematic_keeps_world_quat_when_jaw_tilts():
    from sarm_hand.genesis.grasp import GraspLatch, carry_kinematic
    from sarm_hand.genesis.scene_loader import SceneObjectSpec, SceneProp

    quat_calls: list[np.ndarray] = []

    class FakeEntity:
        def set_pos(self, pos, relative=False, zero_velocity=True):
            pass

        def set_quat(self, quat, relative=False, zero_velocity=True):
            quat_calls.append(np.asarray(quat, dtype=np.float64))

    class FakeLink:
        def get_pos(self, relative=False):
            return np.array([0.0, 0.0, 0.1])

        def get_quat(self, relative=False):
            # jaw pitched ~90° — would stand pen on end if rotation were copied
            return np.array([0.7071, 0.7071, 0.0, 0.0])

    latch = GraspLatch(
        prop_name="pen",
        mode="kinematic",
        anchor_link_name="jaw",
        jaw_link_idx=7,
        prop_link_idx=8,
        offset_local=np.array([0.0, 0.0, -0.05]),
        quat_local=np.array([0.7071, -0.7071, 0.0, 0.0]),
        world_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    prop = SceneProp(
        spec=SceneObjectSpec("pen", "box", (0.17, 0.04, 0.029), size=(0.135, 0.018, 0.018)),
        entity=FakeEntity(),
    )
    carry_kinematic(latch, prop, FakeLink())
    np.testing.assert_allclose(quat_calls[0], [1.0, 0.0, 0.0, 0.0], atol=1e-4)


def test_latch_weld_and_release():
    from sarm_hand.genesis.grasp import latch_weld, release_latch

    weld_calls: list[tuple[int, int]] = []
    delete_calls: list[tuple[int, int]] = []

    class FakeSolver:
        def add_weld_constraint(self, a, b):
            weld_calls.append((a, b))

        def delete_weld_constraint(self, a, b):
            delete_calls.append((a, b))

    class FakeLink:
        idx = 7

    class FakePropLink:
        idx = 8

    class FakeEntity:
        links = [FakePropLink()]

    latch = latch_weld(FakeSolver(), "pen", FakeEntity(), FakeLink(), anchor_link_name="jaw")
    assert latch.mode == "weld"
    assert weld_calls == [(7, 8)]
    release_latch(FakeSolver(), latch)
    assert delete_calls == [(7, 8)]


def test_mirror_steps_while_latched():
    cfg = ProjectConfig.load()
    scene = SO101GenesisScene.__new__(SO101GenesisScene)
    scene.genesis_cfg = cfg.genesis
    scene.genesis_cfg.mirror_grasp_carry = True
    scene.dof_indices = list(range(6))
    scene._gripper_limit_hi = 1.74533
    scene._mirror_cmd_rad = None
    scene.props = {}
    scene._grasp_anchor_links = lambda: ["gripper", "jaw"]
    scene._sim_latch_squeeze_rad = lambda: np.deg2rad(61.0)
    scene._probe_grasp_distances = lambda: (None, None, {})
    scene._set_grasp_diag = lambda **kw: None

    step_calls: list[int] = []

    class FakeLatch:
        mode = "kinematic"
        prop_name = "pen"
        anchor_link_name = "jaw"

    scene._grasp_latch = FakeLatch()
    scene._grasp_carry_link = lambda: None
    scene._update_grasp_carry = lambda *a, **k: None  # type: ignore[method-assign]
    scene.step = lambda n=1, **kw: step_calls.append(n)  # type: ignore[method-assign]

    class FakeRobot:
        def set_dofs_position(self, pos, idx, zero_velocity=True):
            pass

        def control_dofs_position(self, pos, idx):
            pass

    scene.robot = FakeRobot()
    target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, np.deg2rad(70.0)])
    scene._mirror_to_target(target)
    assert step_calls == [max(1, cfg.genesis.mirror_substeps) + cfg.genesis.mirror_grasp_substeps]


def test_mirror_sets_gripper_via_full_pose():
    cfg = ProjectConfig.load()
    scene = SO101GenesisScene.__new__(SO101GenesisScene)
    scene.genesis_cfg = cfg.genesis
    scene.genesis_cfg.mirror_grasp_carry = False
    scene.dof_indices = list(range(6))
    scene._gripper_limit_hi = 1.74533
    scene._mirror_cmd_rad = None
    scene._grasp_latch = None
    scene.props = {}

    set_calls: list[tuple] = []
    control_calls: list[tuple] = []

    class FakeRobot:
        def set_dofs_position(self, pos, idx, zero_velocity=True):
            set_calls.append((np.asarray(pos, dtype=np.float64), list(idx), zero_velocity))

        def control_dofs_position(self, pos, idx):
            control_calls.append((np.asarray(pos, dtype=np.float64), list(idx)))

    scene.robot = FakeRobot()
    scene.step = lambda n=1: None  # type: ignore[method-assign]

    target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    scene._mirror_to_target(target)

    assert len(set_calls) == 1
    pos, idx, _zv = set_calls[0]
    assert idx == list(range(6))
    assert len(pos) == 6
    assert len(control_calls) == 1
    assert control_calls[0][1] == [5]
