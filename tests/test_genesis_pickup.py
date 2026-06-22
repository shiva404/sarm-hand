"""Genesis pickup integration test (requires genesis extra)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sarm_hand.config import JOINT_NAMES, ProjectConfig
from sarm_hand.genesis.tensors import to_numpy

pytestmark = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("genesis"),
    reason="genesis not installed",
)


def _pen_z(scene) -> float:
    return float(to_numpy(scene.props["pen"].entity.get_pos()).reshape(-1)[2])


def _mirror_deg(scene, pose_deg: dict[str, float]) -> None:
    target = np.array([math.radians(pose_deg[n]) for n in JOINT_NAMES], dtype=np.float64)
    scene._mirror_to_target(target)


@pytest.fixture(scope="module")
def genesis_scene():
    from sarm_hand.genesis.scene import SO101GenesisScene

    cfg = ProjectConfig.load()
    cfg.genesis.headless = True
    scene = SO101GenesisScene.create(cfg, apply_home=True)
    yield scene
    scene.close()


def test_gripper_extra_close_narrows_gap():
    from sarm_hand.genesis.scene import SO101GenesisScene

    cfg = ProjectConfig.load()
    scene = SO101GenesisScene.__new__(SO101GenesisScene)
    scene.genesis_cfg = cfg.genesis
    scene._gripper_limit_hi = 1.74533
    open_rad = np.deg2rad(42.0)
    closed_rad = np.deg2rad(70.0)
    extra = float(cfg.genesis.gripper_sim_extra_close_deg)
    assert scene._gripper_target_rad(open_rad) == open_rad
    assert scene._gripper_target_rad(closed_rad) == pytest.approx(
        min(closed_rad + np.deg2rad(extra), 1.74533), abs=1e-4
    )


def test_pickup_latches_at_moderate_gripper_close(genesis_scene):
    scene = genesis_scene
    reach = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -23.0,
        "elbow_flex": 63.0,
        "wrist_flex": 73.0,
        "wrist_roll": 0.0,
        "gripper": 42.0,
    }
    _mirror_deg(scene, reach)
    _mirror_deg(scene, {**reach, "gripper": 50.0})
    assert scene._grasp_latch is not None, "should latch with deliberate squeeze (50°)"


def test_pickup_snap_moves_pen_on_lift(genesis_scene):
    scene = genesis_scene
    from sarm_hand.genesis.grasp import pinch_anchor, link_world_pose

    reach = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -23.0,
        "elbow_flex": 63.0,
        "wrist_flex": 73.0,
        "wrist_roll": 0.0,
        "gripper": 85.0,
    }
    z0 = _pen_z(scene)
    _mirror_deg(scene, reach)
    assert scene._grasp_latch is not None
    pinch = pinch_anchor(scene.robot, scene._grasp_anchor_links())
    pen = scene.props["pen"].entity.get_pos()
    pen_np = to_numpy(pen).reshape(3)
    assert float(np.linalg.norm(pen_np - pinch)) < 0.04, "latched pen should snap near pinch"
    _mirror_deg(scene, {**reach, "shoulder_lift": -33.0, "elbow_flex": 68.0})
    assert _pen_z(scene) > z0 + 0.015, "lifted pen should rise with gripper"


def test_pickup_latch_and_lift(genesis_scene):
    scene = genesis_scene
    # Pose from kinematic search: jaw ~3 cm from pen, over desk.
    reach = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -23.0,
        "elbow_flex": 63.0,
        "wrist_flex": 73.0,
        "wrist_roll": 0.0,
        "gripper": 42.0,
    }
    _mirror_deg(scene, reach)
    _mirror_deg(scene, {**reach, "gripper": 85.0})
    assert scene._grasp_latch is not None, "gripper should latch pen when closed nearby"

    lift = {**reach, "shoulder_lift": -33.0, "elbow_flex": 68.0, "gripper": 85.0}
    _mirror_deg(scene, lift)
    assert _pen_z(scene) > 0.04, "pen should lift with latched gripper"
    assert scene._grasp_latch is not None

    # Opening gripper must release — no permanent attachment.
    _mirror_deg(scene, {**reach, "gripper": 36.0})
    assert scene._grasp_latch is None, "latch must release when leader opens gripper"
