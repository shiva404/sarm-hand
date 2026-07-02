"""Tests for rest pose resolution and capture helpers."""

from __future__ import annotations

from sarm_hand.config import JOINT_NAMES, ProjectConfig
from sarm_hand.poses import compute_rest_pose_from_genesis, format_pose_yaml, patch_pose_in_yaml


def _flat_cal(lo: int = 0, hi: int = 4095) -> dict[str, dict]:
    return {
        joint: {
            "id": i + 1,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": lo,
            "range_max": hi,
        }
        for i, joint in enumerate(JOINT_NAMES)
    }


def test_compute_rest_pose_from_genesis_at_midpoint():
    cfg = ProjectConfig()
    cfg.genesis.home_raw = {joint: 2048 for joint in JOINT_NAMES}

    leader_cal = _flat_cal()
    follower_cal = _flat_cal()

    from sarm_hand import calibration_bridge

    original = calibration_bridge.require_teleop_calibrations
    calibration_bridge.require_teleop_calibrations = lambda _cfg: (leader_cal, follower_cal)
    try:
        leader_pose = compute_rest_pose_from_genesis(cfg, role="leader")
        follower_pose = compute_rest_pose_from_genesis(cfg, role="follower")
    finally:
        calibration_bridge.require_teleop_calibrations = original

    assert leader_pose is not None
    assert follower_pose is not None
    assert leader_pose != cfg.poses["ready"]
    assert follower_pose != cfg.poses["ready"]


def test_resolve_pose_ready_uses_genesis_when_enabled():
    cfg = ProjectConfig()
    cfg._rest_from_genesis = True
    cfg.genesis.home_raw = {joint: 2048 for joint in JOINT_NAMES}
    cfg._poses["ready"] = {joint: 99.0 for joint in JOINT_NAMES}

    from sarm_hand import calibration_bridge

    original = calibration_bridge.require_teleop_calibrations
    calibration_bridge.require_teleop_calibrations = lambda _cfg: (_flat_cal(), _flat_cal())
    try:
        resolved = cfg.resolve_pose("ready", role="follower")
    finally:
        calibration_bridge.require_teleop_calibrations = original

    assert resolved["shoulder_pan"] != 99.0
    assert resolved != cfg._poses["ready"]


def test_patch_pose_in_yaml(tmp_path):
    yaml_text = """poses:
  rest_from_genesis: true
  ready:
    # Fallback when rest_from_genesis is false
    shoulder_pan: 0
    shoulder_lift: -40
    elbow_flex: 40
    wrist_flex: -40
    wrist_roll: 0
    gripper: 50
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)
    pose = {joint: 1.0 for joint in JOINT_NAMES}
    patch_pose_in_yaml(path, "ready", pose, rest_from_genesis=False)
    text = path.read_text()
    assert "rest_from_genesis: false" in text
    assert "shoulder_pan: 1.0" in text


def test_format_pose_yaml_includes_all_joints():
    pose = {joint: float(i) for i, joint in enumerate(JOINT_NAMES)}
    block = format_pose_yaml("ready", pose)
    assert "ready:" in block
    assert block.count("\n") == len(JOINT_NAMES)
