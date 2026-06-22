"""Genesis rest pose from home_raw should stay inside URDF limits."""

from __future__ import annotations

import math

import pytest

from sarm_hand.config import ProjectConfig
from sarm_hand.genesis.calibration import load_calibration
from sarm_hand.genesis.home_pose import clamped_home_joints, home_pose_degrees
from sarm_hand.genesis.urdf_limits import urdf_joint_limits


@pytest.fixture
def cfg() -> ProjectConfig:
    return ProjectConfig.load()


@pytest.fixture
def leader_cal(cfg: ProjectConfig):
    cal = load_calibration("leader", cfg)
    if cal is None:
        pytest.skip("leader calibration not on this machine")
    return cal


def test_home_pose_not_clamped_at_urdf_limits(cfg: ProjectConfig, leader_cal):
    clamped = clamped_home_joints(cfg, calibration=leader_cal)
    assert clamped == []


def test_home_pose_matches_rest_pose(cfg: ProjectConfig, leader_cal):
    """At home_raw, delta mapping returns genesis.rest_pose exactly."""
    if not cfg.genesis.home_raw:
        pytest.skip("home_raw not configured")
    if cfg.genesis.mapping != "delta":
        pytest.skip("rest_pose anchor applies only to delta mapping")
    degrees = home_pose_degrees(cfg, calibration=leader_cal)
    for joint, target in cfg.genesis.rest_pose.items():
        assert degrees[joint] == pytest.approx(float(target), abs=0.5)


def test_home_pose_folded_at_rest(cfg: ProjectConfig, leader_cal):
    """Legacy mapping at home_raw should show folded shoulder/elbow (not reaching)."""
    if not cfg.genesis.home_raw:
        pytest.skip("home_raw not configured")
    if cfg.genesis.mapping != "legacy":
        pytest.skip("folded rest check applies to legacy mapping")
    degrees = home_pose_degrees(cfg, calibration=leader_cal)
    assert degrees["shoulder_lift"] < 0.0
    assert degrees["elbow_flex"] > 0.0


def test_home_pose_within_hard_limits(cfg: ProjectConfig, leader_cal):
    hard = urdf_joint_limits(cfg)
    degrees = home_pose_degrees(cfg, calibration=leader_cal)
    for joint, deg in degrees.items():
        lo, hi = hard[joint]
        assert math.degrees(lo) - 1e-3 <= deg <= math.degrees(hi) + 1e-3
