"""Unit tests for LeRobot ↔ Genesis joint conversion (no genesis-world required)."""

from __future__ import annotations

import math

import pytest

from sarm_hand.config import ProjectConfig
from sarm_hand.genesis.urdf_limits import urdf_joint_limits
from sarm_hand.genesis.units import (
    home_pose_radians,
    norm_to_radians,
    observation_to_radians,
    radians_to_norm,
)


@pytest.fixture
def cfg() -> ProjectConfig:
    return ProjectConfig.load()


@pytest.fixture
def limits(cfg: ProjectConfig) -> dict[str, tuple[float, float]]:
    return urdf_joint_limits(cfg)


def test_norm_zero_maps_to_urdf_midpoint(cfg: ProjectConfig, limits):
    # shoulder_lift / elbow_flex use old-calib mapping limits (not new URDF midpoint).
    for joint in ("shoulder_pan", "wrist_flex"):
        lo, hi = limits[joint]
        mid = (lo + hi) / 2
        rad = norm_to_radians(0.0, joint, cfg, urdf_limits=limits)
        assert rad == pytest.approx(mid, abs=1e-4)


def test_gripper_closed_open(cfg: ProjectConfig, limits):
    lo, hi = limits["gripper"]
    assert norm_to_radians(0.0, "gripper", cfg, urdf_limits=limits) == pytest.approx(lo, abs=1e-4)
    assert norm_to_radians(100.0, "gripper", cfg, urdf_limits=limits) == pytest.approx(hi, abs=1e-4)


def test_norm_roundtrip_shoulder_pan(cfg: ProjectConfig, limits):
    for value in (-50.0, 0.0, 50.0, 100.0):
        rad = norm_to_radians(value, "shoulder_pan", cfg, urdf_limits=limits)
        back = radians_to_norm(rad, "shoulder_pan", cfg, urdf_limits=limits)
        assert abs(back - value) < 1e-3


def test_sign_flip_inverts_travel(cfg: ProjectConfig, limits):
    lo, hi = limits["shoulder_pan"]
    pos = norm_to_radians(80.0, "shoulder_pan", cfg, urdf_limits=limits)
    neg = norm_to_radians(-80.0, "shoulder_pan", cfg, urdf_limits=limits)
    assert pos == pytest.approx(hi - 0.9 * (hi - lo), abs=1e-4)
    assert neg == pytest.approx(lo + 0.9 * (hi - lo), abs=1e-4)
    assert pos < neg


def test_home_pose_uses_home_raw(cfg: ProjectConfig, limits):
    if not cfg.genesis.home_raw:
        pytest.skip("home_raw not configured")
    from sarm_hand.genesis.calibration import load_calibration, startup_pose_norm

    cal = load_calibration(cfg.genesis.calibration_role, cfg)
    if cal is None:
        pytest.skip("calibration not on this machine")
    radians = home_pose_radians(cfg, urdf_limits=limits, calibration=cal)
    expected = observation_to_radians(
        startup_pose_norm(cfg, calibration=cal),
        cfg,
        calibration=cal,
        urdf_limits=limits,
    )
    assert radians == pytest.approx(expected, abs=1e-5)


def test_wrist_roll_midpoint_not_zero(cfg: ProjectConfig, limits):
    lo, hi = limits["wrist_roll"]
    mid = (lo + hi) / 2
    assert abs(mid) > 0.01
    assert norm_to_radians(0.0, "wrist_roll", cfg, urdf_limits=limits) == pytest.approx(mid, abs=1e-4)
