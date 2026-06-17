"""Tests for calibration-aware norm ↔ raw ↔ URDF mapping."""

from __future__ import annotations

import pytest

from sarm_hand.config import ProjectConfig
from sarm_hand.genesis.calibration import load_calibration, norm_to_raw, raw_to_norm
from sarm_hand.genesis.urdf_limits import urdf_joint_limits
from sarm_hand.genesis.units import (
    norm_to_radians,
    observation_to_radians,
    radians_to_norm,
    raw_to_radians,
)


@pytest.fixture
def cfg() -> ProjectConfig:
    return ProjectConfig.load()


@pytest.fixture
def leader_cal(cfg: ProjectConfig):
    cal = load_calibration("leader", cfg)
    if cal is None:
        pytest.skip("leader calibration not on this machine")
    return cal


@pytest.fixture
def limits(cfg: ProjectConfig):
    return urdf_joint_limits(cfg)


def test_norm_raw_roundtrip(leader_cal):
    for joint in leader_cal:
        for norm in (-80.0, 0.0, 40.0) if joint != "gripper" else (10.0, 50.0, 90.0):
            raw = norm_to_raw(norm, joint, leader_cal)
            lo = int(leader_cal[joint]["range_min"])
            hi = int(leader_cal[joint]["range_max"])
            assert lo <= raw <= hi
            back = raw_to_norm(raw, joint, leader_cal)
            assert back == pytest.approx(norm, abs=0.5)


def test_home_raw_maps_via_calibration(cfg: ProjectConfig, leader_cal, limits):
    if not cfg.genesis.home_raw:
        pytest.skip("home_raw not configured")
    from sarm_hand.genesis.calibration import startup_pose_norm

    obs = startup_pose_norm(cfg, calibration=leader_cal)
    radians = observation_to_radians(obs, cfg, calibration=leader_cal, urdf_limits=limits)
    assert len(radians) == 6
    for i, joint in enumerate(
        (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
    ):
        raw = cfg.genesis.home_raw[joint]
        expected = raw_to_radians(raw, joint, cfg, leader_cal, urdf_limits=limits)
        assert radians[i] == pytest.approx(expected, abs=1e-5)


def test_norm_radians_roundtrip_with_calibration(cfg: ProjectConfig, leader_cal, limits):
    for joint in ("shoulder_pan", "wrist_flex", "gripper"):
        norm = 46.6 if joint == "wrist_flex" else (38.99 if joint == "gripper" else 2.6)
        rad = norm_to_radians(norm, joint, cfg, calibration=leader_cal, urdf_limits=limits)
        back = radians_to_norm(rad, joint, cfg, calibration=leader_cal, urdf_limits=limits)
        assert back == pytest.approx(norm, abs=0.5)
