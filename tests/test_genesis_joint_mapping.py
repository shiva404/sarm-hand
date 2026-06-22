"""Realistic joint-mapping checks against calibration + URDF physical limits."""

from __future__ import annotations

import math

import pytest

from sarm_hand.config import JOINT_NAMES, ProjectConfig
from sarm_hand.genesis.calibration import load_calibration, norm_to_raw, raw_to_norm
from sarm_hand.genesis.urdf_limits import mapping_joint_limits, urdf_joint_limits
from sarm_hand.genesis.units import (
    home_pose_radians,
    norm_to_radians,
    observation_to_radians,
    radians_to_norm,
    raw_to_radians,
)

# so101_old_calib.urdf — semantic limits for shoulder_lift / elbow_flex mapping.
_OLD_SHOULDER_LIFT = (-3.31613, 0.174533)
_OLD_ELBOW_FLEX = (-0.174533, 3.14159)


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
def hard_limits(cfg: ProjectConfig):
    return urdf_joint_limits(cfg)


@pytest.fixture
def map_limits(cfg: ProjectConfig, hard_limits):
    return mapping_joint_limits(cfg, urdf_limits=hard_limits)


def test_shoulder_elbow_legacy_limits_for_home_anchor(cfg: ProjectConfig, map_limits):
    """Wide-cal joints keep legacy limits in config for home anchoring only."""
    if cfg.genesis.mapping != "legacy":
        pytest.skip("legacy mapping not enabled in config")
    assert map_limits["shoulder_lift"] == pytest.approx(_OLD_SHOULDER_LIFT, abs=1e-4)
    assert map_limits["elbow_flex"] == pytest.approx(_OLD_ELBOW_FLEX, abs=1e-4)


def test_home_pose_rest_angles(cfg: ProjectConfig, leader_cal, hard_limits):
    """At home_raw, delta mapping reproduces genesis.rest_pose exactly."""
    if not cfg.genesis.home_raw:
        pytest.skip("home_raw not configured")
    if cfg.genesis.mapping != "delta":
        pytest.skip("rest_pose anchor applies only to delta mapping")

    radians = home_pose_radians(cfg, calibration=leader_cal, urdf_limits=hard_limits)
    for i, joint in enumerate(JOINT_NAMES):
        target = cfg.genesis.rest_pose.get(joint)
        if target is None:
            continue
        assert math.degrees(radians[i]) == pytest.approx(float(target), abs=1.0)


def test_wide_cal_norm_gain_near_one_degree_per_unit(cfg: ProjectConfig, leader_cal, hard_limits):
    """Leader norm delta ≈ URDF degree delta for shoulder_lift / elbow_flex (wide_cal mode)."""
    if cfg.genesis.mapping != "wide_cal":
        pytest.skip("wide_cal mapping not enabled in config")
    from sarm_hand.genesis.calibration import raw_to_norm

    for joint in ("shoulder_lift", "elbow_flex"):
        hn = raw_to_norm(cfg.genesis.home_raw[joint], joint, leader_cal)
        hr = norm_to_radians(hn, joint, cfg, calibration=leader_cal, urdf_limits=hard_limits)
        for delta in (-30.0, -15.0, 15.0, 30.0):
            rad = norm_to_radians(
                hn + delta, joint, cfg, calibration=leader_cal, urdf_limits=hard_limits
            )
            assert math.degrees(rad - hr) == pytest.approx(delta, abs=2.5)


def test_home_angles_within_genesis_hard_limits(cfg: ProjectConfig, leader_cal, hard_limits):
    radians = home_pose_radians(cfg, calibration=leader_cal, urdf_limits=hard_limits)
    for i, joint in enumerate(JOINT_NAMES):
        lo, hi = hard_limits[joint]
        assert lo - 1e-6 <= radians[i] <= hi + 1e-6, f"{joint} out of URDF range"


def test_calibrated_sweep_within_hard_limits(cfg: ProjectConfig, leader_cal, hard_limits):
    """Every calibrated raw endpoint must map inside the Genesis URDF file limits."""
    for joint in JOINT_NAMES:
        lo_raw = int(leader_cal[joint]["range_min"])
        hi_raw = int(leader_cal[joint]["range_max"])
        for raw in (lo_raw, hi_raw, (lo_raw + hi_raw) // 2):
            rad = raw_to_radians(
                raw, joint, cfg, leader_cal, urdf_limits=hard_limits, hard_limits=hard_limits
            )
            lo, hi = hard_limits[joint]
            assert lo - 1e-6 <= rad <= hi + 1e-6, f"{joint} raw={raw} -> {math.degrees(rad):.1f}°"


def test_norm_direction_monotonic(cfg: ProjectConfig, leader_cal, hard_limits):
    """Increasing LeRobot norm should move shoulder_lift / elbow_flex consistently."""
    for joint, expect_increasing in (("shoulder_lift", True), ("elbow_flex", True)):
        prev = norm_to_radians(
            -80.0, joint, cfg, calibration=leader_cal, urdf_limits=hard_limits
        )
        for norm in (-40.0, 0.0, 40.0, 80.0):
            cur = norm_to_radians(norm, joint, cfg, calibration=leader_cal, urdf_limits=hard_limits)
            if expect_increasing:
                assert cur >= prev - 1e-6, f"{joint} norm={norm} decreased"
            else:
                assert cur <= prev + 1e-6, f"{joint} norm={norm} increased"
            prev = cur


def test_norm_raw_radians_roundtrip(cfg: ProjectConfig, leader_cal, hard_limits):
    if cfg.genesis.mapping == "delta":
        pytest.skip("delta mapping uses rest_pose anchor, not norm-linear roundtrip")
    cases = {
        "shoulder_lift": (-30.0, 0.0, 30.0),
        "elbow_flex": (-30.0, 0.0, 30.0),
        "wrist_flex": (-60.0, 0.0, 60.0),
    }
    for joint, norms in cases.items():
        for norm in norms:
            rad = norm_to_radians(norm, joint, cfg, calibration=leader_cal, urdf_limits=hard_limits)
            back = radians_to_norm(rad, joint, cfg, calibration=leader_cal, urdf_limits=hard_limits)
            assert back == pytest.approx(norm, abs=1.0)


def test_home_raw_matches_observation_path(cfg: ProjectConfig, leader_cal, hard_limits):
    from sarm_hand.genesis.calibration import startup_pose_norm

    obs = startup_pose_norm(cfg, calibration=leader_cal)
    radians = observation_to_radians(obs, cfg, calibration=leader_cal, urdf_limits=hard_limits)
    for joint in JOINT_NAMES:
        raw = cfg.genesis.home_raw[joint]
        expected = raw_to_radians(
            raw, joint, cfg, leader_cal, urdf_limits=hard_limits, hard_limits=hard_limits
        )
        idx = JOINT_NAMES.index(joint)
        assert radians[idx] == pytest.approx(expected, abs=1e-5)
