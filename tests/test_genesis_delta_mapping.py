"""Delta mapping: encoder pulse travel from home_raw tracks sim 1:1."""

from __future__ import annotations

import math

import pytest

from sarm_hand.config import ProjectConfig
from sarm_hand.genesis.calibration import load_calibration, norm_to_raw, raw_to_norm
from sarm_hand.genesis.units import norm_to_radians, raw_to_radians
from sarm_hand.genesis.urdf_limits import urdf_joint_limits


@pytest.fixture
def cfg() -> ProjectConfig:
    loaded = ProjectConfig.load()
    if loaded.genesis.mapping != "delta":
        pytest.skip("delta mapping not enabled in config")
    return loaded


@pytest.fixture
def leader_cal(cfg: ProjectConfig):
    cal = load_calibration("leader", cfg)
    if cal is None:
        pytest.skip("leader calibration not on this machine")
    return cal


def test_delta_gain_one_degree_per_encoder_degree(cfg: ProjectConfig, leader_cal):
    hard = urdf_joint_limits(cfg)
    resolution = cfg.servo.resolution
    pulse_by_joint = {
        "shoulder_lift": (-256, -128, 64),
        "elbow_flex": (-256, -128, 128, 256),
        "wrist_flex": (-128, -64, 64),
    }
    for joint, pulse_deltas in pulse_by_joint.items():
        home_raw = cfg.genesis.home_raw[joint]
        home_rad = raw_to_radians(home_raw, joint, cfg, leader_cal, hard_limits=hard)
        for pulse_delta in pulse_deltas:
            raw = home_raw + pulse_delta
            rad = raw_to_radians(raw, joint, cfg, leader_cal, hard_limits=hard)
            enc_deg = pulse_delta * 360.0 / resolution
            assert math.degrees(rad - home_rad) == pytest.approx(enc_deg, abs=1.5)


def test_delta_norm_path_matches_raw(cfg: ProjectConfig, leader_cal):
    hard = urdf_joint_limits(cfg)
    joint = "shoulder_lift"
    home_norm = raw_to_norm(cfg.genesis.home_raw[joint], joint, leader_cal)
    for delta in (-20.0, 20.0):
        norm = home_norm + delta
        raw = norm_to_raw(norm, joint, leader_cal)
        from_raw = raw_to_radians(raw, joint, cfg, leader_cal, hard_limits=hard)
        from_norm = norm_to_radians(norm, joint, cfg, calibration=leader_cal, urdf_limits=hard)
        assert from_norm == pytest.approx(from_raw, abs=1e-4)
