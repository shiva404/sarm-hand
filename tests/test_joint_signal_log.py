"""Tests for joint signal analysis (no hardware)."""

from __future__ import annotations

import pytest

from sarm_hand.config import JOINT_NAMES, ProjectConfig
from sarm_hand.genesis.calibration import load_calibration
from sarm_hand.joint_signal_log import analyze_joint_signals


@pytest.fixture
def cfg() -> ProjectConfig:
    return ProjectConfig.load()


@pytest.fixture
def leader_cal(cfg: ProjectConfig):
    cal = load_calibration("leader", cfg)
    if cal is None:
        pytest.skip("leader calibration not on this machine")
    return cal


def test_analyze_returns_all_joints(cfg: ProjectConfig, leader_cal):
    rows = analyze_joint_signals(cfg, role="leader", target_degrees=90.0)
    assert len(rows) == len(JOINT_NAMES)
    assert {r.joint for r in rows} == set(JOINT_NAMES)


def test_encoder_pulses_per_90_is_1024(cfg: ProjectConfig, leader_cal):
    rows = analyze_joint_signals(cfg, role="leader")
    for r in rows:
        assert r.encoder_pulses_per_90 == pytest.approx(1024.0)


def test_wide_cal_joints_sim_overcounts_encoder(cfg: ProjectConfig, leader_cal):
    """Wide-cal: sim expects ~1.8× more pulses than 90° physical rotation."""
    if not cfg.genesis.home_raw:
        pytest.skip("home_raw not configured")
    rows = {r.joint: r for r in analyze_joint_signals(cfg, role="leader", target_degrees=90.0)}
    for joint in ("shoulder_lift", "elbow_flex"):
        r = rows[joint]
        assert r.wide_cal
        assert r.sim_pulses_for_target_deg == pytest.approx(1840.0, abs=80.0)
        assert r.norm_for_encoder_90 == pytest.approx(50.0, abs=3.0)
        assert r.sim_vs_encoder_pulse_ratio == pytest.approx(1.8, abs=0.15)
