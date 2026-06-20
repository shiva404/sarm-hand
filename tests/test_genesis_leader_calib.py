"""Tests for Genesis ↔ leader calibration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from sarm_hand.config import JOINT_NAMES, ProjectConfig
from sarm_hand.genesis.calibration import load_calibration
from sarm_hand.genesis.leader_calib import (
    _encoder_deg_from_home,
    build_leader_sim_rows,
    format_home_raw_yaml,
    format_leader_sim_table,
    patch_home_raw_in_yaml,
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


def test_format_home_raw_yaml_includes_all_joints(cfg: ProjectConfig):
    home = {j: 2000 + i * 10 for i, j in enumerate(JOINT_NAMES)}
    text = format_home_raw_yaml(home)
    assert text.startswith("  home_raw:")
    for joint in JOINT_NAMES:
        assert f"{joint}: {home[joint]}" in text


def test_patch_home_raw_in_yaml(tmp_path: Path):
    src = tmp_path / "cfg.yaml"
    src.write_text(
        "genesis:\n"
        "  urdf: test.urdf\n"
        "  home_raw:\n"
        "    shoulder_pan: 1000\n"
        "    shoulder_lift: 1001\n"
        "    elbow_flex: 1002\n"
        "    wrist_flex: 1003\n"
        "    wrist_roll: 1004\n"
        "    gripper: 1005\n"
        "  backend: auto\n"
    )
    new_home = {j: 3000 + i for i, j in enumerate(JOINT_NAMES)}
    patch_home_raw_in_yaml(src, new_home)
    text = src.read_text()
    assert "shoulder_pan: 3000" in text
    assert "gripper: 3005" in text
    assert "backend: auto" in text


def test_build_leader_sim_rows_and_table(cfg: ProjectConfig, leader_cal):
    if not cfg.genesis.home_raw:
        pytest.skip("home_raw not configured")
    raw = dict(cfg.genesis.home_raw)
    action = {f"{j}.pos": 0.0 for j in JOINT_NAMES}
    sim_deg = {j: 0.0 for j in JOINT_NAMES}
    rows = build_leader_sim_rows(
        raw=raw,
        action=action,
        sim_deg=sim_deg,
        cfg=cfg,
        calibration=leader_cal,
    )
    assert len(rows) == len(JOINT_NAMES)
    table = format_leader_sim_table(rows)
    assert "shoulder_pan" in table
    assert "enc°" in table
    assert "map°" in table
    assert "sim°" in table
    assert "e−s" in table


def test_encoder_deg_from_home_is_raw_pulse_direction(cfg: ProjectConfig):
    if not cfg.genesis.home_raw:
        pytest.skip("home_raw not configured")
    joint = "shoulder_pan"
    home = cfg.genesis.home_raw[joint]
    resolution = cfg.servo.resolution
    pulses_90 = int(round(90.0 * resolution / 360.0))
    at_home = _encoder_deg_from_home(home, joint, cfg, resolution=resolution)
    moved = _encoder_deg_from_home(home + pulses_90, joint, cfg, resolution=resolution)
    assert at_home == pytest.approx(0.0, abs=0.1)
    assert moved == pytest.approx(90.0, abs=1.0)


def test_align_deg_reflects_leader_sim_offset(cfg: ProjectConfig, leader_cal):
    if not cfg.genesis.home_raw:
        pytest.skip("home_raw not configured")
    raw = dict(cfg.genesis.home_raw)
    rows = build_leader_sim_rows(
        raw=raw,
        action={f"{j}.pos": 0.0 for j in JOINT_NAMES},
        sim_deg={j: 90.0 for j in JOINT_NAMES},
        cfg=cfg,
        calibration=leader_cal,
    )
    pan = next(r for r in rows if r.joint == "shoulder_pan")
    assert pan.enc_deg == pytest.approx(0.0, abs=0.5)
    assert pan.sim_deg == pytest.approx(90.0, abs=0.1)
    assert pan.align_deg == pytest.approx(-90.0, abs=1.0)
