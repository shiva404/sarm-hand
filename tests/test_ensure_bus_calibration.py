"""Tests for non-interactive servo calibration loading."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sarm_hand.config import ProjectConfig
from sarm_hand.genesis.calibration import calibration_path, load_calibration
from sarm_hand.robot import ensure_bus_calibration


def test_calibration_path_uses_hf_lerobot_home(tmp_path, monkeypatch):
    cfg = ProjectConfig.load()
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    path = calibration_path("follower", cfg)
    assert path == tmp_path / "calibration" / "robots" / "so_follower" / f"{cfg.robot.id}.json"


def test_load_calibration_falls_back_to_legacy_cache(tmp_path, monkeypatch):
    cfg = ProjectConfig.load()
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    legacy = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "calibration"
        / "robots"
        / "so_follower"
        / f"{cfg.robot.id}.json"
    )
    if not legacy.is_file():
        pytest.skip("no legacy follower calibration on this machine")
    assert load_calibration("follower", cfg) is not None


def test_ensure_bus_calibration_writes_saved_file(tmp_path, monkeypatch):
    cfg = ProjectConfig.load()
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    cal_path = calibration_path("follower", cfg)
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    sample = {
        "shoulder_pan": {
            "id": 1,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 100,
            "range_max": 4000,
        },
        "shoulder_lift": {
            "id": 2,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 100,
            "range_max": 4000,
        },
        "elbow_flex": {
            "id": 3,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 100,
            "range_max": 4000,
        },
        "wrist_flex": {
            "id": 4,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 100,
            "range_max": 4000,
        },
        "wrist_roll": {
            "id": 5,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 0,
            "range_max": 4095,
        },
        "gripper": {
            "id": 6,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 1500,
            "range_max": 2500,
        },
    }
    cal_path.write_text(json.dumps(sample))

    device = MagicMock()
    device.is_calibrated = False
    device.calibration = {}
    device.calibration_fpath = cal_path
    device._load_calibration = MagicMock()

    ensure_bus_calibration(device, "follower", cfg=cfg)

    device.bus.write_calibration.assert_called_once()
    written = device.bus.write_calibration.call_args[0][0]
    assert "shoulder_pan" in written
    assert written["shoulder_pan"].range_min == 100


def test_ensure_bus_calibration_skips_when_already_calibrated():
    device = MagicMock()
    device.is_calibrated = True
    ensure_bus_calibration(device, "follower")
    device.bus.write_calibration.assert_not_called()
