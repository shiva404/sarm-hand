"""Tests for shared ST-3215-C001 servo configuration."""

from __future__ import annotations

from sarm_hand.config import (
    DEFAULT_SERVO_GEAR_RATIO,
    DEFAULT_SERVO_MODEL,
    ProjectConfig,
)
from sarm_hand.servo import export_servo_dict, servo_summary
from sarm_hand.sim_config import export_robot_yaml


def test_default_servo_spec():
    cfg = ProjectConfig.load()
    assert cfg.servo.model == DEFAULT_SERVO_MODEL
    assert cfg.servo.gear_ratio == DEFAULT_SERVO_GEAR_RATIO
    assert cfg.servo.lerobot_type == "sts3215"
    assert cfg.servo.resolution == 4096
    assert cfg.servo.urdf_mechanical_reduction == 1.0
    assert cfg.servo.mujoco_class == "sts3215"


def test_servo_exported_to_browser_sim_yaml():
    cfg = ProjectConfig.load()
    blob = export_robot_yaml(cfg)
    assert blob["servo"]["model"] == "ST-3215-C001"
    assert blob["servo"]["gear_ratio"] == 345


def test_servo_summary():
    assert "ST-3215-C001" in servo_summary()
    assert "1:345" in servo_summary()
    assert export_servo_dict()["gear_ratio"] == 345
