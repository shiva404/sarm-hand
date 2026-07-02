"""Tests for servo connect helpers."""

from __future__ import annotations

from sarm_hand.config import ProjectConfig
from sarm_hand.robot import joint_for_servo_id


def test_joint_for_servo_id_follower_wrist_roll():
    cfg = ProjectConfig.load()
    assert joint_for_servo_id(cfg, "follower", 5) == "wrist_roll"
