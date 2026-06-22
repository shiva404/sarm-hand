"""Genesis leader USB config matches calibrate-genesis and record-sim --leader."""

from __future__ import annotations

from sarm_hand.config import ProjectConfig
from sarm_hand.genesis.leader import so101_leader_config


def test_so101_leader_config_uses_robot_use_degrees():
    cfg = ProjectConfig.load()
    leader_cfg = so101_leader_config(cfg, "/dev/tty.test")
    assert leader_cfg.use_degrees is cfg.robot.use_degrees
    assert leader_cfg.id == cfg.teleop.leader.id
    assert leader_cfg.port == "/dev/tty.test"


def test_so101_leader_config_not_lerobot_default_degrees():
    """LeRobot SO101LeaderConfig defaults use_degrees=True; we must override."""
    cfg = ProjectConfig.load()
    assert cfg.robot.use_degrees is False
    leader_cfg = so101_leader_config(cfg, "/dev/tty.test")
    assert leader_cfg.use_degrees is False
