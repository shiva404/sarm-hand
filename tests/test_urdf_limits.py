"""Tests for URDF joint limit parsing."""

from __future__ import annotations

from sarm_hand.config import ProjectConfig
from sarm_hand.genesis.urdf_limits import parse_urdf_joint_limits, urdf_joint_limits


def test_parse_so101_urdf_limits():
    cfg = ProjectConfig.load()
    limits = urdf_joint_limits(cfg)
    assert "gripper" in limits
    lo, hi = limits["gripper"]
    assert lo < 0
    assert hi > 1.0


def test_parse_cached():
    from sarm_hand.genesis.assets import resolve_urdf

    cfg = ProjectConfig.load()
    path = str(resolve_urdf(cfg.genesis.urdf))
    a = parse_urdf_joint_limits(path)
    b = parse_urdf_joint_limits(path)
    assert a is b
