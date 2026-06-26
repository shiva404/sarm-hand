"""Tests for Genesis + SmolVLA policy wiring helpers."""

from __future__ import annotations

from sarm_hand.config import ProjectConfig
from sarm_hand.policy import _use_genesis_policy, build_policy_rename_map


def test_build_policy_rename_map_identity_from_cameras():
    cfg = ProjectConfig.load()
    mapping = build_policy_rename_map(cfg, genesis=False)
    for name in cfg.cameras:
        assert mapping[f"observation.images.{name}"] == f"observation.images.{name}"


def test_build_policy_rename_map_genesis_identity_from_cameras():
    cfg = ProjectConfig.load()
    mapping = build_policy_rename_map(cfg, genesis=True)
    for name in cfg.genesis.cameras:
        assert mapping[f"observation.images.{name}"] == f"observation.images.{name}"


def test_use_genesis_policy_flag_and_backend():
    cfg = ProjectConfig.load()
    assert _use_genesis_policy(cfg, True) is True
    assert _use_genesis_policy(cfg, False) is False

    original = cfg.robot.backend
    try:
        cfg.robot.backend = "genesis"
        assert _use_genesis_policy(cfg, None) is True
        cfg.robot.backend = "hardware"
        assert _use_genesis_policy(cfg, None) is False
    finally:
        cfg.robot.backend = original
