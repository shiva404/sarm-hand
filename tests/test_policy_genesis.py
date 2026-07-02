"""Tests for Genesis + SmolVLA policy wiring helpers."""

from __future__ import annotations

from sarm_hand.config import ProjectConfig
from sarm_hand.policy import (
    _use_genesis_policy,
    build_policy_rename_map,
    resolve_training_batch_size,
    resolve_training_num_workers,
)


def test_build_policy_rename_map_maps_cameras_to_smolvla_slots():
    cfg = ProjectConfig.load()
    mapping = build_policy_rename_map(cfg, genesis=False)
    for i, name in enumerate(cfg.cameras, start=1):
        assert mapping[f"observation.images.{name}"] == f"observation.images.camera{i}"


def test_build_policy_rename_map_genesis_maps_cameras_to_smolvla_slots():
    cfg = ProjectConfig.load()
    mapping = build_policy_rename_map(cfg, genesis=True)
    for i, name in enumerate(cfg.genesis.cameras, start=1):
        assert mapping[f"observation.images.{name}"] == f"observation.images.camera{i}"


def test_resolve_training_batch_size_caps_mps():
    cfg = ProjectConfig.load()
    assert resolve_training_batch_size("mps", 64, cfg) == 4
    assert resolve_training_batch_size("mps", 4, cfg) == 4
    assert resolve_training_batch_size("cuda", 64, cfg) == 64


def test_resolve_training_num_workers_auto():
    cfg = ProjectConfig.load()
    assert resolve_training_num_workers("mps", None, cfg, kind="smolvla") == 0
    assert resolve_training_num_workers("cuda", None, cfg, kind="smolvla") == 4
    assert resolve_training_num_workers("mps", 2, cfg, kind="act") == 2


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
