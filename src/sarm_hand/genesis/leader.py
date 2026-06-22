"""Shared USB leader setup for Genesis mirror paths (calibrate, record-sim, twin)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import ProjectConfig
from ..joint_signal_log import _read_all_raw

if TYPE_CHECKING:
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    from .scene import SO101GenesisScene


def so101_leader_config(cfg: ProjectConfig, port: str, *, leader_id: str | None = None) -> SO101LeaderConfig:
    """LeRobot leader config aligned with ``robot.use_degrees`` and teleop settings."""
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    return SO101LeaderConfig(
        id=leader_id or cfg.teleop.leader.id,
        port=port,
        use_degrees=cfg.robot.use_degrees,
    )


def sync_leader_to_scene(scene: SO101GenesisScene, leader) -> dict[str, float]:
    """Read leader encoder pulses and mirror into Genesis (bypasses norm round-trip)."""
    raw = _read_all_raw(leader.bus)
    scene.sync_raw_pose(raw)
    return leader.get_action()
