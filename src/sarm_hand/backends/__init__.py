"""Robot backend factory."""

from __future__ import annotations

from ..config import ProjectConfig
from .base import RobotBackend
from .genesis_sim import GenesisSimRobot
from .hardware import HardwareRobot


def build_robot_backend(
    port: str | None = None,
    *,
    config: ProjectConfig | None = None,
    connect: bool = False,
) -> RobotBackend:
    """Create a robot backend from config/default.yaml robot.backend."""
    cfg = config or ProjectConfig.load()
    backend = cfg.robot.backend.lower()

    if backend == "hardware":
        from ..robot import ensure_port

        resolved_port = ensure_port(port or cfg.robot.port, "Follower")
        robot = HardwareRobot(resolved_port, cfg)
    elif backend in ("genesis", "sim"):
        robot = GenesisSimRobot(cfg)
    elif backend == "twin":
        raise ValueError(
            "robot.backend=twin requires sarm-hand twin (use GenesisTwin, not build_robot_backend)"
        )
    else:
        raise ValueError(f"Unknown robot.backend: {backend!r}")

    if connect:
        robot.connect()
    return robot
