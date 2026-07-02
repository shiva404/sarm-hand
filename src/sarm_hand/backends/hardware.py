"""USB hardware robot backend."""

from __future__ import annotations

import time
from typing import Any

from ..config import ProjectConfig
from ..cameras import build_robot_camera_configs, connect_follower_robot
from .base import RobotBackend


class HardwareRobot(RobotBackend):
    """Wraps LeRobot SO101Follower."""

    def __init__(self, port: str, cfg: ProjectConfig, *, use_cameras: bool = True):
        from ..cameras import install_all_camera_patches

        self._cfg = cfg
        install_all_camera_patches(cfg=cfg)
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        if use_cameras and cfg.cameras:
            cameras = build_robot_camera_configs(cfg)
        else:
            cameras = {}
        robot_cfg = SOFollowerRobotConfig(
            id=cfg.robot.id,
            port=port,
            use_degrees=cfg.robot.use_degrees,
            max_relative_target=cfg.robot.max_relative_target,
            disable_torque_on_disconnect=cfg.robot.disable_torque_on_disconnect,
            cameras=cameras,
        )
        self._robot = SO101Follower(robot_cfg)

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def action_features(self) -> dict[str, type]:
        return self._robot.action_features

    @property
    def observation_features(self) -> dict:
        return self._robot.observation_features

    @property
    def cameras(self) -> dict[str, Any]:
        return self._robot.cameras

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    def connect(self) -> None:
        from ..robot import _motor_write_retries

        last_exc: ConnectionError | None = None
        for attempt in range(2):
            try:
                with _motor_write_retries():
                    if self._robot.cameras:
                        connect_follower_robot(
                            self._robot, calibrate=False, cfg=self._cfg
                        )
                    else:
                        self._robot.connect(calibrate=False)
                return
            except ConnectionError as exc:
                last_exc = exc
                if self._robot.is_connected:
                    self._robot.disconnect()
                if attempt == 0:
                    time.sleep(0.5)
        assert last_exc is not None
        raise last_exc

    def disconnect(self) -> None:
        self._robot.disconnect()

    def get_observation(self) -> dict[str, Any]:
        return self._robot.get_observation()

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return self._robot.send_action(action)

    @property
    def inner(self):
        """Underlying LeRobot robot (for teleop compatibility)."""
        return self._robot
