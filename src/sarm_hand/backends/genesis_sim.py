"""LeRobot-compatible Genesis sim robot (pure simulation, no USB)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import JOINT_NAMES, ProjectConfig
from ..genesis.scene import SO101GenesisScene
from ..genesis.units import norm_to_radians, radians_to_observation
from .base import RobotBackend


class GenesisSimRobot(RobotBackend):
    """SO-101 follower implemented in Genesis World."""

    name = "so101_follower_genesis"
    robot_type = "so101_follower"

    def __init__(self, cfg: ProjectConfig | None = None):
        self._cfg = cfg or ProjectConfig.load()
        self._scene: SO101GenesisScene | None = None
        self._connected = False

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in JOINT_NAMES}

    @property
    def observation_features(self) -> dict:
        feats: dict = {f"{j}.pos": float for j in JOINT_NAMES}
        for cam in self._cfg.genesis.cameras:
            h = self._cfg.genesis.cameras[cam].height
            w = self._cfg.genesis.cameras[cam].width
            feats[cam] = (h, w, 3)
        return feats

    @property
    def cameras(self) -> dict[str, Any]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._scene = SO101GenesisScene.create(self._cfg)
        self._scene.apply_home_pose()
        self._connected = True

    def disconnect(self) -> None:
        if self._scene is not None:
            self._scene.close()
        self._connected = False
        self._scene = None

    def get_observation(self) -> dict[str, Any]:
        if self._scene is None:
            raise RuntimeError("Genesis sim not connected")
        self._scene.step(1)
        qpos = self._scene.robot.get_dofs_position(self._scene.dof_indices)
        obs = radians_to_observation(
            list(qpos),
            self._cfg,
            calibration=self._scene.calibration,
        )
        for cam_name in self._scene.cameras:
            frame = self._scene.render_rgb(cam_name)
            if frame is not None:
                obs[cam_name] = frame
        if not self._cfg.genesis.headless:
            self._scene.refresh_previews()
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if self._scene is None:
            raise RuntimeError("Genesis sim not connected")
        radians = [
            norm_to_radians(
                float(action[f"{name}.pos"]),
                name,
                self._cfg,
                calibration=self._scene.calibration,
            )
            for name in JOINT_NAMES
        ]
        self._scene.robot.control_dofs_position(
            np.array(radians, dtype=np.float64),
            self._scene.dof_indices,
        )
        self._scene.step(1)
        return {k: float(v) for k, v in action.items() if k.endswith(".pos")}
