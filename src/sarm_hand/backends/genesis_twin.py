"""Hardware + Genesis digital twin (mirror follower into sim)."""

from __future__ import annotations

from ..config import ProjectConfig
from ..genesis.scene import SO101GenesisScene
from .hardware import HardwareRobot


class GenesisTwin:
    """Mirrors a USB SO-101 follower into a Genesis World scene."""

    def __init__(self, port: str, cfg: ProjectConfig | None = None):
        self.cfg = cfg or ProjectConfig.load()
        self.hardware = HardwareRobot(port, self.cfg)
        self.scene: SO101GenesisScene | None = None

    def start(self) -> None:
        self.hardware.connect()
        self.scene = SO101GenesisScene.create(self.cfg, calibration_role="follower")
        obs = self.hardware.get_observation()
        self.scene.set_joint_positions_norm(obs)
        self.scene.step(5)

    def stop(self) -> None:
        if self.hardware.is_connected:
            self.hardware.disconnect()
        if self.scene is not None:
            self.scene.close()
        self.scene = None

    def sync_hardware_to_sim(self) -> dict:
        """Read hardware joints and update Genesis mirror."""
        obs = self.hardware.get_observation()
        if self.scene is not None:
            self.scene.set_joint_positions_norm(obs)
            self.scene.step(1)
        return obs

    def render_camera(self, name: str = "front"):
        if self.scene is None:
            return None
        return self.scene.render_rgb(name)

    def render_all_cameras(self) -> dict:
        if self.scene is None:
            return {}
        return self.scene.render_all_rgb()
