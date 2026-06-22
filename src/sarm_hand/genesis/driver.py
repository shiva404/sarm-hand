"""SO-101 SceneDriver for lerobot-genesis recording and eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ..config import JOINT_NAMES, ProjectConfig
from .scene import SO101GenesisScene
from .tensors import to_numpy
from .units import radians_to_norm

FloatArray = npt.NDArray[np.float32]


@dataclass
class SO101SceneDriver:
    """SceneDriver for lerobot-genesis: SO-101 pick/place desk scene.

    Implements :class:`lerobot_genesis.env.SceneDriver` — actions are normalised
    to ``[-1, 1]`` and mapped onto URDF joint limits; observations use LeRobot
    joint units in ``agent_pos``.
    """

    cfg: ProjectConfig = field(default_factory=ProjectConfig.load)
    _scene: SO101GenesisScene | None = field(default=None, init=False, repr=False)
    _low: FloatArray | None = field(default=None, init=False, repr=False)
    _high: FloatArray | None = field(default=None, init=False, repr=False)
    _span: FloatArray | None = field(default=None, init=False, repr=False)

    @property
    def state_dim(self) -> int:
        return len(JOINT_NAMES)

    @property
    def action_dim(self) -> int:
        return len(JOINT_NAMES)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        if self.cfg.genesis.cameras:
            cam = next(iter(self.cfg.genesis.cameras.values()))
            return (cam.height, cam.width, 3)
        return (480, 640, 3)

    def reset(self) -> None:
        if self._scene is None:
            self._scene = SO101GenesisScene.create(self.cfg)
            lo, hi = self._scene.calibrated_radian_limits()
            self._low = np.asarray(lo, dtype=np.float32)
            self._high = np.asarray(hi, dtype=np.float32)
            self._span = np.maximum(self._high - self._low, 1e-6)
        else:
            self._scene.reset_props()
        self._scene.apply_home_pose()

    def apply_action(self, action: FloatArray) -> None:
        assert self._scene is not None and self._low is not None
        assert self._span is not None
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target = self._low + (a + 1.0) * 0.5 * self._span
        self._scene.robot.control_dofs_position(
            np.asarray(target, dtype=np.float64),
            self._scene.dof_indices,
        )

    def step(self) -> None:
        assert self._scene is not None
        self._scene.step(1)

    def observe(self) -> tuple[np.ndarray, FloatArray]:
        assert self._scene is not None
        pixels = self._scene.render_rgb()
        if pixels is None:
            h, w, c = self.image_shape
            pixels = np.zeros((h, w, c), dtype=np.uint8)
        qpos = to_numpy(self._scene.robot.get_dofs_position(self._scene.dof_indices))
        agent_pos = np.array(
            [
                radians_to_norm(
                    float(q),
                    JOINT_NAMES[i],
                    self.cfg,
                    calibration=self._scene.calibration,
                )
                for i, q in enumerate(qpos)
            ],
            dtype=np.float32,
        )
        return np.asarray(pixels, dtype=np.uint8), agent_pos

    def is_success(self) -> bool:
        return False

    def close(self) -> None:
        if self._scene is not None:
            self._scene.close()
        self._scene = None
        self._low = self._high = self._span = None
