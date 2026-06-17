"""Genesis camera presets, poses, and URDF link helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import GenesisCameraSettings

URDF_LINK_ALIASES: dict[str, str] = {
    "gripper": "gripper_link",
    "base": "base_link",
    "wrist": "wrist_link",
}

# Default poses for the pick/place desk scene (meters).
CAMERA_PRESETS: dict[str, dict] = {
    "front": {
        "pos": (0.35, -0.55, 0.35),
        "lookat": (0.35, 0.0, 0.12),
        "fov": 55.0,
    },
    "top": {
        "pos": (0.35, 0.0, 1.15),
        "lookat": (0.35, 0.0, 0.05),
        "fov": 50.0,
    },
    "arm": {
        "attach_link": "gripper_link",
        "pos": (0.35, -0.25, 0.45),
        "lookat": (0.35, 0.0, 0.18),
        "fov": 60.0,
    },
}

# Gripper-mounted camera offset (4x4 transform in link frame).
GRIPPER_CAMERA_OFFSET = np.array(
    [
        [1.0, 0.0, 0.0, 0.02],
        [0.0, 1.0, 0.0, -0.10],
        [0.0, 0.0, 1.0, 0.06],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def resolve_link_name(name: str) -> str:
    return URDF_LINK_ALIASES.get(name, name)


def resolve_camera_pose(
    name: str,
    cam_cfg: GenesisCameraSettings,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float, str | None]:
    """Return (pos, lookat, fov, attach_link) for a named Genesis camera."""
    preset = CAMERA_PRESETS.get(name, CAMERA_PRESETS["front"])
    pos_raw = cam_cfg.pos if cam_cfg.pos is not None else preset.get("pos")
    lookat_raw = cam_cfg.lookat if cam_cfg.lookat is not None else preset.get("lookat")
    pos = tuple(float(x) for x in pos_raw)
    lookat = tuple(float(x) for x in lookat_raw)
    fov = float(cam_cfg.fov if cam_cfg.fov is not None else preset.get("fov", 55.0))
    attach = cam_cfg.attach_link or preset.get("attach_link")
    if attach:
        attach = resolve_link_name(attach)
    return pos, lookat, fov, attach


def default_genesis_cameras() -> dict[str, dict]:
    """Default genesis.cameras block for config/default.yaml."""
    return {
        name: {
            "width": 640,
            "height": 480,
            **{k: v for k, v in preset.items() if k != "attach_link"},
            **({"attach_link": preset["attach_link"]} if "attach_link" in preset else {}),
        }
        for name, preset in CAMERA_PRESETS.items()
    }
