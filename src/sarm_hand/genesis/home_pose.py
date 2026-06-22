"""Startup home pose sanity checks for Genesis joint mapping."""

from __future__ import annotations

import math

from typing import Any

from ..config import JOINT_NAMES, ProjectConfig
from .urdf_limits import urdf_joint_limits
from .units import home_pose_radians


def home_pose_degrees(
    cfg: ProjectConfig,
    *,
    calibration: dict[str, dict] | None = None,
) -> dict[str, float]:
    hard = urdf_joint_limits(cfg)
    radians = home_pose_radians(cfg, calibration=calibration, urdf_limits=hard)
    return {name: math.degrees(radians[i]) for i, name in enumerate(JOINT_NAMES)}


def clamped_home_joints(
    cfg: ProjectConfig,
    *,
    calibration: dict[str, dict] | None = None,
    margin_deg: float = 1.0,
) -> list[str]:
    """Joints whose mapped rest pose sits on a URDF hard limit (mapping likely wrong)."""
    hard = urdf_joint_limits(cfg)
    radians = home_pose_radians(cfg, calibration=calibration, urdf_limits=hard)
    margin = math.radians(margin_deg)
    clamped: list[str] = []
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = hard[name]
        q = float(radians[i])
        if q <= lo + margin or q >= hi - margin:
            clamped.append(name)
    return clamped


def format_home_pose_summary(
    cfg: ProjectConfig,
    *,
    calibration: dict[str, dict] | None = None,
) -> str:
    degrees = home_pose_degrees(cfg, calibration=calibration)
    mode = cfg.genesis.mapping or "delta"
    lines = [f"Genesis rest pose (mapping={mode}, leader at home_raw):"]
    for name in JOINT_NAMES:
        lines.append(f"  {name:14} {degrees[name]:+7.1f}°")
    if mode == "delta" and cfg.genesis.rest_pose:
        lines.append("  (anchor: genesis.rest_pose — tune if sim rest ≠ physical rest)")
    clamped = clamped_home_joints(cfg, calibration=calibration)
    if clamped:
        joints = ", ".join(clamped)
        lines.append(f"  WARNING: {joints} at URDF limit — check rest_pose or home_raw")
    return "\n".join(lines)
