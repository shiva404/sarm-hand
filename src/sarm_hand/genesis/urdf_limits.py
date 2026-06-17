"""Parse revolute joint limits from the SO-101 URDF."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from ..config import JOINT_NAMES, ProjectConfig
from .assets import resolve_urdf


@lru_cache(maxsize=4)
def parse_urdf_joint_limits(urdf_path: str) -> dict[str, tuple[float, float]]:
    """Return ``{joint_name: (lower_rad, upper_rad)}`` for revolute joints."""
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is None or upper is None:
            continue
        limits[name] = (float(lower), float(upper))
    return limits


def urdf_joint_limits(cfg: ProjectConfig | None = None) -> dict[str, tuple[float, float]]:
    """Joint limits for ``JOINT_NAMES`` from the configured Genesis URDF."""
    cfg = cfg or ProjectConfig.load()
    urdf = resolve_urdf(cfg.genesis.urdf)
    parsed = parse_urdf_joint_limits(str(urdf))
    missing = [name for name in JOINT_NAMES if name not in parsed]
    if missing:
        raise ValueError(f"URDF {urdf} missing revolute limits for: {', '.join(missing)}")
    return {name: parsed[name] for name in JOINT_NAMES}


def mapping_joint_limits(
    cfg: ProjectConfig,
    *,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Per-joint URDF radian span used for cal raw/norm → sim angle (may override parsed limits)."""
    limits = urdf_limits or urdf_joint_limits(cfg)
    out = dict(limits)
    for joint, spec in cfg.genesis.joints.items():
        if spec.urdf_min is not None and spec.urdf_max is not None:
            out[joint] = (float(spec.urdf_min), float(spec.urdf_max))
    return out


def clamp_to_urdf_limits(
    radians: list[float],
    cfg: ProjectConfig,
    *,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
) -> list[float]:
    """Clamp mapped angles to hard limits from the Genesis URDF file."""
    hard = urdf_limits or urdf_joint_limits(cfg)
    clamped: list[float] = []
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = hard[name]
        clamped.append(max(lo, min(hi, float(radians[i]))))
    return clamped
