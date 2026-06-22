"""Keep movable props resting on desk surfaces (teleport-safe floor clamp)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .grasp import prop_world_pose, quat_rotate

if TYPE_CHECKING:
    from .scene_loader import SceneObjectSpec, SceneProp


def _box_half_xy(spec: SceneObjectSpec) -> tuple[float, float]:
    if spec.size is None:
        return 0.0, 0.0
    return float(spec.size[0]) * 0.5, float(spec.size[1]) * 0.5


def _box_half_z(spec: SceneObjectSpec) -> float:
    if spec.size is None:
        return 0.0
    return float(spec.size[2]) * 0.5


def desk_top_z(desk: SceneObjectSpec) -> float:
    return float(desk.pos[2]) + _box_half_z(desk)


def _xy_bounds_center_half(
    center: np.ndarray,
    half_x: float,
    half_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([center[0] - half_x, center[1] - half_y], dtype=np.float64)
    hi = np.array([center[0] + half_x, center[1] + half_y], dtype=np.float64)
    return lo, hi


def _xy_overlap(lo_a: np.ndarray, hi_a: np.ndarray, lo_b: np.ndarray, hi_b: np.ndarray) -> bool:
    return not (hi_a[0] < lo_b[0] or lo_a[0] > hi_b[0] or hi_a[1] < lo_b[1] or lo_a[1] > hi_b[1])


def prop_xy_bounds(prop: SceneProp) -> tuple[np.ndarray, np.ndarray] | None:
    spec = prop.spec
    if spec.shape != "box" or spec.size is None:
        return None
    pos, quat = prop_world_pose(prop.entity)
    half_x, half_y = _box_half_xy(spec)
    corners = []
    for sx in (-half_x, half_x):
        for sy in (-half_y, half_y):
            corners.append(pos[:2] + quat_rotate(quat, np.array([sx, sy, 0.0]))[:2])
    corners_arr = np.asarray(corners, dtype=np.float64)
    return corners_arr.min(axis=0), corners_arr.max(axis=0)


def min_center_z_on_desk(prop: SceneProp, desk: SceneObjectSpec) -> float | None:
    """Lowest valid prop center Z when resting on the desk (None if not over desk)."""
    bounds = prop_xy_bounds(prop)
    if bounds is None:
        return None
    desk_lo, desk_hi = _xy_bounds_center_half(
        np.asarray(desk.pos, dtype=np.float64),
        *_box_half_xy(desk),
    )
    if not _xy_overlap(bounds[0], bounds[1], desk_lo, desk_hi):
        return None
    return desk_top_z(desk) + _box_half_z(prop.spec)


def clamp_prop_above_desk(
    prop: SceneProp,
    desk: SceneObjectSpec,
    *,
    max_sink_m: float = 0.001,
) -> bool:
    """Raise prop only when meaningfully sunk into the desk (preserves sliding velocity)."""
    min_z = min_center_z_on_desk(prop, desk)
    if min_z is None:
        return False
    pos, _quat = prop_world_pose(prop.entity)
    if float(pos[2]) >= min_z - max_sink_m:
        return False
    pos = pos.copy()
    pos[2] = min_z
    prop.entity.set_pos(pos, relative=False, zero_velocity=False)
    return True


def enforce_desk_support(
    props: dict[str, SceneProp],
    desk_name: str = "desk",
    *,
    exclude: set[str] | frozenset[str] | None = None,
) -> int:
    """Clamp all movable props above the desk surface when overlapping it."""
    desk_prop = props.get(desk_name)
    if desk_prop is None or not desk_prop.spec.fixed:
        return 0
    desk = desk_prop.spec
    skip = exclude or frozenset()
    adjusted = 0
    for name, prop in props.items():
        if name == desk_name or name in skip or prop.spec.fixed:
            continue
        if clamp_prop_above_desk(prop, desk):
            adjusted += 1
    return adjusted


def resting_pen_center_z(desk: SceneObjectSpec, pen: SceneObjectSpec) -> float:
    """Spawn / reset Z so pen bottom sits flush on desk top."""
    return desk_top_z(desk) + _box_half_z(pen)
