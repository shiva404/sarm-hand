"""Grasp latch for teleop mirror — physics weld (solid) or kinematic carry fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from .tensors import to_numpy

if TYPE_CHECKING:
    from .scene_loader import SceneProp

GraspMode = Literal["weld", "kinematic"]


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by unit quaternion ``q`` in ``[w, x, y, z]`` order."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    w, x, y, z = q
    t = 2.0 * np.cross(q[1:], v)
    return v + w * t + np.cross(q[1:], t)


def quat_inv_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return quat_rotate(np.array([q[0], -q[1], -q[2], -q[3]]), v)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


@dataclass
class GraspLatch:
    prop_name: str
    mode: GraspMode
    anchor_link_name: str
    jaw_link_idx: int
    prop_link_idx: int
    offset_local: np.ndarray | None = None
    quat_local: np.ndarray | None = None
    world_quat: np.ndarray | None = None  # freeze prop orientation at latch (flat pen)


def link_world_pose(link) -> tuple[np.ndarray, np.ndarray]:
    pos = to_numpy(link.get_pos(relative=False)).reshape(3)
    quat = to_numpy(link.get_quat(relative=False)).reshape(4)
    return pos, quat


def prop_world_pose(entity) -> tuple[np.ndarray, np.ndarray]:
    pos = to_numpy(entity.get_pos(relative=False)).reshape(3)
    quat = to_numpy(entity.get_quat(relative=False)).reshape(4)
    return pos, quat


def prop_base_link(entity):
    return entity.links[0]


def anchor_distance_to_prop(prop: SceneProp, anchor: np.ndarray) -> float:
    """Distance from anchor to prop surface (box AABB) or center for other shapes."""
    pos, quat = prop_world_pose(prop.entity)
    spec = prop.spec
    if spec.shape == "box" and spec.size is not None:
        half = 0.5 * np.asarray(spec.size, dtype=np.float64)
        local = quat_inv_rotate(quat, anchor - pos)
        closest_local = np.clip(local, -half, half)
        closest_world = pos + quat_rotate(quat, closest_local)
        return float(np.linalg.norm(anchor - closest_world))
    return float(np.linalg.norm(anchor - pos))


def should_latch(
    gripper_rad: float,
    *,
    close_rad: float,
    open_rad: float,
    latched: bool,
) -> bool:
    if latched:
        return gripper_rad >= open_rad
    return gripper_rad >= close_rad


def pinch_anchor(robot, link_names: list[str]) -> np.ndarray:
    """Midpoint between gripper links — better proxy for finger gap than a single link."""
    pts = [link_world_pose(robot.get_link(name))[0] for name in link_names]
    return np.mean(np.asarray(pts, dtype=np.float64), axis=0)


def can_acquire_latch(
    leader_rad: float,
    sim_rad: float,
    *,
    close_rad: float,
    prop_dist: float,
    tight_radius_m: float,
    sim_squeeze_rad: float,
) -> bool:
    """Latch on deliberate leader close, or sim fingers nearly shut while touching the prop."""
    if leader_rad >= close_rad:
        return True
    if prop_dist <= tight_radius_m and sim_rad >= sim_squeeze_rad:
        return True
    return False


def nearest_prop(
    props: dict[str, SceneProp],
    anchor: np.ndarray,
    *,
    radius: float,
) -> tuple[str, float] | None:
    best_name: str | None = None
    best_dist = float(radius)
    for name, prop in props.items():
        if prop.spec.fixed:
            continue
        dist = anchor_distance_to_prop(prop, anchor)
        if dist <= best_dist:
            best_dist = dist
            best_name = name
    if best_name is None:
        return None
    return best_name, best_dist


def nearest_prop_to_links(
    robot,
    props: dict[str, SceneProp],
    link_names: list[str],
    *,
    radius: float,
) -> tuple[str, float, str] | None:
    """Find movable prop closest to any gripper link or the pinch midpoint."""
    best_name: str | None = None
    best_link: str | None = None
    best_dist = float(radius)
    anchors: list[tuple[np.ndarray, str]] = []
    for link_name in link_names:
        anchor, _ = link_world_pose(robot.get_link(link_name))
        anchors.append((anchor, link_name))
    if link_names:
        anchors.append((pinch_anchor(robot, link_names), link_names[0]))
    for anchor, link_name in anchors:
        found = nearest_prop(props, anchor, radius=radius)
        if found is None:
            continue
        name, dist = found
        if dist <= best_dist:
            best_dist = dist
            best_name = name
            best_link = link_name
    if best_name is None or best_link is None:
        return None
    return best_name, best_dist, best_link


def latch_kinematic(
    prop_name: str,
    prop_entity,
    robot,
    link_names: list[str],
    *,
    anchor_link_name: str,
    snap_to_pinch: bool = True,
) -> GraspLatch:
    link = robot.get_link(anchor_link_name)
    lp, lq = link_world_pose(link)
    _, pq = prop_world_pose(prop_entity)
    if snap_to_pinch and link_names:
        target_world = pinch_anchor(robot, link_names)
    else:
        target_world, _ = prop_world_pose(prop_entity)
    prop_link = prop_base_link(prop_entity)
    return GraspLatch(
        prop_name=prop_name,
        mode="kinematic",
        anchor_link_name=anchor_link_name,
        jaw_link_idx=link.idx,
        prop_link_idx=prop_link.idx,
        offset_local=quat_inv_rotate(lq, target_world - lp),
        quat_local=quat_mul(quat_inv(lq), pq),
        world_quat=pq.copy(),
    )


def latch_weld(solver, prop_name: str, prop_entity, link, *, anchor_link_name: str) -> GraspLatch:
    prop_link = prop_base_link(prop_entity)
    solver.add_weld_constraint(link.idx, prop_link.idx)
    return GraspLatch(
        prop_name=prop_name,
        mode="weld",
        anchor_link_name=anchor_link_name,
        jaw_link_idx=link.idx,
        prop_link_idx=prop_link.idx,
    )


def release_latch(solver, latch: GraspLatch) -> None:
    if latch.mode == "weld":
        solver.delete_weld_constraint(latch.jaw_link_idx, latch.prop_link_idx)


def carry_kinematic(
    latch: GraspLatch,
    prop: SceneProp,
    link,
) -> None:
    if latch.offset_local is None or latch.quat_local is None:
        raise ValueError("kinematic latch missing pose offset")
    lp, lq = link_world_pose(link)
    new_pos = lp + quat_rotate(lq, latch.offset_local)
    new_quat = latch.world_quat if latch.world_quat is not None else quat_mul(lq, latch.quat_local)
    prop.entity.set_pos(new_pos, relative=False, zero_velocity=True)
    prop.entity.set_quat(new_quat, relative=False, zero_velocity=True)
