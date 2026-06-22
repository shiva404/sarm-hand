#!/usr/bin/env python3
"""Headless Genesis pickup smoke test — move gripper to pen, close, lift."""

from __future__ import annotations

import math
import sys

import numpy as np

from sarm_hand.config import JOINT_NAMES, ProjectConfig
from sarm_hand.genesis.grasp import anchor_distance_to_prop, link_world_pose
from sarm_hand.genesis.scene import SO101GenesisScene
from sarm_hand.genesis.tensors import to_numpy


def _pen_z(scene: SO101GenesisScene) -> float:
    return float(to_numpy(scene.props["pen"].entity.get_pos()).reshape(-1)[2])


def _gripper_deg(scene: SO101GenesisScene) -> float:
    q = float(to_numpy(scene.robot.get_dofs_position([scene._gripper_dof_index()])))
    return math.degrees(q)


def _jaw_pen_dist(scene: SO101GenesisScene) -> float:
    link = scene.robot.get_link(scene.genesis_cfg.grasp_link)
    anchor, _ = link_world_pose(link)
    found = anchor_distance_to_prop(scene.props["pen"], anchor)
    return found


def _mirror_gripper(scene: SO101GenesisScene, gripper_deg: float) -> None:
    cur = to_numpy(scene.robot.get_dofs_position(scene.dof_indices))
    target = cur.copy()
    target[-1] = math.radians(gripper_deg)
    scene._mirror_to_target(target)


def _mirror_pose(scene: SO101GenesisScene, pose_deg: dict[str, float]) -> None:
    target = np.array([math.radians(pose_deg[n]) for n in JOINT_NAMES], dtype=np.float64)
    scene._mirror_to_target(target)


def main() -> int:
    cfg = ProjectConfig.load()
    cfg.genesis.headless = True
    scene = SO101GenesisScene.create(cfg, apply_home=True)
    try:
        # Reach pose over pen (tuned manually from desk layout).
        reach = {
            "shoulder_pan": 8.0,
            "shoulder_lift": -18.0,
            "elbow_flex": 38.0,
            "wrist_flex": 52.0,
            "wrist_roll": 0.0,
            "gripper": 42.0,
        }
        _mirror_pose(scene, reach)
        print(f"at reach: jaw→pen dist={_jaw_pen_dist(scene):.4f}m pen_z={_pen_z(scene):.4f}")

        _mirror_gripper(scene, 80.0)
        print(
            f"closed: gripper={_gripper_deg(scene):.1f}° jaw→pen={_jaw_pen_dist(scene):.4f}m "
            f"latch={scene._grasp_latch!r}"
        )

        lift = dict(reach)
        lift["shoulder_lift"] = -32.0
        lift["elbow_flex"] = 48.0
        _mirror_pose(scene, lift)
        pen_z1 = _pen_z(scene)
        print(f"after lift: pen_z={pen_z1:.4f} latch={scene._grasp_latch!r}")

        ok = pen_z1 > 0.04 and scene._grasp_latch is not None
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        scene.close()


if __name__ == "__main__":
    sys.exit(main())
