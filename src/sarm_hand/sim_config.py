"""Build simulator-facing config from config/default.yaml."""

from __future__ import annotations

from typing import Any

import yaml

from .config import JOINT_NAMES, PROJECT_ROOT, ProjectConfig
from .kinematics import ArmGeometry, JointMap, KINEMATIC_JOINTS
from .servo import export_servo_dict

GEOMETRY_SCALAR_KEYS = (
    "units",
    "shoulder_height",
    "upper_arm",
    "forearm",
    "wrist_rot_offset",
    "hand",
    "gripper_offset",
    "gripper_motor",
    "elbow",
)
GEOMETRY_JOINT_KEYS = KINEMATIC_JOINTS


def _require_key(d: dict[str, Any], key: str, *, section: str) -> Any:
    if key not in d:
        raise ValueError(f"{section}: missing required key '{key}'")
    return d[key]


def _joint_map_from_dict(
    d: dict[str, Any] | None,
    limits: tuple[float, float] | None = None,
) -> JointMap:
    d = d or {}
    lo, hi = limits if limits is not None else (None, None)
    return JointMap(
        zero=float(d.get("zero", d.get("zero_deg", 0.0))),
        sign=float(d.get("sign", 1.0)),
        min_val=lo,
        max_val=hi,
    )


def geometry_from_config(config: ProjectConfig) -> ArmGeometry:
    raw = config.sim_geometry()
    for key in GEOMETRY_SCALAR_KEYS:
        _require_key(raw, key, section="geometry")
    joints = raw.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("geometry: missing required 'joints' mapping block")
    for jn in GEOMETRY_JOINT_KEYS:
        if jn not in joints:
            raise ValueError(f"geometry.joints: missing required joint '{jn}'")

    limits = config.sim_joint_limits()
    return ArmGeometry(
        units=str(_require_key(raw, "units", section="geometry")),
        shoulder_height=float(_require_key(raw, "shoulder_height", section="geometry")),
        upper_arm=float(_require_key(raw, "upper_arm", section="geometry")),
        forearm=float(_require_key(raw, "forearm", section="geometry")),
        wrist_rot_offset=float(_require_key(raw, "wrist_rot_offset", section="geometry")),
        hand=float(_require_key(raw, "hand", section="geometry")),
        gripper_offset=float(_require_key(raw, "gripper_offset", section="geometry")),
        gripper_motor=float(_require_key(raw, "gripper_motor", section="geometry")),
        elbow=str(_require_key(raw, "elbow", section="geometry")),
        shoulder_pan_map=_joint_map_from_dict(
            joints["shoulder_pan"], limits.get("shoulder_pan")
        ),
        shoulder_lift_map=_joint_map_from_dict(
            joints["shoulder_lift"], limits.get("shoulder_lift")
        ),
        elbow_flex_map=_joint_map_from_dict(joints["elbow_flex"], limits.get("elbow_flex")),
        wrist_flex_map=_joint_map_from_dict(joints["wrist_flex"], limits.get("wrist_flex")),
        wrist_roll_map=_joint_map_from_dict(joints["wrist_roll"], limits.get("wrist_roll")),
    )


def export_robot_yaml(config: ProjectConfig) -> dict[str, Any]:
    """Dict served to the browser as /robot.yaml (no hardcoded defaults)."""
    geom_raw = config.sim_geometry()
    joint_limits = config.sim_joint_limits()
    joint_meta = config.sim_joint_meta()
    value_suffix = config.sim_value_suffix()

    joints_list = []
    home_pose = config.poses.get("home", {})
    home_values: dict[str, float] = {}
    servos_home: dict[str, float] = {}

    for name in JOINT_NAMES:
        lo, hi = joint_limits[name]
        meta = joint_meta.get(name, {})
        home = float(home_pose.get(name, (lo + hi) / 2))
        home_values[name] = home
        servos_home[name] = home
        joints_list.append(
            {
                "name": name,
                "min": lo,
                "max": hi,
                "home": home,
                **({k: meta[k] for k in ("min_pulse_us", "max_pulse_us", "invert") if k in meta}),
            }
        )

    poses: dict[str, dict[str, float]] = {}
    for pose_name, pose_joints in config.poses.items():
        if pose_name == "sequence":
            continue
        merged = dict(home_values)
        for joint, value in pose_joints.items():
            if joint in JOINT_NAMES:
                merged[joint] = float(value)
        poses[pose_name] = merged

    reach_steps = config.sim_reach_steps()
    visual = config.sim_visual()

    return {
        "geometry": geom_raw,
        "servo": export_servo_dict(config),
        "joints": joints_list,
        "poses": poses,
        "home": home_values,
        "sim": {
            "brand_title": config.sim_brand_title(),
            "brand_subtitle": config.sim_brand_subtitle(),
            "value_suffix": value_suffix,
            "reach_steps": reach_steps,
            "reach_z_max": config.sim_reach_z_max(),
            "reach_z_tolerance": config.sim_reach_z_tolerance(),
            "reach_go_tol_mm": config.sim_reach_go_tol_mm(),
        },
        "visual": visual,
    }


def dump_robot_yaml(config: ProjectConfig) -> str:
    return yaml.safe_dump(export_robot_yaml(config), sort_keys=False, default_flow_style=False)
