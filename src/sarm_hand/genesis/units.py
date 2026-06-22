"""Convert between LeRobot normalized joint units and Genesis radians."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..config import JOINT_NAMES, ProjectConfig
from .calibration import norm_to_raw, raw_to_norm, startup_pose_norm
from .tensors import to_numpy
from .urdf_limits import clamp_to_urdf_limits, mapping_joint_limits, urdf_joint_limits

# LeRobot SO-101 normalized range (unless use_degrees).
_NORM_MIN = -100.0
_NORM_MAX = 100.0
_GRIPPER_MIN = 0.0
_GRIPPER_MAX = 100.0
# Cal files that span ~full encoder counts (0..4095) compress ~2× vs real joint travel.
_WIDE_CAL_ENCODER_SPAN = 4000
_full_cal_home_offset_cache: dict[int, dict[str, float]] = {}


def _joint_meta(cfg: ProjectConfig, joint: str) -> dict[str, Any]:
    joints = cfg.sim_joint_meta()
    return joints.get(joint, {})


def _joint_limits_norm(cfg: ProjectConfig, joint: str) -> tuple[float, float]:
    meta = _joint_meta(cfg, joint)
    if joint == "gripper":
        return (
            float(meta.get("min", _GRIPPER_MIN)),
            float(meta.get("max", _GRIPPER_MAX)),
        )
    return (
        float(meta.get("min", _NORM_MIN)),
        float(meta.get("max", _NORM_MAX)),
    )


def _genesis_joint_map(cfg: ProjectConfig, joint: str) -> dict[str, Any]:
    spec = cfg.genesis.joints.get(joint)
    if spec is None:
        return {"sign": 1.0, "urdf_offset": 0.0, "frame_offset": 0.0}
    return {
        "sign": float(spec.sign),
        "urdf_offset": float(spec.urdf_offset),
        "frame_offset": float(spec.frame_offset),
    }


def _is_wide_calibration(joint: str, calibration: dict[str, dict[str, Any]]) -> bool:
    joint_cal = calibration[joint]
    lo_raw = int(joint_cal["range_min"])
    hi_raw = int(joint_cal["range_max"])
    return (hi_raw - lo_raw) >= _WIDE_CAL_ENCODER_SPAN


def _mapping_mode(cfg: ProjectConfig) -> str:
    return (cfg.genesis.mapping or "delta").lower()


def _use_delta_mapping(cfg: ProjectConfig) -> bool:
    return _mapping_mode(cfg) == "delta"


def _use_wide_cal_mapping(
    joint: str,
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
) -> bool:
    """Wide-cal linear remap (new_calib); legacy path preserves physical joint semantics."""
    if _use_delta_mapping(cfg) or _mapping_mode(cfg) != "wide_cal":
        return False
    return _is_wide_calibration(joint, calibration) and cfg.genesis.home_raw.get(joint) is not None


def _rest_pose_radians(cfg: ProjectConfig, joint: str) -> float:
    deg = cfg.genesis.rest_pose.get(joint, 0.0)
    return math.radians(float(deg))


def _encoder_delta_radians(raw: int, joint: str, cfg: ProjectConfig) -> float:
    """Shaft rotation from ``home_raw`` in URDF radians (sign from genesis.joints)."""
    home = cfg.genesis.home_raw.get(joint)
    if home is None:
        return 0.0
    resolution = cfg.servo.resolution
    delta_raw = int(raw) - int(home)
    sign = float(_genesis_joint_map(cfg, joint)["sign"])
    return sign * delta_raw * (2.0 * math.pi / resolution)


def _delta_raw_to_radians(
    raw: int,
    joint: str,
    cfg: ProjectConfig,
    hard_limits: dict[str, tuple[float, float]],
) -> float:
    rad = _rest_pose_radians(cfg, joint) + _encoder_delta_radians(raw, joint, cfg)
    lo, hi = hard_limits[joint]
    return max(lo, min(hi, rad))


def _delta_radians_to_raw(
    rad: float,
    joint: str,
    cfg: ProjectConfig,
    hard_limits: dict[str, tuple[float, float]],
) -> int:
    home = cfg.genesis.home_raw.get(joint)
    if home is None:
        return 0
    sign = float(_genesis_joint_map(cfg, joint)["sign"])
    if sign == 0:
        sign = 1.0
    lo, hi = hard_limits[joint]
    bounded = max(lo, min(hi, float(rad)))
    delta_rad = bounded - _rest_pose_radians(cfg, joint)
    delta_raw = delta_rad * cfg.servo.resolution / (2.0 * math.pi * sign)
    return int(round(int(home) + delta_raw))


def _legacy_mapping_limits(
    cfg: ProjectConfig,
    joint: str,
    hard_limits: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Old-calib semantic limits for home anchoring (wide-cal joints only)."""
    spec = cfg.genesis.joints.get(joint)
    if spec is not None and spec.urdf_min is not None and spec.urdf_max is not None:
        return (float(spec.urdf_min), float(spec.urdf_max))
    return hard_limits[joint]


def _linear_norm_to_hard(
    norm: float,
    joint: str,
    cfg: ProjectConfig,
    hard_limits: dict[str, tuple[float, float]],
) -> float:
    """Map LeRobot norm linearly onto Genesis URDF hard limits."""
    lo_rad, hi_rad = hard_limits[joint]
    lo_n, hi_n = _joint_limits_norm(cfg, joint)
    span_n = hi_n - lo_n
    if span_n == 0:
        t = 0.5
    else:
        t = (float(norm) - lo_n) / span_n
    sign = float(_genesis_joint_map(cfg, joint)["sign"])
    if sign < 0:
        t = 1.0 - t
    offset = float(_genesis_joint_map(cfg, joint)["urdf_offset"])
    return lo_rad + t * (hi_rad - lo_rad) + offset


def _raw_to_legacy_radians(
    raw: int,
    joint: str,
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
    legacy_limits: tuple[float, float],
) -> float:
    lo_rad, hi_rad = legacy_limits
    t = _raw_fraction(raw, joint, calibration)
    sign = float(_genesis_joint_map(cfg, joint)["sign"])
    if sign < 0:
        t = 1.0 - t
    offset = float(_genesis_joint_map(cfg, joint)["urdf_offset"])
    return lo_rad + t * (hi_rad - lo_rad) + offset


def _full_cal_home_offset(
    joint: str,
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
    hard_limits: dict[str, tuple[float, float]],
) -> float:
    """URDF offset so home_raw matches legacy rest pose in the new_calib frame."""
    cache_key = id(cfg)
    cached = _full_cal_home_offset_cache.get(cache_key)
    if cached is not None and joint in cached:
        return cached[joint]

    home_raw = cfg.genesis.home_raw.get(joint)
    if home_raw is None:
        return 0.0

    home_norm = raw_to_norm(home_raw, joint, calibration)
    legacy_limits = _legacy_mapping_limits(cfg, joint, hard_limits)
    legacy_home = _raw_to_legacy_radians(home_raw, joint, cfg, calibration, legacy_limits)
    frame_offset = float(_genesis_joint_map(cfg, joint)["frame_offset"])
    target_home = legacy_home + frame_offset
    linear_home = _linear_norm_to_hard(home_norm, joint, cfg, hard_limits)
    offset = target_home - linear_home

    if cache_key not in _full_cal_home_offset_cache:
        _full_cal_home_offset_cache[cache_key] = {}
    _full_cal_home_offset_cache[cache_key][joint] = offset
    return offset


def _norm_to_radians_wide_cal(
    norm: float,
    joint: str,
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
    hard_limits: dict[str, tuple[float, float]],
) -> float:
    lo_hard, hi_hard = hard_limits[joint]
    offset = _full_cal_home_offset(joint, cfg, calibration, hard_limits)
    rad = _linear_norm_to_hard(norm, joint, cfg, hard_limits) + offset
    return max(lo_hard, min(hi_hard, rad))


def _radians_to_norm_wide_cal(
    rad: float,
    joint: str,
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
    hard_limits: dict[str, tuple[float, float]],
) -> float:
    lo_hard, hi_hard = hard_limits[joint]
    offset = _full_cal_home_offset(joint, cfg, calibration, hard_limits)
    bounded = max(lo_hard, min(hi_hard, float(rad))) - offset
    lo_rad, hi_rad = hard_limits[joint]
    explicit_offset = float(_genesis_joint_map(cfg, joint)["urdf_offset"])
    linear_rad = bounded - explicit_offset
    span = hi_rad - lo_rad
    if span == 0:
        t = 0.5
    else:
        t = (linear_rad - lo_rad) / span
    sign = float(_genesis_joint_map(cfg, joint)["sign"])
    if sign < 0:
        t = 1.0 - t
    lo_n, hi_n = _joint_limits_norm(cfg, joint)
    return lo_n + t * (hi_n - lo_n)


def _norm_fraction(value: float, joint: str, cfg: ProjectConfig) -> float:
    """Map a LeRobot joint value to fraction along the norm range [0, 1]."""
    lo_n, hi_n = _joint_limits_norm(cfg, joint)
    span = hi_n - lo_n
    if span == 0:
        return 0.5
    return (float(value) - lo_n) / span


def _raw_fraction(
    raw: int,
    joint: str,
    calibration: dict[str, dict[str, Any]],
) -> float:
    """Map a servo count to fraction [0, 1] along calibrated min..max."""
    joint_cal = calibration[joint]
    lo_raw = int(joint_cal["range_min"])
    hi_raw = int(joint_cal["range_max"])
    span = hi_raw - lo_raw
    bounded = max(lo_raw, min(hi_raw, int(raw)))
    if span == 0:
        return 0.5
    return (bounded - lo_raw) / span


def _fraction_to_raw(
    t: float,
    joint: str,
    calibration: dict[str, dict[str, Any]],
) -> int:
    """Inverse of :func:`_raw_fraction`."""
    joint_cal = calibration[joint]
    lo_raw = int(joint_cal["range_min"])
    hi_raw = int(joint_cal["range_max"])
    t = max(0.0, min(1.0, float(t)))
    raw = lo_raw + t * (hi_raw - lo_raw)
    return int(round(max(lo_raw, min(hi_raw, raw))))


def raw_to_radians(
    raw: int,
    joint: str,
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
    *,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
    hard_limits: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Map a servo encoder count to URDF radians using cal min/max → mapping limits."""
    hard = hard_limits or urdf_limits or urdf_joint_limits(cfg)
    if _use_delta_mapping(cfg) and cfg.genesis.home_raw.get(joint) is not None:
        return _delta_raw_to_radians(int(raw), joint, cfg, hard)
    if _use_wide_cal_mapping(joint, cfg, calibration):
        norm = raw_to_norm(raw, joint, calibration)
        return _norm_to_radians_wide_cal(norm, joint, cfg, calibration, hard)

    map_limits = mapping_joint_limits(cfg, urdf_limits=urdf_limits)
    lo_rad, hi_rad = map_limits[joint]
    t = _raw_fraction(raw, joint, calibration)

    sign = float(_genesis_joint_map(cfg, joint)["sign"])
    if sign < 0:
        t = 1.0 - t

    offset = float(_genesis_joint_map(cfg, joint)["urdf_offset"])
    rad = lo_rad + t * (hi_rad - lo_rad) + offset
    lo_hard, hi_hard = hard[joint]
    return max(lo_hard, min(hi_hard, rad))


def radians_to_raw(
    value: float,
    joint: str,
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
    *,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
    hard_limits: dict[str, tuple[float, float]] | None = None,
) -> int:
    """Map URDF radians back to a servo encoder count."""
    hard = hard_limits or urdf_limits or urdf_joint_limits(cfg)
    if _use_delta_mapping(cfg) and cfg.genesis.home_raw.get(joint) is not None:
        return _delta_radians_to_raw(float(value), joint, cfg, hard)
    if _use_wide_cal_mapping(joint, cfg, calibration):
        norm = _radians_to_norm_wide_cal(value, joint, cfg, calibration, hard)
        return norm_to_raw(norm, joint, calibration)

    map_limits = mapping_joint_limits(cfg, urdf_limits=urdf_limits)
    lo_rad, hi_rad = map_limits[joint]
    lo_hard, hi_hard = hard[joint]
    offset = float(_genesis_joint_map(cfg, joint)["urdf_offset"])
    rad = max(lo_hard, min(hi_hard, float(value))) - offset
    span = hi_rad - lo_rad
    if span == 0:
        t = 0.5
    else:
        t = (rad - lo_rad) / span

    sign = float(_genesis_joint_map(cfg, joint)["sign"])
    if sign < 0:
        t = 1.0 - t

    return _fraction_to_raw(t, joint, calibration)


def calibrated_urdf_limits(
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
    *,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """URDF radian span matching each joint's calibrated servo min..max."""
    map_limits = mapping_joint_limits(cfg, urdf_limits=urdf_limits)
    hard = urdf_limits or urdf_joint_limits(cfg)
    out: dict[str, tuple[float, float]] = {}
    for joint in JOINT_NAMES:
        lo_raw = int(calibration[joint]["range_min"])
        hi_raw = int(calibration[joint]["range_max"])
        rad_lo = raw_to_radians(
            lo_raw, joint, cfg, calibration, urdf_limits=urdf_limits, hard_limits=hard
        )
        rad_hi = raw_to_radians(
            hi_raw, joint, cfg, calibration, urdf_limits=urdf_limits, hard_limits=hard
        )
        out[joint] = (min(rad_lo, rad_hi), max(rad_lo, rad_hi))
    return out


def norm_to_radians(
    value: float,
    joint: str,
    cfg: ProjectConfig,
    *,
    calibration: dict[str, dict[str, Any]] | None = None,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Map LeRobot norm → URDF radians (via servo counts when calibration is set)."""
    hard = urdf_limits or urdf_joint_limits(cfg)
    if calibration is not None:
        if _use_wide_cal_mapping(joint, cfg, calibration):
            return _norm_to_radians_wide_cal(float(value), joint, cfg, calibration, hard)
        raw = norm_to_raw(value, joint, calibration)
        return raw_to_radians(raw, joint, cfg, calibration, urdf_limits=urdf_limits)

    map_limits = mapping_joint_limits(cfg, urdf_limits=urdf_limits)
    lo_rad, hi_rad = map_limits[joint]
    t = _norm_fraction(value, joint, cfg)
    sign = float(_genesis_joint_map(cfg, joint)["sign"])
    if sign < 0:
        t = 1.0 - t
    offset = float(_genesis_joint_map(cfg, joint)["urdf_offset"])
    rad = lo_rad + t * (hi_rad - lo_rad) + offset
    lo_hard, hi_hard = hard[joint]
    return max(lo_hard, min(hi_hard, rad))


def radians_to_norm(
    value: float,
    joint: str,
    cfg: ProjectConfig,
    *,
    calibration: dict[str, dict[str, Any]] | None = None,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Map URDF radians → LeRobot norm (via servo counts when calibration is set)."""
    if cfg.robot.use_degrees:
        return math.degrees(float(value))

    hard = urdf_limits or urdf_joint_limits(cfg)
    if calibration is not None:
        if _use_wide_cal_mapping(joint, cfg, calibration):
            return _radians_to_norm_wide_cal(float(value), joint, cfg, calibration, hard)
        raw = radians_to_raw(value, joint, cfg, calibration, urdf_limits=urdf_limits)
        return raw_to_norm(raw, joint, calibration)

    map_limits = mapping_joint_limits(cfg, urdf_limits=urdf_limits)
    lo_rad, hi_rad = map_limits[joint]
    offset = float(_genesis_joint_map(cfg, joint)["urdf_offset"])
    rad = float(value) - offset
    span = hi_rad - lo_rad
    if span == 0:
        t = 0.5
    else:
        t = (rad - lo_rad) / span
    sign = float(_genesis_joint_map(cfg, joint)["sign"])
    if sign < 0:
        t = 1.0 - t
    lo_n, hi_n = _joint_limits_norm(cfg, joint)
    return lo_n + t * (hi_n - lo_n)


def observation_to_radians(
    obs: dict[str, float],
    cfg: ProjectConfig,
    *,
    calibration: dict[str, dict[str, Any]] | None = None,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
) -> list[float]:
    """Convert robot observation dict to radians list in JOINT_NAMES order."""
    radians = [
        norm_to_radians(
            float(obs[f"{name}.pos"]),
            name,
            cfg,
            calibration=calibration,
            urdf_limits=urdf_limits,
        )
        for name in JOINT_NAMES
    ]
    return clamp_to_urdf_limits(radians, cfg, urdf_limits=urdf_limits)


def radians_to_observation(
    values: list[float],
    cfg: ProjectConfig,
    *,
    calibration: dict[str, dict[str, Any]] | None = None,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Convert radians list to LeRobot-style observation dict."""
    return {
        f"{name}.pos": radians_to_norm(
            values[i],
            name,
            cfg,
            calibration=calibration,
            urdf_limits=urdf_limits,
        )
        for i, name in enumerate(JOINT_NAMES)
    }


def action_dict_to_vector(action: dict[str, float]) -> list[float]:
    """Flatten a LeRobot action dict to a vector in JOINT_NAMES order."""
    return [float(action[f"{name}.pos"]) for name in JOINT_NAMES]


def agent_pos_from_qpos(
    qpos,
    cfg: ProjectConfig,
    *,
    calibration: dict[str, dict[str, Any]] | None = None,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Convert Genesis DOF positions to a float32 state vector."""
    qpos = to_numpy(qpos)
    return np.array(
        [
            radians_to_norm(
                float(q),
                JOINT_NAMES[i],
                cfg,
                calibration=calibration,
                urdf_limits=urdf_limits,
            )
            for i, q in enumerate(qpos)
        ],
        dtype=np.float32,
    )


def home_pose_radians(
    cfg: ProjectConfig,
    pose_name: str = "home",
    *,
    calibration: dict[str, dict[str, Any]] | None = None,
    urdf_limits: dict[str, tuple[float, float]] | None = None,
    role: str | None = None,
) -> list[float]:
    """Return URDF radians for startup (``genesis.home_raw``) or a named pose."""
    if pose_name == "home" and cfg.genesis.home_raw:
        obs = startup_pose_norm(cfg, role=role, calibration=calibration)
    else:
        pose = cfg.poses.get(pose_name, {})
        obs = {
            f"{name}.pos": float(pose.get(name, 50.0 if name == "gripper" else 0.0))
            for name in JOINT_NAMES
        }
    return observation_to_radians(
        obs,
        cfg,
        calibration=calibration,
        urdf_limits=urdf_limits,
    )
