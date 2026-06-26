"""Load LeRobot motor calibration and convert raw encoder counts to norm."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from ..config import JOINT_NAMES, ProjectConfig

_LEGACY_LEROBOT_CAL_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"


def lerobot_calibration_root() -> Path:
    """LeRobot calibration directory (respects HF_LEROBOT_HOME when set)."""
    home = os.environ.get("HF_LEROBOT_HOME")
    if home:
        return Path(home).expanduser() / "calibration"
    cal = os.environ.get("HF_LEROBOT_CALIBRATION")
    if cal:
        return Path(cal).expanduser()
    return _LEGACY_LEROBOT_CAL_ROOT


def calibration_path(role: str, cfg: ProjectConfig) -> Path:
    """Path to the saved LeRobot calibration JSON for follower or leader."""
    root = lerobot_calibration_root()
    if role == "leader":
        return root / "teleoperators" / "so_leader" / f"{cfg.teleop.leader.id}.json"
    return root / "robots" / "so_follower" / f"{cfg.robot.id}.json"


def _legacy_calibration_path(role: str, cfg: ProjectConfig) -> Path:
    if role == "leader":
        return (
            _LEGACY_LEROBOT_CAL_ROOT
            / "teleoperators"
            / "so_leader"
            / f"{cfg.teleop.leader.id}.json"
        )
    return _LEGACY_LEROBOT_CAL_ROOT / "robots" / "so_follower" / f"{cfg.robot.id}.json"


def load_calibration(role: str, cfg: ProjectConfig) -> dict[str, dict[str, Any]] | None:
    for path in (calibration_path(role, cfg), _legacy_calibration_path(role, cfg)):
        if path.is_file():
            data = json.loads(path.read_text())
            return {k: v for k, v in data.items() if k in JOINT_NAMES}
    return None


def require_calibration(role: str, cfg: ProjectConfig) -> dict[str, dict[str, Any]]:
    """Load calibration or exit with a helpful message."""
    cal = load_calibration(role, cfg)
    if cal is None:
        print(
            f"Genesis sim requires {role} calibration at {calibration_path(role, cfg)}.\n"
            f"Run:  sarm-hand calibrate --role {role}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    missing = [j for j in JOINT_NAMES if j not in cal]
    if missing:
        print(f"Calibration for {role!r} is missing joints: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)
    return cal


def raw_to_norm(raw: int, joint: str, calibration: dict[str, dict[str, Any]]) -> float:
    """Convert a Feetech Present_Position count to LeRobot normalized units."""
    joint_cal = calibration[joint]
    lo = int(joint_cal["range_min"])
    hi = int(joint_cal["range_max"])
    if hi == lo:
        raise ValueError(f"Invalid calibration for {joint}: range_min == range_max")
    bounded = max(lo, min(hi, int(raw)))
    drive_mode = bool(joint_cal.get("drive_mode", 0))
    if joint == "gripper":
        norm = ((bounded - lo) / (hi - lo)) * 100.0
        return 100.0 - norm if drive_mode else norm
    norm = (((bounded - lo) / (hi - lo)) * 200.0) - 100.0
    return -norm if drive_mode else norm


def norm_to_raw(norm: float, joint: str, calibration: dict[str, dict[str, Any]]) -> int:
    """Inverse of :func:`raw_to_norm` (matches LeRobot Feetech bus)."""
    joint_cal = calibration[joint]
    lo = int(joint_cal["range_min"])
    hi = int(joint_cal["range_max"])
    if hi == lo:
        raise ValueError(f"Invalid calibration for {joint}: range_min == range_max")
    drive_mode = bool(joint_cal.get("drive_mode", 0))
    if joint == "gripper":
        val = float(norm)
        if drive_mode:
            val = 100.0 - val
        val = max(0.0, min(100.0, val))
        return int(round((val / 100.0) * (hi - lo) + lo))
    val = float(norm)
    if drive_mode:
        val = -val
    val = max(-100.0, min(100.0, val))
    return int(round(((val + 100.0) / 200.0) * (hi - lo) + lo))


def travel_fraction(raw: int, joint: str, calibration: dict[str, dict[str, Any]]) -> float:
    """Fraction [0, 1] along calibrated joint travel (independent of homing offset)."""
    joint_cal = calibration[joint]
    lo = int(joint_cal["range_min"])
    hi = int(joint_cal["range_max"])
    span = hi - lo
    bounded = max(lo, min(hi, int(raw)))
    if span == 0:
        return 0.5
    t = (bounded - lo) / span
    if bool(joint_cal.get("drive_mode", 0)):
        t = 1.0 - t
    return float(t)


def norm_from_travel_fraction(
    fraction: float,
    joint: str,
    calibration: dict[str, dict[str, Any]],
) -> float:
    """Map travel fraction to LeRobot norm on this arm's calibrated range."""
    joint_cal = calibration[joint]
    lo = int(joint_cal["range_min"])
    hi = int(joint_cal["range_max"])
    span = hi - lo
    t = max(0.0, min(1.0, float(fraction)))
    if bool(joint_cal.get("drive_mode", 0)):
        t = 1.0 - t
    raw = int(round(lo + t * span)) if span else lo
    return raw_to_norm(raw, joint, calibration)


def raw_positions_to_norm(
    raw: dict[str, int],
    calibration: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Map raw encoder counts to ``{joint.pos: norm}`` observation keys."""
    return {
        f"{joint}.pos": raw_to_norm(raw[joint], joint, calibration)
        for joint in JOINT_NAMES
        if joint in raw and joint in calibration
    }


def startup_pose_norm(
    cfg: ProjectConfig,
    *,
    role: str | None = None,
    calibration: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Normalized joint pose for Genesis startup (from ``genesis.home_raw``)."""
    home_raw = cfg.genesis.home_raw
    if not home_raw:
        pose = cfg.poses.get("home", {})
        return {
            f"{name}.pos": float(pose.get(name, 50.0 if name == "gripper" else 0.0))
            for name in JOINT_NAMES
        }

    cal_role = role or cfg.genesis.calibration_role
    cal = calibration or require_calibration(cal_role, cfg)
    obs = raw_positions_to_norm(home_raw, cal)
    missing = [name for name in JOINT_NAMES if f"{name}.pos" not in obs]
    if missing:
        raise ValueError(f"genesis.home_raw missing joints: {', '.join(missing)}")
    return obs


def calibration_summary(calibration: dict[str, dict[str, Any]]) -> str:
    """One-line per joint for startup logs."""
    parts = []
    for joint in JOINT_NAMES:
        c = calibration[joint]
        parts.append(f"{joint}={c['range_min']}..{c['range_max']}")
    return ", ".join(parts)
