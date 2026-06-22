"""Leader ↔ follower calibration alignment for teleop."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import JOINT_NAMES, ProjectConfig
from .genesis.calibration import (
    calibration_path,
    load_calibration,
    norm_from_travel_fraction,
    norm_to_raw,
    require_calibration,
    travel_fraction,
)

_WIDE_CAL_ENCODER_SPAN = 4000
_RANGE_KEYS = ("range_min", "range_max", "drive_mode")


def _cal_entry_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return dict(entry)
    return {
        "id": int(entry.id),
        "drive_mode": int(entry.drive_mode),
        "homing_offset": int(entry.homing_offset),
        "range_min": int(entry.range_min),
        "range_max": int(entry.range_max),
    }


def _normalize_cal(cal: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: _cal_entry_dict(entry) for name, entry in cal.items() if name in JOINT_NAMES}


def merge_calibration_ranges(
    source: dict[str, dict[str, Any]],
    target: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Copy travel ranges from source; keep target homing_offset and motor ids."""
    target_norm = _normalize_cal(target)
    source_norm = _normalize_cal(source)
    merged = deepcopy(target_norm)
    for joint in JOINT_NAMES:
        if joint not in source_norm or joint not in merged:
            continue
        for key in _RANGE_KEYS:
            if key in source_norm[joint]:
                merged[joint][key] = source_norm[joint][key]
    return merged


def remap_leader_action_to_follower(
    action: dict[str, float],
    *,
    leader_cal: dict[str, dict[str, Any]],
    follower_cal: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Map leader norm → same travel fraction on follower (homing offsets may differ)."""
    out = dict(action)
    for joint in JOINT_NAMES:
        key = f"{joint}.pos"
        if key not in action:
            continue
        leader_raw = norm_to_raw(float(action[key]), joint, leader_cal)
        fraction = travel_fraction(leader_raw, joint, leader_cal)
        out[key] = norm_from_travel_fraction(fraction, joint, follower_cal)
    return out


def leader_pose_to_follower_action(
    *,
    joints: dict[str, float],
    raw: dict[str, int] | None,
    leader_cal: dict[str, dict[str, Any]],
    follower_cal: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Build follower goal from a recorded leader frame (prefers encoder counts when saved)."""
    out: dict[str, float] = {}
    for joint in JOINT_NAMES:
        key = f"{joint}.pos"
        if raw is not None and joint in raw:
            fraction = travel_fraction(int(raw[joint]), joint, leader_cal)
        else:
            leader_raw = norm_to_raw(float(joints[joint]), joint, leader_cal)
            fraction = travel_fraction(leader_raw, joint, leader_cal)
        out[key] = norm_from_travel_fraction(fraction, joint, follower_cal)
    return out


def _joint_span(cal: dict[str, dict[str, Any]], joint: str) -> int:
    entry = cal[joint]
    return int(entry["range_max"]) - int(entry["range_min"])


def calibration_mismatch_report(
    leader_cal: dict[str, dict[str, Any]],
    follower_cal: dict[str, dict[str, Any]],
) -> list[str]:
    """Human-readable issues when follower cal diverges from leader."""
    issues: list[str] = []
    for joint in JOINT_NAMES:
        ls = _joint_span(leader_cal, joint)
        fs = _joint_span(follower_cal, joint)
        if joint != "wrist_roll" and fs >= _WIDE_CAL_ENCODER_SPAN:
            f = follower_cal[joint]
            issues.append(
                f"{joint}: follower not swept ({f['range_min']}..{f['range_max']}) — "
                "run sync-calibration --from leader"
            )
        elif joint != "wrist_roll" and abs(fs - ls) > max(400, int(0.35 * ls)):
            issues.append(
                f"{joint}: leader span {ls} vs follower span {fs} — "
                "run sync-calibration --from leader"
            )
    return issues


def sync_calibration_file(
    cfg: ProjectConfig,
    *,
    from_role: str = "leader",
    to_role: str = "follower",
) -> Path:
    """Merge travel ranges from one role into the other's calibration JSON."""
    if from_role not in ("leader", "follower") or to_role not in ("leader", "follower"):
        raise ValueError("from_role and to_role must be 'leader' or 'follower'")
    src = require_calibration(from_role, cfg)
    dst_path = calibration_path(to_role, cfg)
    target = load_calibration(to_role, cfg) or deepcopy(src)
    merged = merge_calibration_ranges(src, target)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(merged, indent=4) + "\n", encoding="utf-8")
    return dst_path


def write_calibration_ranges_to_port(
    role: str,
    port: str,
    range_source: dict[str, dict[str, Any]],
    *,
    cfg: ProjectConfig | None = None,
) -> None:
    """Flash travel ranges to servos; preserve each motor's existing homing offset."""
    from lerobot.motors import MotorCalibration

    from .robot import _make_calibrate_device, _motor_write_retries, require_all_motors

    cfg = cfg or ProjectConfig.load()
    robot_id = cfg.robot.id if role == "follower" else cfg.teleop.leader.id
    require_all_motors(role, port, context="sync-calibration")
    device = _make_calibrate_device(role, port, robot_id, cfg)
    with _motor_write_retries():
        device.connect(calibrate=False)
        try:
            on_motors = device.bus.read_calibration()
            merged = merge_calibration_ranges(range_source, on_motors)
            motor_cal = {
                joint: MotorCalibration(
                    id=int(entry["id"]),
                    drive_mode=int(entry.get("drive_mode", 0)),
                    homing_offset=int(entry["homing_offset"]),
                    range_min=int(entry["range_min"]),
                    range_max=int(entry["range_max"]),
                )
                for joint, entry in merged.items()
                if joint in JOINT_NAMES
            }
            device.bus.write_calibration(motor_cal)
        finally:
            device.disconnect()


def run_sync_calibration(
    *,
    from_role: str = "leader",
    to_role: str = "follower",
    port: str | None = None,
    write_motors: bool = False,
) -> None:
    """CLI entry: merge travel ranges into cal file; optionally flash ranges to servos."""
    cfg = ProjectConfig.load()
    src = require_calibration(from_role, cfg)
    old_target = load_calibration(to_role, cfg)
    if to_role == "follower" and old_target is not None:
        issues = calibration_mismatch_report(src, old_target)
        if issues:
            print("Previous follower calibration issues:")
            for line in issues:
                print(f"  - {line}")
    dst = sync_calibration_file(cfg, from_role=from_role, to_role=to_role)
    print(f"Merged {from_role} travel ranges → {dst}")
    print("  (homing_offset on each arm is unchanged — never copy across arms)")

    if not write_motors:
        print("\nMotors unchanged. Re-run with --write-motors to flash ranges to servos.")
        return

    from .robot import ensure_port, resolve_role_port

    target_port = ensure_port(port or resolve_role_port(to_role, None), to_role.title())
    print(f"\nWriting travel ranges to {to_role} servos on {target_port} (keeping homing offsets)...")
    write_calibration_ranges_to_port(to_role, target_port, src, cfg=cfg)
    print("Done.")


def require_teleop_calibrations(cfg: ProjectConfig) -> tuple[dict, dict]:
    """Load leader + follower cal files or exit with instructions."""
    leader_cal = load_calibration("leader", cfg)
    follower_cal = load_calibration("follower", cfg)
    if leader_cal is None:
        print(
            f"Leader calibration missing: {calibration_path('leader', cfg)}\n"
            "Run:  sarm-hand calibrate --role leader",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if follower_cal is None:
        print(
            f"Follower calibration missing: {calibration_path('follower', cfg)}\n"
            "Run:  sarm-hand calibrate --role follower\n"
            "  or:  sarm-hand sync-calibration --from leader",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return leader_cal, follower_cal
