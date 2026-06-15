"""Predefined poses and pose-sequence validation for S-ARM101."""

from __future__ import annotations

import sys
import time
from typing import Any

from .config import JOINT_NAMES, ProjectConfig
from .robot import build_robot, ensure_port


def list_poses(config: ProjectConfig | None = None) -> None:
    """Print available poses and joint targets from config."""
    cfg = config or ProjectConfig.load()
    print("Available poses (normalized units, tunable in config/default.yaml):\n")
    for name in cfg.pose_names():
        pose = cfg.poses[name]
        joints = ", ".join(f"{j}={pose[j]:.0f}" for j in JOINT_NAMES)
        print(f"  {name:6}  {joints}")
    print(f"\nDefault test sequence: {' → '.join(cfg.pose_sequence)}")


def _pose_action(pose: dict[str, float]) -> dict[str, float]:
    return {f"{joint}.pos": pose[joint] for joint in JOINT_NAMES}


def _joint_tolerance(joint: str, default: float) -> float:
    return default * 1.5 if joint == "gripper" else default


def _move_to_pose(robot, target: dict[str, float], *, duration_s: float, fps: int) -> None:
    target_action = _pose_action(target)
    start_obs = robot.get_observation()
    steps = max(int(duration_s * fps), 1)

    for step in range(1, steps + 1):
        alpha = step / steps
        action = {}
        for joint in JOINT_NAMES:
            key = f"{joint}.pos"
            start_val = float(start_obs[key])
            action[key] = start_val + alpha * (target_action[key] - start_val)
        robot.send_action(action)
        time.sleep(1.0 / fps)


def _verify_pose(
    observation: dict[str, Any],
    target: dict[str, float],
    tolerance: float,
) -> tuple[bool, list[tuple[str, float, float, float]]]:
    """Return (passed, rows of joint, target, actual, error)."""
    rows: list[tuple[str, float, float, float]] = []
    passed = True

    for joint in JOINT_NAMES:
        target_val = target[joint]
        actual_val = float(observation[f"{joint}.pos"])
        error = abs(actual_val - target_val)
        tol = _joint_tolerance(joint, tolerance)
        if error > tol:
            passed = False
        rows.append((joint, target_val, actual_val, error))

    return passed, rows


def _print_pose_result(name: str, passed: bool, rows: list[tuple[str, float, float, float]], tolerance: float) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"\n{name} — {status}")
    print(f"{'Joint':<15} {'Target':>8} {'Actual':>8} {'Error':>8} {'Tol':>6}")
    print("-" * 50)
    for joint, target, actual, error in rows:
        tol = _joint_tolerance(joint, tolerance)
        flag = "!" if error > tol else " "
        print(f"{joint:<15} {target:8.1f} {actual:8.1f} {error:8.1f} {tol:6.1f}{flag}")


def test_poses(
    port: str | None = None,
    *,
    pose: str | None = None,
    sequence: list[str] | None = None,
    duration_s: float = 3.0,
    settle_s: float = 0.5,
    tolerance: float = 8.0,
    list_only: bool = False,
) -> None:
    """Move through predefined poses and verify the arm reaches each target."""
    cfg = ProjectConfig.load()

    if list_only:
        list_poses(cfg)
        return

    resolved_port = ensure_port(port or cfg.robot.port, "Follower")
    run_sequence = [pose] if pose else list(sequence or cfg.pose_sequence)

    for name in run_sequence:
        if name not in cfg.poses:
            print(f"Unknown pose '{name}'. Available: {', '.join(cfg.pose_names())}", file=sys.stderr)
            sys.exit(1)

    print("Pose validation test (follower arm)")
    print(f"  Port:      {resolved_port}")
    print(f"  Sequence:  {' → '.join(run_sequence)}")
    print(f"  Tolerance: ±{tolerance} (gripper ±{tolerance * 1.5:.0f})")
    print("\nKeep clear of the arm. Ctrl+C to abort.\n")

    robot = build_robot(resolved_port, cfg)

    if not robot.is_calibrated:
        robot.disconnect()
        print(
            "Arm is not calibrated. Run calibrate first:\n"
            f"  uv run sarm-hand calibrate --role follower --port {resolved_port}",
            file=sys.stderr,
        )
        sys.exit(1)

    all_passed = True
    try:
        for name in run_sequence:
            target = cfg.poses[name]
            print(f"Moving to '{name}' ({duration_s:.1f}s)...")
            _move_to_pose(robot, target, duration_s=duration_s, fps=cfg.dataset.fps)
            time.sleep(settle_s)

            obs = robot.get_observation()
            passed, rows = _verify_pose(obs, target, tolerance)
            _print_pose_result(name, passed, rows, tolerance)
            all_passed = all_passed and passed
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
    finally:
        robot.disconnect()

    if all_passed:
        print("\nAll poses reached within tolerance.")
    else:
        print(
            "\nSome poses missed their targets."
            "\n  - Re-run calibrate if joints drift from center at home"
            "\n  - Tune poses in config/default.yaml for your arm geometry"
            "\n  - Increase --tolerance slightly if motion is close but not exact"
        )
        sys.exit(1)
