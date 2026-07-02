"""Predefined poses and pose-sequence validation for S-ARM101."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

from .calibration_bridge import leader_pose_to_follower_action, require_teleop_calibrations
from .config import JOINT_NAMES, PROJECT_ROOT, ProjectConfig
from .genesis.calibration import raw_to_norm
from .robot import build_robot, disable_arm_torque, ensure_port, resolve_role_port

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def compute_rest_pose_from_genesis(
    cfg: ProjectConfig,
    *,
    role: str = "follower",
) -> dict[str, float] | None:
    """Map ``genesis.home_raw`` (leader at folded rest) to LeRobot norm for leader or follower."""
    home_raw = cfg.genesis.home_raw
    if not home_raw or not all(j in home_raw for j in JOINT_NAMES):
        return None
    try:
        leader_cal, follower_cal = require_teleop_calibrations(cfg)
    except SystemExit:
        return None

    leader_joints = {
        joint: raw_to_norm(int(home_raw[joint]), joint, leader_cal) for joint in JOINT_NAMES
    }
    if role == "leader":
        return leader_joints

    follower_action = leader_pose_to_follower_action(
        joints=leader_joints,
        raw={joint: int(home_raw[joint]) for joint in JOINT_NAMES},
        leader_cal=leader_cal,
        follower_cal=follower_cal,
    )
    return {joint: float(follower_action[f"{joint}.pos"]) for joint in JOINT_NAMES}


def format_pose_yaml(name: str, pose: dict[str, float]) -> str:
    lines = [f"  {name}:"]
    for joint in JOINT_NAMES:
        lines.append(f"    {joint}: {pose[joint]:.1f}")
    return "\n".join(lines)


def _pose_block_pattern(name: str) -> re.Pattern[str]:
    """Match a poses.<name> block (allows comment lines between joints)."""
    return re.compile(
        rf"^  {re.escape(name)}:\n"
        rf"(?:    (?:#.*|\w+: [-0-9.]+)\n)+",
        re.MULTILINE,
    )


def patch_pose_in_yaml(
    path: Path,
    name: str,
    pose: dict[str, float],
    *,
    rest_from_genesis: bool | None = None,
) -> None:
    """Replace a pose block under ``poses:`` in config/default.yaml."""
    text = path.read_text()
    block = format_pose_yaml(name, pose)
    pattern = _pose_block_pattern(name)
    if not pattern.search(text):
        raise ValueError(f"No poses.{name} block found in {path}")
    updated = pattern.sub(block + "\n", text, count=1)
    if rest_from_genesis is not None:
        flag = "true" if rest_from_genesis else "false"
        if "rest_from_genesis:" in updated:
            updated = re.sub(
                r"^  rest_from_genesis: .*$",
                f"  rest_from_genesis: {flag}",
                updated,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            updated = updated.replace(
                "poses:\n",
                f"poses:\n  rest_from_genesis: {flag}\n",
                1,
            )
    path.write_text(updated)


def read_role_pose_norm(
    role: str,
    port: str,
    cfg: ProjectConfig,
) -> dict[str, float]:
    """Read current joint norms with torque off (physical rest pose)."""
    if role == "follower":
        robot = build_robot(port, cfg, use_cameras=False)
        try:
            robot.bus.disable_torque(num_retry=5)
            time.sleep(0.3)
            obs = robot.get_observation()
        finally:
            robot.disconnect()
        return {joint: float(obs[f"{joint}.pos"]) for joint in JOINT_NAMES}

    from lerobot.teleoperators.so_leader import SO101Leader
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    disable_arm_torque(role, port)
    leader_cfg = SO101LeaderConfig(
        id=cfg.teleop.leader.id,
        port=port,
        use_degrees=cfg.robot.use_degrees,
    )
    leader = SO101Leader(leader_cfg)
    try:
        leader.connect()
        leader.bus.disable_torque(num_retry=5)
        time.sleep(0.3)
        action = leader.get_action()
    finally:
        leader.disconnect()
    return {joint: float(action[f"{joint}.pos"]) for joint in JOINT_NAMES}


def capture_pose(
    *,
    role: str,
    name: str,
    port: str | None = None,
    save: bool = False,
    config_path: Path | None = None,
) -> dict[str, float]:
    """Read the arm's current pose and optionally save it under poses.<name> in config."""
    if name not in ("home", "ready", "park"):
        print(f"Unknown pose name {name!r}. Use: home, ready, park", file=sys.stderr)
        raise SystemExit(1)

    cfg = ProjectConfig.load()
    resolved_port = resolve_role_port(role, port)
    print(f"Place the {role} arm in its physical rest pose (torque will be disabled to read).")
    pose = read_role_pose_norm(role, resolved_port, cfg)

    print(f"Captured {role} pose for poses.{name} (torque off, normalized units):\n")
    print(format_pose_yaml(name, pose))
    if role == "follower" and name == "ready":
        genesis = compute_rest_pose_from_genesis(cfg, role="follower")
        if genesis:
            print("\nFor comparison, genesis.home_raw → follower rest:")
            print(format_pose_yaml("ready (genesis)", genesis))

    if save:
        path = config_path or DEFAULT_CONFIG_PATH
        rest_flag = False if name == "ready" else None
        patch_pose_in_yaml(path, name, pose, rest_from_genesis=rest_flag)
        print(f"\nUpdated {path}")
        if name == "ready":
            print("  Set poses.rest_from_genesis: false (using captured values)")
    else:
        print("\nAdd to config with:  --save")

    return pose


def move_connected_follower_to_pose(
    robot,
    cfg: ProjectConfig,
    pose_name: str,
    *,
    duration_s: float = 3.0,
) -> None:
    """Move an already-connected follower to a named pose."""
    target = cfg.resolve_pose(pose_name, role="follower")
    print(f"Moving to '{pose_name}' ({duration_s:.1f}s)...")
    _move_to_pose(robot, target, duration_s=duration_s, fps=cfg.dataset.fps)


def move_follower_to_pose(
    cfg: ProjectConfig,
    pose_name: str,
    *,
    port: str | None = None,
    duration_s: float = 3.0,
) -> None:
    """Smoothly move follower to a named pose (uses resolve_pose for ``ready``)."""
    resolved_port = ensure_port(port or cfg.robot.port, "Follower")
    target = cfg.resolve_pose(pose_name, role="follower")
    robot = build_robot(resolved_port, cfg, use_cameras=False)
    try:
        move_connected_follower_to_pose(robot, cfg, pose_name, duration_s=duration_s)
    finally:
        robot.disconnect()


def list_poses(config: ProjectConfig | None = None) -> None:
    """Print available poses and joint targets from config."""
    cfg = config or ProjectConfig.load()
    print("Available poses (normalized units, tunable in config/default.yaml):\n")
    for name in cfg.pose_names():
        pose = cfg.resolve_pose(name, role="follower") if name == "ready" else cfg.poses[name]
        joints = ", ".join(f"{j}={pose[j]:.1f}" for j in JOINT_NAMES)
        suffix = ""
        if name == "ready" and cfg.rest_from_genesis:
            suffix = "  (from genesis.home_raw rest)"
        print(f"  {name:6}  {joints}{suffix}")
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
    if "ready" in run_sequence and cfg.rest_from_genesis:
        print("  ready:     physical rest from genesis.home_raw")
    print(f"  Tolerance: ±{tolerance} (gripper ±{tolerance * 1.5:.0f})")
    print("\nKeep clear of the arm. Ctrl+C to abort.\n")

    robot = build_robot(resolved_port, cfg, use_cameras=False)

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
            target = cfg.resolve_pose(name, role="follower")
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
            "\n  - Capture rest: sarm-hand capture-pose --role follower --name ready --save"
            "\n  - Increase --tolerance slightly if motion is close but not exact"
        )
        sys.exit(1)
