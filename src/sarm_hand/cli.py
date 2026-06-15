"""Command-line interface for S-ARM101 project."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import JOINT_NAMES, ProjectConfig, parse_initial_ids
from .data import dataset_export_episode, dataset_info, dataset_push, dataset_sample
from .lelab_ui import (
    install_lelab,
    launch_lelab,
    lelab_info,
    open_dataset_viz_hub,
    viz_dataset_local,
)
from .record import record_leader, record_policy, record_quest
from .sim_api import launch_sim
from .poses import list_poses, test_poses
from .robot import calibrate, disable_arm_torque, find_port, resolve_role_port, setup_motors, test_motors
from .teleop import teleop_leader, teleop_quest, teleop_quest_instructions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sarm-hand",
        description="S-ARM101 arm: USB control, Quest 2 teleop, LeRobot data collection",
    )
    parser.add_argument("-v", "--version", action="version", version=f"sarm-hand {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("find-port", help="Find USB serial port for the arm")
    sub.add_parser("config-show", help="Show loaded project configuration")
    sub.add_parser("teleop-quest-help", help="Quest 2 teleoperation instructions")

    p = sub.add_parser(
        "setup-motors",
        help="Assign servo IDs per joint row (uses config/default.yaml motor map)",
    )
    p.add_argument("--role", choices=["follower", "leader"], required=True)
    p.add_argument(
        "--port",
        default=None,
        help="USB port (default: robot.port or teleop.leader.port from config/default.yaml)",
    )
    p.add_argument(
        "--initial-ids",
        default=None,
        help="Override current servo IDs per joint, e.g. shoulder_pan=1,gripper=6",
    )
    p.add_argument(
        "--scan",
        action="store_true",
        help="Scan the bus and print the motor table without programming",
    )
    p.add_argument(
        "--one-at-a-time",
        action="store_true",
        help="Legacy mode: connect one motor at a time (factory-fresh arms)",
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=list(JOINT_NAMES),
        metavar="JOINT",
        help="Program only these joints (connect that motor alone to the controller)",
    )

    p = sub.add_parser("calibrate", help="Calibrate follower or leader arm")
    p.add_argument("--role", choices=["follower", "leader"], required=True)
    p.add_argument(
        "--port",
        default=None,
        help="USB port (default: robot.port or teleop.leader.port from config/default.yaml)",
    )
    p.add_argument("--robot-id", default=None)

    p = sub.add_parser("test-motors", help="Ping/read/torque test each servo individually")
    p.add_argument("--role", choices=["follower", "leader"], required=True)
    p.add_argument(
        "--port",
        default=None,
        help="USB port (default: robot.port or teleop.leader.port from config/default.yaml)",
    )
    p.add_argument("--retries", type=int, default=3)

    p = sub.add_parser(
        "leader-free",
        help="Disable leader arm torque so it moves freely by hand (run before teleop if stiff)",
    )
    p.add_argument(
        "--port",
        default=None,
        help="Leader USB port (default: teleop.leader.port from config/default.yaml)",
    )

    p = sub.add_parser("test-poses", help="Move through home/ready/park poses and verify targets")
    p.add_argument("--port", default=None)
    p.add_argument(
        "--pose",
        default=None,
        help="Run a single pose instead of the full sequence (home, ready, park)",
    )
    p.add_argument(
        "--sequence",
        nargs="+",
        default=None,
        help="Custom pose order, e.g. --sequence home ready park home",
    )
    p.add_argument("--duration", type=float, default=3.0, help="Seconds to reach each pose")
    p.add_argument("--settle", type=float, default=0.5, help="Pause after each pose (seconds)")
    p.add_argument("--tolerance", type=float, default=8.0, help="Max joint error in normalized units")
    p.add_argument("--list", action="store_true", dest="list_poses", help="Show configured poses")

    p = sub.add_parser("teleop-leader", help="Leader-follower teleoperation via USB")
    p.add_argument("--follower-port", default=None)
    p.add_argument("--leader-port", default=None)
    p.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--with-cameras",
        action="store_true",
        help="Include cameras from config/default.yaml (optional; not needed for teleop)",
    )

    p = sub.add_parser("teleop-quest", help="Start phosphobot for Quest 2 VR teleop")
    p.add_argument("--follower-port", default=None)
    p.add_argument("--no-open-dashboard", action="store_true")

    p = sub.add_parser("record-leader", help="Record dataset with leader-follower teleop")
    p.add_argument("--follower-port", default=None)
    p.add_argument("--leader-port", default=None)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument("--single-task", default=None)
    p.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=None)

    p = sub.add_parser("record-quest", help="Quest 2 recording instructions")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=None)

    p = sub.add_parser("record-policy", help="Record policy evaluation rollouts")
    p.add_argument("--follower-port", default=None)
    p.add_argument("--policy-path", required=True)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--num-episodes", type=int, default=10)

    p = sub.add_parser("data-info", help="Show dataset metadata")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--root", default=None)

    p = sub.add_parser("data-sample", help="Print one dataset frame summary")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--index", type=int, default=0)

    p = sub.add_parser("data-export", help="Export episode to CSV")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output-dir", default="data/exports")

    p = sub.add_parser("data-push", help="Upload dataset to Hugging Face Hub")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--root", default=None)

    p = sub.add_parser("lelab", help="Launch LeLab web UI (datasets, 3D teleop, training)")
    p.add_argument("--install", action="store_true", help="Install LeLab via uv tool")
    p.add_argument("--dev", action="store_true", help="LeLab dev mode (Vite HMR + reload)")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--info", action="store_true", help="Show LeLab integration status")

    p = sub.add_parser("viz-dataset", help="Visualize a dataset locally (Rerun) or open HF Space")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument(
        "--hub",
        action="store_true",
        help="Open Hugging Face dataset visualizer (3D URDF) in browser",
    )

    p = sub.add_parser("sim", help="Launch SO-ARM101 3D joint simulator (web UI + IK API)")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-browser", action="store_true")

    return parser


def _show_config() -> None:
    cfg = ProjectConfig.load()
    print(f"Robot:   {cfg.robot.type} @ {cfg.robot.port or '(auto)'} id={cfg.robot.id}")
    print(f"Leader:  {cfg.teleop.leader.type} @ {cfg.teleop.leader.port or '(auto)'}")
    print(f"Quest:   phosphobot on port {cfg.teleop.quest.port}")
    print(f"Dataset: {cfg.dataset.repo_id} → {cfg.resolve_dataset_root()}")
    print(f"LeLab:   HF_LEROBOT_HOME → {cfg.lelab.resolve_hf_lerobot_home(cfg)}")
    print(f"Cameras: {list(cfg.cameras.keys()) or '(none)'}")
    for role in ("follower", "leader"):
        motor_map = cfg.motor_map(role)
        ids = ", ".join(f"{j}={motor_map.ids[j]}" for j in JOINT_NAMES)
        print(f"Motors ({role}): {ids}")


def main() -> None:
    args = _build_parser().parse_args()

    match args.command:
        case "find-port":
            find_port()
        case "config-show":
            _show_config()
        case "teleop-quest-help":
            teleop_quest_instructions()
        case "setup-motors":
            try:
                overrides = parse_initial_ids(args.initial_ids)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                sys.exit(1)
            setup_motors(
                args.role,
                resolve_role_port(args.role, args.port),
                initial_ids=overrides or None,
                one_at_a_time=args.one_at_a_time,
                scan_only=args.scan,
                only_joints=args.only,
            )
        case "calibrate":
            calibrate(args.role, resolve_role_port(args.role, args.port), args.robot_id)
        case "test-motors":
            test_motors(args.role, resolve_role_port(args.role, args.port), retries=args.retries)
        case "leader-free":
            disable_arm_torque("leader", resolve_role_port("leader", args.port))
        case "test-poses":
            test_poses(
                args.port,
                pose=args.pose,
                sequence=args.sequence,
                duration_s=args.duration,
                settle_s=args.settle,
                tolerance=args.tolerance,
                list_only=args.list_poses,
            )
        case "teleop-leader":
            teleop_leader(
                args.follower_port,
                args.leader_port,
                args.display_data,
                with_cameras=args.with_cameras,
            )
        case "teleop-quest":
            teleop_quest(args.follower_port, open_dashboard=not args.no_open_dashboard)
        case "record-leader":
            record_leader(
                args.follower_port,
                args.leader_port,
                args.repo_id,
                args.num_episodes,
                args.single_task,
                args.push_to_hub,
            )
        case "record-quest":
            record_quest(args.repo_id, args.push_to_hub)
        case "record-policy":
            record_policy(args.follower_port, args.policy_path, args.repo_id, args.num_episodes)
        case "data-info":
            dataset_info(args.repo_id, args.root)
        case "data-sample":
            dataset_sample(args.repo_id, args.root, args.index)
        case "data-export":
            dataset_export_episode(args.repo_id, args.root, args.episode, args.output_dir)
        case "data-push":
            dataset_push(args.repo_id, args.root)
        case "lelab":
            if args.install:
                install_lelab()
            elif args.info:
                lelab_info()
            else:
                launch_lelab(dev=args.dev, open_browser=not args.no_browser)
        case "viz-dataset":
            if args.hub:
                open_dataset_viz_hub(args.repo_id)
            else:
                viz_dataset_local(args.repo_id, args.root, args.episode)
        case "sim":
            launch_sim(
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
        case _:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
