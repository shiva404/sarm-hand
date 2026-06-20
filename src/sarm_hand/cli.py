"""Command-line interface for S-ARM101 project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .cameras import describe_camera, list_usb_cameras, preview_camera, test_configured_cameras
from .config import JOINT_NAMES, ProjectConfig, parse_initial_ids
from .data import dataset_export_episode, dataset_info, dataset_push, dataset_sample
from .joint_signal_log import run_joint_signal_log
from .lelab_ui import (
    install_lelab,
    launch_lelab,
    lelab_info,
    open_dataset_viz_hub,
    viz_dataset_local,
)
from .policy import run_smolvla, train_smolvla
from .poses import test_poses
from .record import record_leader, record_policy, record_quest
from .record_sim import record_sim, record_twin
from .robot import (
    calibrate,
    disable_arm_torque,
    find_port,
    resolve_role_port,
    setup_motors,
    test_motors,
)
from .sim_api import launch_sim
from .task_motion import list_task_motions, record_task_motion, replay_task_motion, show_task_motion
from .teleop import teleop_leader, teleop_quest, teleop_quest_instructions
from .twin import run_genesis_spike, run_twin


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sarm-hand",
        description="S-ARM101 arm: USB control, Quest 2 teleop, LeRobot data collection",
    )
    parser.add_argument("-v", "--version", action="version", version=f"sarm-hand {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("find-port", help="Find USB serial port for the arm")
    sub.add_parser("list-cameras", help="List USB cameras detected by OpenCV")
    sub.add_parser("config-show", help="Show loaded project configuration")
    sub.add_parser("teleop-quest-help", help="Quest 2 teleoperation instructions")

    p = sub.add_parser("camera-preview", help="Preview a USB camera or HTTP/RTSP stream")
    p.add_argument("--name", default=None, help="Camera name from config/default.yaml")
    p.add_argument("--index", type=int, default=None, help="USB camera device index (e.g. 0)")
    p.add_argument("--url", default=None, help="HTTP or RTSP stream URL")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--output", default=None, help="Save first frame to this image path")
    p.add_argument("--seconds", type=float, default=5.0, help="Preview duration (default: 5)")
    p.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open a GUI window (use with --output for headless capture)",
    )

    sub.add_parser("camera-test", help="Test all cameras defined in config/default.yaml")

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

    task_p = sub.add_parser(
        "task",
        help="Record leader demos and replay them on the follower by task name",
    )
    task_sub = task_p.add_subparsers(dest="task_command", required=True)

    p = task_sub.add_parser("record", help="Record a leader-arm demo for a task")
    p.add_argument(
        "--task",
        required=True,
        help='Task label, e.g. "Pick up the cube and place it in the box"',
    )
    p.add_argument("--leader-port", default=None)
    p.add_argument("--follower-port", default=None)
    p.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Sample rate (default: tasks.fps in config)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Max seconds (default: until Ctrl+C)",
    )
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="Record leader only; do not mirror to follower during capture",
    )

    p = task_sub.add_parser("replay", help="Replay a saved task demo on the follower")
    p.add_argument("--task", default=None, help="Task label (uses slug derived from text)")
    p.add_argument("--task-slug", default=None, help="Task folder slug, e.g. pick_up_the_cube")
    p.add_argument(
        "--demo",
        default="latest",
        help="Demo id or 'latest' (default: latest)",
    )
    p.add_argument("--follower-port", default=None)
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.add_argument("--loop", action="store_true", help="Repeat until Ctrl+C")
    p.add_argument("--pause", type=float, default=2.0, help="Seconds before motion starts")

    task_sub.add_parser("list", help="List recorded task demos")

    p = task_sub.add_parser("info", help="Show metadata for one task demo")
    p.add_argument("--task", default=None)
    p.add_argument("--task-slug", default=None)
    p.add_argument("--demo", default="latest")

    p = sub.add_parser("record-quest", help="Quest 2 recording instructions")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=None)

    p = sub.add_parser("record-policy", help="Record policy evaluation rollouts")
    p.add_argument("--follower-port", default=None)
    p.add_argument("--policy-path", required=True)
    p.add_argument("--task", default=None, help="Language task query for VLA policies")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--num-episodes", type=int, default=10)

    p = sub.add_parser(
        "run-smolvla",
        help="Run SmolVLA policy with a natural-language task on the follower arm",
    )
    p.add_argument(
        "--task",
        default=None,
        help="Task query, e.g. 'Pick up the cube and place it in the box'",
    )
    p.add_argument("--follower-port", default=None)
    p.add_argument("--policy-path", default=None, help="HF model id or local checkpoint")
    p.add_argument("--device", default=None, help="cuda, mps, or cpu")
    p.add_argument("--episode-time", type=float, default=None, help="Seconds per episode")
    p.add_argument("--interactive", action="store_true", help="Prompt for task queries")
    p.add_argument(
        "--record",
        action="store_true",
        help="Record eval episodes with lerobot-record instead of live-only run",
    )
    p.add_argument("--repo-id", default=None, help="Eval dataset repo id when --record")
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--genesis",
        action="store_true",
        help="Run in Genesis sim (no USB arm; uses genesis.cameras for vision)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Genesis without viewer windows (with --genesis)",
    )

    p = sub.add_parser("train-smolvla", help="Fine-tune SmolVLA on a recorded dataset")
    p.add_argument("--dataset-repo-id", default=None)
    p.add_argument("--policy-path", default=None, help="Base model (default: lerobot/smolvla_base)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)

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

    p = sub.add_parser(
        "genesis-spike",
        help="Smoke-test Genesis World (load SO-101 URDF, step physics)",
    )
    p.add_argument("--headless", action="store_true")

    p = sub.add_parser(
        "twin",
        help="Digital twin: mirror USB follower joints in Genesis World",
    )
    p.add_argument("--follower-port", default=None)
    p.add_argument("--rate", type=float, default=None, help="Sync rate Hz (default: twin.rate_hz)")
    p.add_argument("--duration", type=float, default=None, help="Seconds (default: until Ctrl+C)")

    p = sub.add_parser("record-sim", help="Record LeRobot dataset in Genesis simulation")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument("--episode-time", type=float, default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--headless", action="store_true", help="Hide Genesis viewer (overrides config)")
    p.add_argument("--gui", action="store_true", help="Show Genesis viewer (overrides config)")
    p.add_argument("--random-actions", action="store_true", help="Random policy (wiring test)")
    p.add_argument(
        "--leader",
        action="store_true",
        help="Drive Genesis sim from USB leader arm (uses teleop.leader.port from config)",
    )
    p.add_argument("--leader-port", default=None, help="Leader USB port (overrides config)")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append episodes to an existing dataset (requires --repo-id)",
    )
    p.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Use fixed repo id without timestamp (fails if dataset exists)",
    )

    p = sub.add_parser(
        "log-joint-signal",
        help="Log encoder pulses vs LeRobot norm vs Genesis sim angles (90° travel gap)",
    )
    p.add_argument("--role", choices=["follower", "leader"], default="leader")
    p.add_argument("--port", default=None)
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Print expected pulses table only (no USB hardware)",
    )
    p.add_argument(
        "--no-live",
        action="store_true",
        help="Skip live hardware logging (same as --analyze-only)",
    )
    p.add_argument("--duration", type=float, default=45.0, help="Live log duration (seconds)")
    p.add_argument("--rate", type=float, default=10.0, help="Live sample rate (Hz)")
    p.add_argument("--output", default=None, help="JSONL output path for live samples")
    p.add_argument(
        "--target-degrees",
        type=float,
        default=90.0,
        help="Sim travel angle for expected-pulse column (default: 90)",
    )

    p = sub.add_parser(
        "calibrate-genesis",
        help="Mirror leader into Genesis; compare pulses, norm, and sim angles",
    )
    p.add_argument("--leader-port", default=None, help="Leader USB port (overrides config)")
    p.add_argument("--rate", type=float, default=15.0, help="Live update rate (Hz)")
    p.add_argument("--duration", type=float, default=None, help="Stop after N seconds (default: until Ctrl+C)")
    p.add_argument(
        "--capture-home",
        action="store_true",
        help="Read leader rest pose and print genesis.home_raw YAML (no live loop)",
    )
    p.add_argument(
        "--save-home",
        action="store_true",
        help="With --capture-home, patch config/default.yaml genesis.home_raw",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Config YAML path for --save-home (default: config/default.yaml)",
    )
    p.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip static pulse/norm analysis table at startup",
    )
    p.add_argument(
        "--measure",
        action="store_true",
        help="Interactive per-joint ~90° travel check before live mirror",
    )
    p.add_argument("--headless", action="store_true", help="Genesis without viewer window")

    p = sub.add_parser(
        "record-twin",
        help="Record hardware joints + Genesis-rendered camera to LeRobot dataset",
    )
    p.add_argument("--follower-port", default=None)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--episode-time", type=float, default=None)
    p.add_argument("--task", default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append episodes to an existing dataset (requires --repo-id)",
    )
    p.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Use fixed repo id without timestamp (fails if dataset exists)",
    )

    return parser


def _show_config() -> None:
    cfg = ProjectConfig.load()
    print(f"Robot:   {cfg.robot.type} backend={cfg.robot.backend} @ {cfg.robot.port or '(auto)'}")
    print(f"Leader:  {cfg.teleop.leader.type} @ {cfg.teleop.leader.port or '(auto)'}")
    print(f"Quest:   phosphobot on port {cfg.teleop.quest.port}")
    print(f"Dataset: {cfg.dataset.repo_id} → {cfg.resolve_dataset_root()}")
    print(f"Policy:  {cfg.policy.path} (device={cfg.policy.device or 'auto'})")
    print(f"Genesis: scene={cfg.genesis.scene} backend={cfg.genesis.backend}")
    print(f"Twin:    {cfg.twin.sync_mode} @ {cfg.twin.rate_hz} Hz")
    print(f"Cameras: {list(cfg.cameras.keys()) or '(none)'}")
    for name, cam in cfg.cameras.items():
        print(f"  {describe_camera(name, cam)}")
    for role in ("follower", "leader"):
        motor_map = cfg.motor_map(role)
        ids = ", ".join(f"{j}={motor_map.ids[j]}" for j in JOINT_NAMES)
        print(f"Motors ({role}): {ids}")


def main() -> None:
    args = _build_parser().parse_args()

    match args.command:
        case "find-port":
            find_port()
        case "list-cameras":
            list_usb_cameras()
        case "config-show":
            _show_config()
        case "teleop-quest-help":
            teleop_quest_instructions()
        case "camera-preview":
            preview_camera(
                name=args.name,
                index=args.index,
                url=args.url,
                width=args.width,
                height=args.height,
                fps=args.fps,
                output=args.output,
                seconds=args.seconds,
                show_window=not args.no_window,
            )
        case "camera-test":
            test_configured_cameras()
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
        case "task":
            match args.task_command:
                case "record":
                    record_task_motion(
                        task=args.task,
                        leader_port=args.leader_port,
                        follower_port=args.follower_port,
                        fps=args.fps,
                        duration_s=args.duration,
                        mirror_follower=not args.no_mirror,
                    )
                case "replay":
                    if not args.task and not args.task_slug:
                        print("Provide --task or --task-slug.", file=sys.stderr)
                        sys.exit(2)
                    replay_task_motion(
                        task=args.task,
                        task_slug=args.task_slug,
                        demo_id=args.demo,
                        follower_port=args.follower_port,
                        speed=args.speed,
                        loop=args.loop,
                        pause_s=args.pause,
                    )
                case "list":
                    list_task_motions()
                case "info":
                    if not args.task and not args.task_slug:
                        print("Provide --task or --task-slug.", file=sys.stderr)
                        sys.exit(2)
                    show_task_motion(
                        task=args.task,
                        task_slug=args.task_slug,
                        demo_id=args.demo,
                    )
        case "record-quest":
            record_quest(args.repo_id, args.push_to_hub)
        case "record-policy":
            record_policy(
                args.follower_port,
                args.policy_path,
                args.repo_id,
                args.num_episodes,
                single_task=args.task,
            )
        case "run-smolvla":
            run_smolvla(
                args.task,
                follower_port=args.follower_port,
                policy_path=args.policy_path,
                episode_time_s=args.episode_time,
                display_data=args.display_data,
                device=args.device,
                record=args.record,
                repo_id=args.repo_id,
                num_episodes=args.num_episodes,
                interactive=args.interactive,
                genesis=args.genesis if args.genesis else None,
                headless=args.headless if args.headless else None,
            )
        case "train-smolvla":
            train_smolvla(
                args.dataset_repo_id,
                policy_path=args.policy_path,
                output_dir=args.output_dir,
                steps=args.steps,
                batch_size=args.batch_size,
                device=args.device,
            )
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
        case "genesis-spike":
            run_genesis_spike(headless=args.headless)
        case "twin":
            run_twin(
                follower_port=args.follower_port,
                rate_hz=args.rate,
                duration_s=args.duration,
            )
        case "record-sim":
            if args.gui and args.headless:
                print("Use only one of --gui or --headless.", file=sys.stderr)
                raise SystemExit(2)
            if args.leader and args.random_actions:
                print("Use only one of --leader or --random-actions.", file=sys.stderr)
                raise SystemExit(2)
            sim_headless = None
            if args.gui:
                sim_headless = False
            elif args.headless:
                sim_headless = True
            leader_port = args.leader_port
            if args.leader:
                from .config import ProjectConfig

                cfg = ProjectConfig.load()
                leader_port = leader_port or cfg.teleop.leader.port
            record_sim(
                repo_id=args.repo_id,
                num_episodes=args.num_episodes,
                episode_time_s=args.episode_time,
                single_task=args.task,
                headless=sim_headless,
                random_actions=args.random_actions,
                leader_port=leader_port,
                resume=args.resume,
                timestamp=not args.no_timestamp,
            )
        case "log-joint-signal":
            run_joint_signal_log(
                role=args.role,
                port=args.port,
                analyze_only=args.analyze_only or args.no_live,
                live=not (args.analyze_only or args.no_live),
                duration_s=args.duration,
                rate_hz=args.rate,
                output=Path(args.output) if args.output else None,
                target_degrees=args.target_degrees,
            )
        case "calibrate-genesis":
            from .genesis.leader_calib import run_genesis_leader_calib

            run_genesis_leader_calib(
                leader_port=args.leader_port,
                rate_hz=args.rate,
                duration_s=args.duration,
                capture_home=args.capture_home,
                save_home=args.save_home,
                config_path=Path(args.config) if args.config else None,
                print_analysis=not args.no_analysis,
                measure_joints=args.measure,
                headless=args.headless,
            )
        case "record-twin":
            record_twin(
                follower_port=args.follower_port,
                repo_id=args.repo_id,
                num_episodes=args.num_episodes,
                episode_time_s=args.episode_time,
                single_task=args.task,
                resume=args.resume,
                timestamp=not args.no_timestamp,
            )
        case _:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
