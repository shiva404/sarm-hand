"""Dataset recording for S-ARM101 using LeRobot."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

from .calibration_bridge import (
    calibration_mismatch_report,
    remap_leader_action_to_follower,
    require_teleop_calibrations,
)
from .cameras import (
    build_robot_camera_configs,
    connect_follower_robot,
    install_all_camera_patches,
    prepare_opencv_platform,
)
from .config import ProjectConfig
from .data import (
    _count_dataset_artifacts,
    configure_local_lerobot_env,
    write_latest_session_pointer,
    write_session_manifest,
)
from .dataset_session import (
    build_robot_dataset_features,
    camera_feature_keys,
    create_recording_dataset,
    resolve_recording_paths,
)
from .keyboard_control import init_recording_keyboard_listener
from .policy import _smolvla_record_flags
from .recording_ui import print_episode_banner, print_session_ready
from .rerun_viz import leader_record_loop
from .robot import _motor_write_retries, disable_arm_torque, ensure_bus_calibration, ensure_port, require_all_motors


def install_record_feedback_patch() -> None:
    """Log each saved episode so it is obvious when disk writes happen."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if getattr(LeRobotDataset, "_sarm_save_feedback", False):
        return

    original_save = LeRobotDataset.save_episode

    def save_episode(self, *args, **kwargs):
        frame_count = 0
        if self.writer is not None and self.writer.episode_buffer is not None:
            frame_count = int(self.writer.episode_buffer.get("size", 0))
        if frame_count == 0:
            raise ValueError(
                "Episode has 0 frames — move the leader arm during recording, "
                "then press → Right arrow to save."
            )
        original_save(self, *args, **kwargs)
        episode_index = max(0, self.meta.total_episodes - 1)
        parquet_n, mp4_n = _count_dataset_artifacts(Path(self.root))
        print(
            f"\n✓ Saved episode {episode_index} ({frame_count} frames) "
            f"→ {self.root}\n"
            f"  parquet files: {parquet_n}  video files: {mp4_n}\n"
            f"  Inspect: uv run sarm-hand data-info --repo-id {self.repo_id}\n"
        )

    LeRobotDataset.save_episode = save_episode
    LeRobotDataset._sarm_save_feedback = True


def _print_recording_guide(
    *,
    session_repo_id: str,
    dataset_dir: Path,
    num_episodes: int,
    episode_time_s: float,
    reset_time_s: float,
    fps: int,
    task: str,
    camera_names: list[str],
) -> None:
    print(f"Recording {num_episodes} episodes → {session_repo_id}")
    print(f"Local dataset: {dataset_dir}")
    print(f"Task: {task!r}  @ {fps} fps (dataset)")
    print("  Follower control: teleop.control_fps in config/default.yaml (smoother than dataset fps)")
    if camera_names:
        print(f"Cameras (video): {', '.join(camera_names)}")
    print("  Layout: same as record-sim — data/*.parquet + videos/observation.images.*/")
    print(
        f"\nData is written to disk after each episode finishes "
        f"({episode_time_s:.0f}s max, or press → Right arrow to end early)."
    )
    print("Rerun preview is live only — it does not replace saved dataset files.")
    print("  Default: Rerun off during record-leader (use --rerun for joint preview).\n")
    print("Terminal controls (always work — click this window first):")
    print("  s              save episode now")
    print("  r              discard episode and re-record")
    print("  q              stop all recording")
    print("Optional (needs macOS Accessibility for Cursor/Terminal):")
    print("  →  Right arrow   save episode")
    print("  ←  Left arrow     re-record")
    print("  Esc              stop all")
    print("  Ctrl+C           stop and save partial episode (if any frames buffered)")
    print("  Or wait for the episode timer — saves automatically.\n")
    print(f"After each episode: {reset_time_s:.0f}s reset window (not saved).")
    print("When done, inspect: uv run sarm-hand data-info --latest")
    print("Train ACT:          uv run sarm-hand train-act  # front + wrist, uses this session")
    print("Train SmolVLA:      uv run sarm-hand train-smolvla\n")
    print("Move the leader arm to demonstrate each episode.\n")


def _warn_macos_keyboard_access() -> None:
    """Note that pynput 'not trusted' is OK — stdin s/r/q still works."""
    if platform.system() != "Darwin":
        return
    print(
        "\n(pynput 'not trusted' warning is normal in Cursor without Accessibility.\n"
        " Use s / r / q keys in this terminal instead of arrow keys.)\n",
        file=sys.stderr,
    )


def _verify_dataset_saved(dataset_dir: Path, session_repo_id: str) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        print(f"\nNo dataset metadata at {info_path}", file=sys.stderr)
        sys.exit(1)

    info = json.loads(info_path.read_text())
    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    parquet_n, mp4_n = _count_dataset_artifacts(dataset_dir)

    if episodes == 0 or frames == 0:
        print("\nNo episodes were saved to disk.", file=sys.stderr)
        print(
            "  Finish at least one episode (wait for the timer or press → Right arrow).",
            file=sys.stderr,
        )
        print(
            f"  Then run: uv run sarm-hand data-info --repo-id {session_repo_id}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDataset ready: {episodes} episode(s), {frames} frame(s)")
    print(f"  Path:    {dataset_dir}")
    print(f"  Parquet: {parquet_n} file(s) under data/")
    print(f"  Video:   {mp4_n} file(s) under videos/")
    print(f"  Train:   uv run sarm-hand train-act --dataset-repo-id {session_repo_id}")
    print(f"  Inspect: uv run sarm-hand data-info --repo-id {session_repo_id}")
    print(f"  Viz:     uv run sarm-hand viz-dataset --repo-id {session_repo_id} --episode 0")


def _save_episode_if_buffered(
    dataset,
    *,
    cfg: ProjectConfig,
    session_repo_id: str,
    dataset_dir: Path,
    label: str = "episode",
) -> bool:
    """Flush the in-memory episode buffer to disk; return True if anything was saved."""
    if not dataset.has_pending_frames():
        return False
    frame_count = 0
    if dataset.writer is not None and dataset.writer.episode_buffer is not None:
        frame_count = int(dataset.writer.episode_buffer.get("size", 0))
    if frame_count == 0:
        return False
    dataset.save_episode()
    write_latest_session_pointer(cfg, session_repo_id, dataset_dir)
    parquet_n, mp4_n = _count_dataset_artifacts(dataset_dir)
    print(
        f"\n✓ Saved {label} ({frame_count} frames) → {dataset_dir}\n"
        f"  parquet files: {parquet_n}  video files: {mp4_n}\n"
    )
    return True


def _run_hardware_record_session(
    cfg: ProjectConfig,
    *,
    session_repo_id: str,
    dataset_dir: Path,
    follower_port: str,
    leader_port: str,
    num_episodes: int,
    task: str,
    episode_time_s: float,
    reset_time_s: float,
    push_to_hub: bool,
    display_data: bool = True,
) -> None:
    """In-process leader recording — same LeRobotDataset writer as record-sim."""
    from lerobot.datasets.video_utils import VideoEncodingManager
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader import SO101Leader
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.utils.control_utils import is_headless
    from lerobot.utils.utils import init_logging, log_say
    from lerobot.utils.visualization_utils import init_rerun

    leader_cal, follower_cal = require_teleop_calibrations(cfg)
    mismatch = calibration_mismatch_report(leader_cal, follower_cal)
    if mismatch:
        print("Calibration mismatch (recording remaps via encoder counts):")
        for line in mismatch:
            print(f"  - {line}")
        print("  Fix permanently:  sarm-hand sync-calibration --from leader --write-motors\n")

    def _remap(action: dict[str, float]) -> dict[str, float]:
        return remap_leader_action_to_follower(
            action,
            leader_cal=leader_cal,
            follower_cal=follower_cal,
        )

    control_fps = max(cfg.teleop.control_fps, cfg.dataset.fps)

    install_all_camera_patches(cfg=cfg)
    install_record_feedback_patch()
    prepare_opencv_platform()

    require_all_motors("leader", leader_port, context="record-leader")
    require_all_motors("follower", follower_port, context="record-leader")
    disable_arm_torque("leader", leader_port)

    robot_cfg = SOFollowerRobotConfig(
        id=cfg.robot.id,
        port=follower_port,
        use_degrees=cfg.robot.use_degrees,
        max_relative_target=cfg.robot.max_relative_target,
        disable_torque_on_disconnect=cfg.robot.disable_torque_on_disconnect,
        cameras=build_robot_camera_configs(cfg) if cfg.cameras else {},
    )
    leader_cfg = SO101LeaderConfig(
        id=cfg.teleop.leader.id,
        port=leader_port,
        use_degrees=cfg.robot.use_degrees,
    )

    robot = SO101Follower(robot_cfg)
    teleop = SO101Leader(leader_cfg)
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    features = build_robot_dataset_features(
        robot,
        teleop_action_processor,
        robot_observation_processor,
        use_videos=cfg.dataset.video,
    )
    dataset = create_recording_dataset(
        cfg,
        session_repo_id,
        dataset_dir,
        features,
        num_cameras=len(cfg.cameras),
    )
    write_session_manifest(
        dataset_dir,
        session_repo_id,
        fps=cfg.dataset.fps,
        task=task,
    )

    init_logging()
    if display_data:
        init_rerun(session_name="recording")
        print("Rerun: joint preview only (cameras saved to dataset videos/, not streamed to Rerun).\n")

    listener = None
    interrupted = False
    recorded_episodes = 0
    try:
        with _motor_write_retries():
            if robot.cameras:
                connect_follower_robot(robot, calibrate=False)
            else:
                robot.bus.connect()
                ensure_bus_calibration(robot, "follower", cfg=cfg)
                robot.configure()
                from .robot import sync_follower_goals_to_present

                sync_follower_goals_to_present(robot)
            teleop.connect(calibrate=False)
            ensure_bus_calibration(teleop, "leader", cfg=cfg)

        listener, events, _stdin_thread = init_recording_keyboard_listener()
        _warn_macos_keyboard_access()
        print_session_ready(num_episodes=num_episodes, episode_time_s=episode_time_s)

        with VideoEncodingManager(dataset):
            while recorded_episodes < num_episodes and not events["stop_recording"]:
                print_episode_banner(
                    episode_index=dataset.num_episodes,
                    num_episodes=num_episodes,
                    task=task,
                    duration_s=episode_time_s,
                    phase="record",
                )
                log_say(f"Recording episode {dataset.num_episodes}", True)
                try:
                    frames_in_episode = leader_record_loop(
                        teleop=teleop,
                        robot=robot,
                        events=events,
                        control_fps=control_fps,
                        control_time_s=episode_time_s,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        dataset=dataset,
                        dataset_fps=cfg.dataset.fps,
                        single_task=task,
                        remap_action=_remap,
                        action_smoothing=cfg.teleop.action_smoothing,
                        display_data=display_data,
                        camera_names=list(cfg.cameras) if cfg.cameras else None,
                        show_countdown=True,
                        phase="record",
                    )
                except KeyboardInterrupt:
                    interrupted = True
                    events["stop_recording"] = True
                    print("\nCtrl+C — stopping recording.", file=sys.stderr)
                    if _save_episode_if_buffered(
                        dataset,
                        cfg=cfg,
                        session_repo_id=session_repo_id,
                        dataset_dir=dataset_dir,
                        label="partial episode",
                    ):
                        recorded_episodes += 1
                    break

                if frames_in_episode == 0 and not events.get("rerecord_episode"):
                    print(
                        "\nNo dataset frames captured this episode "
                        f"(expected ~{int(episode_time_s * cfg.dataset.fps)} at {cfg.dataset.fps} fps).\n"
                        "  Move the leader arm during recording; press → Right arrow to save early.\n",
                        file=sys.stderr,
                    )

                if not events["stop_recording"] and (
                    recorded_episodes < num_episodes - 1 or events["rerecord_episode"]
                ):
                    print_episode_banner(
                        episode_index=dataset.num_episodes,
                        num_episodes=num_episodes,
                        task=task,
                        duration_s=reset_time_s,
                        phase="reset",
                    )
                    log_say("Reset the environment", True)
                    try:
                        leader_record_loop(
                            teleop=teleop,
                            robot=robot,
                            events=events,
                            control_fps=control_fps,
                            control_time_s=reset_time_s,
                            teleop_action_processor=teleop_action_processor,
                            robot_action_processor=robot_action_processor,
                            robot_observation_processor=robot_observation_processor,
                            remap_action=_remap,
                            action_smoothing=cfg.teleop.action_smoothing,
                            display_data=display_data,
                            camera_names=list(cfg.cameras) if cfg.cameras else None,
                            show_countdown=True,
                            phase="reset",
                        )
                    except KeyboardInterrupt:
                        interrupted = True
                        events["stop_recording"] = True
                        print("\nCtrl+C — stopping recording.", file=sys.stderr)
                        break

                if events["rerecord_episode"]:
                    log_say("Re-record episode", True)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                if _save_episode_if_buffered(
                    dataset,
                    cfg=cfg,
                    session_repo_id=session_repo_id,
                    dataset_dir=dataset_dir,
                ):
                    recorded_episodes += 1
    except KeyboardInterrupt:
        interrupted = True
        print("\nCtrl+C — stopping recording.", file=sys.stderr)
        _save_episode_if_buffered(
            dataset,
            cfg=cfg,
            session_repo_id=session_repo_id,
            dataset_dir=dataset_dir,
            label="partial episode",
        )
    finally:
        log_say("Stop recording", True, blocking=True)
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if listener is not None and not is_headless():
            listener.stop()
        if push_to_hub:
            dataset.push_to_hub()

    if interrupted:
        info_path = dataset_dir / "meta" / "info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            episodes = int(info.get("total_episodes", 0))
            frames = int(info.get("total_frames", 0))
            if episodes > 0 and frames > 0:
                parquet_n, mp4_n = _count_dataset_artifacts(dataset_dir)
                print(
                    f"\nRecording stopped (Ctrl+C). Saved data at {dataset_dir}\n"
                    f"  {episodes} episode(s), {frames} frame(s), "
                    f"{parquet_n} parquet, {mp4_n} video(s)\n"
                    f"  Inspect: uv run sarm-hand data-info --repo-id {session_repo_id}\n"
                )
                return
        print(
            "\nRecording stopped (Ctrl+C) with no saved frames.\n"
            "  Move the leader during an episode, then press → Right arrow or wait before stopping.\n",
            file=sys.stderr,
        )
        raise SystemExit(130)

    _verify_dataset_saved(dataset_dir, session_repo_id)


def _lerobot_dataset_flags(cfg: ProjectConfig) -> list[str]:
    """CLI flags for lerobot-record subprocess (record-policy)."""
    ds = cfg.dataset
    flags = [
        f"--dataset.video={'true' if ds.video else 'false'}",
        "--dataset.streaming_encoding=false",
        f"--dataset.vcodec={ds.vcodec}",
        f"--dataset.num_image_writer_threads_per_camera={ds.num_image_writer_threads_per_camera}",
        f"--dataset.video_encoding_batch_size={ds.video_encoding_batch_size}",
        "--dataset.private=false",
    ]
    if ds.encoder_threads is not None:
        flags.append(f"--dataset.encoder_threads={ds.encoder_threads}")
    return flags


def _run_lerobot_record(cmd: list[str], *, dataset_dir: Path, session_repo_id: str) -> None:
    """Run lerobot-record in-process (policy eval recording)."""
    install_all_camera_patches(cfg=ProjectConfig.load())
    install_record_feedback_patch()
    sys.argv = list(cmd)
    from lerobot.scripts.lerobot_record import main

    main()
    _verify_dataset_saved(dataset_dir, session_repo_id)


def record_leader(
    follower_port: str | None = None,
    leader_port: str | None = None,
    repo_id: str | None = None,
    num_episodes: int | None = None,
    single_task: str | None = None,
    push_to_hub: bool | None = None,
    episode_time_s: float | None = None,
    reset_time_s: float | None = None,
    display_data: bool | None = None,
) -> None:
    """Record demonstrations with leader-follower teleoperation."""
    cfg = ProjectConfig.load()
    configure_local_lerobot_env(cfg)
    follower_port = ensure_port(follower_port or cfg.robot.port, "Follower")
    leader_port = ensure_port(leader_port or cfg.teleop.leader.port, "Leader")

    base_repo = repo_id or cfg.dataset.repo_id
    resolved_episodes = num_episodes if num_episodes is not None else cfg.dataset.num_episodes
    resolved_task = single_task or cfg.dataset.single_task
    resolved_push = push_to_hub if push_to_hub is not None else cfg.dataset.push_to_hub
    resolved_episode_time = (
        episode_time_s if episode_time_s is not None else cfg.dataset.episode_time_s
    )
    resolved_reset_time = reset_time_s if reset_time_s is not None else cfg.dataset.reset_time_s
    resolved_display = (
        display_data if display_data is not None else cfg.dataset.display_rerun
    )

    session_repo_id, dataset_dir = resolve_recording_paths(
        base_repo=base_repo,
        root=cfg.resolve_dataset_root(),
        repo_id=repo_id,
        resume=False,
        timestamp=True,
    )
    write_latest_session_pointer(cfg, session_repo_id, dataset_dir)

    cam_keys = camera_feature_keys(cfg)
    _print_recording_guide(
        session_repo_id=session_repo_id,
        dataset_dir=dataset_dir,
        num_episodes=resolved_episodes,
        episode_time_s=resolved_episode_time,
        reset_time_s=resolved_reset_time,
        fps=cfg.dataset.fps,
        task=resolved_task,
        camera_names=cam_keys,
    )
    if resolved_push:
        print("Hugging Face: will upload when recording finishes (--push-to-hub)")
    else:
        print("Hugging Face: disabled (local only)")
        print("  Upload later: uv run sarm-hand data-push --repo-id", session_repo_id)
        print()

    _run_hardware_record_session(
        cfg,
        session_repo_id=session_repo_id,
        dataset_dir=dataset_dir,
        follower_port=follower_port,
        leader_port=leader_port,
        num_episodes=resolved_episodes,
        task=resolved_task,
        episode_time_s=resolved_episode_time,
        reset_time_s=resolved_reset_time,
        push_to_hub=resolved_push,
        display_data=resolved_display,
    )


def record_quest(
    repo_id: str | None = None,
    push_to_hub: bool | None = None,
) -> None:
    """Guide user to record via Quest 2 phospho app (records through phosphobot)."""
    cfg = ProjectConfig.load()
    resolved_repo_id = repo_id or cfg.dataset.repo_id
    resolved_push = push_to_hub if push_to_hub is not None else cfg.dataset.push_to_hub

    print(
        f"""
Quest 2 Data Collection
=======================

Recording is done through the phospho Quest app while phosphobot is running.

Steps:
  1. Start the server:  sarm-hand teleop quest
  2. Connect Quest 2 to phosphobot
  3. Press B on the controller to start/stop recording each episode
  4. Press Y to discard a bad episode

Target dataset repo_id: {resolved_repo_id}
Push to Hugging Face Hub: {resolved_push}

After recording, inspect data with:
  sarm-hand data-info --repo-id {resolved_repo_id}
"""
    )


def record_policy(
    follower_port: str | None = None,
    policy_path: str = "",
    repo_id: str | None = None,
    num_episodes: int = 10,
    single_task: str | None = None,
) -> None:
    """Record evaluation rollouts using a trained policy (no teleop device)."""
    if not policy_path:
        print("Error: --policy-path is required for policy recording.", file=sys.stderr)
        sys.exit(1)

    cfg = ProjectConfig.load()
    configure_local_lerobot_env(cfg)
    follower_port = ensure_port(follower_port or cfg.robot.port, "Follower")
    base_repo = repo_id or f"{cfg.dataset.repo_id}-eval"
    resolved_task = single_task or cfg.dataset.single_task
    session_repo_id, dataset_dir = resolve_recording_paths(
        base_repo=base_repo,
        root=cfg.resolve_dataset_root(),
        repo_id=repo_id,
        resume=False,
        timestamp=True,
    )
    write_latest_session_pointer(cfg, session_repo_id, dataset_dir)

    cmd = [
        "lerobot-record",
        f"--robot.type={cfg.robot.type}",
        f"--robot.port={follower_port}",
        f"--robot.id={cfg.robot.id}",
        f"--dataset.repo_id={session_repo_id}",
        f"--dataset.root={dataset_dir.resolve()}",
        f"--dataset.fps={cfg.dataset.fps}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={resolved_task}",
        f"--policy.path={policy_path}",
        "--dataset.push_to_hub=false",
        "--display_data=true",
        *_lerobot_dataset_flags(cfg),
    ]

    cameras = cfg.cameras_lerobot_dict()
    if cameras:
        cmd.append(f"--robot.cameras={cameras!r}")
    cmd.extend(_smolvla_record_flags(cfg))

    _run_lerobot_record(cmd, dataset_dir=dataset_dir, session_repo_id=session_repo_id)
