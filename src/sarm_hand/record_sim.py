"""Record LeRobot datasets in Genesis simulation."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import JOINT_NAMES, ProjectConfig
from .genesis.deps import ensure_lerobot_genesis
from .genesis.driver import SO101SceneDriver
from .genesis.scene import SO101GenesisScene
from .genesis.shutdown import (
    check_shutdown,
    ensure_shutdown_handlers,
    exit_after_interrupt,
    install_shutdown_handlers,
    interruptible_sleep,
    shutdown_requested,
)
from .genesis.units import action_dict_to_vector, agent_pos_from_qpos


def _recording_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def resolve_recording_paths(
    *,
    base_repo: str,
    root: Path,
    repo_id: str | None,
    resume: bool,
    timestamp: bool,
) -> tuple[str, Path]:
    """Pick a dataset repo_id and directory for a recording session.

    By default each run uses a fresh timestamped folder so prior datasets are never
    overwritten. Pass ``resume=True`` with an explicit ``repo_id`` to append episodes.
    """
    base = repo_id or base_repo
    if resume:
        path = root / base
        if not path.exists():
            print(f"Cannot resume: dataset not found at {path}", file=sys.stderr)
            raise SystemExit(1)
        return base, path

    if timestamp:
        stamped = f"{base}-{_recording_stamp()}"
        return stamped, root / stamped

    path = root / base
    if path.exists():
        print(f"Dataset already exists: {path}", file=sys.stderr)
        print("Use default timestamped recording, or pass --resume to append.", file=sys.stderr)
        raise SystemExit(1)
    return base, path


def _genesis_dataset_features(
    cfg: ProjectConfig,
    *,
    state_dim: int,
    action_dim: int,
) -> dict:
    features: dict = {}
    for name, cam in cfg.genesis.cameras.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channels"],
        }
    features["observation.state"] = {"dtype": "float32", "shape": (state_dim,), "names": None}
    features["action"] = {"dtype": "float32", "shape": (action_dim,), "names": None}
    return features


class _GenesisDatasetSink:
    """EpisodeSink for Genesis recording; uses LeRobotDataset.resume() when appending."""

    def __init__(
        self,
        repo_id: str,
        *,
        fps: int,
        root: Path,
        cfg: ProjectConfig,
        state_dim: int,
        action_dim: int,
        task: str,
        robot_type: str,
        resume: bool,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._task = task
        if resume:
            self._dataset = LeRobotDataset.resume(repo_id, root=root)
            return

        self._dataset = LeRobotDataset.create(
            repo_id,
            fps,
            features=_genesis_dataset_features(
                cfg, state_dim=state_dim, action_dim=action_dim
            ),
            root=root,
            robot_type=robot_type,
            use_videos=True,
        )

    def add_frame(self, frame: dict) -> None:
        self._dataset.add_frame({**frame, "task": frame.get("task", self._task)})

    def save_episode(self) -> None:
        self._dataset.save_episode()

    def finalize(self) -> None:
        self._dataset.finalize()


def _hold_policy_action(obs: dict) -> np.ndarray:
    lo, hi = -100.0, 100.0
    pos = np.asarray(obs["agent_pos"], dtype=np.float32)
    return np.clip(2.0 * (pos - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def _record_genesis_episodes(
    env,
    driver: SO101SceneDriver,
    policy,
    sink: _GenesisDatasetSink,
    *,
    n_episodes: int,
) -> int:
    """Record episodes with all Genesis camera streams."""
    frames = 0
    interrupted = False
    try:
        for _ in range(n_episodes):
            check_shutdown()
            obs, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                check_shutdown()
                action = np.asarray(policy(obs), dtype=np.float32)
                images = driver._scene.render_all_rgb() if driver._scene else {}
                frame: dict = {
                    "observation.state": np.asarray(obs["agent_pos"], dtype=np.float32),
                    "action": action,
                }
                for name, rgb in images.items():
                    frame[f"observation.images.{name}"] = rgb
                sink.add_frame(frame)
                frames += 1
                obs, _, terminated, truncated, _ = env.step(action)
            sink.save_episode()
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped early.")
    finally:
        driver.close()
        try:
            sink.finalize()
        except KeyboardInterrupt:
            print("Skipping dataset finalize.")
        if interrupted or shutdown_requested():
            exit_after_interrupt()
    return frames


def _record_genesis_leader_episodes(
    *,
    cfg: ProjectConfig,
    leader_port: str,
    sink: _GenesisDatasetSink,
    n_episodes: int,
    episode_time_s: float,
    reset_time_s: float,
    fps: int,
    grasp_log_path: Path | None = None,
) -> int:
    """Record Genesis sim episodes driven by a USB leader arm."""
    from lerobot.teleoperators.so_leader import SO101Leader

    from .genesis.grasp_diag import GraspLogWriter
    from .genesis.leader import so101_leader_config, sync_leader_to_scene
    from .genesis.scene import SO101GenesisScene
    from .robot import _motor_write_retries, disable_arm_torque, ensure_port, require_all_motors

    port = ensure_port(leader_port, "Leader")
    require_all_motors("leader", port, context="record-sim")
    disable_arm_torque("leader", port)

    leader_cfg = so101_leader_config(cfg, port)
    leader = SO101Leader(leader_cfg)
    install_shutdown_handlers()
    cal_role = cfg.genesis.calibration_role or "leader"
    scene = SO101GenesisScene.create(cfg, calibration_role=cal_role, apply_home=False)
    ensure_shutdown_handlers()
    interval = 1.0 / fps
    frames = 0
    interrupted = False
    grasp_log: GraspLogWriter | None = None
    if grasp_log_path is not None:
        grasp_log = GraspLogWriter(grasp_log_path)
        print(f"  Grasp log: {grasp_log_path}")

    try:
        with _motor_write_retries():
            leader.connect()
        sync_leader_to_scene(scene, leader)
        for ep in range(n_episodes):
            check_shutdown()
            print(f"Episode {ep + 1}/{n_episodes} — move the leader arm (Ctrl+C to stop early)")
            scene.reset_props()
            sync_leader_to_scene(scene, leader)
            deadline = time.perf_counter() + episode_time_s
            while time.perf_counter() < deadline:
                check_shutdown()
                loop_start = time.perf_counter()
                action = sync_leader_to_scene(scene, leader)
                if grasp_log is not None and scene._last_grasp_diag is not None:
                    grasp_log.write(
                        scene._last_grasp_diag,
                        episode=ep + 1,
                        frame=frames,
                    )
                    grasp_log.maybe_print_transition(scene._last_grasp_diag)
                qpos = scene.robot.get_dofs_position(scene.dof_indices)
                state = agent_pos_from_qpos(qpos, cfg, calibration=scene.calibration)
                action_vec = np.asarray(action_dict_to_vector(action), dtype=np.float32)
                frame: dict = {"observation.state": state, "action": action_vec}
                for name, rgb in scene.render_all_rgb().items():
                    frame[f"observation.images.{name}"] = rgb
                sink.add_frame(frame)
                frames += 1
                elapsed = time.perf_counter() - loop_start
                interruptible_sleep(max(interval - elapsed, 0.0))
            sink.save_episode()
            if ep + 1 < n_episodes and reset_time_s > 0:
                print(f"  Reset ({reset_time_s:.0f}s) — reposition, then next episode...")
                interruptible_sleep(reset_time_s)
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped early.")
    finally:
        if grasp_log is not None:
            grasp_log.close()
        scene.close()
        try:
            sink.finalize()
        except KeyboardInterrupt:
            print("Skipping dataset finalize.")
        if leader.is_connected:
            leader.disconnect()
        if interrupted or shutdown_requested():
            exit_after_interrupt()
    return frames


def record_sim(
    *,
    repo_id: str | None = None,
    num_episodes: int | None = None,
    episode_time_s: float | None = None,
    single_task: str | None = None,
    headless: bool | None = None,
    random_actions: bool = False,
    leader_port: str | None = None,
    resume: bool = False,
    timestamp: bool = True,
    grasp_log: bool | None = None,
) -> None:
    """Record episodes in Genesis with all configured camera streams."""
    if random_actions and leader_port:
        print("Use only one of --random-actions or --leader-port.", file=sys.stderr)
        raise SystemExit(2)
    if resume and not repo_id:
        print("--resume requires --repo-id pointing at an existing dataset.", file=sys.stderr)
        raise SystemExit(2)

    ensure_lerobot_genesis()
    install_shutdown_handlers()

    cfg = ProjectConfig.load()
    cfg.genesis.headless = cfg.genesis.headless if headless is None else headless

    base_repo = f"{cfg.dataset.repo_id}-genesis"
    resolved_episodes = num_episodes if num_episodes is not None else cfg.dataset.num_episodes
    resolved_task = single_task or cfg.dataset.single_task
    resolved_time = episode_time_s if episode_time_s is not None else cfg.dataset.episode_time_s
    reset_time = cfg.dataset.reset_time_s
    fps = cfg.dataset.fps
    root = cfg.resolve_dataset_root()
    resolved_repo, dataset_path = resolve_recording_paths(
        base_repo=base_repo,
        root=root,
        repo_id=repo_id,
        resume=resume,
        timestamp=timestamp,
    )

    driver = SO101SceneDriver(cfg=cfg)
    sink = _GenesisDatasetSink(
        resolved_repo,
        fps=fps,
        root=dataset_path,
        cfg=cfg,
        state_dim=driver.state_dim,
        action_dim=driver.action_dim,
        task=resolved_task,
        robot_type="so101_follower",
        resume=resume,
    )

    mode = "resume" if resume else ("timestamped" if timestamp else "create")
    control = "leader USB" if leader_port else ("random" if random_actions else "hold pose")
    print("Genesis sim recording")
    print(f"  Dataset:   {resolved_repo}")
    print(f"  Root:      {dataset_path}")
    print(f"  Mode:      {mode}")
    print(f"  Control:   {control}")
    print(f"  Episodes:  {resolved_episodes}")
    print(f"  Steps/ep:  {max(1, int(resolved_time * fps))}")
    print(f"  Task:      {resolved_task!r}")
    print(f"  Cameras:   {', '.join(cfg.genesis.cameras)} (video)")
    print(f"  Headless:  {cfg.genesis.headless}")
    if leader_port:
        from .robot import ensure_port

        print(f"  Leader:    {ensure_port(leader_port, 'Leader')}")
    log_grasp = cfg.genesis.grasp_log if grasp_log is None else grasp_log
    grasp_log_path: Path | None = None
    if leader_port and log_grasp:
        grasp_log_path = dataset_path / "grasp_log.jsonl"
    if not cfg.genesis.headless:
        print("  Preview:   OpenCV windows sarm-hand: front | top | arm")
    print()

    if leader_port:
        frame_count = _record_genesis_leader_episodes(
            cfg=cfg,
            leader_port=leader_port,
            sink=sink,
            n_episodes=resolved_episodes,
            episode_time_s=resolved_time,
            reset_time_s=reset_time,
            fps=fps,
            grasp_log_path=grasp_log_path,
        )
    else:
        from lerobot_genesis import GenesisEnv

        max_episode_steps = max(1, int(resolved_time * fps))
        env = GenesisEnv(
            driver,
            task=resolved_task,
            max_episode_steps=max_episode_steps,
        )

        def policy(obs: dict) -> np.ndarray:
            if random_actions:
                return env.action_space.sample()
            return _hold_policy_action(obs)

        frame_count = _record_genesis_episodes(
            env,
            driver,
            policy,
            sink,
            n_episodes=resolved_episodes,
        )

    print(f"Recorded {resolved_episodes} episode(s), {frame_count} frames → {dataset_path}")


def record_twin(
    *,
    follower_port: str | None = None,
    repo_id: str | None = None,
    num_episodes: int = 1,
    episode_time_s: float | None = None,
    single_task: str | None = None,
    resume: bool = False,
    timestamp: bool = True,
) -> None:
    """Record hardware joint state + Genesis-rendered camera frames."""
    if resume and not repo_id:
        print("--resume requires --repo-id pointing at an existing dataset.", file=sys.stderr)
        raise SystemExit(2)

    ensure_lerobot_genesis()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, OBS_STR

    from .backends.genesis_twin import GenesisTwin
    from .genesis.deps import ensure_genesis
    from .robot import ensure_port, _motor_write_retries

    ensure_genesis()
    install_shutdown_handlers()
    cfg = ProjectConfig.load()
    port = ensure_port(follower_port or cfg.robot.port, "Follower")
    base_repo = f"{cfg.dataset.repo_id}-twin"
    resolved_task = single_task or cfg.dataset.single_task
    resolved_time = episode_time_s if episode_time_s is not None else cfg.dataset.episode_time_s
    fps = cfg.dataset.fps
    root = cfg.resolve_dataset_root()
    resolved_repo, dataset_path = resolve_recording_paths(
        base_repo=base_repo,
        root=root,
        repo_id=repo_id,
        resume=resume,
        timestamp=timestamp,
    )

    _, _, robot_observation_processor = make_default_processors()
    twin = GenesisTwin(port, cfg)

    if resume:
        dataset = LeRobotDataset.resume(resolved_repo, root=dataset_path)
        ds_features = dataset.meta.features
    else:
        ds_features = _twin_dataset_features(cfg)
        dataset = LeRobotDataset.create(
            repo_id=resolved_repo,
            fps=fps,
            root=dataset_path,
            robot_type="so101_follower",
            features=ds_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=0,
        )

    print("Twin recording (hardware joints + Genesis camera)")
    print(f"  Dataset: {resolved_repo}")
    print(f"  Root:    {dataset_path}")
    print(f"  Mode:    {'resume' if resume else ('timestamped' if timestamp else 'create')}")
    print(f"  Cameras: {', '.join(cfg.genesis.cameras)}")
    print(f"  Task:    {resolved_task!r}\n")

    interval = 1.0 / fps
    interrupted = False
    try:
        with _motor_write_retries():
            twin.start()
        ensure_shutdown_handlers()
        for ep in range(num_episodes):
            check_shutdown()
            print(f"Episode {ep + 1}/{num_episodes} — move the arm, Ctrl+C to stop early")
            deadline = time.perf_counter() + resolved_time
            while time.perf_counter() < deadline:
                check_shutdown()
                loop_start = time.perf_counter()
                obs = twin.sync_hardware_to_sim()
                obs_processed = robot_observation_processor(obs)
                frame = build_dataset_frame(ds_features, obs_processed, prefix=OBS_STR)
                frame["task"] = resolved_task
                for name, rgb in twin.render_all_cameras().items():
                    frame[f"observation.images.{name}"] = rgb
                action_frame = build_dataset_frame(ds_features, obs_processed, prefix=ACTION)
                dataset.add_frame({**frame, **action_frame})
                elapsed = time.perf_counter() - loop_start
                interruptible_sleep(max(interval - elapsed, 0.0))
            dataset.save_episode()
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped early.")
    finally:
        twin.stop()
        try:
            dataset.finalize()
        except KeyboardInterrupt:
            print("Skipping dataset finalize.")
        if interrupted or shutdown_requested():
            exit_after_interrupt()

    print(f"Done → {dataset_path}")


def _twin_dataset_features(cfg: ProjectConfig) -> dict:
    from lerobot.datasets.feature_utils import combine_feature_dicts
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.processor import make_default_processors

    obs_features = {f"{j}.pos": float for j in JOINT_NAMES}
    for name, cam in cfg.genesis.cameras.items():
        obs_features[name] = (cam.height, cam.width, 3)
    action_features = {f"{j}.pos": float for j in JOINT_NAMES}
    _, teleop_proc, rob_obs_proc = make_default_processors()
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            teleop_proc,
            create_initial_features(action=action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            rob_obs_proc,
            create_initial_features(observation=obs_features),
            use_videos=True,
        ),
    )
