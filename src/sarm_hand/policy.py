"""SmolVLA and other LeRobot policy inference / training for S-ARM101."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .cameras import build_robot_camera_configs, connect_follower_robot, install_all_camera_patches
from .config import ProjectConfig
from .robot import _motor_write_retries, ensure_port, require_all_motors


def _policy_fps(cfg: ProjectConfig) -> int:
    return cfg.policy.control_fps


def resolve_camera_map(cfg: ProjectConfig, *, genesis: bool = False) -> dict[str, str]:
    """Camera name → policy image key. Default: identity from cameras: / genesis.cameras:."""
    if cfg.policy.camera_map:
        return dict(cfg.policy.camera_map)
    names = cfg.genesis.cameras if genesis else cfg.cameras
    return {name: name for name in names}


def build_policy_rename_map(cfg: ProjectConfig, *, genesis: bool = False) -> dict[str, str]:
    """Map observation.images.<robot_cam> → observation.images.<policy_cam>."""
    mapping = resolve_camera_map(cfg, genesis=genesis)
    return {
        f"observation.images.{src}": f"observation.images.{dst}"
        for src, dst in mapping.items()
    }


def apply_smolvla_policy_overrides(policy_cfg, cfg: ProjectConfig, *, genesis: bool = False) -> None:
    """Tune SmolVLA config for the number of physical cameras available."""
    empty = cfg.policy.empty_cameras
    cam_count = len(cfg.genesis.cameras) if genesis else len(cfg.cameras)
    if empty is None and cam_count == 1:
        empty = 2
    if empty is not None:
        policy_cfg.empty_cameras = empty


def _processor_uses_buffer_stats(policy_path: str) -> bool:
    """True when pretrained processor safetensors use multi-robot buffer stat keys."""
    try:
        from pathlib import Path

        from safetensors.torch import load_file

        local = Path(policy_path)
        if local.is_dir():
            state_path = local / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
            if not state_path.is_file():
                return False
        else:
            from huggingface_hub import hf_hub_download

            state_path = hf_hub_download(
                policy_path,
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            )
        return any(".buffer." in key for key in load_file(state_path).keys())
    except Exception:
        return False


def load_smolvla_buffer_stats(policy_path: str, buffer_key: str) -> dict[str, dict]:
    """Remap smolvla_base buffer stats (e.g. so100.buffer.action) to standard feature keys."""
    from pathlib import Path

    from safetensors.torch import load_file

    from lerobot.utils.constants import ACTION, OBS_STR

    state_files = [
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    prefix = f"{buffer_key}."
    raw: dict[str, dict] = {}

    for filename in state_files:
        local = Path(policy_path)
        if local.is_dir():
            path = local / filename
            if not path.is_file():
                continue
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(policy_path, filename)

        for key, value in load_file(path).items():
            if not key.startswith(prefix):
                continue
            feature, stat = key[len(prefix) :].rsplit(".", 1)
            raw.setdefault(feature, {})[stat] = value

    mapped: dict[str, dict] = {}
    if "action" in raw:
        mapped[ACTION] = raw["action"]
    state_key = f"{OBS_STR}.state"
    if "observation.state" in raw:
        mapped[state_key] = raw["observation.state"]
    return mapped


def resolve_policy_normalization_stats(
    cfg: ProjectConfig,
    policy_path: str,
    *,
    rename_map: dict[str, str],
) -> dict | None:
    """Pick normalization stats for inference (dataset > smolvla buffer remap > none)."""
    import json

    from lerobot.processor.rename_processor import rename_stats

    train_id = cfg.policy.train_dataset
    if not train_id:
        from .data import read_latest_session_pointer

        session = read_latest_session_pointer(cfg)
        train_id = session[0] if session else cfg.dataset.repo_id
    stats_path = cfg.resolve_dataset_path(train_id) / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        if stats:
            return rename_stats(stats, rename_map)

    if _processor_uses_buffer_stats(policy_path):
        return load_smolvla_buffer_stats(policy_path, cfg.policy.stats_buffer)

    return None


def _log_action_step(step: int, obs: dict, sent: dict) -> None:
    """Print goal-vs-present joint delta so silent no-ops are visible."""
    present = {k: v for k, v in obs.items() if k.endswith(".pos")}
    if not present:
        return
    deltas = {
        joint: abs(float(sent.get(joint, present[joint])) - float(present[joint]))
        for joint in present
    }
    max_joint = max(deltas, key=deltas.get)
    print(
        f"  step {step}: max joint delta {deltas[max_joint]:.1f} "
        f"({max_joint}, goal={float(sent.get(max_joint, 0)):.1f})"
    )


def _smolvla_record_flags(cfg: ProjectConfig, *, genesis: bool = False) -> list[str]:
    """Extra lerobot-record flags for SmolVLA camera mapping."""
    flags: list[str] = []
    rename_map = build_policy_rename_map(cfg, genesis=genesis)
    if rename_map:
        flags.append(f"--dataset.rename_map={rename_map!r}")
    empty = cfg.policy.empty_cameras
    cam_count = len(cfg.genesis.cameras) if genesis else len(cfg.cameras)
    if empty is None and cam_count == 1:
        empty = 2
    if empty is not None:
        flags.append(f"--policy.empty_cameras={empty}")
    return flags


def _use_genesis_policy(cfg: ProjectConfig, genesis: bool | None) -> bool:
    if genesis is not None:
        return genesis
    return cfg.robot.backend.lower() in ("genesis", "sim")


def _require_policy_cameras(cfg: ProjectConfig, *, genesis: bool) -> None:
    if genesis:
        if not cfg.genesis.cameras:
            print(
                "Genesis SmolVLA needs at least one camera under genesis.cameras in config/default.yaml.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return
    _require_cameras(cfg)


def ensure_smolvla(*, genesis: bool = False) -> None:
    """Verify SmolVLA dependencies are installed."""
    missing: list[str] = []
    for module in ("transformers", "torch"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        extras = "sim-policy" if genesis else "smolvla"
        alt = "genesis --extra smolvla" if genesis else "smolvla"
        print(
            f"SmolVLA is missing: {', '.join(missing)}\n"
            f"Install with:  uv sync --extra {extras}\n"
            f"  (or: uv sync --extra {alt})\n"
            "\nuv extras are per-sync — passing only --extra genesis does not install smolvla.",
            file=sys.stderr,
        )
        sys.exit(1)


def resolve_device(device: str | None) -> str:
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _move_tree_to_device(value, device):
    """Recursively move tensors in nested dicts/lists to *device*."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_tree_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_tree_to_device(v, device) for v in value)
    return value


def _predict_action_on_device(
    observation: dict,
    *,
    policy,
    device: torch.device,
    preprocessor,
    postprocessor,
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
):
    """Like ``lerobot.utils.control_utils.predict_action`` but fixes CPU/MPS drift after normalizer."""
    from contextlib import nullcontext
    from copy import copy

    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)
        observation = _move_tree_to_device(observation, device)
        action = policy.select_action(observation)
        action = postprocessor(action)
    return action


def _require_cameras(cfg: ProjectConfig) -> None:
    if not cfg.cameras:
        print(
            "SmolVLA needs at least one camera in config/default.yaml.\n"
            "Add a front camera (USB or HTTP stream) under cameras:.",
            file=sys.stderr,
        )
        sys.exit(1)


def _build_dataset_features(robot, teleop_action_processor, robot_observation_processor):
    from lerobot.datasets.feature_utils import combine_feature_dicts
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )

    # LeRobot's aggregate_pipeline_dataset_features skips image keys when use_videos=False.
    # Match lerobot-record (video=True) so observation.images.* appear in features.
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )


def _policy_episode(
    *,
    robot,
    policy,
    preprocessor,
    postprocessor,
    robot_action_processor,
    robot_observation_processor,
    features: dict,
    task: str,
    fps: int,
    episode_time_s: float,
    display_data: bool,
) -> None:
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import log_rerun_data

    policy_device = get_safe_torch_device(policy.config.device)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    deadline = time.perf_counter() + episode_time_s
    interval = 1.0 / fps

    print(f"Running task: {task!r}")
    print(f"Duration: {episode_time_s:.0f}s @ {fps} fps (Ctrl+C to stop early)\n")

    step = 0
    try:
        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)

            action_tensor = _predict_action_on_device(
                observation_frame,
                policy=policy,
                device=policy_device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
            robot_action = make_robot_action(action_tensor, features)
            action_to_send = robot_action_processor((robot_action, obs))
            sent = robot.send_action(action_to_send)

            if step == 0 or step % max(fps * 5, 1) == 0:
                _log_action_step(step, obs, sent)

            if display_data:
                log_rerun_data(observation=obs_processed, action=robot_action)

            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(interval - elapsed, 0.0))
            step += 1
    except KeyboardInterrupt:
        print("\nStopped early.")


def _load_smolvla_stack(
    cfg: ProjectConfig,
    *,
    policy_path: str,
    device: str,
    robot,
    genesis: bool,
):
    """Build policy, processors, and dataset features for one inference episode."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor import make_default_processors

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )
    features = _build_dataset_features(
        robot, teleop_action_processor, robot_observation_processor
    )

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    apply_smolvla_policy_overrides(policy_cfg, cfg, genesis=genesis)
    rename_map = build_policy_rename_map(cfg, genesis=genesis)

    norm_stats = resolve_policy_normalization_stats(cfg, policy_path, rename_map=rename_map)

    dataset_root = Path(tempfile.mkdtemp(prefix="sarm-hand-smolvla-")) / "local/sarm101-inference-scratch"
    dataset = LeRobotDataset.create(
        repo_id="local/sarm101-inference-scratch",
        fps=_policy_fps(cfg),
        root=dataset_root,
        robot_type=robot.name,
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    policy = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=rename_map)

    preprocessor_overrides: dict = {
        "device_processor": {"device": policy_cfg.device},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    postprocessor_overrides: dict = {}
    if norm_stats:
        preprocessor_overrides["normalizer_processor"] = {
            "stats": norm_stats,
            "device": device,
        }
        postprocessor_overrides["unnormalizer_processor"] = {"stats": norm_stats}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    return {
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "robot_action_processor": robot_action_processor,
        "robot_observation_processor": robot_observation_processor,
        "features": features,
        "rename_map": rename_map,
        "policy_cfg": policy_cfg,
        "norm_stats": norm_stats,
    }


def _print_smolvla_header(
    *,
    policy_path: str,
    device: str,
    task: str,
    genesis: bool,
    cfg: ProjectConfig,
    rename_map: dict[str, str],
    policy_cfg,
    norm_stats,
    follower_port: str | None = None,
) -> None:
    print("SmolVLA inference")
    print(f"  Policy:  {policy_path}")
    print(f"  Device:  {device}")
    if genesis:
        print(f"  Backend: Genesis ({cfg.genesis.scene})")
        print(f"  Cameras: {list(cfg.genesis.cameras.keys())} (rendered)")
        print(f"  Headless: {cfg.genesis.headless}")
    else:
        print(f"  Robot:   {follower_port}")
        print(f"  Cameras: {list(cfg.cameras.keys())}")
    if rename_map:
        print(f"  Camera map: {rename_map}")
    if getattr(policy_cfg, "empty_cameras", 0):
        print(f"  Empty camera slots: {policy_cfg.empty_cameras} (zero-padded)")
    if policy_path == "lerobot/smolvla_base":
        print(
            "\n  Note: smolvla_base is pretrained on community data. "
            "For reliable SO-101 tasks, fine-tune on your demos:\n"
            "    uv sync --extra smolvla\n"
            "    sarm-hand train-smolvla --dataset-repo-id local/your-dataset\n"
        )
    if norm_stats:
        print(f"  Norm stats: {cfg.policy.stats_buffer} buffer remap")
    print(f"  Task:    {task!r}\n")


def run_smolvla(
    task: str | None = None,
    *,
    follower_port: str | None = None,
    policy_path: str | None = None,
    episode_time_s: float | None = None,
    display_data: bool = True,
    device: str | None = None,
    record: bool = False,
    repo_id: str | None = None,
    num_episodes: int = 1,
    interactive: bool = False,
    genesis: bool | None = None,
    headless: bool | None = None,
) -> None:
    """Run SmolVLA with a natural-language task on hardware or Genesis sim."""
    cfg = ProjectConfig.load()
    use_genesis = _use_genesis_policy(cfg, genesis)
    ensure_smolvla(genesis=use_genesis)
    if use_genesis:
        from .genesis.deps import ensure_genesis

        ensure_genesis()
    _require_policy_cameras(cfg, genesis=use_genesis)

    if record and use_genesis:
        print(
            "Genesis policy recording is not supported yet. "
            "Use: sarm-hand run-smolvla --genesis --task \"...\"",
            file=sys.stderr,
        )
        raise SystemExit(1)

    resolved_follower_port: str | None = None
    if use_genesis:
        if headless is not None:
            cfg.genesis.headless = headless
    else:
        resolved_follower_port = ensure_port(follower_port or cfg.robot.port, "Follower")

    resolved_policy = policy_path or cfg.policy.path
    resolved_device = resolve_device(device or cfg.policy.device)
    resolved_episode_s = (
        episode_time_s if episode_time_s is not None else cfg.policy.episode_time_s
    )

    if record:
        _run_smolvla_record(
            cfg,
            follower_port=resolved_follower_port,
            policy_path=resolved_policy,
            task=task or cfg.dataset.single_task,
            episode_time_s=resolved_episode_s,
            repo_id=repo_id,
            num_episodes=num_episodes,
            display_data=display_data,
        )
        return

    run_one = (
        _run_smolvla_genesis_inference if use_genesis else _run_smolvla_inference
    )

    if interactive:
        label = "Genesis sim" if use_genesis else "follower arm"
        print(f"SmolVLA interactive mode ({label}) — enter a task query per episode.")
        print("Press Enter on an empty line to quit.\n")
        while True:
            query = input("Task: ").strip()
            if not query:
                break
            if use_genesis:
                run_one(
                    cfg,
                    task=query,
                    policy_path=resolved_policy,
                    episode_time_s=resolved_episode_s,
                    display_data=display_data,
                    device=resolved_device,
                )
            else:
                run_one(
                    cfg,
                    task=query,
                    follower_port=resolved_follower_port,
                    policy_path=resolved_policy,
                    episode_time_s=resolved_episode_s,
                    display_data=display_data,
                    device=resolved_device,
                )
            print()
        return

    if not task:
        print("Error: provide --task or use --interactive.", file=sys.stderr)
        raise SystemExit(1)

    if use_genesis:
        run_one(
            cfg,
            task=task,
            policy_path=resolved_policy,
            episode_time_s=resolved_episode_s,
            display_data=display_data,
            device=resolved_device,
        )
    else:
        run_one(
            cfg,
            task=task,
            follower_port=resolved_follower_port,
            policy_path=resolved_policy,
            episode_time_s=resolved_episode_s,
            display_data=display_data,
            device=resolved_device,
        )


def _run_smolvla_inference(
    cfg: ProjectConfig,
    *,
    task: str,
    follower_port: str,
    policy_path: str,
    episode_time_s: float,
    display_data: bool,
    device: str,
) -> None:
    install_all_camera_patches()
    import rerun as rr
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import init_rerun

    init_logging()
    require_all_motors("follower", follower_port, context="run SmolVLA")

    robot_cfg = SOFollowerRobotConfig(
        id=cfg.robot.id,
        port=follower_port,
        use_degrees=cfg.robot.use_degrees,
        max_relative_target=cfg.robot.max_relative_target,
        disable_torque_on_disconnect=cfg.robot.disable_torque_on_disconnect,
        cameras=build_robot_camera_configs(cfg),
    )
    robot = SO101Follower(robot_cfg)
    stack = _load_smolvla_stack(
        cfg, policy_path=policy_path, device=device, robot=robot, genesis=False
    )
    _print_smolvla_header(
        policy_path=policy_path,
        device=device,
        task=task,
        genesis=False,
        cfg=cfg,
        rename_map=stack["rename_map"],
        policy_cfg=stack["policy_cfg"],
        norm_stats=stack["norm_stats"],
        follower_port=follower_port,
    )

    if display_data:
        init_rerun(session_name="smolvla")

    try:
        with _motor_write_retries():
            if cfg.cameras:
                connect_follower_robot(robot, calibrate=False)
            else:
                robot.connect()
        _policy_episode(
            robot=robot,
            policy=stack["policy"],
            preprocessor=stack["preprocessor"],
            postprocessor=stack["postprocessor"],
            robot_action_processor=stack["robot_action_processor"],
            robot_observation_processor=stack["robot_observation_processor"],
            features=stack["features"],
            task=task,
            fps=_policy_fps(cfg),
            episode_time_s=episode_time_s,
            display_data=display_data,
        )
    except ConnectionError as exc:
        print(
            "\nLost contact with a servo while connecting the follower.\n"
            "This is usually a loose daisy-chain cable or insufficient 12V power.\n"
            "\n  sarm-hand test-motors --role follower\n"
            "\nReseat the 3-pin cable at the joint mentioned in the error, then retry.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    finally:
        if display_data:
            rr.rerun_shutdown()
        if robot.is_connected:
            robot.disconnect()

    print("Done.")


def _run_smolvla_genesis_inference(
    cfg: ProjectConfig,
    *,
    task: str,
    policy_path: str,
    episode_time_s: float,
    display_data: bool,
    device: str,
) -> None:
    import rerun as rr
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import init_rerun

    from .backends.genesis_sim import GenesisSimRobot
    from .genesis.deps import ensure_genesis
    from .genesis.shutdown import (
        ensure_shutdown_handlers,
        exit_after_interrupt,
        install_shutdown_handlers,
        shutdown_requested,
    )

    ensure_genesis()
    install_shutdown_handlers()
    init_logging()

    robot = GenesisSimRobot(cfg)
    stack = _load_smolvla_stack(
        cfg, policy_path=policy_path, device=device, robot=robot, genesis=True
    )
    _print_smolvla_header(
        policy_path=policy_path,
        device=device,
        task=task,
        genesis=True,
        cfg=cfg,
        rename_map=stack["rename_map"],
        policy_cfg=stack["policy_cfg"],
        norm_stats=stack["norm_stats"],
    )

    if display_data:
        init_rerun(session_name="smolvla-genesis")

    interrupted = False
    try:
        ensure_shutdown_handlers()
        robot.connect()
        _policy_episode(
            robot=robot,
            policy=stack["policy"],
            preprocessor=stack["preprocessor"],
            postprocessor=stack["postprocessor"],
            robot_action_processor=stack["robot_action_processor"],
            robot_observation_processor=stack["robot_observation_processor"],
            features=stack["features"],
            task=task,
            fps=_policy_fps(cfg),
            episode_time_s=episode_time_s,
            display_data=display_data,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped.")
    finally:
        if display_data:
            rr.rerun_shutdown()
        if robot.is_connected:
            robot.disconnect()
        if interrupted or shutdown_requested():
            exit_after_interrupt()

    print("Done.")


def _run_smolvla_record(
    cfg: ProjectConfig,
    *,
    follower_port: str,
    policy_path: str,
    task: str,
    episode_time_s: float,
    repo_id: str | None,
    num_episodes: int,
    display_data: bool,
) -> None:
    resolved_repo_id = repo_id or f"{cfg.dataset.repo_id}-smolvla-eval"
    from .data import configure_local_lerobot_env, write_latest_session_pointer, write_session_manifest
    from .dataset_session import resolve_recording_paths
    from .record import _lerobot_dataset_flags, _run_lerobot_record

    configure_local_lerobot_env(cfg)
    session_repo_id, dataset_dir = resolve_recording_paths(
        base_repo=resolved_repo_id,
        root=cfg.resolve_dataset_root(),
        repo_id=repo_id,
        resume=False,
        timestamp=True,
    )
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    write_latest_session_pointer(cfg, session_repo_id, dataset_dir)

    cmd = [
        "lerobot-record",
        f"--robot.type={cfg.robot.type}",
        f"--robot.port={follower_port}",
        f"--robot.id={cfg.robot.id}",
        f"--dataset.repo_id={session_repo_id}",
        f"--dataset.root={dataset_dir.resolve()}",
        f"--dataset.fps={_policy_fps(cfg)}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={task}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={cfg.dataset.reset_time_s}",
        "--dataset.push_to_hub=false",
        f"--policy.path={policy_path}",
        f"--display_data={'true' if display_data else 'false'}",
        *_lerobot_dataset_flags(cfg),
    ]

    cameras = cfg.cameras_lerobot_dict()
    if cameras:
        cmd.append(f"--robot.cameras={cameras!r}")
    cmd.extend(_smolvla_record_flags(cfg))

    print(f"Recording {num_episodes} SmolVLA episode(s) → {session_repo_id}")
    print(f"Task: {task!r}\n")
    _run_lerobot_record(cmd, dataset_dir=dataset_dir, session_repo_id=session_repo_id)


def train_smolvla(
    dataset_repo_id: str | None = None,
    *,
    policy_path: str | None = None,
    output_dir: str | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> None:
    """Fine-tune SmolVLA on a recorded LeRobot dataset."""
    ensure_smolvla()
    cfg = ProjectConfig.load()
    from .data import configure_local_lerobot_env, resolve_training_dataset

    configure_local_lerobot_env(cfg)
    override = dataset_repo_id or cfg.policy.train_dataset
    resolved_dataset, dataset_dir = resolve_training_dataset(cfg, override, require_frames=True)

    resolved_policy = policy_path or cfg.policy.path
    resolved_output = output_dir or cfg.policy.output_dir
    resolved_steps = steps if steps is not None else cfg.policy.train_steps
    resolved_batch = batch_size if batch_size is not None else cfg.policy.train_batch_size
    resolved_device = resolve_device(device or cfg.policy.device)

    cmd = [
        "lerobot-train",
        f"--policy.path={resolved_policy}",
        f"--dataset.repo_id={resolved_dataset}",
        f"--dataset.root={dataset_dir.resolve()}",
        f"--batch_size={resolved_batch}",
        f"--steps={resolved_steps}",
        f"--output_dir={resolved_output}",
        "--job_name=sarm101_smolvla",
        f"--policy.device={resolved_device}",
        "--wandb.enable=false",
    ]

    print("Fine-tuning SmolVLA")
    print(f"  Base model:  {resolved_policy}")
    print(f"  Dataset:     {resolved_dataset}")
    print(f"  Local path:  {dataset_dir}")
    print(f"  Steps:       {resolved_steps}")
    print(f"  Output:      {resolved_output}")
    print(f"  Device:      {resolved_device}\n")
    subprocess.run(cmd, check=True)

    print("\nTraining complete. Run inference with:\n")
    print(f"  sarm-hand run-smolvla --task \"your task\" --policy-path {resolved_output}")
