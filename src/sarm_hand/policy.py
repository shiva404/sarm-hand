"""SmolVLA and other LeRobot policy inference / training for S-ARM101."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .cameras import build_robot_camera_configs
from .config import ProjectConfig
from .robot import _motor_write_retries, ensure_port, require_all_motors


def _policy_fps(cfg: ProjectConfig) -> int:
    return cfg.policy.control_fps


def build_policy_rename_map(cfg: ProjectConfig) -> dict[str, str]:
    """Map observation.images.<robot_cam> → observation.images.<policy_cam>."""
    mapping = dict(cfg.policy.camera_map)
    if not mapping and cfg.cameras:
        first = next(iter(cfg.cameras))
        mapping[first] = "camera1"
    return {
        f"observation.images.{src}": f"observation.images.{dst}"
        for src, dst in mapping.items()
    }


def apply_smolvla_policy_overrides(policy_cfg, cfg: ProjectConfig) -> None:
    """Tune SmolVLA config for the number of physical cameras available."""
    empty = cfg.policy.empty_cameras
    if empty is None and len(cfg.cameras) == 1:
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

    dataset_root = cfg.resolve_dataset_root()
    train_id = cfg.policy.train_dataset or cfg.dataset.repo_id
    stats_path = dataset_root / train_id / "meta" / "stats.json"
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


def _smolvla_record_flags(cfg: ProjectConfig) -> list[str]:
    """Extra lerobot-record flags for SmolVLA camera mapping."""
    flags: list[str] = []
    rename_map = build_policy_rename_map(cfg)
    if rename_map:
        flags.append(f"--dataset.rename_map={rename_map!r}")
    empty = cfg.policy.empty_cameras
    if empty is None and len(cfg.cameras) == 1:
        empty = 2
    if empty is not None:
        flags.append(f"--policy.empty_cameras={empty}")
    return flags


def ensure_smolvla() -> None:
    """Verify SmolVLA dependencies are installed."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        print(
            "SmolVLA requires extra dependencies.\n"
            "Install with:  uv sync --extra smolvla",
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
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import log_rerun_data

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

            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
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
) -> None:
    """Run SmolVLA on the follower arm with a natural-language task query."""
    ensure_smolvla()
    cfg = ProjectConfig.load()
    _require_cameras(cfg)

    follower_port = ensure_port(follower_port or cfg.robot.port, "Follower")
    resolved_policy = policy_path or cfg.policy.path
    resolved_device = resolve_device(device or cfg.policy.device)
    resolved_episode_s = (
        episode_time_s if episode_time_s is not None else cfg.policy.episode_time_s
    )

    if record:
        _run_smolvla_record(
            cfg,
            follower_port=follower_port,
            policy_path=resolved_policy,
            task=task or cfg.dataset.single_task,
            episode_time_s=resolved_episode_s,
            repo_id=repo_id,
            num_episodes=num_episodes,
            display_data=display_data,
        )
        return

    if interactive:
        print("SmolVLA interactive mode — enter a task query per episode.")
        print("Press Enter on an empty line to quit.\n")
        while True:
            query = input("Task: ").strip()
            if not query:
                break
            _run_smolvla_inference(
                cfg,
                task=query,
                follower_port=follower_port,
                policy_path=resolved_policy,
                episode_time_s=resolved_episode_s,
                display_data=display_data,
                device=resolved_device,
            )
            print()
        return

    if not task:
        print("Error: provide --task or use --interactive.", file=sys.stderr)
        sys.exit(1)

    _run_smolvla_inference(
        cfg,
        task=task,
        follower_port=follower_port,
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
    import rerun as rr
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor import make_default_processors
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

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    robot = SO101Follower(robot_cfg)
    features = _build_dataset_features(
        robot, teleop_action_processor, robot_observation_processor
    )

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    apply_smolvla_policy_overrides(policy_cfg, cfg)
    rename_map = build_policy_rename_map(cfg)

    print("SmolVLA inference")
    print(f"  Policy:  {policy_path}")
    print(f"  Device:  {device}")
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

    norm_stats = resolve_policy_normalization_stats(cfg, policy_path, rename_map=rename_map)
    if norm_stats:
        print(f"  Norm stats: {cfg.policy.stats_buffer} buffer remap")

    if display_data:
        init_rerun(session_name="smolvla")

    with tempfile.TemporaryDirectory(prefix="sarm-hand-smolvla-") as tmp:
        # LeRobotDataset.create() mkdirs root itself — use a child path, not tmp directly.
        dataset_root = Path(tmp) / "local/sarm101-inference-scratch"
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
            preprocessor_overrides["normalizer_processor"] = {"stats": norm_stats}
            postprocessor_overrides["unnormalizer_processor"] = {"stats": norm_stats}

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )

        try:
            with _motor_write_retries():
                robot.connect()
            _policy_episode(
                robot=robot,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                features=features,
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
    dataset_root = cfg.resolve_dataset_root()
    dataset_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lerobot-record",
        f"--robot.type={cfg.robot.type}",
        f"--robot.port={follower_port}",
        f"--robot.id={cfg.robot.id}",
        f"--dataset.repo_id={resolved_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.fps={_policy_fps(cfg)}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={task}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={cfg.dataset.reset_time_s}",
        "--dataset.push_to_hub=false",
        f"--policy.path={policy_path}",
        f"--display_data={'true' if display_data else 'false'}",
    ]

    cameras = cfg.cameras_lerobot_dict()
    if cameras:
        cmd.append(f"--robot.cameras={cameras!r}")
    cmd.extend(_smolvla_record_flags(cfg))

    print(f"Recording {num_episodes} SmolVLA episode(s) → {resolved_repo_id}")
    print(f"Task: {task!r}\n")
    subprocess.run(cmd, check=True)


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
    resolved_dataset = dataset_repo_id or cfg.policy.train_dataset or cfg.dataset.repo_id
    resolved_policy = policy_path or cfg.policy.path
    resolved_output = output_dir or cfg.policy.output_dir
    resolved_steps = steps if steps is not None else cfg.policy.train_steps
    resolved_batch = batch_size if batch_size is not None else cfg.policy.train_batch_size
    resolved_device = resolve_device(device or cfg.policy.device)
    dataset_root = cfg.resolve_dataset_root()

    cmd = [
        "lerobot-train",
        f"--policy.path={resolved_policy}",
        f"--dataset.repo_id={resolved_dataset}",
        f"--dataset.root={dataset_root}",
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
    print(f"  Steps:       {resolved_steps}")
    print(f"  Output:      {resolved_output}")
    print(f"  Device:      {resolved_device}\n")
    subprocess.run(cmd, check=True)

    print("\nTraining complete. Run inference with:\n")
    print(f"  sarm-hand run-smolvla --task \"your task\" --policy-path {resolved_output}")
