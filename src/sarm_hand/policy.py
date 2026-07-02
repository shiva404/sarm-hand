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


def _policy_fps(cfg: ProjectConfig, *, kind: str) -> int:
    if kind == "act":
        return cfg.policies.act.control_fps
    return cfg.policies.smolvla.control_fps


def resolve_camera_map(cfg: ProjectConfig, *, genesis: bool = False) -> dict[str, str]:
    """Camera name → SmolVLA image key (camera1, camera2, …)."""
    smolvla = cfg.policies.smolvla
    if smolvla.camera_map:
        return dict(smolvla.camera_map)
    names = cfg.genesis.cameras if genesis else cfg.cameras
    return {name: f"camera{i}" for i, name in enumerate(names, start=1)}


def build_policy_rename_map(cfg: ProjectConfig, *, genesis: bool = False) -> dict[str, str]:
    """Map observation.images.<robot_cam> → observation.images.<policy_cam>."""
    mapping = resolve_camera_map(cfg, genesis=genesis)
    return {
        f"observation.images.{src}": f"observation.images.{dst}"
        for src, dst in mapping.items()
    }


def apply_smolvla_policy_overrides(policy_cfg, cfg: ProjectConfig, *, genesis: bool = False) -> None:
    """Tune SmolVLA config for the number of physical cameras available."""
    empty = cfg.policies.smolvla.empty_cameras
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

    train_id = cfg.policies.train_dataset
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
        return load_smolvla_buffer_stats(policy_path, cfg.policies.smolvla.stats_buffer)

    return None


def apply_act_inference_overrides(policy_cfg, act) -> None:
    """Apply ACT inference-time overrides to a loaded checkpoint config."""
    if act.temporal_ensemble_coeff is not None:
        policy_cfg.temporal_ensemble_coeff = act.temporal_ensemble_coeff
        policy_cfg.n_action_steps = 1
    elif act.inference_n_action_steps > 0:
        policy_cfg.n_action_steps = act.inference_n_action_steps


def _inference_blend_alpha(
    inference_step: int,
    *,
    inference_blend_steps: int,
    replan_blend_steps: int,
    n_action_steps: int,
    use_temporal_ensemble: bool,
) -> float:
    """Policy weight 0→1 after startup hold, then soften replan chunk starts."""
    if inference_blend_steps > 0 and inference_step < inference_blend_steps:
        return (inference_step + 1) / inference_blend_steps
    if use_temporal_ensemble or replan_blend_steps <= 0:
        return 1.0
    chunk_pos = inference_step % max(n_action_steps, 1)
    if chunk_pos >= replan_blend_steps:
        return 1.0
    return (chunk_pos + 1) / replan_blend_steps


def _blend_action_with_present(
    robot_action: dict[str, float],
    obs: dict,
    alpha: float,
) -> dict[str, float]:
    """Blend policy goals toward present joints. alpha=0 holds pose; alpha=1 uses policy."""
    alpha = max(0.0, min(1.0, alpha))
    if alpha >= 1.0:
        return robot_action
    present = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
    if alpha <= 0.0:
        return {k: present[k] for k in present}
    return {
        key: present[key] + alpha * (float(goal) - present[key])
        if key.endswith(".pos") and key in present
        else goal
        for key, goal in robot_action.items()
    }


def _log_action_step(
    step: int,
    obs: dict,
    sent: dict,
    *,
    raw: dict | None = None,
    ramp_steps: int = 0,
) -> None:
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
        f"({max_joint}, present={float(present[max_joint]):.1f} "
        f"goal={float(sent.get(max_joint, present[max_joint])):.1f})"
    )
    if step == 0 and raw and ramp_steps > 0:
        raw_deltas = {
            joint: abs(float(raw.get(joint, present[joint])) - float(present[joint]))
            for joint in present
        }
        raw_max = max(raw_deltas, key=raw_deltas.get)
        print(
            f"  step 0: policy wanted Δ{raw_deltas[raw_max]:.1f} on {raw_max} "
            f"(goal={float(raw.get(raw_max, 0)):.1f}); "
            f"startup ramp ({ramp_steps} steps) holds current pose first"
        )
    elif step == 0 and deltas[max_joint] > 20:
        print(
            "  step 0: large jump — policy output, not move-to-ready. "
            "Match your physical start pose to demo recordings (see warning above)."
        )
    elif deltas[max_joint] < 3 and step > 0 and step % max(150, 1) == 0:
        print(
            "  (policy goals ≈ present joints — checkpoint may be undertrained or "
            "trained on the wrong dataset; see warnings above)"
        )


def _read_checkpoint_training_dataset(
    policy_path: str,
) -> tuple[str, Path, dict] | None:
    """Return (repo_id, path, info) from a checkpoint's train_config.json."""
    import json

    from .data import _read_dataset_info

    pretrained = ensure_local_pretrained_dir(policy_path)
    train_cfg_path = pretrained / "train_config.json"
    if not train_cfg_path.is_file():
        return None
    try:
        train_cfg = json.loads(train_cfg_path.read_text())
    except json.JSONDecodeError:
        return None
    dataset = train_cfg.get("dataset") or {}
    repo_id = dataset.get("repo_id")
    root = dataset.get("root")
    if not repo_id or not root:
        return None
    path = Path(root)
    info = _read_dataset_info(path)
    if info is None:
        return None
    return str(repo_id), path, info


def _warn_act_checkpoint_dataset(policy_path: str, cfg: ProjectConfig) -> None:
    """Warn when the checkpoint was trained on a tiny or test subsample dataset."""
    from .data import _best_training_dataset, _dataset_summary, _is_test_subsample_dataset

    trained = _read_checkpoint_training_dataset(policy_path)
    if not trained:
        return
    repo_id, _path, info = trained
    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    tiny = episodes < 5 or frames < 1000
    test_set = _is_test_subsample_dataset(repo_id)
    if not tiny and not test_set:
        print(f"  Trained on: {_dataset_summary(repo_id, info)}")
        return

    print("\nCheckpoint was trained on a tiny/test dataset:", file=sys.stderr)
    print(f"  {_dataset_summary(repo_id, info)}", file=sys.stderr)
    if alt := _best_training_dataset(cfg):
        alt_id, _alt_path, alt_info = alt
        if int(alt_info.get("total_frames", 0)) > frames:
            print(f"  Your full recordings: {_dataset_summary(alt_id, alt_info)}", file=sys.stderr)
            print(
                "\n  Retrain on the full dataset, then run-act again:\n"
                f"    uv run sarm-hand train-act --dataset-repo-id {alt_id}\n",
                file=sys.stderr,
            )
            return
    print(
        "\n  Record more demos (record-leader), then train-act on that dataset.\n",
        file=sys.stderr,
    )


def _warn_act_pose_vs_training(robot, cfg: ProjectConfig, *, policy_path: str) -> None:
    """Warn when the arm is far from where training demos begin."""
    from .data import dataset_demo_start_action_mean

    trained = _read_checkpoint_training_dataset(policy_path)
    if trained:
        _repo_id, dataset_dir, _info = trained
    else:
        from .data import resolve_training_dataset

        train_id = cfg.policies.train_dataset
        _repo_id, dataset_dir = resolve_training_dataset(cfg, train_id)
    demo_start = dataset_demo_start_action_mean(dataset_dir)
    if not demo_start:
        return

    obs = robot.get_observation()
    present = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
    mismatches = [
        (key, present.get(key, demo_start[key]), demo_val)
        for key, demo_val in demo_start.items()
        if abs(present.get(key, demo_val) - demo_val) > 20
    ]
    if not mismatches:
        return

    print("\nStart pose differs from training demos:", file=sys.stderr)
    print("  ACT outputs absolute joint goals from (cameras + current joints).", file=sys.stderr)
    print("  Your demos typically start at:", file=sys.stderr)
    for key, demo_val in demo_start.items():
        short = key.replace(".pos", "")
        print(f"    {short:14} {demo_val:6.1f}", file=sys.stderr)
    print("  You are at:", file=sys.stderr)
    for key, cur, demo_val in mismatches:
        short = key.replace(".pos", "")
        print(f"    {short:14} {cur:6.1f}  (demo start {demo_val:.1f})", file=sys.stderr)
    print(
        "\n  Fix: move the arm to the same folded rest pose used during record-leader,\n"
        "  or re-record demos starting from the pose you want at inference.\n"
        "  Checkpoint 002000 is also very early — try a later checkpoint or finish train-act.\n",
        file=sys.stderr,
    )


def _smolvla_dataset_flags(cfg: ProjectConfig, *, genesis: bool = False, train: bool = False) -> list[str]:
    """Extra lerobot flags for SmolVLA camera mapping (record or train)."""
    flags: list[str] = []
    rename_map = build_policy_rename_map(cfg, genesis=genesis)
    if rename_map:
        key = "rename_map" if train else "dataset.rename_map"
        flags.append(f"--{key}={rename_map!r}")
    empty = cfg.policies.smolvla.empty_cameras
    cam_count = len(cfg.genesis.cameras) if genesis else len(cfg.cameras)
    if empty is None and cam_count == 1:
        empty = 2
    if empty is not None:
        flags.append(f"--policy.empty_cameras={empty}")
    return flags


def _smolvla_record_flags(cfg: ProjectConfig, *, genesis: bool = False) -> list[str]:
    """Extra lerobot-record flags for SmolVLA camera mapping."""
    return _smolvla_dataset_flags(cfg, genesis=genesis, train=False)


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


def ensure_training() -> None:
    """Verify PyTorch is available for policy training."""
    try:
        __import__("torch")
    except ImportError:
        print(
            "Training requires PyTorch.\n"
            "Install with:  uv sync --extra training\n"
            "  (or: uv sync --extra smolvla for SmolVLA)",
            file=sys.stderr,
        )
        sys.exit(1)


def ensure_act_vision_backbone_weights() -> None:
    """Pre-cache ResNet18 ImageNet weights used as the ACT vision backbone."""
    from .data import configure_ssl_certificates

    configure_ssl_certificates()
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    cache = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if cache.is_file():
        return
    print("Downloading ACT vision backbone (ResNet18 ImageNet weights)...")
    try:
        resnet18(weights=weights)
    except Exception as exc:
        print(
            f"Failed to download ResNet18 weights: {exc}\n"
            "  macOS fix: open /Applications/Python 3.12/Install Certificates.command\n"
            "  Or retry after: uv run python -c \"import certifi; print(certifi.where())\"",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    print(f"  Cached: {cache}")


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


def resolve_policy_checkpoint_path(policy_path: str) -> str:
    """Resolve a LeRobot train output dir to checkpoints/.../pretrained_model."""
    if not policy_path:
        return policy_path

    # Bare hub model ids (no path separators).
    if "/" not in policy_path and "\\" not in policy_path:
        return policy_path

    path = Path(policy_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if not path.exists():
        fallback = _latest_checkpoint_from_ancestor(path)
        return str(fallback) if fallback is not None else policy_path

    if (path / "config.json").is_file():
        return str(path)

    if (path / "pretrained_model" / "config.json").is_file():
        return str((path / "pretrained_model").resolve())

    for candidate in (
        path / "checkpoints" / "last" / "pretrained_model",
        path / "pretrained_model",
    ):
        if (candidate / "config.json").is_file():
            return str(candidate.resolve())

    checkpoints = path / "checkpoints" if path.name != "checkpoints" else path
    latest = _latest_numbered_pretrained_model(checkpoints)
    if latest is not None:
        return str(latest)

    if path.parent.name == "last" and path.parent.parent.name == "checkpoints":
        latest = _latest_numbered_pretrained_model(path.parent.parent)
        if latest is not None:
            return str(latest)

    return policy_path


def _latest_checkpoint_from_ancestor(path: Path) -> Path | None:
    """When a direct checkpoint path is missing, search upward for checkpoints/."""
    for parent in [path, *path.parents]:
        if parent.name == "checkpoints":
            latest = _latest_numbered_pretrained_model(parent)
            if latest is not None:
                return latest
        ckpt_dir = parent / "checkpoints"
        if ckpt_dir.is_dir():
            latest = _latest_numbered_pretrained_model(ckpt_dir)
            if latest is not None:
                return latest
    return None


def _latest_numbered_pretrained_model(checkpoints: Path) -> Path | None:
    """Newest checkpoints/NNNNNN/pretrained_model with config.json."""
    if not checkpoints.is_dir():
        return None
    numbered = sorted(
        (p for p in checkpoints.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for ckpt in reversed(numbered):
        model = ckpt / "pretrained_model"
        if (model / "config.json").is_file():
            return model.resolve()
    return None


def _looks_like_hub_model_id(policy_path: str) -> bool:
    """True for Hugging Face repo ids like ``lerobot/smolvla_base``."""
    if "\\" in policy_path or policy_path.startswith((".", "/", "~")):
        return False
    if policy_path.startswith(("outputs/", "data/", "local/")):
        return False
    parts = policy_path.split("/")
    return len(parts) == 2 and all(parts)


def ensure_local_pretrained_dir(policy_path: str) -> Path:
    """Resolve a local pretrained_model directory or exit with a clear message."""
    resolved = resolve_policy_checkpoint_path(policy_path)
    path = Path(resolved).expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if path.is_dir() and (path / "config.json").is_file():
        return path

    original = Path(policy_path).expanduser()
    if not original.is_absolute():
        original = original.resolve()
    if not original.exists() and resolved == policy_path and _looks_like_hub_model_id(policy_path):
        return Path(policy_path)

    print(f"Policy checkpoint not found: {policy_path}", file=sys.stderr)
    if str(path) != policy_path:
        print(f"  Resolved to: {path}", file=sys.stderr)
    latest = _latest_checkpoint_from_ancestor(original)
    if latest is not None:
        print(f"  Latest checkpoint on disk: {latest}", file=sys.stderr)
        print(f"  Try: --policy-path {latest}", file=sys.stderr)
    else:
        print(
            "  Training may not have saved yet, or the path is wrong.\n"
            "  Use the train output dir, e.g. outputs/train/sarm101_act",
            file=sys.stderr,
        )
    raise SystemExit(1)


def resolve_training_batch_size(
    device: str,
    batch_size: int | None,
    cfg: ProjectConfig,
) -> int:
    """Pick a batch size that fits the device (SmolVLA + 3 cameras is memory-heavy on MPS)."""
    resolved = batch_size if batch_size is not None else cfg.policies.smolvla.train_batch_size
    if device == "mps" and resolved > 4:
        print(
            f"MPS memory: reducing batch size {resolved} → 4 "
            f"(use --batch-size 2 if OOM persists)",
            file=sys.stderr,
        )
        return 4
    return resolved


def resolve_training_num_workers(
    device: str,
    num_workers: int | None,
    cfg: ProjectConfig,
    *,
    kind: str,
) -> int:
    if num_workers is not None:
        return num_workers
    settings = cfg.policies.act if kind == "act" else cfg.policies.smolvla
    if settings.train_num_workers is not None:
        return settings.train_num_workers
    return 0 if device in ("mps", "cpu") else 4


def _training_output_has_checkpoints(output_dir: Path) -> bool:
    checkpoints = output_dir / "checkpoints"
    return checkpoints.is_dir() and any(checkpoints.iterdir())


def _resolve_training_config_path(output_dir: Path) -> Path:
    config_path = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    if config_path.is_file():
        return config_path
    print(f"No training checkpoint to resume in {output_dir}", file=sys.stderr)
    print("  Start fresh:  rm -rf", output_dir, file=sys.stderr)
    raise SystemExit(1)


def _resolve_last_checkpoint_dir(output_dir: Path) -> Path:
    last = output_dir / "checkpoints" / "last"
    if not last.exists():
        raise FileNotFoundError(f"No checkpoint to resume in {output_dir}")
    return last.resolve() if last.is_symlink() else last


def _apply_act_learning_rate(
    checkpoint_dir: Path,
    lr: float,
    *,
    backbone_lr: float | None = None,
) -> None:
    """Patch saved train config + optimizer param groups (resume ignores CLI LR otherwise)."""
    import json

    from lerobot.utils.constants import OPTIMIZER_PARAM_GROUPS, PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR

    backbone = backbone_lr if backbone_lr is not None else lr

    train_cfg_path = checkpoint_dir / PRETRAINED_MODEL_DIR / "train_config.json"
    if train_cfg_path.is_file():
        data = json.loads(train_cfg_path.read_text())
        policy = data.setdefault("policy", {})
        policy["optimizer_lr"] = lr
        policy["optimizer_lr_backbone"] = backbone
        data.setdefault("optimizer", {})["lr"] = lr
        train_cfg_path.write_text(json.dumps(data, indent=4) + "\n")

    pg_path = checkpoint_dir / TRAINING_STATE_DIR / OPTIMIZER_PARAM_GROUPS
    if pg_path.is_file():
        groups = json.loads(pg_path.read_text())
        for index, group in enumerate(groups):
            group["lr"] = backbone if index == 1 and len(groups) > 1 else lr
        pg_path.write_text(json.dumps(groups, indent=4) + "\n")


def _act_lerobot_lr_flags(lr: float, *, backbone_lr: float | None = None) -> list[str]:
    backbone = backbone_lr if backbone_lr is not None else lr
    return [
        f"--optimizer.lr={lr}",
        f"--policy.optimizer_lr={lr}",
        f"--policy.optimizer_lr_backbone={backbone}",
    ]


def resolve_act_training_batch_size(
    device: str,
    batch_size: int | None,
    cfg: ProjectConfig,
) -> int:
    """ACT is lightweight — use larger batches than SmolVLA on MPS."""
    resolved = batch_size if batch_size is not None else cfg.policies.act.train_batch_size
    if device == "mps" and resolved > 8:
        print(
            f"MPS: reducing ACT batch size {resolved} → 8",
            file=sys.stderr,
        )
        return 8
    return resolved


def estimate_dataset_frames(cfg: ProjectConfig) -> int:
    """Expected frame count from dataset recording settings."""
    ds = cfg.dataset
    return int(ds.num_episodes) * int(ds.episode_time_s) * int(ds.fps)


def resolve_act_training_steps(
    act,
    cfg: ProjectConfig,
    dataset_dir: Path,
    batch_size: int,
    *,
    steps_override: int | None = None,
) -> tuple[int, int, int]:
    """Return (total_steps, steps_per_epoch, total_frames)."""
    from .data import _read_dataset_info

    info = _read_dataset_info(dataset_dir) or {}
    total_frames = int(info.get("total_frames", 0))
    if total_frames <= 0:
        total_frames = estimate_dataset_frames(cfg)
    steps_per_epoch = max(1, (total_frames + batch_size - 1) // batch_size)

    if steps_override is not None:
        return steps_override, steps_per_epoch, total_frames
    if act.train_steps is not None:
        return int(act.train_steps), steps_per_epoch, total_frames
    return act.train_epochs * steps_per_epoch, steps_per_epoch, total_frames


def _require_training_output_dir(
    output_dir: Path,
    *,
    resume: bool,
    train_cmd: str = "train-smolvla",
) -> None:
    if not output_dir.is_dir() or resume:
        return
    print(
        f"Output directory already exists: {output_dir}\n",
        file=sys.stderr,
    )
    if _training_output_has_checkpoints(output_dir):
        print(
            "  Resume from checkpoint:\n"
            f"    uv run sarm-hand {train_cmd} --resume --output-dir {output_dir}\n",
            file=sys.stderr,
        )
    print(
        "  Start fresh (new directory):\n"
        f"    uv run sarm-hand {train_cmd} --output-dir {output_dir.parent / (output_dir.name + '-fresh')}\n"
        "  Or remove the old run:\n"
        f"    rm -rf {output_dir}\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


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
    task: str | None,
    fps: int,
    episode_time_s: float,
    display_data: bool,
    startup_ramp_steps: int = 0,
    action_smoothing: float = 1.0,
    inference_blend_steps: int = 0,
    replan_blend_steps: int = 0,
) -> None:
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import log_rerun_data

    from .rerun_viz import smooth_action_targets

    policy_device = get_safe_torch_device(policy.config.device)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    deadline = time.perf_counter() + episode_time_s
    interval = 1.0 / fps

    print(f"Running policy for {episode_time_s:.0f}s @ {fps} fps (Ctrl+C to stop early)")
    if task:
        print(f"Task: {task!r}")
    if startup_ramp_steps > 0:
        print(
            f"Startup hold: {startup_ramp_steps} steps (~{startup_ramp_steps / fps:.1f}s) "
            "— cameras settle, policy not called yet"
        )
    print()

    step = 0
    ramp_logged = False
    smoothed_action: dict[str, float] | None = None
    try:
        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            present_action = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}

            # Hold pose without calling ACT — select_action fills a 100-step queue we must not burn.
            if startup_ramp_steps > 0 and step < startup_ramp_steps:
                action_to_send = robot_action_processor((present_action, obs))
                sent = robot.send_action(action_to_send)
                if step == 0 or step % max(fps * 5, 1) == 0:
                    _log_action_step(step, obs, sent)
                if display_data:
                    log_rerun_data(observation=obs_processed, action=present_action)
                elapsed = time.perf_counter() - loop_start
                precise_sleep(max(interval - elapsed, 0.0))
                step += 1
                continue

            if startup_ramp_steps > 0 and not ramp_logged:
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
                    replan_note = (
                        f"temporal ensemble coeff {policy.config.temporal_ensemble_coeff}"
                    )
                else:
                    replan_note = (
                        f"replan every {getattr(policy.config, 'n_action_steps', '?')} steps"
                    )
                print(
                    f"Startup hold done — policy inference from step {step} ({replan_note})"
                )
                ramp_logged = True

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
            inference_step = max(0, step - startup_ramp_steps)
            blend_alpha = _inference_blend_alpha(
                inference_step,
                inference_blend_steps=inference_blend_steps,
                replan_blend_steps=replan_blend_steps,
                n_action_steps=getattr(policy.config, "n_action_steps", 1),
                use_temporal_ensemble=getattr(
                    policy.config, "temporal_ensemble_coeff", None
                )
                is not None,
            )
            if blend_alpha < 1.0:
                robot_action = _blend_action_with_present(robot_action, obs, blend_alpha)
            if action_smoothing < 1.0:
                robot_action = smooth_action_targets(
                    smoothed_action, robot_action, alpha=action_smoothing
                )
                smoothed_action = robot_action
            action_to_send = robot_action_processor((robot_action, obs))
            sent = robot.send_action(action_to_send)

            if step == 0 or step % max(fps * 5, 1) == 0:
                _log_action_step(step, obs, sent)

            if display_data:
                log_rerun_data(observation=obs_processed, action=sent)

            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(interval - elapsed, 0.0))
            step += 1
    except KeyboardInterrupt:
        print("\nStopped early.")


def _load_policy_stack(
    cfg: ProjectConfig,
    *,
    policy_path: str,
    device: str,
    robot,
    genesis: bool,
    smolvla: bool,
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

    pretrained_dir = ensure_local_pretrained_dir(policy_path)
    pretrained_path = str(pretrained_dir)

    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    policy_cfg.pretrained_path = pretrained_path
    policy_cfg.device = device
    if not smolvla:
        apply_act_inference_overrides(policy_cfg, cfg.policies.act)
    rename_map: dict[str, str] = {}
    if smolvla:
        apply_smolvla_policy_overrides(policy_cfg, cfg, genesis=genesis)
        rename_map = build_policy_rename_map(cfg, genesis=genesis)

    norm_stats = (
        resolve_policy_normalization_stats(
            cfg, pretrained_path, rename_map=rename_map if smolvla else {}
        )
        if smolvla
        else None  # ACT: use stats baked into checkpoint preprocessor (dataset override → oscillation)
    )

    scratch = "smolvla" if smolvla else "act"
    dataset_root = Path(tempfile.mkdtemp(prefix=f"sarm-hand-{scratch}-")) / "local/sarm101-inference-scratch"
    dataset = LeRobotDataset.create(
        repo_id="local/sarm101-inference-scratch",
        fps=_policy_fps(cfg, kind="smolvla" if smolvla else "act"),
        root=dataset_root,
        robot_type=robot.name,
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    policy = make_policy(
        policy_cfg,
        ds_meta=dataset.meta,
        rename_map=rename_map if rename_map else None,
    )

    preprocessor_overrides: dict = {"device_processor": {"device": policy_cfg.device}}
    if rename_map:
        preprocessor_overrides["rename_observations_processor"] = {"rename_map": rename_map}
    postprocessor_overrides: dict = {}
    if norm_stats:
        preprocessor_overrides["normalizer_processor"] = {
            "stats": norm_stats,
            "device": device,
        }
        postprocessor_overrides["unnormalizer_processor"] = {"stats": norm_stats}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
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


def _load_smolvla_stack(
    cfg: ProjectConfig,
    *,
    policy_path: str,
    device: str,
    robot,
    genesis: bool,
):
    return _load_policy_stack(
        cfg,
        policy_path=policy_path,
        device=device,
        robot=robot,
        genesis=genesis,
        smolvla=True,
    )


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
        print(f"  Norm stats: {cfg.policies.smolvla.stats_buffer} buffer remap")
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

    smolvla = cfg.policies.smolvla
    resolved_policy = resolve_policy_checkpoint_path(policy_path or smolvla.path)
    if policy_path and resolved_policy != policy_path:
        print(f"Using checkpoint: {resolved_policy}")
    resolved_device = resolve_device(device or smolvla.device)
    resolved_episode_s = (
        episode_time_s if episode_time_s is not None else smolvla.episode_time_s
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
    install_all_camera_patches(cfg=cfg)
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
            fps=_policy_fps(cfg, kind="smolvla"),
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
            fps=_policy_fps(cfg, kind="smolvla"),
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
        f"--dataset.fps={_policy_fps(cfg, kind='smolvla')}",
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
    num_workers: int | None = None,
    device: str | None = None,
    resume: bool = False,
) -> None:
    """Fine-tune SmolVLA on a recorded LeRobot dataset."""
    ensure_smolvla()
    cfg = ProjectConfig.load()
    from .data import configure_local_lerobot_env, resolve_training_dataset, training_subprocess_env

    configure_local_lerobot_env(cfg)
    override = dataset_repo_id or cfg.policies.train_dataset
    resolved_dataset, dataset_dir = resolve_training_dataset(cfg, override, require_frames=True)
    smolvla = cfg.policies.smolvla

    resolved_policy = policy_path or smolvla.path
    resolved_output = output_dir or smolvla.output_dir
    resolved_steps = steps if steps is not None else smolvla.train_steps
    resolved_device = resolve_device(device or smolvla.device)
    resolved_batch = resolve_training_batch_size(resolved_device, batch_size, cfg)
    resolved_workers = resolve_training_num_workers(
        resolved_device, num_workers, cfg, kind="smolvla"
    )
    output_path = Path(resolved_output)
    if resume:
        config_path = _resolve_training_config_path(output_path)
    else:
        _require_training_output_dir(output_path, resume=False, train_cmd="train-smolvla")
        config_path = None

    if resume:
        cmd = [
            "lerobot-train",
            f"--config_path={config_path}",
            "--resume=true",
            f"--batch_size={resolved_batch}",
            f"--num_workers={resolved_workers}",
            f"--steps={resolved_steps}",
            f"--output_dir={resolved_output}",
            "--save_freq=2000",
            "--dataset.video_backend=pyav",
            f"--policy.device={resolved_device}",
            "--wandb.enable=false",
        ]
    else:
        cmd = [
            "lerobot-train",
            f"--policy.path={resolved_policy}",
            f"--dataset.repo_id={resolved_dataset}",
            f"--dataset.root={dataset_dir.resolve()}",
            f"--batch_size={resolved_batch}",
            f"--num_workers={resolved_workers}",
            f"--steps={resolved_steps}",
            f"--output_dir={resolved_output}",
            "--save_freq=2000",
            "--dataset.video_backend=pyav",
            "--job_name=sarm101_smolvla",
            f"--policy.device={resolved_device}",
            "--policy.push_to_hub=false",
            "--wandb.enable=false",
        ]
    cmd.extend(_smolvla_dataset_flags(cfg, train=True))

    print("Fine-tuning SmolVLA")
    if resume:
        print(f"  Resume:      {config_path}")
    else:
        print(f"  Base model:  {resolved_policy}")
    print(f"  Dataset:     {resolved_dataset}")
    print(f"  Local path:  {dataset_dir}")
    print(f"  Steps:       {resolved_steps}")
    print(f"  Batch size:  {resolved_batch}")
    print(f"  Workers:     {resolved_workers}")
    print(f"  Output:      {resolved_output}")
    print(f"  Device:      {resolved_device}\n")
    subprocess.run(cmd, check=True, env=training_subprocess_env())

    checkpoint = resolve_policy_checkpoint_path(str(resolved_output))
    print("\nTraining complete. Run inference with:\n")
    print(f"  sarm-hand run-smolvla --task \"your task\" --policy-path {checkpoint}")


def _require_act_cameras(cfg: ProjectConfig) -> None:
    if not cfg.cameras:
        print(
            "ACT needs cameras in config/default.yaml (front + wrist recommended).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    names = set(cfg.cameras)
    if not names.intersection({"front", "wrist"}):
        print(
            f"ACT config expects front + wrist cameras; got: {sorted(names)}",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _print_act_header(
    *,
    policy_path: str,
    device: str,
    cfg: ProjectConfig,
    follower_port: str,
) -> None:
    print("ACT inference")
    print(f"  Policy:  {policy_path}")
    print(f"  Device:  {device}")
    print(f"  Robot:   {follower_port}")
    print(f"  Cameras: {list(cfg.cameras.keys())}")
    print(f"  Rate:    {cfg.policies.act.control_fps} fps")
    act = cfg.policies.act
    clamp = act.max_relative_target
    ramp = act.startup_ramp_steps
    if clamp is None:
        print("  Start:   current pose (no move-to-ready; policy goals unclamped)")
    else:
        print(f"  Start:   current pose (max step clamp {clamp})")
    if ramp > 0:
        print(f"  Hold:    {ramp} steps (~{ramp / act.control_fps:.1f}s) before first inference")
    if act.temporal_ensemble_coeff is not None:
        print(f"  Smooth:  temporal ensemble coeff {act.temporal_ensemble_coeff} (replan every step)")
    elif act.inference_n_action_steps > 0:
        print(f"  Replan:  every {act.inference_n_action_steps} control steps")
    if act.inference_blend_steps > 0:
        print(
            f"  Ease-in: {act.inference_blend_steps} steps "
            f"(~{act.inference_blend_steps / act.control_fps:.1f}s) after hold"
        )
    if act.replan_blend_steps > 0 and act.temporal_ensemble_coeff is None:
        print(f"  Replan blend: first {act.replan_blend_steps} steps of each chunk")
    if act.action_smoothing < 1.0:
        print(f"  Filter:  action EMA alpha {act.action_smoothing}")
    print()


def run_act(
    *,
    follower_port: str | None = None,
    policy_path: str | None = None,
    episode_time_s: float | None = None,
    display_data: bool = True,
    device: str | None = None,
) -> None:
    """Run a fine-tuned ACT policy on the follower arm (front + wrist cameras)."""
    ensure_training()
    cfg = ProjectConfig.load()
    _require_act_cameras(cfg)

    act = cfg.policies.act
    resolved_port = ensure_port(follower_port or cfg.robot.port, "Follower")
    resolved_policy = resolve_policy_checkpoint_path(
        policy_path or act.output_dir
    )
    if policy_path and resolved_policy != policy_path:
        print(f"Using checkpoint: {resolved_policy}")
    resolved_device = resolve_device(device or act.device)
    resolved_episode_s = (
        episode_time_s if episode_time_s is not None else act.episode_time_s
    )

    _run_act_inference(
        cfg,
        follower_port=resolved_port,
        policy_path=resolved_policy,
        episode_time_s=resolved_episode_s,
        display_data=display_data,
        device=resolved_device,
    )


def _run_act_inference(
    cfg: ProjectConfig,
    *,
    follower_port: str,
    policy_path: str,
    episode_time_s: float,
    display_data: bool,
    device: str,
) -> None:
    install_all_camera_patches(cfg=cfg)
    import rerun as rr
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import init_rerun

    init_logging()
    require_all_motors("follower", follower_port, context="run ACT")
    act = cfg.policies.act

    robot_cfg = SOFollowerRobotConfig(
        id=cfg.robot.id,
        port=follower_port,
        use_degrees=cfg.robot.use_degrees,
        max_relative_target=act.max_relative_target,
        disable_torque_on_disconnect=cfg.robot.disable_torque_on_disconnect,
        cameras=build_robot_camera_configs(cfg),
    )
    robot = SO101Follower(robot_cfg)
    stack = _load_policy_stack(
        cfg,
        policy_path=policy_path,
        device=device,
        robot=robot,
        genesis=False,
        smolvla=False,
    )
    _print_act_header(
        policy_path=policy_path,
        device=device,
        cfg=cfg,
        follower_port=follower_port,
    )

    if display_data:
        init_rerun(session_name="act")

    try:
        with _motor_write_retries():
            connect_follower_robot(robot, calibrate=False, cfg=cfg)
        _warn_act_checkpoint_dataset(policy_path, cfg)
        _warn_act_pose_vs_training(robot, cfg, policy_path=policy_path)
        _policy_episode(
            robot=robot,
            policy=stack["policy"],
            preprocessor=stack["preprocessor"],
            postprocessor=stack["postprocessor"],
            robot_action_processor=stack["robot_action_processor"],
            robot_observation_processor=stack["robot_observation_processor"],
            features=stack["features"],
            task=None,
            fps=_policy_fps(cfg, kind="act"),
            episode_time_s=episode_time_s,
            display_data=display_data,
            startup_ramp_steps=act.startup_ramp_steps,
            action_smoothing=act.action_smoothing,
            inference_blend_steps=act.inference_blend_steps,
            replan_blend_steps=act.replan_blend_steps,
        )
    except ConnectionError as exc:
        print(
            "\nLost contact with a servo while connecting the follower.\n"
            "  sarm-hand test-motors --role follower",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    finally:
        if display_data:
            rr.rerun_shutdown()
        if robot.is_connected:
            robot.disconnect()

    print("Done.")


def train_act(
    dataset_repo_id: str | None = None,
    *,
    output_dir: str | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: str | None = None,
    learning_rate: float | None = None,
    resume: bool = False,
) -> None:
    """Train ACT on a recorded LeRobot dataset (front + wrist cameras)."""
    ensure_training()
    cfg = ProjectConfig.load()
    _require_act_cameras(cfg)
    ensure_act_vision_backbone_weights()
    from .data import configure_local_lerobot_env, resolve_training_dataset, training_subprocess_env

    configure_local_lerobot_env(cfg)
    override = dataset_repo_id or cfg.policies.train_dataset
    resolved_dataset, dataset_dir = resolve_training_dataset(cfg, override, require_frames=True)

    act = cfg.policies.act
    resolved_output = output_dir or act.output_dir
    resolved_device = resolve_device(device or act.device)
    resolved_batch = resolve_act_training_batch_size(resolved_device, batch_size, cfg)
    resolved_steps, steps_per_epoch, total_frames = resolve_act_training_steps(
        act,
        cfg,
        dataset_dir,
        resolved_batch,
        steps_override=steps,
    )
    resolved_workers = resolve_training_num_workers(
        resolved_device, num_workers, cfg, kind="act"
    )
    resolved_lr = learning_rate if learning_rate is not None else act.learning_rate
    output_path = Path(resolved_output)
    if resume:
        config_path = _resolve_training_config_path(output_path)
    else:
        _require_training_output_dir(output_path, resume=False, train_cmd="train-act")
        config_path = None

    if resume:
        _apply_act_learning_rate(_resolve_last_checkpoint_dir(output_path), resolved_lr)
        cmd = [
            "lerobot-train",
            f"--config_path={config_path}",
            "--resume=true",
            f"--batch_size={resolved_batch}",
            f"--num_workers={resolved_workers}",
            f"--steps={resolved_steps}",
            f"--output_dir={resolved_output}",
            f"--save_freq={act.save_freq}",
            "--dataset.video_backend=pyav",
            f"--policy.device={resolved_device}",
            *_act_lerobot_lr_flags(resolved_lr),
            "--wandb.enable=false",
        ]
    else:
        cmd = [
            "lerobot-train",
            "--policy.type=act",
            f"--dataset.repo_id={resolved_dataset}",
            f"--dataset.root={dataset_dir.resolve()}",
            f"--batch_size={resolved_batch}",
            f"--num_workers={resolved_workers}",
            f"--steps={resolved_steps}",
            f"--output_dir={resolved_output}",
            f"--save_freq={act.save_freq}",
            "--dataset.video_backend=pyav",
            "--job_name=sarm101_act",
            f"--policy.device={resolved_device}",
            *_act_lerobot_lr_flags(resolved_lr),
            "--policy.push_to_hub=false",
            "--wandb.enable=false",
        ]

    print("Training ACT")
    if resume:
        print(f"  Resume:      {config_path}")
    from .data import _read_dataset_info

    print(f"  Dataset:     {resolved_dataset}")
    print(f"  Local path:  {dataset_dir}")
    info = _read_dataset_info(dataset_dir) or {}
    episodes = int(info.get("total_episodes", 0))
    if episodes:
        print(f"  Episodes:    {episodes}")
    print(f"  Frames:      {total_frames} (~{steps_per_epoch} steps/epoch)")
    print(f"  Epochs:      {act.train_epochs}")
    print(f"  Steps:       {resolved_steps} ({act.train_epochs} × {steps_per_epoch})")
    print(f"  Save every:  {act.save_freq} steps")
    print(f"  Cameras:     {list(cfg.cameras.keys())}")
    print(f"  Batch size:  {resolved_batch}")
    print(f"  Workers:     {resolved_workers}")
    print(f"  LR:          {resolved_lr}")
    print(f"  Output:      {resolved_output}")
    print(f"  Device:      {resolved_device}\n")
    subprocess.run(cmd, check=True, env=training_subprocess_env())

    checkpoint = resolve_policy_checkpoint_path(str(resolved_output))
    print("\nTraining complete. Run inference with:\n")
    print(f"  sarm-hand run-act --policy-path {checkpoint}")
