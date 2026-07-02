"""LeRobot dataset access utilities for S-ARM101 recordings (local-first)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import ProjectConfig


def _local_hub_offline() -> None:
    """Prefer on-disk datasets; never fall back to Hugging Face Hub for reads."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _read_dataset_info(path: Path) -> dict[str, Any] | None:
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        return json.loads(info_path.read_text())
    except json.JSONDecodeError:
        return None


def list_local_datasets(cfg: ProjectConfig) -> list[tuple[str, Path, dict[str, Any]]]:
    """Return (repo_id, path, info.json) for every local dataset, newest first."""
    base = cfg.resolve_dataset_root()
    if not base.is_dir():
        return []

    seen: dict[Path, tuple[str, Path, dict[str, Any]]] = {}
    for info_path in base.rglob("meta/info.json"):
        path = info_path.parent.parent
        if path in seen:
            continue
        info = _read_dataset_info(path)
        if info is None:
            continue
        rel = path.relative_to(base)
        seen[path] = (rel.as_posix(), path, info)

    return sorted(seen.values(), key=lambda row: row[1].stat().st_mtime, reverse=True)


def read_latest_session_pointer(cfg: ProjectConfig) -> tuple[str, Path] | None:
    """Last session written by record-leader (``.latest_session`` under dataset root)."""
    pointer = cfg.resolve_dataset_root() / ".latest_session"
    if not pointer.is_file():
        return None
    repo_id: str | None = None
    path: Path | None = None
    for line in pointer.read_text().splitlines():
        if line.startswith("repo_id="):
            repo_id = line.split("=", 1)[1].strip()
        elif line.startswith("path="):
            path = Path(line.split("=", 1)[1].strip())
    if repo_id and path and (path / "meta" / "info.json").is_file():
        return repo_id, path
    return None


def _session_has_frames(path: Path) -> bool:
    info = _read_dataset_info(path)
    return info is not None and int(info.get("total_frames", 0)) > 0


def _newest_session_with_frames(
    cfg: ProjectConfig,
) -> tuple[str, Path] | None:
    for repo_id, path, info in list_local_datasets(cfg):
        if int(info.get("total_frames", 0)) > 0:
            return repo_id, path
    return None


def _is_test_subsample_dataset(repo_id: str) -> bool:
    return "test-subsample" in repo_id


def _dataset_summary(repo_id: str, info: dict[str, Any]) -> str:
    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    fps = int(info.get("fps", 0))
    return f"{repo_id} ({episodes} episodes, {frames} frames{f' @ {fps} fps' if fps else ''})"


def _best_training_dataset(
    cfg: ProjectConfig,
) -> tuple[str, Path, dict[str, Any]] | None:
    """Largest non-test-subsample local dataset with saved frames."""
    best: tuple[str, Path, dict[str, Any]] | None = None
    for repo_id, path, info in list_local_datasets(cfg):
        if _is_test_subsample_dataset(repo_id):
            continue
        if int(info.get("total_frames", 0)) <= 0:
            continue
        if best is None or int(info.get("total_frames", 0)) > int(best[2].get("total_frames", 0)):
            best = (repo_id, path, info)
    return best


def _should_prefer_alternate_training_dataset(repo_id: str, info: dict[str, Any]) -> bool:
    episodes = int(info.get("total_episodes", 0))
    return _is_test_subsample_dataset(repo_id) or episodes < 5


def configure_ssl_certificates() -> None:
    """Use certifi CA bundle (fixes macOS Python.org SSL verify failures)."""
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def training_subprocess_env() -> dict[str, str]:
    """Environment for lerobot-train subprocesses (SSL + offline HF)."""
    configure_ssl_certificates()
    return os.environ.copy()


def configure_local_lerobot_env(cfg: ProjectConfig) -> Path:
    """Force LeRobot to read/write datasets under the project tree (never Hub by default)."""
    dataset_home = cfg.resolve_dataset_root()
    dataset_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(dataset_home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return dataset_home


def write_session_manifest(
    dataset_dir: Path,
    repo_id: str,
    *,
    fps: int,
    task: str,
) -> None:
    """Write session metadata after LeRobotDataset.create (meta/ must already exist)."""
    meta = dataset_dir / "meta"
    if not meta.is_dir():
        return
    payload = {
        "repo_id": repo_id,
        "session_id": repo_id.split("/")[-1],
        "path": str(dataset_dir),
        "fps": fps,
        "task": task,
    }
    (meta / "session.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_latest_session_pointer(cfg: ProjectConfig, repo_id: str, path: Path) -> None:
    pointer = cfg.resolve_dataset_root() / ".latest_session"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(f"repo_id={repo_id}\npath={path}\n")


def _count_dataset_artifacts(dataset_dir: Path) -> tuple[int, int]:
    parquet = sum(1 for _ in dataset_dir.rglob("*.parquet")) if (dataset_dir / "data").is_dir() else 0
    mp4 = sum(1 for _ in dataset_dir.rglob("*.mp4")) if (dataset_dir / "videos").is_dir() else 0
    return parquet, mp4


def dataset_demo_start_action_mean(dataset_dir: Path) -> dict[str, float] | None:
    """Mean first-frame action across episodes — typical pose at the start of demos."""
    data_dir = dataset_dir / "data"
    info_path = dataset_dir / "meta" / "info.json"
    if not data_dir.is_dir() or not info_path.is_file():
        return None

    import numpy as np
    import pandas as pd

    info = json.loads(info_path.read_text())
    names = info["features"]["action"]["names"]
    parquets = sorted(data_dir.rglob("*.parquet"))
    if not parquets:
        return None

    frames: list[np.ndarray] = []
    for path in parquets:
        df = pd.read_parquet(path, columns=["action", "episode_index"])
        for ep in df["episode_index"].unique():
            frames.append(np.asarray(df.loc[df["episode_index"] == ep].iloc[0]["action"], dtype=float))

    if not frames:
        return None

    mean = np.mean(frames, axis=0)
    return {name: float(val) for name, val in zip(names, mean)}


def resolve_training_dataset(
    cfg: ProjectConfig,
    repo_id: str | None = None,
    *,
    require_frames: bool = True,
) -> tuple[str, Path]:
    """Resolve dataset for training — defaults to latest record-leader session with data."""
    if repo_id:
        repo_id = repo_id.strip("/")
        path = cfg.resolve_dataset_path(repo_id)
        if not (path / "meta" / "info.json").is_file():
            _require_local_dataset(cfg, repo_id)
        return repo_id, path

    resolved_repo_id, path = resolve_dataset_lookup(cfg, latest=True)
    if not (path / "meta" / "info.json").is_file():
        _require_local_dataset(cfg, resolved_repo_id, latest=True)

    info = _read_dataset_info(path) or {}
    if _should_prefer_alternate_training_dataset(resolved_repo_id, info):
        if alt := _best_training_dataset(cfg):
            alt_id, alt_path, alt_info = alt
            if alt_path != path and int(alt_info.get("total_frames", 0)) > int(
                info.get("total_frames", 0)
            ):
                print(
                    "Training dataset: skipping "
                    f"{_dataset_summary(resolved_repo_id, info)}",
                    file=sys.stderr,
                )
                print(
                    f"  Using instead: {_dataset_summary(alt_id, alt_info)}",
                    file=sys.stderr,
                )
                print(
                    "  (.latest_session pointed at a test subsample or tiny run; "
                    "pass --dataset-repo-id to override.)\n",
                    file=sys.stderr,
                )
                return alt_id, alt_path

    if require_frames:
        info = _read_dataset_info(path) or {}
        if int(info.get("total_frames", 0)) <= 0:
            print(f"Latest session has no saved frames yet: {path}", file=sys.stderr)
            if alt := _newest_session_with_frames(cfg):
                alt_id, alt_path = alt
                if alt_path != path:
                    alt_info = _read_dataset_info(alt_path) or {}
                    print(
                        f"  Using newest session with data instead: {alt_id}"
                        f" ({alt_info.get('total_episodes', 0)} episodes,"
                        f" {alt_info.get('total_frames', 0)} frames)",
                        file=sys.stderr,
                    )
                    return alt_id, alt_path
            print("  Finish at least one episode during record-leader (→ Right arrow).", file=sys.stderr)
            print(f"  repo_id: {resolved_repo_id}", file=sys.stderr)
            raise SystemExit(1)
    return resolved_repo_id, path


def resolve_dataset_lookup(
    cfg: ProjectConfig,
    repo_id: str | None = None,
    *,
    latest: bool = False,
) -> tuple[str, Path]:
    """Resolve repo_id and on-disk path; prefer latest session with saved frames."""
    if latest:
        if session := read_latest_session_pointer(cfg):
            if _session_has_frames(session[1]):
                return session
        if with_data := _newest_session_with_frames(cfg):
            return with_data
        if session := read_latest_session_pointer(cfg):
            return session
        datasets = list_local_datasets(cfg)
        if not datasets:
            raise SystemExit("No local datasets found. Record with: uv run sarm-hand record-leader")
        return datasets[0][0], datasets[0][1]

    if repo_id is None:
        if session := read_latest_session_pointer(cfg):
            if _session_has_frames(session[1]):
                return session
        if with_data := _newest_session_with_frames(cfg):
            return with_data
        if session := read_latest_session_pointer(cfg):
            return session
        datasets = list_local_datasets(cfg)
        if datasets:
            return datasets[0][0], datasets[0][1]

    resolved = repo_id or cfg.dataset.repo_id
    return resolved, cfg.resolve_dataset_path(resolved)


def resolve_local_dataset_path(
    cfg: ProjectConfig,
    repo_id: str,
    root: str | None = None,
) -> Path:
    """Expected on-disk path for ``repo_id`` (may not exist yet)."""
    if root:
        path = Path(root)
        if not path.is_absolute():
            path = cfg.resolve_dataset_root().parent / path
        if (path / "meta" / "info.json").is_file():
            return path
    return cfg.resolve_dataset_path(repo_id)


def _list_local_datasets(cfg: ProjectConfig) -> list[Path]:
    return [path for _, path, _ in list_local_datasets(cfg)]


def _require_local_dataset(
    cfg: ProjectConfig,
    repo_id: str | None = None,
    root: str | None = None,
    *,
    latest: bool = False,
) -> Path:
    if root:
        path = resolve_local_dataset_path(cfg, repo_id or cfg.dataset.repo_id, root)
        if (path / "meta" / "info.json").is_file():
            return path
    else:
        _, path = resolve_dataset_lookup(cfg, repo_id, latest=latest)
        if (path / "meta" / "info.json").is_file():
            return path

    resolved = repo_id or cfg.dataset.repo_id
    print(f"No local dataset at {path if root else cfg.resolve_dataset_path(resolved)}", file=sys.stderr)
    if resolved:
        print(f"  repo_id: {resolved}", file=sys.stderr)
    print("\nRecord hardware demos with:", file=sys.stderr)
    print("  uv run sarm-hand record-leader", file=sys.stderr)
    print("  (finish one episode: wait or press → Right arrow; then check data-info --latest)", file=sys.stderr)

    others = list_local_datasets(cfg)
    if others:
        print("\nLocal datasets found:", file=sys.stderr)
        for rid, ds, info in others[:8]:
            eps = info.get("total_episodes", 0)
            frames = info.get("total_frames", 0)
            print(
                f"  uv run sarm-hand data-info --repo-id {rid}"
                f"   ({eps} episodes, {frames} frames)",
                file=sys.stderr,
            )
        print("\nLatest session:  uv run sarm-hand data-info --latest", file=sys.stderr)
    raise SystemExit(1)


def _load_local_metadata(
    cfg: ProjectConfig,
    repo_id: str | None = None,
    root: str | None = None,
    *,
    latest: bool = False,
):
    """Load dataset metadata from disk only (no Hub download)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    _local_hub_offline()
    if root:
        dataset_path = _require_local_dataset(cfg, repo_id, root, latest=False)
        resolved_repo_id = repo_id or cfg.dataset.repo_id
    else:
        resolved_repo_id, dataset_path = resolve_dataset_lookup(cfg, repo_id, latest=latest)
        if not (dataset_path / "meta" / "info.json").is_file():
            _require_local_dataset(cfg, resolved_repo_id, latest=latest)
    return LeRobotDatasetMetadata(
        resolved_repo_id,
        root=dataset_path,
        force_cache_sync=False,
    ), dataset_path


def load_dataset(
    repo_id: str | None = None,
    root: str | None = None,
    episodes: list[int] | None = None,
    download_videos: bool = False,
    *,
    latest: bool = False,
):
    """Load a LeRobotDataset from the local project dataset directory."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = ProjectConfig.load()
    _local_hub_offline()
    configure_local_lerobot_env(cfg)
    if root:
        resolved_repo_id = repo_id or cfg.dataset.repo_id
        dataset_path = _require_local_dataset(cfg, resolved_repo_id, root)
    else:
        resolved_repo_id, dataset_path = resolve_dataset_lookup(cfg, repo_id, latest=latest)
        if not (dataset_path / "meta" / "info.json").is_file():
            dataset_path = _require_local_dataset(cfg, resolved_repo_id, latest=latest)

    kwargs: dict[str, Any] = {
        "repo_id": resolved_repo_id,
        "download_videos": download_videos,
        "root": dataset_path,
        "force_cache_sync": False,
    }

    if episodes is not None:
        kwargs["episodes"] = episodes

    return LeRobotDataset(**kwargs)


def dataset_list() -> None:
    """List local recording sessions."""
    cfg = ProjectConfig.load()
    rows = list_local_datasets(cfg)
    if not rows:
        print("No local datasets yet. Record with: uv run sarm-hand record-leader")
        return
    latest = read_latest_session_pointer(cfg)
    newest_with_data = _newest_session_with_frames(cfg)
    print(f"Datasets under {cfg.resolve_dataset_root()}:\n")
    for repo_id, path, info in rows:
        eps = info.get("total_episodes", 0)
        frames = info.get("total_frames", 0)
        markers: list[str] = []
        if latest and latest[0] == repo_id:
            markers.append("latest session")
        if newest_with_data and newest_with_data[0] == repo_id:
            markers.append("newest with data")
        marker = f"  ← {', '.join(markers)}" if markers else ""
        print(f"  {repo_id}")
        print(f"    path: {path}")
        print(f"    episodes: {eps}  frames: {frames}{marker}\n")


def dataset_info(
    repo_id: str | None = None,
    root: str | None = None,
    *,
    latest: bool = False,
) -> None:
    """Print metadata for a local dataset (never contacts Hugging Face Hub)."""
    cfg = ProjectConfig.load()

    try:
        if root:
            resolved_repo_id = repo_id or cfg.dataset.repo_id
            dataset_path = resolve_local_dataset_path(cfg, resolved_repo_id, root)
            if not (dataset_path / "meta" / "info.json").is_file():
                _require_local_dataset(cfg, resolved_repo_id, root)
        else:
            resolved_repo_id, dataset_path = resolve_dataset_lookup(cfg, repo_id, latest=latest)
            if not (dataset_path / "meta" / "info.json").is_file():
                _require_local_dataset(cfg, resolved_repo_id, latest=latest)
    except SystemExit:
        raise

    info = _read_dataset_info(dataset_path)
    if info is None:
        print(f"Could not read {dataset_path / 'meta' / 'info.json'}", file=sys.stderr)
        sys.exit(1)

    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    features = info.get("features", {})
    pointer = read_latest_session_pointer(cfg)
    pointer_id = pointer[0] if pointer else None

    print(f"Dataset: {resolved_repo_id}")
    print(f"  Path:        {dataset_path}")
    print(f"  Episodes:    {episodes}")
    print(f"  Frames:      {frames}")
    if frames == 0:
        print(
            "  Note:        no frames saved yet — finish at least one episode during record-leader"
            " (→ Right arrow or wait for timer)"
        )
        if alt := _newest_session_with_frames(cfg):
            alt_id, alt_path = alt
            if alt_path != dataset_path:
                alt_info = _read_dataset_info(alt_path) or {}
                print(
                    f"  Tip:         newest session with data is {alt_id}"
                    f" ({alt_info.get('total_episodes', 0)} episodes,"
                    f" {alt_info.get('total_frames', 0)} frames)"
                )
                print(f"               uv run sarm-hand data-info --repo-id {alt_id}")
    elif pointer_id and pointer_id != resolved_repo_id:
        print(f"  Note:        .latest_session points at empty run {pointer_id}")
    print(f"  FPS:         {info.get('fps', '?')}")
    print(f"  Robot type:  {info.get('robot_type', 'unknown')}")
    print("  Features:")
    for key, spec in features.items():
        print(f"    - {key}: {spec.get('dtype', spec.get('shape', spec))}")

    video_keys = [k for k, spec in features.items() if spec.get("dtype") == "video"]
    if video_keys:
        videos_dir = dataset_path / "videos"
        if videos_dir.is_dir():
            mp4_count = sum(1 for _ in videos_dir.rglob("*.mp4"))
            print(f"  Video files: {mp4_count} under videos/")
        elif frames > 0:
            print("  Video files: (encoding may still be in progress or failed)")

    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        try:
            import pandas as pd

            tasks = pd.read_parquet(tasks_path)
            if len(tasks):
                print(f"  Tasks ({len(tasks)}):")
                for task in tasks.index[:10]:
                    print(f"    - {task}")
                if len(tasks) > 10:
                    print(f"    ... and {len(tasks) - 10} more")
        except Exception:
            pass


def dataset_sample(
    repo_id: str | None = None,
    root: str | None = None,
    index: int = 0,
) -> None:
    """Print one dataset frame as JSON-serializable summary."""
    dataset = load_dataset(repo_id=repo_id, root=root, download_videos=False)
    if index < 0 or index >= len(dataset):
        print(f"Index {index} out of range (0..{len(dataset) - 1})", file=sys.stderr)
        sys.exit(1)

    frame = dataset[index]
    summary: dict[str, Any] = {}
    for key, value in frame.items():
        if hasattr(value, "shape"):
            summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            summary[key] = value

    print(json.dumps(summary, indent=2, default=str))


def dataset_export_episode(
    repo_id: str | None = None,
    root: str | None = None,
    episode: int = 0,
    output_dir: str = "data/exports",
) -> None:
    """Export one episode's actions and state to CSV files."""
    import csv

    cfg = ProjectConfig.load()
    dataset = load_dataset(repo_id=repo_id, root=root, episodes=[episode])

    out = Path(output_dir)
    if not out.is_absolute():
        out = cfg.resolve_dataset_root().parent / out
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        frame = dataset[idx]
        row = {k: (v.item() if hasattr(v, "item") and v.numel() == 1 else str(v)) for k, v in frame.items()}
        rows.append(row)

    if not rows:
        print("No frames found in episode.", file=sys.stderr)
        sys.exit(1)

    csv_path = out / f"episode_{episode:04d}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported episode {episode} ({len(rows)} frames) → {csv_path}")


def subsample_stride(source_fps: int, target_fps: int) -> int:
    """Return keep-every-N stride to downsample ``source_fps`` → ``target_fps``."""
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"fps must be positive (got {source_fps} → {target_fps})")
    if target_fps > source_fps:
        raise ValueError(
            f"target_fps ({target_fps}) must be <= source_fps ({source_fps}); "
            "use record-leader at a higher rate instead"
        )
    if source_fps % target_fps != 0:
        raise ValueError(
            f"source_fps ({source_fps}) must be evenly divisible by target_fps ({target_fps}); "
            f"e.g. 30→10 (stride 3) or 30→15 (stride 2)"
        )
    return source_fps // target_fps


def frame_image_to_hwc_uint8(img: Any) -> Any:
    """Convert a dataset image (CHW float tensor) to HWC uint8 for LeRobot ``add_frame``."""
    import numpy as np
    import torch

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr


def _episode_frame_indices(dataset_path: Path) -> dict[int, list[tuple[int, int]]]:
    """Map episode_index → [(global_index, frame_index_within_episode), ...]."""
    import pandas as pd

    grouped: dict[int, list[tuple[int, int]]] = {}
    data_dir = dataset_path / "data"
    if not data_dir.is_dir():
        return grouped

    for path in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(path, columns=["episode_index", "index", "frame_index"])
        for row in df.itertuples(index=False):
            ep = int(row.episode_index)
            grouped.setdefault(ep, []).append((int(row.index), int(row.frame_index)))

    for ep in grouped:
        grouped[ep].sort(key=lambda pair: pair[1])
    return grouped


def _default_subsample_repo_id(source_repo_id: str, target_fps: int) -> str:
    if source_repo_id.endswith(f"-fps{target_fps}"):
        return source_repo_id
    return f"{source_repo_id}-fps{target_fps}"


def _subsample_output_features(features: dict[str, Any], target_fps: int) -> dict[str, Any]:
    import copy

    out = copy.deepcopy(features)
    for spec in out.values():
        if spec.get("dtype") != "video":
            continue
        info = spec.setdefault("info", {})
        info["video.fps"] = target_fps
    return out


def _frame_for_writer(item: dict[str, Any], feature_keys: set[str]) -> dict[str, Any]:
    import numpy as np
    import torch

    frame: dict[str, Any] = {"task": item["task"]}
    for key in feature_keys:
        if key not in item:
            continue
        value = item[key]
        if key.startswith("observation.images."):
            frame[key] = frame_image_to_hwc_uint8(value)
        elif isinstance(value, torch.Tensor):
            frame[key] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            frame[key] = value
        else:
            frame[key] = np.asarray(value)
    return frame


def dataset_subsample(
    repo_id: str | None = None,
    root: str | None = None,
    *,
    target_fps: int = 10,
    output_repo_id: str | None = None,
    output_root: str | None = None,
    episodes: list[int] | None = None,
    latest: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """Downsample a LeRobot dataset by keeping every Nth frame (e.g. 30 fps → 10 fps).

  Works on local recordings and external datasets copied under ``--root``.
  Re-encodes videos at the new fps and recomputes dataset stats.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = ProjectConfig.load()
    configure_local_lerobot_env(cfg)

    if root:
        resolved_repo_id = repo_id or cfg.dataset.repo_id
        source_path = _require_local_dataset(cfg, resolved_repo_id, root)
    else:
        resolved_repo_id, source_path = resolve_dataset_lookup(cfg, repo_id, latest=latest)
        if not (source_path / "meta" / "info.json").is_file():
            source_path = _require_local_dataset(cfg, resolved_repo_id, latest=latest)

    info = _read_dataset_info(source_path)
    if info is None:
        print(f"Could not read {source_path / 'meta' / 'info.json'}", file=sys.stderr)
        sys.exit(1)

    source_fps = int(info.get("fps", 0))
    stride = subsample_stride(source_fps, target_fps)
    episode_map = _episode_frame_indices(source_path)
    if not episode_map:
        print(f"No parquet frames found under {source_path / 'data'}", file=sys.stderr)
        sys.exit(1)

    selected_eps = sorted(episode_map) if episodes is None else sorted(episodes)
    missing = [ep for ep in selected_eps if ep not in episode_map]
    if missing:
        print(f"Episode(s) not in dataset: {missing}", file=sys.stderr)
        sys.exit(1)

    kept_per_ep = {
        ep: [(gi, fi) for gi, fi in episode_map[ep] if fi % stride == 0] for ep in selected_eps
    }
    total_in = sum(len(episode_map[ep]) for ep in selected_eps)
    total_out = sum(len(rows) for rows in kept_per_ep.values())
    out_repo_id = output_repo_id or _default_subsample_repo_id(resolved_repo_id, target_fps)

    if output_root:
        out_path = Path(output_root)
        if not out_path.is_absolute():
            out_path = cfg.resolve_dataset_root().parent / out_path
    else:
        out_path = cfg.resolve_dataset_path(out_repo_id)

    print(f"Subsample {resolved_repo_id}")
    print(f"  Source:      {source_path}")
    print(f"  Source fps:  {source_fps}")
    print(f"  Target fps:  {target_fps}  (keep every {stride} frame(s))")
    print(f"  Episodes:    {len(selected_eps)}")
    print(f"  Frames:      {total_in} → {total_out}")
    print(f"  Output:      {out_repo_id}")
    print(f"  Path:        {out_path}")

    if dry_run:
        return out_path

    if out_path.exists() and any(out_path.iterdir()):
        print(f"Output path already exists: {out_path}", file=sys.stderr)
        print("  Pass --output-repo-id or remove the directory first.", file=sys.stderr)
        sys.exit(1)

    src = LeRobotDataset(
        resolved_repo_id,
        root=source_path,
        episodes=selected_eps,
        download_videos=True,
        force_cache_sync=False,
        video_backend="pyav",
    )

    vcodec = "h264"
    for spec in src.meta.features.values():
        if spec.get("dtype") == "video":
            vcodec = spec.get("info", {}).get("video.codec", vcodec)
            break

    write_keys = {
        key
        for key in src.meta.features
        if key == "action" or key.startswith("observation.")
    }

    dst = LeRobotDataset.create(
        repo_id=out_repo_id,
        fps=target_fps,
        features=_subsample_output_features(src.meta.features, target_fps),
        root=out_path,
        robot_type=src.meta.robot_type,
        use_videos=True,
        vcodec=vcodec,
        image_writer_processes=0,
        image_writer_threads=4,
        video_backend="pyav",
    )

    for ep_i, ep in enumerate(selected_eps, start=1):
        kept = kept_per_ep[ep]
        if not kept:
            print(f"  Episode {ep}: no frames after subsample — skipped", file=sys.stderr)
            continue
        print(f"  Episode {ep_i}/{len(selected_eps)}: {len(kept)} frames", flush=True)
        for global_idx, _frame_idx in kept:
            item = src[global_idx]
            dst.add_frame(_frame_for_writer(item, write_keys))
        dst.save_episode(parallel_encoding=False)

    dst.finalize()
    write_session_manifest(out_path, out_repo_id, fps=target_fps, task=_read_source_task(source_path))
    write_latest_session_pointer(cfg, out_repo_id, out_path)

    out_info = _read_dataset_info(out_path) or {}
    print(
        f"\nDone: {out_repo_id} — {out_info.get('total_episodes', '?')} episodes,"
        f" {out_info.get('total_frames', '?')} frames @ {out_info.get('fps', target_fps)} fps"
    )
    print(f"  Train with: uv run sarm-hand train-act --dataset-repo-id {out_repo_id}")
    return out_path


def _read_source_task(dataset_path: Path) -> str:
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return "task"
    try:
        import pandas as pd

        tasks = pd.read_parquet(tasks_path)
        if len(tasks):
            return str(tasks.index[0])
    except Exception:
        pass
    return "task"


def dataset_push(
    repo_id: str | None = None,
    root: str | None = None,
    *,
    push_to_hub: bool | None = None,
) -> None:
    """Upload a local dataset to Hugging Face Hub (explicit opt-in only)."""
    cfg = ProjectConfig.load()
    resolved_repo_id = repo_id or cfg.dataset.repo_id
    allow = push_to_hub if push_to_hub is not None else cfg.dataset.push_to_hub
    if not allow:
        print(
            "Hugging Face upload disabled (dataset.push_to_hub: false).\n"
            "Re-run with:  uv run sarm-hand data-push --push-to-hub",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_path = _require_local_dataset(cfg, resolved_repo_id, root)
    dataset = load_dataset(repo_id=resolved_repo_id, root=str(dataset_path), download_videos=False)
    print(f"Uploading local dataset → Hugging Face Hub")
    print(f"  Local:   {dataset_path}")
    print(f"  repo_id: {dataset.repo_id}")
    dataset.push_to_hub()
    print("Upload complete.")
