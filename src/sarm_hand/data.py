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


def resolve_training_dataset(
    cfg: ProjectConfig,
    repo_id: str | None = None,
    *,
    require_frames: bool = True,
) -> tuple[str, Path]:
    """Resolve dataset for training — defaults to latest record-leader session with data."""
    if repo_id:
        path = cfg.resolve_dataset_path(repo_id)
        if not (path / "meta" / "info.json").is_file():
            _require_local_dataset(cfg, repo_id)
        return repo_id, path

    resolved_repo_id, path = resolve_dataset_lookup(cfg, latest=True)
    if not (path / "meta" / "info.json").is_file():
        _require_local_dataset(cfg, resolved_repo_id, latest=True)

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
