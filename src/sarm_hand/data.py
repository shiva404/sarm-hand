"""LeRobot dataset access utilities for S-ARM101 recordings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .config import ProjectConfig


def _resolve_root(root: str | None, cfg: ProjectConfig) -> Path | None:
    if root:
        path = Path(root)
        return path if path.is_absolute() else cfg.resolve_dataset_root().parent / path
    default = cfg.resolve_dataset_root()
    return default if default.exists() else None


def load_dataset(
    repo_id: str | None = None,
    root: str | None = None,
    episodes: list[int] | None = None,
    download_videos: bool = True,
):
    """Load a LeRobotDataset from disk or Hugging Face Hub."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = ProjectConfig.load()
    resolved_repo_id = repo_id or cfg.dataset.repo_id
    resolved_root = _resolve_root(root, cfg)

    kwargs: dict[str, Any] = {
        "repo_id": resolved_repo_id,
        "download_videos": download_videos,
    }
    if resolved_root and (resolved_root / resolved_repo_id.split("/")[-1]).exists():
        kwargs["root"] = resolved_root / resolved_repo_id.split("/")[-1]
    elif resolved_root and resolved_root.exists():
        kwargs["root"] = resolved_root

    if episodes is not None:
        kwargs["episodes"] = episodes

    return LeRobotDataset(**kwargs)


def dataset_info(repo_id: str | None = None, root: str | None = None) -> None:
    """Print dataset metadata summary."""
    cfg = ProjectConfig.load()
    resolved_repo_id = repo_id or cfg.dataset.repo_id

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        resolved_root = _resolve_root(root, cfg)
        meta_kwargs: dict[str, Any] = {"repo_id": resolved_repo_id}
        if resolved_root:
            meta_kwargs["root"] = resolved_root

        meta = LeRobotDatasetMetadata(**meta_kwargs)
    except Exception as exc:
        print(f"Could not load dataset metadata: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset: {resolved_repo_id}")
    print(f"  Episodes:    {meta.total_episodes}")
    print(f"  Frames:      {meta.total_frames}")
    print(f"  FPS:         {meta.fps}")
    print(f"  Robot type:  {meta.info.get('robot_type', 'unknown')}")
    print(f"  Features:")
    for key, spec in meta.features.items():
        print(f"    - {key}: {spec.get('dtype', spec.get('shape', spec))}")

    if meta.tasks:
        print(f"  Tasks ({len(meta.tasks)}):")
        for task in list(meta.tasks)[:10]:
            print(f"    - {task}")
        if len(meta.tasks) > 10:
            print(f"    ... and {len(meta.tasks) - 10} more")


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

    # Collect scalar/tensor columns from the episode
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


def dataset_push(repo_id: str | None = None, root: str | None = None) -> None:
    """Upload a local dataset to Hugging Face Hub."""
    dataset = load_dataset(repo_id=repo_id, root=root, download_videos=True)
    print(f"Uploading {dataset.repo_id} to Hugging Face Hub...")
    dataset.push_to_hub()
    print("Upload complete.")
