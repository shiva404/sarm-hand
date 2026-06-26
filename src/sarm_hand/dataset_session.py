"""Shared local LeRobot dataset session paths and creation (hardware + sim)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig


def recording_stamp(when: datetime | None = None) -> str:
    """Timestamp suffix matching record-sim (``YYYYMMDD-HHMMSS-micro``)."""
    return (when or datetime.now()).strftime("%Y%m%d-%H%M%S-%f")


def resolve_recording_paths(
    *,
    base_repo: str,
    root: Path,
    repo_id: str | None,
    resume: bool,
    timestamp: bool,
    when: datetime | None = None,
) -> tuple[str, Path]:
    """Pick repo_id and directory for a recording session (same rules as record-sim).

    Timestamped runs write under ``root / local/name-YYYYMMDD-HHMMSS-ffffff``.
    """
    base = repo_id or base_repo
    if resume:
        path = root / base
        if not path.exists():
            print(f"Cannot resume: dataset not found at {path}", file=sys.stderr)
            raise SystemExit(1)
        return base, path

    if timestamp:
        while True:
            stamped = f"{base}-{recording_stamp(when)}"
            path = root / stamped
            if not path.exists():
                return stamped, path
            when = None

    path = root / base
    if path.exists():
        print(f"Dataset already exists: {path}", file=sys.stderr)
        print("Use timestamped recording (default), or pass --resume to append.", file=sys.stderr)
        raise SystemExit(1)
    return base, path


def build_robot_dataset_features(
    robot: Any,
    teleop_action_processor: Any,
    robot_observation_processor: Any,
    *,
    use_videos: bool,
) -> dict:
    """Feature schema for lerobot-record-style hardware datasets (video + joint state)."""
    from lerobot.datasets.feature_utils import combine_feature_dicts
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )

    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=use_videos,
        ),
    )


def create_recording_dataset(
    cfg: ProjectConfig,
    repo_id: str,
    root: Path,
    features: dict,
    *,
    num_cameras: int,
    resume: bool = False,
) -> Any:
    """Create or resume a local LeRobot dataset using the same writer as record-sim."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import resolve_vcodec

    ds = cfg.dataset
    vcodec = resolve_vcodec(ds.vcodec)
    threads = ds.num_image_writer_threads_per_camera * max(num_cameras, 1)

    if resume:
        return LeRobotDataset.resume(
            repo_id,
            root=root,
            batch_encoding_size=ds.video_encoding_batch_size,
            vcodec=vcodec,
        )

    return LeRobotDataset.create(
        repo_id,
        ds.fps,
        root=root,
        robot_type=cfg.robot.type,
        features=features,
        use_videos=ds.video,
        image_writer_processes=0,
        image_writer_threads=threads,
        batch_encoding_size=ds.video_encoding_batch_size,
        vcodec=vcodec,
        streaming_encoding=False,
        encoder_threads=ds.encoder_threads,
    )


def camera_feature_keys(cfg: ProjectConfig) -> list[str]:
    return [f"observation.images.{name}" for name in cfg.cameras]
