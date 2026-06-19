"""Tests for task motion record/replay storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sarm_hand.config import ProjectConfig
from sarm_hand.task_motion import (
    TaskMotionDemo,
    TaskMotionFrame,
    list_demos,
    load_demo,
    resolve_demo_path,
    save_demo,
    task_slug,
)


def test_task_slug_normalizes_text():
    assert task_slug("Pick up the cube!") == "pick_up_the_cube"


def test_task_slug_rejects_empty():
    with pytest.raises(ValueError):
        task_slug("   ")


def test_save_and_load_demo_roundtrip(tmp_path: Path):
    demo = TaskMotionDemo(
        version=1,
        task="Pick and place",
        task_slug="pick_and_place",
        demo_id="demo_test",
        recorded_at="2026-01-01T00:00:00+00:00",
        fps=30,
        source="leader",
        frames=[
            TaskMotionFrame(
                t=0.0,
                joints={
                    "shoulder_pan": 0.0,
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 0.0,
                    "gripper": 50.0,
                },
            ),
            TaskMotionFrame(
                t=0.1,
                joints={
                    "shoulder_pan": 5.0,
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 0.0,
                    "gripper": 55.0,
                },
            ),
        ],
    )
    path = tmp_path / "pick_and_place" / "demo_test.json"
    save_demo(demo, path)
    loaded = load_demo(path)
    assert loaded.task == demo.task
    assert len(loaded.frames) == 2
    assert loaded.frames[1].joints["shoulder_pan"] == 5.0


def test_resolve_demo_path_latest(tmp_path: Path):
    cfg = ProjectConfig.load()
    cfg.tasks.root = str(tmp_path)

    slug = "pick_and_place"
    first = tmp_path / slug / "demo_20260101_120000.json"
    second = tmp_path / slug / "demo_20260101_130000.json"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "task": "Pick and place",
                    "task_slug": slug,
                    "demo_id": path.stem,
                    "recorded_at": "",
                    "fps": 30,
                    "source": "leader",
                    "frames": [
                        {
                            "t": 0.0,
                            "joints": {
                                "shoulder_pan": 0.0,
                                "shoulder_lift": 0.0,
                                "elbow_flex": 0.0,
                                "wrist_flex": 0.0,
                                "wrist_roll": 0.0,
                                "gripper": 50.0,
                            },
                        }
                    ],
                }
            )
        )

    assert resolve_demo_path(cfg, task_slug=slug, demo_id="latest") == second
    assert list_demos(cfg, slug) == [first, second]
