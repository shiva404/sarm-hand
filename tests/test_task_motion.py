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


def test_leader_pose_to_follower_prefers_raw():
    from sarm_hand.calibration_bridge import leader_pose_to_follower_action
    from sarm_hand.genesis.calibration import norm_to_raw, raw_to_norm

    leader = {
        "shoulder_lift": {"range_min": 853, "range_max": 3078, "drive_mode": 0},
        "gripper": {"range_min": 1746, "range_max": 2333, "drive_mode": 0},
    }
    follower = {
        "shoulder_lift": {"range_min": 0, "range_max": 4095, "drive_mode": 0},
        "gripper": {"range_min": 1728, "range_max": 2540, "drive_mode": 0},
    }
    joints = {"shoulder_lift": 10.0, "gripper": 50.0}
    raw = {"shoulder_lift": 2000, "gripper": 2043}
    action = leader_pose_to_follower_action(
        joints=joints,
        raw=raw,
        leader_cal=leader,
        follower_cal=follower,
    )
    assert action["shoulder_lift.pos"] == raw_to_norm(2000, "shoulder_lift", follower)
    assert action["gripper.pos"] == raw_to_norm(2043, "gripper", follower)


def test_replay_remaps_leader_norm_to_follower(monkeypatch, tmp_path: Path):
    """Replay converts leader-normalized frames to follower goals via encoder counts."""
    cfg = ProjectConfig.load()
    slug = "pick_and_place"
    demo_path = tmp_path / slug / "demo_test.json"
    demo_path.parent.mkdir(parents=True)
    demo_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task": "Pick and place",
                "task_slug": slug,
                "demo_id": "demo_test",
                "recorded_at": "",
                "fps": 30,
                "source": "leader",
                "frames": [
                    {
                        "t": 0.0,
                        "joints": {
                            "shoulder_pan": 25.0,
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

    sent_actions: list[dict[str, float]] = []

    class FakeRobot:
        config = type("Cfg", (), {"max_relative_target": 10})()
        is_calibrated = True
        is_connected = True

        def send_action(self, action):
            sent_actions.append(dict(action))
            return action

        def disconnect(self):
            pass

    leader_cal = {
        j: {"range_min": 1000, "range_max": 3000, "drive_mode": 0}
        for j in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
    }
    follower_cal = {
        j: {"range_min": 0, "range_max": 4095, "drive_mode": 0}
        for j in leader_cal
    }

    monkeypatch.setattr("sarm_hand.task_motion.ProjectConfig.load", lambda: cfg)
    monkeypatch.setattr("sarm_hand.task_motion.resolve_demo_path", lambda *a, **k: demo_path)
    monkeypatch.setattr("sarm_hand.task_motion.ensure_port", lambda port, label: port or "/dev/tty.test")
    monkeypatch.setattr("sarm_hand.task_motion.require_all_motors", lambda *a, **k: None)
    monkeypatch.setattr("sarm_hand.task_motion.build_robot", lambda *a, **k: FakeRobot())
    monkeypatch.setattr("sarm_hand.task_motion.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "sarm_hand.task_motion.require_teleop_calibrations",
        lambda cfg: (leader_cal, follower_cal),
    )

    from sarm_hand.task_motion import replay_task_motion

    replay_task_motion(task_slug=slug, demo_id="demo_test", pause_s=0)

    assert sent_actions
    assert sent_actions[0]["shoulder_pan.pos"] != 25.0


def test_replay_skips_cameras(monkeypatch, tmp_path: Path):
    cfg = ProjectConfig.load()
    slug = "pick_and_place"
    demo_path = tmp_path / slug / "demo_test.json"
    demo_path.parent.mkdir(parents=True)
    demo_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task": "Pick and place",
                "task_slug": slug,
                "demo_id": "demo_test",
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

    captured: dict[str, bool] = {}

    class FakeRobot:
        config = type("Cfg", (), {"max_relative_target": 10})()
        is_calibrated = True
        is_connected = True

        def send_action(self, action):
            return action

        def disconnect(self):
            pass

    def fake_build_robot(port, config, *, use_cameras=True):
        captured["use_cameras"] = use_cameras
        return FakeRobot()

    monkeypatch.setattr("sarm_hand.task_motion.ProjectConfig.load", lambda: cfg)
    monkeypatch.setattr("sarm_hand.task_motion.resolve_demo_path", lambda *a, **k: demo_path)
    monkeypatch.setattr("sarm_hand.task_motion.ensure_port", lambda port, label: port or "/dev/tty.test")
    monkeypatch.setattr("sarm_hand.task_motion.require_all_motors", lambda *a, **k: None)
    monkeypatch.setattr("sarm_hand.task_motion.build_robot", fake_build_robot)
    monkeypatch.setattr("sarm_hand.task_motion.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "sarm_hand.task_motion.require_teleop_calibrations",
        lambda cfg: (
            {j: {"range_min": 1000, "range_max": 3000, "drive_mode": 0} for j in (
                "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
            )},
            {j: {"range_min": 0, "range_max": 4095, "drive_mode": 0} for j in (
                "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
            )},
        ),
    )

    from sarm_hand.task_motion import replay_task_motion

    replay_task_motion(task_slug=slug, demo_id="demo_test", pause_s=0)

    assert captured.get("use_cameras") is False
