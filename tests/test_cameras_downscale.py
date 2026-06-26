"""Tests for USB camera downscale-from-native capture."""

from __future__ import annotations

from sarm_hand.cameras import (
    _wants_downscale,
    build_lerobot_camera_config,
    camera_to_lerobot_dict,
)
from sarm_hand.config import CameraSettings


def test_wants_downscale_when_auto_resolution_and_output_size():
    cam = CameraSettings(
        type="opencv",
        index_or_path=0,
        auto_resolution=True,
        width=640,
        height=480,
    )
    assert _wants_downscale(cam)


def test_no_downscale_for_direct_capture():
    cam = CameraSettings(type="opencv", index_or_path=0, width=640, height=480, fps=30)
    assert not _wants_downscale(cam)


def test_lerobot_dict_keeps_output_dimensions_when_downscaling():
    cam = CameraSettings(
        type="opencv",
        index_or_path=2,
        auto_resolution=True,
        width=640,
        height=480,
        warmup_s=3,
    )
    payload = camera_to_lerobot_dict(cam)
    assert payload["width"] == 640
    assert payload["height"] == 480
    assert payload["fps"] == 30
    assert payload["index_or_path"] == 2


def test_build_lerobot_camera_config_satisfies_robot_validation():
    from lerobot.robots.so_follower import SOFollowerRobotConfig

    cam = CameraSettings(
        type="opencv",
        index_or_path=0,
        auto_resolution=True,
        width=640,
        height=480,
        fps=None,
        warmup_s=3,
    )
    cfg = build_lerobot_camera_config(cam)
    assert cfg.width == 640
    assert cfg.height == 480
    assert cfg.fps == 30
    SOFollowerRobotConfig(
        port="/dev/null",
        id="test",
        cameras={"front": cfg},
    )
