"""Tests for ESP32-CAM UDP JPEG reassembly."""

from __future__ import annotations

import struct

import numpy as np

from sarm_hand.cameras import (
    build_lerobot_camera_config,
    camera_to_lerobot_dict,
    is_udp_camera,
    resolve_udp_endpoint,
)
from sarm_hand.config import CameraSettings
from sarm_hand.esp32_udp_stream import CHUNK_HDR, CHUNK_MAGIC, FrameAssembler, decode_jpeg, should_hold_udp_frames


def _chunk(frame_id: int, total_len: int, idx: int, total: int, payload: bytes) -> bytes:
    header = struct.pack("<IHHBBH", CHUNK_MAGIC, frame_id, total_len, idx, total, len(payload))
    return header + payload


def test_frame_assembler_rebuilds_jpeg():
    jpg = b"\xff\xd8\xff\xd9"
    asm = FrameAssembler()
    assert asm.feed(_chunk(1, len(jpg), 0, 1, jpg)) == jpg


def test_decode_jpeg_returns_rgb():
    import cv2

    bgr = np.zeros((10, 12, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    rgb = decode_jpeg(
        encoded.tobytes(),
        rotate_180=False,
        flip_horizontal=False,
        width=12,
        height=10,
    )
    assert rgb is not None
    assert rgb.shape == (10, 12, 3)
    assert rgb.dtype == np.uint8


def test_should_hold_udp_frames_when_stream_slower_than_target():
    assert should_hold_udp_frames(target_fps=5.0, net_fps=2.0, hold_enabled=True)
    assert should_hold_udp_frames(target_fps=5.0, net_fps=0.0, hold_enabled=True)
    assert not should_hold_udp_frames(target_fps=5.0, net_fps=5.0, hold_enabled=True)
    assert not should_hold_udp_frames(target_fps=5.0, net_fps=2.0, hold_enabled=False)


def test_udp_output_frame_duplicates_when_holding():
    from sarm_hand.esp32_udp_camera import Esp32UdpCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cfg = OpenCVCameraConfig(
        index_or_path="esp32udp:192.168.0.58:82",
        width=4,
        height=3,
        fps=5,
    )
    cam = Esp32UdpCamera(cfg)
    cam._connected = True
    cam._latest_rgb = np.full((3, 4, 3), 7, dtype=np.uint8)
    cam._latest_time = 0.0
    cam._receiver = type(
        "R",
        (),
        {"fps": 2.0, "get_latest": lambda self, stale_sec: (None, True, 2.0)},
    )()

    out1 = cam._output_frame(max_age_ms=100)
    out2 = cam._output_frame(max_age_ms=100)
    assert out1.shape == (3, 4, 3)
    assert np.array_equal(out1, out2)
    assert cam._hold_emit_count == 2


def test_opencv_delegates_esp32udp_source():
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    from sarm_hand.cameras import install_udp_camera_patch
    from sarm_hand.esp32_udp_camera import Esp32UdpCamera, is_esp32_udp_source, normalize_esp32_udp_source

    install_udp_camera_patch()
    cfg = OpenCVCameraConfig(
        index_or_path="esp32udp:192.168.0.58:82",
        width=640,
        height=480,
        fps=5,
        warmup_s=1,
    )
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    cam = OpenCVCamera(cfg)
    assert is_esp32_udp_source(str(cam.index_or_path))
    assert not hasattr(cam, "_sarm_udp_active")
    # Delegate is created lazily on connect; verify class routing exists.
    assert getattr(OpenCVCamera, "_sarm_udp_delegate_patched", False)
    delegate = Esp32UdpCamera(cfg)
    assert delegate.fps == 5

    # lerobot-record CLI collapses // to / — still recognized.
    assert normalize_esp32_udp_source("esp32udp:/192.168.0.58:82") == "esp32udp:192.168.0.58:82"
    assert is_esp32_udp_source("esp32udp:/192.168.0.58:82")


def test_udp_camera_config_serializes_to_esp32udp_source():
    cam = CameraSettings(type="udp", host="192.168.0.58", port=82, width=640, height=480, fps=5)
    assert is_udp_camera(cam)
    assert resolve_udp_endpoint(cam) == ("192.168.0.58", 82)
    payload = camera_to_lerobot_dict(cam)
    assert payload["index_or_path"] == "esp32udp:192.168.0.58:82"
    cfg = build_lerobot_camera_config(cam)
    assert str(cfg.index_or_path) == "esp32udp:192.168.0.58:82"
