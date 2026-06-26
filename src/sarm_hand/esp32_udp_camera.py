"""LeRobot camera backend for ESP32-CAM raw UDP JPEG streams."""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any

import cv2
import numpy as np
from lerobot.cameras.camera import Camera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from .esp32_udp_stream import (
    SUBSCRIBE,
    UdpStreamOptions,
    UdpStreamReceiver,
    decode_jpeg,
    open_udp_socket,
    should_hold_udp_frames,
    subscribe_loop,
)

logger = logging.getLogger(__name__)

# No slashes — lerobot-record CLI collapses esp32udp:// to esp32udp:/ and breaks routing.
ESP32_UDP_PREFIX = "esp32udp:"


def normalize_esp32_udp_source(source: str | int) -> str | None:
    """Return canonical esp32udp:host:port, or None if not an ESP32 UDP source."""
    if not isinstance(source, str):
        return None
    text = source.strip()
    if not text.startswith("esp32udp"):
        return None
    if text.startswith("esp32udp://"):
        rest = text[len("esp32udp://") :]
    elif text.startswith("esp32udp:/"):
        rest = text[len("esp32udp:/") :]
    elif text.startswith(ESP32_UDP_PREFIX):
        rest = text[len(ESP32_UDP_PREFIX) :]
    else:
        return None
    host, sep, port_str = rest.rpartition(":")
    if not sep or not host:
        return None
    return f"{ESP32_UDP_PREFIX}{host}:{port_str}"


_UDP_OPTIONS: dict[str, UdpStreamOptions] = {}


def register_udp_options(source: str, options: UdpStreamOptions) -> None:
    key = normalize_esp32_udp_source(source)
    if key is None:
        raise ValueError(f"Not an ESP32 UDP source: {source!r}")
    _UDP_OPTIONS[key] = options


def parse_udp_source(source: str) -> UdpStreamOptions:
    key = normalize_esp32_udp_source(source)
    if key is None:
        raise ValueError(f"Not an ESP32 UDP source: {source!r}")
    if key in _UDP_OPTIONS:
        return _UDP_OPTIONS[key]
    rest = key[len(ESP32_UDP_PREFIX) :]
    host, sep, port_str = rest.rpartition(":")
    if not sep:
        raise ValueError(f"Invalid ESP32 UDP source (expected host:port): {source!r}")
    return UdpStreamOptions(host=host, port=int(port_str))


def is_esp32_udp_source(source: str | int) -> bool:
    return normalize_esp32_udp_source(source) is not None


class Esp32UdpCamera(Camera):
    """Decode ESP32 chunked UDP JPEG on the host; exposes LeRobot camera API."""

    def __init__(self, config: OpenCVCameraConfig):
        super().__init__(config)
        self.config = config
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.warmup_s = config.warmup_s
        self.index_or_path = config.index_or_path
        self._opts = parse_udp_source(str(config.index_or_path))
        self._sock: socket.socket | None = None
        self._receiver: UdpStreamReceiver | None = None
        self._subscribe_stop: threading.Event | None = None
        self._subscribe_thread: threading.Thread | None = None
        self._connected = False
        self._latest_rgb: np.ndarray | None = None
        self._latest_time = 0.0
        self._shown_jpg: bytes | None = None
        self._hold_emit_count = 0
        self._sarm_black_fallback = False

    def _target_fps(self) -> float:
        if self._opts.target_fps is not None:
            return max(1.0, float(self._opts.target_fps))
        return max(1.0, float(self.fps or 5))

    def _net_fps(self) -> float:
        if self._receiver is None:
            return 0.0
        return float(self._receiver.fps)

    def _should_hold_frames(self) -> bool:
        return should_hold_udp_frames(
            target_fps=self._target_fps(),
            net_fps=self._net_fps(),
            hold_enabled=self._opts.hold_fps,
        )

    def _output_frame(self, *, max_age_ms: int) -> np.ndarray:
        """Return latest decode, duplicating the previous frame when the stream is below target fps."""
        self._refresh_frame()
        if self._latest_rgb is None:
            raise RuntimeError(f"{self} connected but no UDP frame decoded yet")

        if self._should_hold_frames():
            self._hold_emit_count += 1
            return self._latest_rgb.copy()

        age_ms = (time.monotonic() - self._latest_time) * 1000.0
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is {age_ms:.0f} ms old (max {max_age_ms} ms)")
        self._hold_emit_count = 0
        return self._latest_rgb.copy()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._opts.host}:{self._opts.port})"

    @property
    def is_connected(self) -> bool:
        if getattr(self, "_sarm_black_fallback", False):
            return True
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        cv2.setNumThreads(1)
        self._sock = open_udp_socket()
        local_port = self._sock.getsockname()[1]
        esp_addr = (self._opts.host, self._opts.port)

        self._subscribe_stop = threading.Event()
        self._subscribe_thread = threading.Thread(
            target=subscribe_loop,
            args=(self._sock, esp_addr, self._subscribe_stop, self._opts.subscribe_interval_s),
            daemon=True,
        )
        self._subscribe_thread.start()
        self._sock.sendto(SUBSCRIBE, esp_addr)

        self._receiver = UdpStreamReceiver(self._sock, fps_window=self._opts.fps_window)
        self._receiver.start()
        self._connected = True

        grace_s = max(float(self.warmup_s or 0), self._opts.connect_grace_s)
        if warmup and grace_s > 0:
            deadline = time.time() + grace_s
            while time.time() < deadline:
                self._sock.sendto(SUBSCRIBE, esp_addr)
                if self._refresh_frame():
                    break
                time.sleep(0.05)
            if self._latest_rgb is None:
                raise ConnectionError(
                    f"{self} failed to receive JPEG frames within {grace_s:.0f}s "
                    f"(listening UDP :{local_port}, subscribed to {self._opts.host}:{self._opts.port})"
                )

        net_fps = self._receiver.fps if self._receiver else 0.0
        hold_note = ""
        if self._opts.hold_fps and should_hold_udp_frames(
            target_fps=self._target_fps(), net_fps=net_fps, hold_enabled=True
        ):
            hold_note = f", hold → {self._target_fps():.0f} fps"
        logger.info(f"{self} connected (local UDP :{local_port}, net ~{net_fps:.1f} fps{hold_note}).")

    def _refresh_frame(self) -> bool:
        if self._receiver is None:
            return False
        jpg, stale, _net_fps = self._receiver.get_latest(stale_sec=self._opts.stale_sec)
        if jpg is None or stale or jpg is self._shown_jpg:
            return jpg is not None and not stale
        rgb = decode_jpeg(
            jpg,
            rotate_180=self._opts.rotate_180,
            flip_horizontal=self._opts.flip_horizontal,
            width=self.width,
            height=self.height,
        )
        if rgb is None:
            return False
        self._latest_rgb = rgb
        self._latest_time = time.monotonic()
        self._shown_jpg = jpg
        return True

    @check_if_not_connected
    def read(self) -> np.ndarray:
        deadline = time.time() + max(self.warmup_s or 3.0, 1.0)
        while time.time() < deadline:
            try:
                return self._output_frame(max_age_ms=int(1000 / self._target_fps()) + 50)
            except (RuntimeError, TimeoutError):
                time.sleep(0.01)
        raise TimeoutError(f"{self} timed out waiting for UDP frame")

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                return self._output_frame(max_age_ms=int(timeout_ms))
            except (RuntimeError, TimeoutError):
                time.sleep(0.005)
        raise TimeoutError(f"{self} timed out waiting for UDP frame ({timeout_ms} ms)")

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        hold_max_age = max(max_age_ms, int(1000 / self._target_fps()) * 3)
        return self._output_frame(max_age_ms=hold_max_age if self._should_hold_frames() else max_age_ms)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._subscribe_stop is not None:
            self._subscribe_stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._receiver = None
        self._subscribe_thread = None
        self._subscribe_stop = None
        self._connected = False
        self._latest_rgb = None
        self._shown_jpg = None
        logger.info(f"{self} disconnected.")
