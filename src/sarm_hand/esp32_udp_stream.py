"""Receive chunked JPEG frames from ESP32-CAM over UDP (client-side decode).

Adapted from esp32cam/viewer/udp_viewer.py — assembles UDP chunks into JPEG,
then OpenCV decodes to BGR/RGB on the host.

Chunk header (little-endian, 12 bytes):
  uint32 magic = 0xCAFEA002
  uint16 frame_id
  uint16 total_len
  uint8  chunk_idx
  uint8  chunk_total
  uint16 payload_len
  uint8[payload_len]
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

SUBSCRIBE_MAGIC = 0xCAFE0001
CHUNK_MAGIC = 0xCAFEA002
CHUNK_HDR = struct.Struct("<IHHBBH")
SUBSCRIBE = struct.pack("<I", SUBSCRIBE_MAGIC)
DEFAULT_STALE_SEC = 0.6
DEFAULT_CONNECT_GRACE_SEC = 10.0
DEFAULT_FPS_WINDOW = 5


def should_hold_udp_frames(*, target_fps: float, net_fps: float, hold_enabled: bool) -> bool:
    """True when the stream is slower than target and duplicates should fill the gap."""
    if not hold_enabled or target_fps <= 0:
        return False
    if net_fps <= 0:
        return True
    return net_fps < target_fps - 0.25


@dataclass(frozen=True)
class UdpStreamOptions:
    host: str
    port: int = 82
    rotate_180: bool = True
    flip_horizontal: bool = True
    stale_sec: float = DEFAULT_STALE_SEC
    subscribe_interval_s: float = 2.0
    connect_grace_s: float = DEFAULT_CONNECT_GRACE_SEC
    fps_window: int = DEFAULT_FPS_WINDOW
    hold_fps: bool = True
    target_fps: float | None = None


class FrameAssembler:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.frame_id: int | None = None
        self.total_len = 0
        self.chunk_total = 0
        self.chunks: dict[int, bytes] = {}

    def feed(self, packet: bytes) -> bytes | None:
        if len(packet) < CHUNK_HDR.size:
            return None

        magic, frame_id, total_len, chunk_idx, chunk_total, payload_len = CHUNK_HDR.unpack_from(packet)
        if magic != CHUNK_MAGIC:
            return None

        payload = packet[CHUNK_HDR.size : CHUNK_HDR.size + payload_len]
        if len(payload) != payload_len:
            return None

        if chunk_idx == 0 or frame_id != self.frame_id:
            self.frame_id = frame_id
            self.total_len = total_len
            self.chunk_total = chunk_total
            self.chunks = {}

        self.chunks[chunk_idx] = payload

        if len(self.chunks) != self.chunk_total:
            return None

        parts = [self.chunks.get(i) for i in range(self.chunk_total)]
        if any(p is None for p in parts):
            return None

        jpg = b"".join(parts)
        self.reset()
        if len(jpg) == total_len:
            return jpg
        return None


class UdpStreamReceiver(threading.Thread):
    """Background thread: recv UDP packets, keep latest complete JPEG."""

    def __init__(self, sock: socket.socket, *, fps_window: int = DEFAULT_FPS_WINDOW):
        super().__init__(daemon=True)
        self.sock = sock
        self.lock = threading.Lock()
        self.latest_jpg: bytes | None = None
        self.last_rx_time = 0.0
        self.assembler = FrameAssembler()
        self._frame_times: deque[float] = deque(maxlen=max(2, fps_window))
        self.fps = 0.0

    def run(self) -> None:
        sock = self.sock
        while True:
            try:
                packet, _addr = sock.recvfrom(2048)
            except OSError:
                break
            jpg = self.assembler.feed(packet)
            if jpg is None:
                continue
            now = time.monotonic()
            with self.lock:
                self.latest_jpg = jpg
                self.last_rx_time = now
                self._frame_times.append(now)
                if len(self._frame_times) >= 2:
                    span = self._frame_times[-1] - self._frame_times[0]
                    if span > 0:
                        self.fps = (len(self._frame_times) - 1) / span

    def get_latest(self, *, stale_sec: float) -> tuple[bytes | None, bool, float]:
        with self.lock:
            stale = (
                (time.monotonic() - self.last_rx_time) > stale_sec if self.last_rx_time else True
            )
            return self.latest_jpg, stale, self.fps


def orient_frame(frame: np.ndarray, *, rotate_180: bool, flip_horizontal: bool) -> np.ndarray:
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if flip_horizontal:
        frame = cv2.flip(frame, 1)
    return frame


def decode_jpeg(
    jpg: bytes,
    *,
    rotate_180: bool,
    flip_horizontal: bool,
    width: int | None,
    height: int | None,
) -> np.ndarray | None:
    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    frame = orient_frame(frame, rotate_180=rotate_180, flip_horizontal=flip_horizontal)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if width and height and (frame.shape[1], frame.shape[0]) != (width, height):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def subscribe_loop(sock: socket.socket, esp_addr: tuple[str, int], stop: threading.Event, interval_s: float) -> None:
    while not stop.wait(interval_s):
        try:
            sock.sendto(SUBSCRIBE, esp_addr)
        except OSError:
            break


def open_udp_socket(*, rcvbuf: int = 512 * 1024) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    sock.bind(("", 0))
    return sock
