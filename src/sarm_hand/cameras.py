"""Camera integration: USB devices and HTTP/RTSP streams via LeRobot OpenCV backend."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import CameraSettings, ProjectConfig

STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://")
STREAM_TYPES = frozenset({"http", "https", "rtsp", "rtmp", "stream"})

# macOS/Windows: stop scanning after this many consecutive misses (not 0..59).
_MAX_CONSECUTIVE_MISSES = 2
_MAX_INDEX_DARWIN = 8
_MAX_INDEX_OTHER = 16


@contextlib.contextmanager
def _quiet_opencv():
    """Suppress OpenCV stderr noise while probing camera indices."""
    import cv2

    saved_level = None
    if hasattr(cv2, "getLogLevel") and hasattr(cv2, "setLogLevel"):
        silent = getattr(cv2, "LOG_LEVEL_SILENT", 0)
        saved_level = cv2.getLogLevel()
        cv2.setLogLevel(silent)

    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    try:
        yield
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)
        os.close(devnull)
        if saved_level is not None:
            cv2.setLogLevel(saved_level)


def _probe_usb_camera(cv2, target: int | str) -> dict[str, Any] | None:
    camera = cv2.VideoCapture(target)
    try:
        if not camera.isOpened():
            return None

        default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        default_fps = camera.get(cv2.CAP_PROP_FPS)
        default_format = camera.get(cv2.CAP_PROP_FORMAT)
        default_fourcc_code = int(camera.get(cv2.CAP_PROP_FOURCC))
        default_fourcc = "".join(
            chr((default_fourcc_code >> 8 * i) & 0xFF) for i in range(4)
        ).strip("\x00")

        return {
            "name": f"OpenCV Camera @ {target}",
            "type": "OpenCV",
            "id": target,
            "backend_api": camera.getBackendName(),
            "default_stream_profile": {
                "format": default_format,
                "fourcc": default_fourcc or "?",
                "width": default_width,
                "height": default_height,
                "fps": default_fps,
            },
        }
    finally:
        camera.release()


def find_usb_cameras() -> list[dict[str, Any]]:
    """Discover USB cameras without spamming OpenCV errors."""
    import cv2

    found: list[dict[str, Any]] = []

    with _quiet_opencv():
        if platform.system() == "Linux":
            targets = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            for path in targets:
                info = _probe_usb_camera(cv2, str(path))
                if info:
                    found.append(info)
        else:
            max_index = _MAX_INDEX_DARWIN if platform.system() == "Darwin" else _MAX_INDEX_OTHER
            consecutive_misses = 0
            for index in range(max_index):
                info = _probe_usb_camera(cv2, index)
                if info:
                    found.append(info)
                    consecutive_misses = 0
                else:
                    consecutive_misses += 1
                    if consecutive_misses >= _MAX_CONSECUTIVE_MISSES:
                        break

    return found


def _macos_camera_names() -> list[str]:
    """Best-effort human-readable camera names from system_profiler."""
    if platform.system() != "Darwin":
        return []

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        names: list[str] = []
        for entry in data.get("SPCameraDataType", []):
            for key in entry:
                if key.startswith("_"):
                    continue
                names.append(key)
        return names
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return []


def list_usb_cameras() -> None:
    """List USB cameras detected by OpenCV."""
    cameras = find_usb_cameras()
    if not cameras:
        print("No USB cameras found.")
        print("Tips:")
        print("  - Check the USB cable and permissions")
        if platform.system() == "Darwin":
            print("  - macOS: System Settings → Privacy & Security → Camera → allow Terminal/IDE")
        elif platform.system() == "Linux":
            print("  - Linux: ensure your user is in the 'video' group")
        print("  - For HTTP streams, use type: http with a url in config/default.yaml")
        return

    print(f"Found {len(cameras)} USB camera(s):\n")
    mac_names = _macos_camera_names()
    for i, info in enumerate(cameras):
        profile = info.get("default_stream_profile", {})
        label = info["name"]
        if i < len(mac_names):
            label = f"{mac_names[i]} ({info['id']!r})"
        print(f"  {label}")
        print(f"    id:      {info['id']!r}")
        print(f"    backend: {info.get('backend_api', 'unknown')}")
        print(
            f"    default: {profile.get('width')}x{profile.get('height')} "
            f"@ {profile.get('fps')} fps ({profile.get('fourcc', '?')})"
        )
        print()


def is_stream_url(source: str) -> bool:
    return source.lower().startswith(STREAM_PREFIXES)


def is_stream_camera(cam: CameraSettings) -> bool:
    if cam.type.lower() in STREAM_TYPES:
        return True
    source = cam.url or (cam.index_or_path if isinstance(cam.index_or_path, str) else None)
    return bool(source and is_stream_url(source))


def resolve_camera_source(cam: CameraSettings) -> int | str | Path:
    """Return the OpenCV VideoCapture source for a camera config."""
    cam_type = cam.type.lower()

    if cam_type in STREAM_TYPES:
        if not cam.url:
            raise ValueError(f"Camera type '{cam.type}' requires 'url' in config")
        return cam.url

    if cam.url:
        return cam.url

    if isinstance(cam.index_or_path, str) and is_stream_url(cam.index_or_path):
        return cam.index_or_path

    if isinstance(cam.index_or_path, str) and cam.index_or_path.startswith("/"):
        return Path(cam.index_or_path)

    return cam.index_or_path


_STREAM_MAX_FRAME_AGE_MS: dict[str, int] = {}


def _default_stream_max_frame_age(fps: float | int | None) -> int:
    """LeRobot read_latest() defaults to 500 ms — too tight for HTTP/MJPEG jitter."""
    rate = max(float(fps or 10), 1.0)
    return max(1500, int(1000.0 / rate * 15))


def effective_camera_settings(cam: CameraSettings) -> CameraSettings:
    """Apply auto_resolution and platform defaults before opening a camera."""
    updates: dict[str, Any] = {}
    if cam.auto_resolution:
        updates.update(width=None, height=None, fps=None)
    if is_stream_camera(cam):
        # Declared width/height/fps satisfy LeRobot robot config; connect uses native
        # stream resolution via install_stream_camera_patch().
        if cam.width is None:
            updates["width"] = 640
        if cam.height is None:
            updates["height"] = 480
        if cam.fps is None:
            updates["fps"] = 10
        if cam.warmup_s is None:
            updates["warmup_s"] = 3
        if cam.max_frame_age_ms is None:
            updates["max_frame_age_ms"] = _default_stream_max_frame_age(cam.fps)
    elif cam.warmup_s is None and platform.system() == "Darwin":
        updates["warmup_s"] = 3
    if updates:
        return replace(cam, **updates)
    return cam


_STREAM_CAMERA_PATCHED = False


def install_stream_camera_patch() -> None:
    """Skip OpenCV set(width/height/fps) for HTTP/RTSP URLs (set() fails on streams)."""
    global _STREAM_CAMERA_PATCHED
    if _STREAM_CAMERA_PATCHED:
        return

    import cv2
    from lerobot.cameras.opencv import camera_opencv
    from lerobot.utils.errors import DeviceNotConnectedError

    original = camera_opencv.OpenCVCamera._configure_capture_settings

    def _configure_capture_settings(self) -> None:
        if not is_stream_url(str(self.index_or_path)):
            original(self)
            return

        if self.config.fourcc is not None:
            self._validate_fourcc()
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        default_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.width, self.height = default_width, default_height
        self.capture_width, self.capture_height = default_width, default_height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.width, self.height = default_height, default_width
            self.capture_width, self.capture_height = default_width, default_height

        actual_fps = float(self.videocapture.get(cv2.CAP_PROP_FPS))
        if actual_fps > 0:
            self.fps = actual_fps
        elif self.fps is None:
            self.fps = 10.0

    camera_opencv.OpenCVCamera._configure_capture_settings = _configure_capture_settings

    original_read_latest = camera_opencv.OpenCVCamera.read_latest

    def read_latest(self, max_age_ms: int = 500):
        url = str(self.index_or_path)
        if is_stream_url(url):
            stream_max = _STREAM_MAX_FRAME_AGE_MS.get(url)
            if stream_max is None:
                stream_max = _default_stream_max_frame_age(self.fps)
            max_age_ms = max(max_age_ms, stream_max)
        try:
            return original_read_latest(self, max_age_ms)
        except TimeoutError:
            if is_stream_url(url):
                return self.read()
            raise

    camera_opencv.OpenCVCamera.read_latest = read_latest
    _STREAM_CAMERA_PATCHED = True


def native_usb_profile(source: int | str) -> dict[str, Any] | None:
    for info in find_usb_cameras():
        if info["id"] == source:
            return info.get("default_stream_profile")
    return None


def camera_to_lerobot_dict(cam: CameraSettings) -> dict[str, Any]:
    """Serialize one camera for lerobot-record CLI flags."""
    cam = effective_camera_settings(cam)
    payload: dict[str, Any] = {
        "type": "opencv",
        "index_or_path": resolve_camera_source(cam),
    }
    if cam.fps is not None:
        payload["fps"] = cam.fps
    if cam.width is not None:
        payload["width"] = cam.width
    if cam.height is not None:
        payload["height"] = cam.height
    if cam.warmup_s is not None:
        payload["warmup_s"] = cam.warmup_s
    if cam.fourcc is not None:
        payload["fourcc"] = cam.fourcc
    return payload


def build_lerobot_camera_config(cam: CameraSettings):
    """Build a LeRobot OpenCVCameraConfig from project settings."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    install_stream_camera_patch()
    cam = effective_camera_settings(cam)
    source = resolve_camera_source(cam)
    if is_stream_camera(cam) and cam.max_frame_age_ms is not None:
        _STREAM_MAX_FRAME_AGE_MS[str(source)] = cam.max_frame_age_ms
    kwargs: dict[str, Any] = {"index_or_path": source}
    if cam.fps is not None:
        kwargs["fps"] = cam.fps
    if cam.width is not None:
        kwargs["width"] = cam.width
    if cam.height is not None:
        kwargs["height"] = cam.height
    if cam.warmup_s is not None:
        kwargs["warmup_s"] = cam.warmup_s
    if cam.fourcc is not None:
        kwargs["fourcc"] = cam.fourcc
    return OpenCVCameraConfig(**kwargs)


def build_robot_camera_configs(cfg: ProjectConfig) -> dict:
    """Build LeRobot camera configs keyed by name (for teleop / recording)."""
    return {name: build_lerobot_camera_config(cam) for name, cam in cfg.cameras.items()}


def cameras_lerobot_dict(cfg: ProjectConfig) -> dict[str, dict[str, Any]]:
    return {name: camera_to_lerobot_dict(cam) for name, cam in cfg.cameras.items()}


def describe_camera(name: str, cam: CameraSettings) -> str:
    source = resolve_camera_source(cam)
    kind = "stream" if is_stream_camera(cam) else "usb"
    effective = effective_camera_settings(cam)
    parts = [f"{name}: {kind} ({cam.type})", f"source={source!r}"]
    if effective.auto_resolution:
        parts.append("auto_resolution")
    elif effective.width and effective.height:
        parts.append(f"{effective.width}x{effective.height}")
    if effective.fps:
        parts.append(f"@{effective.fps}fps")
    if is_stream_camera(cam):
        parts.append("(native at connect)")
    return " ".join(parts)


def _camera_failure_hints(settings: CameraSettings) -> str:
    hints = []
    if not is_stream_camera(settings):
        profile = native_usb_profile(resolve_camera_source(settings))
        if profile:
            hints.append(
                f"  Native device profile: {profile.get('width')}x{profile.get('height')} "
                f"@ {profile.get('fps')} fps"
            )
        if platform.system() == "Darwin":
            hints.extend([
                "  macOS: System Settings → Privacy & Security → Camera → allow your terminal",
                "  macOS: set auto_resolution: true (or width/height/fps: null) in config",
                "  macOS: close FaceTime/Zoom if the camera is in use elsewhere",
            ])
        elif not settings.auto_resolution and (settings.width or settings.fps):
            hints.append(
                "  Try auto_resolution: true if the camera rejects the requested size/fps"
            )
    return "\n".join(hints)


def _read_one_frame(settings: CameraSettings):
    """Connect via LeRobot and return one RGB frame."""
    import logging

    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    logging.getLogger("lerobot.cameras.opencv.camera_opencv").setLevel(logging.ERROR)

    camera = OpenCVCamera(build_lerobot_camera_config(settings))
    try:
        with _quiet_opencv():
            camera.connect()
            return camera.read()
    finally:
        if camera.is_connected:
            camera.disconnect()


def _settings_from_cli(
    *,
    name: str | None,
    index: int | None,
    url: str | None,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> CameraSettings:
    cfg = ProjectConfig.load()

    if name:
        if name not in cfg.cameras:
            available = ", ".join(cfg.cameras) or "(none)"
            raise ValueError(f"Unknown camera '{name}'. Configured: {available}")
        return cfg.cameras[name]

    if url:
        return CameraSettings(type="http", url=url, width=width, height=height, fps=fps)

    if index is not None:
        return CameraSettings(
            type="opencv", index_or_path=index, width=width, height=height, fps=fps
        )

    if cfg.cameras:
        first_name = next(iter(cfg.cameras))
        print(f"No camera specified — using configured camera '{first_name}'")
        return cfg.cameras[first_name]

    raise ValueError("Specify --name, --index, --url, or add cameras to config/default.yaml")


def preview_camera(
    *,
    name: str | None = None,
    index: int | None = None,
    url: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    output: str | None = None,
    seconds: float = 5.0,
    show_window: bool = True,
) -> None:
    """Capture frames from a USB camera or HTTP stream."""
    import cv2
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    try:
        settings = _settings_from_cli(
            name=name, index=index, url=url, width=width, height=height, fps=fps
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    config = build_lerobot_camera_config(settings)
    camera = OpenCVCamera(config)

    print(describe_camera(name or "preview", settings))
    print("Connecting...")

    try:
        camera.connect()
    except ConnectionError as exc:
        print(f"Failed to connect: {exc}", file=sys.stderr)
        if is_stream_camera(settings):
            print(
                "\nHTTP/RTSP stream tips:",
                "  - Confirm the URL opens in a browser or VLC",
                "  - For MJPEG, try appending /video or /shot.jpg per your camera docs",
                "  - Set width/height/fps to null in config for auto-detected stream size",
                sep="\n",
                file=sys.stderr,
            )
        sys.exit(1)

    print(f"Connected — {camera.width}x{camera.height} @ {camera.fps} fps")
    output_path = Path(output) if output else None
    deadline = time.time() + seconds
    frame_count = 0

    try:
        while time.time() < deadline:
            frame = camera.read()
            frame_count += 1

            if output_path and frame_count == 1:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
                cv2.imwrite(str(output_path), bgr)
                print(f"Saved snapshot → {output_path}")

            if show_window:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("sarm-hand camera preview (q to quit)", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if show_window:
            cv2.destroyAllWindows()
        camera.disconnect()

    print(f"Captured {frame_count} frame(s)")


def test_configured_cameras() -> None:
    """Connect to each camera in config/default.yaml and grab one frame."""
    cfg = ProjectConfig.load()
    if not cfg.cameras:
        print("No cameras configured in config/default.yaml")
        sys.exit(1)

    print(f"Testing {len(cfg.cameras)} configured camera(s)...\n")
    failed = False

    for name, settings in cfg.cameras.items():
        print(describe_camera(name, settings))
        attempts: list[tuple[str, CameraSettings]] = [
            ("configured", effective_camera_settings(settings)),
        ]
        if not settings.auto_resolution and not is_stream_camera(settings):
            auto = replace(settings, auto_resolution=True, width=None, height=None, fps=None)
            attempts.append(("auto_resolution", effective_camera_settings(auto)))

        last_error: Exception | None = None
        for label, attempt_settings in attempts:
            try:
                frame = _read_one_frame(attempt_settings)
                h, w = frame.shape[:2]
                if label != "configured":
                    print(f"  OK ({label}) — frame {w}x{h}, {frame.dtype}")
                    print(
                        "  Tip: add auto_resolution: true to config for reliable capture\n"
                    )
                else:
                    print(f"  OK — frame {w}x{h}, {frame.dtype}\n")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if label == "configured" and len(attempts) > 1:
                    print(f"  configured settings failed ({exc}); retrying with auto_resolution...")

        if last_error is not None:
            failed = True
            print(f"  FAIL — {last_error}\n", file=sys.stderr)
            hints = _camera_failure_hints(settings)
            if hints:
                print(hints, file=sys.stderr)

    if failed:
        sys.exit(1)
    print("All cameras passed.")
