"""Camera integration: USB, HTTP/RTSP, and ESP32 UDP JPEG via LeRobot backends."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import CameraSettings, ProjectConfig

logger = logging.getLogger(__name__)

STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://")
STREAM_TYPES = frozenset({"http", "https", "rtsp", "rtmp", "stream"})
UDP_TYPES = frozenset({"udp", "esp32_udp", "esp32udp"})

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


# Common USB resolutions to try (width, height). Order: high → low for display.
COMMON_PROBE_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1600, 900),
    (1280, 720),
    (1024, 576),
    (960, 540),
    (800, 600),
    (640, 480),
    (640, 360),
    (480, 270),
    (320, 240),
)


def _opencv_usb_backend() -> int | None:
    import cv2

    if platform.system() == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return None


def _probe_usb_resolution(
    cv2,
    target: int | str,
    width: int,
    height: int,
    *,
    backend: int | None = None,
    fps: float | None = None,
) -> dict[str, Any] | None:
    """Try one capture size; return actual dims + frame size when a frame is read."""
    api = backend if backend is not None else _opencv_usb_backend()
    if api is None:
        api = cv2.CAP_ANY
    camera = cv2.VideoCapture(target, api)
    try:
        if not camera.isOpened():
            return None
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None and fps > 0:
            camera.set(cv2.CAP_PROP_FPS, float(fps))
        actual_w = int(round(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_h = int(round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = float(camera.get(cv2.CAP_PROP_FPS))
        frame_w = frame_h = None
        for _ in range(25):
            ok, frame = camera.read()
            if ok and frame is not None:
                frame_h, frame_w = frame.shape[:2]
                break
            time.sleep(0.04)
        if frame_w is None:
            return None
        return {
            "requested": (width, height),
            "actual": (actual_w, actual_h),
            "frame": (frame_w, frame_h),
            "fps": actual_fps if actual_fps > 0 else None,
        }
    finally:
        camera.release()


def probe_usb_camera_resolutions(
    target: int | str,
    *,
    fps: float | None = None,
    resolutions: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, Any]:
    """Probe default profile and common resolutions for one USB camera."""
    import cv2

    prepare_opencv_platform()
    backend = _opencv_usb_backend()
    result: dict[str, Any] = {"id": target, "default": None, "modes": [], "working": []}

    with _quiet_opencv():
        default = _probe_usb_camera(cv2, target, backend=backend)
        if default:
            profile = default.get("default_stream_profile", {})
            result["default"] = profile
            default_probe = _probe_usb_resolution(
                cv2,
                target,
                int(profile.get("width") or 0),
                int(profile.get("height") or 0),
                backend=backend,
                fps=fps,
            )
            if default_probe:
                default_probe["label"] = "default"
                result["working"].append(default_probe)

        seen: set[tuple[int, int]] = set()
        for width, height in resolutions or COMMON_PROBE_RESOLUTIONS:
            if (width, height) in seen:
                continue
            seen.add((width, height))
            mode = _probe_usb_resolution(
                cv2, target, width, height, backend=backend, fps=fps
            )
            if mode is None:
                result["modes"].append(
                    {
                        "requested": (width, height),
                        "works": False,
                    }
                )
                continue
            mode["works"] = True
            result["modes"].append(mode)
            frame_key = mode["frame"]
            if frame_key not in {m["frame"] for m in result["working"]}:
                result["working"].append(mode)

    result["working"].sort(key=lambda m: m["frame"][0] * m["frame"][1], reverse=True)
    return result


def probe_usb_cameras_together(
    targets: list[int | str],
    width: int,
    height: int,
    *,
    fps: float | None = None,
) -> dict[str, Any]:
    """Open all USB cameras at once and verify each delivers a frame."""
    import cv2

    prepare_opencv_platform()
    backend = _opencv_usb_backend()
    api = backend if backend is not None else cv2.CAP_ANY
    cameras: list[tuple[int | str, Any]] = []
    per_camera: list[dict[str, Any]] = []

    with _quiet_opencv():
        try:
            for target in targets:
                camera = cv2.VideoCapture(target, api)
                if not camera.isOpened():
                    per_camera.append({"id": target, "opened": False})
                    raise RuntimeError(f"index {target!r} failed to open")
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
                if fps is not None and fps > 0:
                    camera.set(cv2.CAP_PROP_FPS, float(fps))
                actual_w = int(round(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
                actual_h = int(round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                cameras.append((target, camera))
                per_camera.append(
                    {
                        "id": target,
                        "opened": True,
                        "actual": (actual_w, actual_h),
                        "frame": None,
                    }
                )
                if len(cameras) > 1:
                    time.sleep(0.5 if platform.system() == "Darwin" else 0.1)

            ok_all = True
            for i, (target, camera) in enumerate(cameras):
                frame_ok = False
                for _ in range(25):
                    ok, frame = camera.read()
                    if ok and frame is not None:
                        fh, fw = frame.shape[:2]
                        per_camera[i]["frame"] = (fw, fh)
                        frame_ok = True
                        break
                    time.sleep(0.04)
                if not frame_ok:
                    ok_all = False
                    per_camera[i]["frame"] = None
        finally:
            for _, camera in cameras:
                camera.release()

    return {
        "requested": (width, height),
        "fps": fps,
        "targets": targets,
        "ok": ok_all and all(row.get("frame") for row in per_camera),
        "cameras": per_camera,
    }


def _recommend_capture_mode(
    probe: dict[str, Any],
    *,
    output_width: int = 640,
    output_height: int = 480,
    prefer_below: tuple[int, int] | None = (1280, 720),
) -> dict[str, Any] | None:
    """Pick a working capture mode — prefer <= prefer_below pixels, else smallest working."""
    working = probe.get("working") or []
    if not working:
        return None

    def pixels(mode: dict[str, Any]) -> int:
        fw, fh = mode["frame"]
        return fw * fh

    if prefer_below:
        pw, ph = prefer_below
        cap = pw * ph
        candidates = [m for m in working if pixels(m) <= cap]
        if candidates:
            return max(candidates, key=pixels)

    return min(working, key=pixels)


def _format_probe_yaml_snippet(
    name: str,
    cam: CameraSettings,
    mode: dict[str, Any],
    *,
    output_width: int,
    output_height: int,
) -> str:
    fw, fh = mode["frame"]
    lines = [
        f"  {name}:",
        f"    type: {cam.type}",
    ]
    if cam.url:
        lines.append(f"    url: {cam.url}")
    elif cam.host:
        lines.append(f"    host: {cam.host}")
        lines.append(f"    port: {cam.port or 82}")
    else:
        lines.append(f"    index_or_path: {cam.index_or_path}")
    if fw == output_width and fh == output_height:
        lines.extend([
            f"    width: {output_width}",
            f"    height: {output_height}",
            "    auto_resolution: false",
        ])
    else:
        lines.extend([
            f"    capture_width: {fw}",
            f"    capture_height: {fh}",
            f"    width: {output_width}          # dataset output (downscaled)",
            f"    height: {output_height}",
            "    auto_resolution: false",
        ])
    if cam.fps is not None:
        lines.append(f"    fps: {cam.fps}")
    return "\n".join(lines)


def probe_camera_resolutions(
    *,
    index: int | None = None,
    name: str | None = None,
    all_usb: bool = False,
    together: bool = False,
    output_width: int = 640,
    output_height: int = 480,
    fps: float | None = None,
) -> None:
    """Print supported USB capture resolutions and optional multi-cam check."""
    cfg = ProjectConfig.load()
    prepare_opencv_platform()

    usb_targets: list[tuple[str, int | str, CameraSettings | None]] = []
    if index is not None:
        usb_targets.append((f"index-{index}", index, None))
    elif name:
        if name not in cfg.cameras:
            raise SystemExit(f"Unknown camera {name!r}. Configured: {', '.join(cfg.cameras)}")
        cam = cfg.cameras[name]
        if is_stream_camera(cam) or is_udp_camera(cam):
            raise SystemExit(f"Camera {name!r} is a network stream — resolution probe is USB-only.")
        usb_targets.append((name, resolve_camera_source(cam), cam))
    elif all_usb or not cfg.cameras:
        for info in find_usb_cameras(verify_capture=False):
            usb_targets.append((str(info["id"]), info["id"], None))
    else:
        for cam_name, cam in cfg.cameras.items():
            if is_stream_camera(cam) or is_udp_camera(cam):
                print(f"Skipping {cam_name} ({cam.type}) — network camera\n")
                continue
            usb_targets.append((cam_name, resolve_camera_source(cam), cam))

    if not usb_targets:
        print("No USB cameras to probe.")
        sys.exit(1)

    probes: list[tuple[str, dict[str, Any], CameraSettings | None]] = []
    for label, target, cam in usb_targets:
        print(f"=== {label} (source={target!r}) ===")
        probe = probe_usb_camera_resolutions(target, fps=fps)
        probes.append((label, probe, cam))
        default = probe.get("default") or {}
        if default:
            print(
                f"Driver default: {default.get('width')}x{default.get('height')} "
                f"@ {default.get('fps')} fps ({default.get('fourcc', '?')})"
            )
        print("\nResolution probe (requested → OpenCV actual → frame):")
        for mode in probe.get("modes", []):
            if not mode.get("works"):
                req_w, req_h = mode["requested"]
                print(f"  ✗ {req_w}x{req_h} — no frames")
                continue
            req_w, req_h = mode["requested"]
            act_w, act_h = mode["actual"]
            frm_w, frm_h = mode["frame"]
            fps_note = f" @ {mode['fps']:.1f}fps" if mode.get("fps") else ""
            match = "exact" if (req_w, req_h) == (act_w, act_h) == (frm_w, frm_h) else "approx"
            print(
                f"  ✓ {req_w}x{req_h} → {act_w}x{act_h} → frame {frm_w}x{frm_h}{fps_note} ({match})"
            )

        rec = _recommend_capture_mode(
            probe, output_width=output_width, output_height=output_height
        )
        if rec:
            fw, fh = rec["frame"]
            print(f"\nSuggested capture: {fw}x{fh} → output {output_width}x{output_height}")
            if cam is not None:
                print("\n" + _format_probe_yaml_snippet(
                    label, cam, rec, output_width=output_width, output_height=output_height
                ))
        else:
            print("\nNo working capture modes found.")
        print()

    if together and len(usb_targets) > 1:
        ids = [target for _, target, _ in usb_targets]
        print(f"=== Multi-camera together ({len(ids)} USB) ===")
        best: dict[str, Any] | None = None
        for width, height in COMMON_PROBE_RESOLUTIONS:
            trial = probe_usb_cameras_together(ids, width, height, fps=fps)
            status = "OK" if trial["ok"] else "FAIL"
            print(f"  {status}  {width}x{height}")
            for row in trial["cameras"]:
                frame = row.get("frame")
                actual = row.get("actual")
                frame_s = f"{frame[0]}x{frame[1]}" if frame else "no frame"
                actual_s = f"{actual[0]}x{actual[1]}" if actual else "?"
                print(f"       id {row['id']!r}: actual {actual_s}, frame {frame_s}")
            if trial["ok"]:
                best = trial
        if best:
            w, h = best["requested"]
            print(
                f"\nAll cameras work together at {w}x{h} "
                f"(lowest verified common mode — use as capture_width/height)."
            )
        else:
            print("\nNo common resolution worked for all cameras simultaneously.")
            print("Try fewer cameras, different USB ports, or lower output sizes.")


def _probe_usb_camera(cv2, target: int | str, *, backend: int | None = None) -> dict[str, Any] | None:
    api = backend if backend is not None else cv2.CAP_ANY
    camera = cv2.VideoCapture(target, api)
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


def _verify_usb_capture(cv2, target: int | str, *, backend: int | None = None) -> bool:
    """True when OpenCV can read at least one frame (opens-only is not enough on macOS)."""
    api = backend if backend is not None else cv2.CAP_ANY
    if platform.system() == "Darwin":
        api = cv2.CAP_AVFOUNDATION
    camera = cv2.VideoCapture(target, api)
    try:
        if not camera.isOpened():
            return False
        for _ in range(20):
            ok, _ = camera.read()
            if ok:
                return True
            time.sleep(0.05)
        return False
    finally:
        camera.release()


def find_usb_cameras(*, verify_capture: bool = False) -> list[dict[str, Any]]:
    """Discover USB cameras without spamming OpenCV errors."""
    import cv2

    prepare_opencv_platform()
    found: list[dict[str, Any]] = []

    with _quiet_opencv():
        if platform.system() == "Linux":
            targets = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            for path in targets:
                info = _probe_usb_camera(cv2, str(path))
                if info:
                    if verify_capture:
                        info["captures"] = _verify_usb_capture(cv2, str(path))
                    found.append(info)
        else:
            max_index = _MAX_INDEX_DARWIN if platform.system() == "Darwin" else _MAX_INDEX_OTHER
            consecutive_misses = 0
            for index in range(max_index):
                info = _probe_usb_camera(
                    cv2, index, backend=cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else None
                )
                if info:
                    if verify_capture:
                        info["captures"] = _verify_usb_capture(cv2, index)
                    found.append(info)
                    consecutive_misses = 0
                else:
                    consecutive_misses += 1
                    if consecutive_misses >= _MAX_CONSECUTIVE_MISSES:
                        break

    return found


def working_usb_camera_ids() -> list[int | str]:
    """USB indices/paths that both open and deliver at least one frame."""
    return [cam["id"] for cam in find_usb_cameras(verify_capture=True) if cam.get("captures")]


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
    print("Scanning USB cameras (verifying frame capture)...")
    cameras = find_usb_cameras(verify_capture=True)
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

    working = [cam for cam in cameras if cam.get("captures")]
    print(f"Found {len(cameras)} device(s), {len(working)} capture frames:\n")
    mac_names = _macos_camera_names()
    for i, info in enumerate(cameras):
        profile = info.get("default_stream_profile", {})
        label = info["name"]
        if i < len(mac_names):
            label = f"{mac_names[i]} ({info['id']!r})"
        capture = "yes" if info.get("captures") else "NO — opens but no frames"
        print(f"  {label}")
        print(f"    id:       {info['id']!r}")
        print(f"    captures: {capture}")
        print(f"    backend:  {info.get('backend_api', 'unknown')}")
        print(
            f"    default:  {profile.get('width')}x{profile.get('height')} "
            f"@ {profile.get('fps')} fps ({profile.get('fourcc', '?')})"
        )
        print()

    if len(working) < len(cameras):
        print(
            "Some indices open but never deliver frames — reseat USB, try another port,\n"
            "or close FaceTime/Zoom. Use only indices with captures: yes in config/default.yaml.\n"
            "Run: uv run sarm-hand camera-probe --together"
        )


def is_udp_camera(cam: CameraSettings) -> bool:
    if cam.type.lower() in UDP_TYPES:
        return True
    source = cam.url or (cam.index_or_path if isinstance(cam.index_or_path, str) else None)
    if source:
        from .esp32_udp_camera import is_esp32_udp_source

        if is_esp32_udp_source(str(source)):
            return True
    return bool(cam.host)


def resolve_udp_endpoint(cam: CameraSettings) -> tuple[str, int]:
    if cam.host:
        return cam.host, int(cam.port or 82)
    source = cam.url or (cam.index_or_path if isinstance(cam.index_or_path, str) else "")
    from .esp32_udp_camera import normalize_esp32_udp_source

    key = normalize_esp32_udp_source(str(source))
    if key is not None:
        rest = key[len("esp32udp:") :]
        host, sep, port_str = rest.rpartition(":")
        if not sep:
            raise ValueError(f"Invalid ESP32 UDP source: {source!r}")
        return host, int(port_str or 82)
    raise ValueError(f"Camera type '{cam.type}' requires 'host' in config")


def _esp32_udp_source(cam: CameraSettings) -> str:
    host, port = resolve_udp_endpoint(cam)
    return f"esp32udp:{host}:{port}"


def _register_udp_camera(cam: CameraSettings) -> str:
    from .esp32_udp_camera import register_udp_options
    from .esp32_udp_stream import UdpStreamOptions

    host, port = resolve_udp_endpoint(cam)
    source = f"esp32udp:{host}:{port}"
    register_udp_options(
        source,
        UdpStreamOptions(
            host=host,
            port=port,
            rotate_180=cam.rotate_180,
            flip_horizontal=cam.flip_horizontal,
            stale_sec=cam.stale_sec,
            connect_grace_s=cam.connect_grace_s,
            fps_window=cam.fps_window,
            hold_fps=True,
            target_fps=float(cam.fps) if cam.fps else None,
        ),
    )
    return source


def is_stream_url(source: str) -> bool:
    return source.lower().startswith(STREAM_PREFIXES)


def is_stream_camera(cam: CameraSettings) -> bool:
    if is_udp_camera(cam):
        return False
    if cam.type.lower() in STREAM_TYPES:
        return True
    source = cam.url or (cam.index_or_path if isinstance(cam.index_or_path, str) else None)
    return bool(source and is_stream_url(source))


def resolve_camera_source(cam: CameraSettings) -> int | str | Path:
    """Return capture source: USB index/path, HTTP URL, or esp32udp:host:port."""
    cam_type = cam.type.lower()

    if is_udp_camera(cam):
        return _register_udp_camera(cam)

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
# USB capture at native/explicit resolution, downscale to output size on read.
_OUTPUT_SIZE_BY_SOURCE: dict[str, tuple[int, int]] = {}
_CAPTURE_SIZE_BY_SOURCE: dict[str, tuple[int, int]] = {}


def _source_key(source: int | str | Path) -> str:
    return str(source)


def _register_output_size(source: int | str | Path, width: int, height: int) -> None:
    _OUTPUT_SIZE_BY_SOURCE[_source_key(source)] = (int(width), int(height))


def _register_capture_size(source: int | str | Path, width: int, height: int) -> None:
    _CAPTURE_SIZE_BY_SOURCE[_source_key(source)] = (int(width), int(height))


def _explicit_capture_dims(cam: CameraSettings) -> tuple[int, int] | None:
    if cam.capture_width and cam.capture_height:
        return int(cam.capture_width), int(cam.capture_height)
    return None


def _output_dims(cam: CameraSettings) -> tuple[int, int]:
    return int(cam.width or 640), int(cam.height or 480)


def _wants_downscale(cam: CameraSettings) -> bool:
    """Deliver width×height frames, capturing natively or at capture_width×capture_height."""
    if is_stream_camera(cam) or is_udp_camera(cam):
        return False
    out_w, out_h = _output_dims(cam)
    if cam.width is None or cam.height is None:
        return False
    if cam.auto_resolution:
        return True
    cap = _explicit_capture_dims(cam)
    if cap and cap != (out_w, out_h):
        return True
    return False


def _declared_camera_dims(cam: CameraSettings) -> tuple[int, int, int]:
    """Width, height, fps for LeRobot robot config (output / dataset dimensions)."""
    w = int(cam.width or 640)
    h = int(cam.height or 480)
    fps = int(cam.fps if cam.fps is not None else 30)
    return w, h, fps


def _default_stream_max_frame_age(fps: float | int | None) -> int:
    """LeRobot read_latest() defaults to 500 ms — too tight for HTTP/MJPEG jitter."""
    rate = max(float(fps or 10), 1.0)
    return max(1500, int(1000.0 / rate * 15))


def effective_camera_settings(cam: CameraSettings) -> CameraSettings:
    """Apply auto_resolution and platform defaults before opening a camera."""
    updates: dict[str, Any] = {}
    if cam.auto_resolution and not (cam.capture_width and cam.capture_height):
        updates.update(width=None, height=None, fps=None)
    if is_stream_camera(cam):
        # Declared width/height/fps satisfy LeRobot robot config; connect uses native
        # stream resolution via install_stream_camera_patch().
        if cam.width is None:
            updates["width"] = 640
        if cam.height is None:
            updates["height"] = 480
        if cam.fps is None:
            updates["fps"] = 5
        if cam.warmup_s is None:
            updates["warmup_s"] = 3
        if cam.max_frame_age_ms is None:
            updates["max_frame_age_ms"] = _default_stream_max_frame_age(cam.fps or 5)
    elif is_udp_camera(cam):
        if cam.width is None:
            updates["width"] = 640
        if cam.height is None:
            updates["height"] = 480
        if cam.fps is None:
            updates["fps"] = 5
        if cam.warmup_s is None:
            updates["warmup_s"] = 3
        if cam.max_frame_age_ms is None:
            updates["max_frame_age_ms"] = _default_stream_max_frame_age(cam.fps or 5)
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


_USB_DOWNSCALE_PATCHED = False


def install_usb_downscale_patch() -> None:
    """Downscale native USB frames to configured output width×height after capture."""
    global _USB_DOWNSCALE_PATCHED
    if _USB_DOWNSCALE_PATCHED:
        return

    import cv2
    from lerobot.cameras.opencv import camera_opencv
    from lerobot.utils.errors import DeviceNotConnectedError

    original_configure = camera_opencv.OpenCVCamera._configure_capture_settings

    def _configure_capture_settings(self) -> None:
        out = _OUTPUT_SIZE_BY_SOURCE.get(_source_key(self.index_or_path))
        cap = _CAPTURE_SIZE_BY_SOURCE.get(_source_key(self.index_or_path))
        if out is None:
            original_configure(self)
            return

        if self.config.fourcc is not None:
            self._validate_fourcc()
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if cap is not None:
            self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(cap[0]))
            self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cap[1]))

        native_w = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        native_h = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.capture_width, self.capture_height = native_w, native_h
        actual_fps = float(self.videocapture.get(cv2.CAP_PROP_FPS))
        if actual_fps > 0:
            self.fps = actual_fps

    original_post = camera_opencv.OpenCVCamera._postprocess_image

    def _postprocess_image(self, image):
        processed = original_post(self, image)
        out = _OUTPUT_SIZE_BY_SOURCE.get(_source_key(self.index_or_path))
        if out is None:
            return processed
        w, h = out
        ph, pw = processed.shape[:2]
        if (pw, ph) != (w, h):
            processed = cv2.resize(processed, (w, h), interpolation=cv2.INTER_AREA)
        return processed

    original_connect = camera_opencv.OpenCVCamera.connect

    def connect(self, warmup: bool = True) -> None:
        original_connect(self, warmup=warmup)
        out = _OUTPUT_SIZE_BY_SOURCE.get(_source_key(self.index_or_path))
        if out is not None:
            self.width, self.height = out

    camera_opencv.OpenCVCamera._postprocess_image = _postprocess_image
    camera_opencv.OpenCVCamera.connect = connect
    camera_opencv.OpenCVCamera._configure_capture_settings = _configure_capture_settings
    _USB_DOWNSCALE_PATCHED = True


def native_usb_profile(source: int | str) -> dict[str, Any] | None:
    for info in find_usb_cameras():
        if info["id"] == source:
            return info.get("default_stream_profile")
    return None


def camera_to_lerobot_dict(cam: CameraSettings) -> dict[str, Any]:
    """Serialize one camera for lerobot-record CLI flags."""
    capture = effective_camera_settings(cam)
    source = resolve_camera_source(cam)
    out_w, out_h, out_fps = _declared_camera_dims(cam)
    if _wants_downscale(cam):
        install_usb_downscale_patch()
        _register_output_size(source, out_w, out_h)
        cap = _explicit_capture_dims(cam)
        if cap is not None:
            _register_capture_size(source, cap[0], cap[1])
    payload: dict[str, Any] = {
        "type": "opencv",
        "index_or_path": source,
        "width": out_w if _wants_downscale(cam) else capture.width,
        "height": out_h if _wants_downscale(cam) else capture.height,
        "fps": out_fps if _wants_downscale(cam) else capture.fps,
    }
    if payload.get("width") is None:
        payload.pop("width", None)
    if payload.get("height") is None:
        payload.pop("height", None)
    if payload.get("fps") is None:
        payload.pop("fps", None)
    if capture.warmup_s is not None:
        payload["warmup_s"] = capture.warmup_s
    if cam.fourcc is not None:
        payload["fourcc"] = cam.fourcc
    return payload


def build_lerobot_camera_config(cam: CameraSettings):
    """Build a LeRobot OpenCVCameraConfig from project settings."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    install_stream_camera_patch()
    install_usb_downscale_patch()
    capture = effective_camera_settings(cam)
    source = resolve_camera_source(cam)
    out_w, out_h, out_fps = _declared_camera_dims(cam)
    if _wants_downscale(cam):
        _register_output_size(source, out_w, out_h)
        cap = _explicit_capture_dims(cam)
        if cap is not None:
            _register_capture_size(source, cap[0], cap[1])
    if is_stream_camera(cam) and cam.max_frame_age_ms is not None:
        _STREAM_MAX_FRAME_AGE_MS[str(source)] = cam.max_frame_age_ms
    if is_udp_camera(cam) and cam.max_frame_age_ms is not None:
        _STREAM_MAX_FRAME_AGE_MS[str(source)] = cam.max_frame_age_ms
    kwargs: dict[str, Any] = {"index_or_path": source}
    if _wants_downscale(cam):
        kwargs.update(width=out_w, height=out_h, fps=out_fps)
    else:
        if capture.fps is not None:
            kwargs["fps"] = capture.fps
        if capture.width is not None:
            kwargs["width"] = capture.width
        if capture.height is not None:
            kwargs["height"] = capture.height
    if capture.warmup_s is not None:
        kwargs["warmup_s"] = capture.warmup_s
    elif _wants_downscale(cam) and platform.system() == "Darwin":
        kwargs["warmup_s"] = 3
    if cam.fourcc is not None:
        kwargs["fourcc"] = cam.fourcc
    if platform.system() == "Darwin" and not is_stream_camera(cam):
        from lerobot.cameras.configs import Cv2Backends

        kwargs["backend"] = Cv2Backends.AVFOUNDATION
    return OpenCVCameraConfig(**kwargs)


_RESILIENT_CAMERA_PATCHED = False
_FOLLOWER_CONNECT_PATCHED = False
_UDP_CAMERA_PATCHED = False


def black_frame(height: int, width: int):
    """RGB uint8 frame filled with black (failed / missing camera placeholder)."""
    import numpy as np

    return np.zeros((int(height), int(width), 3), dtype=np.uint8)


def _fallback_dims(camera: Any) -> tuple[int, int, int]:
    w = int(getattr(camera, "width", None) or getattr(camera.config, "width", None) or 640)
    h = int(getattr(camera, "height", None) or getattr(camera.config, "height", None) or 480)
    fps = int(getattr(camera, "fps", None) or getattr(camera.config, "fps", None) or 30)
    return w, h, fps


def _enable_black_fallback(camera: Any, reason: Exception, *, label: str | None = None) -> None:
    w, h, fps = _fallback_dims(camera)
    camera.width = w
    camera.height = h
    camera.fps = fps
    camera._sarm_black_fallback = True  # noqa: SLF001
    tag = label or camera
    print(f"  {tag}: using black {w}x{h} frames ({reason})")


def install_resilient_camera_patch() -> None:
    """On connect/read failure, emit black frames so recording can continue."""
    global _RESILIENT_CAMERA_PATCHED
    if _RESILIENT_CAMERA_PATCHED:
        return

    from lerobot.cameras.opencv import camera_opencv

    original_connect = camera_opencv.OpenCVCamera.connect
    original_disconnect = camera_opencv.OpenCVCamera.disconnect
    original_is_connected = camera_opencv.OpenCVCamera.is_connected.fget
    original_read = camera_opencv.OpenCVCamera.read
    original_async_read = camera_opencv.OpenCVCamera.async_read
    original_read_latest = camera_opencv.OpenCVCamera.read_latest

    def is_connected(self) -> bool:
        if getattr(self, "_sarm_black_fallback", False):
            return True
        return original_is_connected(self)

    def connect(self, warmup: bool = True) -> None:
        if getattr(self, "_sarm_black_fallback", False):
            return
        try:
            original_connect(self, warmup=warmup)
        except Exception as exc:
            _enable_black_fallback(self, exc)

    def disconnect(self) -> None:
        if getattr(self, "_sarm_black_fallback", False):
            self._sarm_black_fallback = False
            return
        original_disconnect(self)

    def _read_black_or_live(read_fn, self, *args, **kwargs):
        if getattr(self, "_sarm_black_fallback", False):
            return black_frame(self.height, self.width)
        try:
            return read_fn(self, *args, **kwargs)
        except (TimeoutError, RuntimeError, ConnectionError) as exc:
            warnings.warn(f"{self} read failed — black frame ({exc})", stacklevel=2)
            return black_frame(self.height, self.width)

    camera_opencv.OpenCVCamera.is_connected = property(is_connected)
    camera_opencv.OpenCVCamera.connect = connect
    camera_opencv.OpenCVCamera.disconnect = disconnect
    camera_opencv.OpenCVCamera.read = lambda self: _read_black_or_live(original_read, self)
    camera_opencv.OpenCVCamera.async_read = lambda self, timeout_ms=200: _read_black_or_live(
        original_async_read, self, timeout_ms
    )
    camera_opencv.OpenCVCamera.read_latest = lambda self, max_age_ms=500: _read_black_or_live(
        original_read_latest, self, max_age_ms
    )

    from .esp32_udp_camera import Esp32UdpCamera

    original_udp_connect = Esp32UdpCamera.connect
    original_udp_disconnect = Esp32UdpCamera.disconnect
    original_udp_read = Esp32UdpCamera.read
    original_udp_read_latest = Esp32UdpCamera.read_latest
    original_udp_is_connected = Esp32UdpCamera.is_connected.fget

    def udp_is_connected(self) -> bool:
        if getattr(self, "_sarm_black_fallback", False):
            return True
        return original_udp_is_connected(self)

    def udp_connect(self, warmup: bool = True) -> None:
        if getattr(self, "_sarm_black_fallback", False):
            return
        try:
            original_udp_connect(self, warmup=warmup)
        except Exception as exc:
            _enable_black_fallback(self, exc)

    def udp_disconnect(self) -> None:
        if getattr(self, "_sarm_black_fallback", False):
            self._sarm_black_fallback = False
            return
        original_udp_disconnect(self)

    Esp32UdpCamera.is_connected = property(udp_is_connected)
    Esp32UdpCamera.connect = udp_connect
    Esp32UdpCamera.disconnect = udp_disconnect
    Esp32UdpCamera.read = lambda self: _read_black_or_live(original_udp_read, self)
    Esp32UdpCamera.read_latest = lambda self, max_age_ms=500: _read_black_or_live(
        original_udp_read_latest, self, max_age_ms
    )

    _RESILIENT_CAMERA_PATCHED = True


def install_udp_camera_patch() -> None:
    """Route esp32udp: sources to Esp32UdpCamera instead of OpenCV VideoCapture."""
    global _UDP_CAMERA_PATCHED
    if _UDP_CAMERA_PATCHED:
        return

    import lerobot.cameras.utils as cam_utils
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.robots.so_follower import so_follower as sf_module

    from .esp32_udp_camera import Esp32UdpCamera, is_esp32_udp_source

    original_make = cam_utils.make_cameras_from_configs

    def make_cameras_from_configs(camera_configs):
        cameras: dict[str, Any] = {}
        passthrough: dict[str, Any] = {}
        for key, cfg in camera_configs.items():
            if cfg.type == "opencv" and is_esp32_udp_source(getattr(cfg, "index_or_path", "")):
                cameras[key] = Esp32UdpCamera(cfg)
            else:
                passthrough[key] = cfg
        if passthrough:
            cameras.update(original_make(passthrough))
        return cameras

    patched_make = make_cameras_from_configs
    cam_utils.make_cameras_from_configs = patched_make
    sf_module.make_cameras_from_configs = patched_make

    if getattr(OpenCVCamera, "_sarm_udp_delegate_patched", False):
        _UDP_CAMERA_PATCHED = True
        return

    original_connect = OpenCVCamera.connect
    original_disconnect = OpenCVCamera.disconnect
    original_is_connected = OpenCVCamera.is_connected.fget
    original_read = OpenCVCamera.read
    original_read_latest = OpenCVCamera.read_latest

    def _udp_delegate(self) -> Esp32UdpCamera:
        delegate = getattr(self, "_esp32_udp_delegate", None)
        if delegate is None:
            delegate = Esp32UdpCamera(self.config)
            self._esp32_udp_delegate = delegate
        return delegate

    def _uses_udp(self) -> bool:
        return is_esp32_udp_source(str(getattr(self, "index_or_path", "")))

    def opencv_connect(self, warmup: bool = True) -> None:
        if _uses_udp(self):
            _udp_delegate(self).connect(warmup=warmup)
            self._sarm_udp_active = True
            return
        return original_connect(self, warmup=warmup)

    def opencv_disconnect(self) -> None:
        if getattr(self, "_sarm_udp_active", False):
            _udp_delegate(self).disconnect()
            self._sarm_udp_active = False
            return
        return original_disconnect(self)

    def opencv_is_connected(self) -> bool:
        if getattr(self, "_sarm_udp_active", False):
            return _udp_delegate(self).is_connected
        return original_is_connected(self)

    def opencv_read(self) -> Any:
        if getattr(self, "_sarm_udp_active", False):
            return _udp_delegate(self).read()
        return original_read(self)

    def opencv_read_latest(self, max_age_ms: int = 500) -> Any:
        if getattr(self, "_sarm_udp_active", False):
            return _udp_delegate(self).read_latest(max_age_ms=max_age_ms)
        return original_read_latest(self, max_age_ms=max_age_ms)

    OpenCVCamera.connect = opencv_connect
    OpenCVCamera.disconnect = opencv_disconnect
    OpenCVCamera.is_connected = property(opencv_is_connected)
    OpenCVCamera.read = opencv_read
    OpenCVCamera.read_latest = opencv_read_latest
    OpenCVCamera._sarm_udp_delegate_patched = True

    _UDP_CAMERA_PATCHED = True


def build_camera_from_config(cam: CameraSettings):
    """Instantiate a LeRobot Camera (USB, HTTP, or ESP32 UDP) from project settings."""
    install_udp_camera_patch()
    install_stream_camera_patch()
    install_usb_downscale_patch()
    from lerobot.cameras.utils import make_cameras_from_configs

    config = build_lerobot_camera_config(cam)
    return make_cameras_from_configs({"_preview": config})["_preview"]


def install_all_camera_patches(
    *,
    cfg: ProjectConfig | None = None,
    resilient: bool | None = None,
) -> None:
    """Install stream, UDP, downscale, optional resilient, and follower-connect camera patches."""
    cfg = cfg or ProjectConfig.load()
    if resilient is None:
        resilient = not cfg.camera.fail_on_error
    install_udp_camera_patch()
    install_stream_camera_patch()
    install_usb_downscale_patch()
    if resilient:
        install_resilient_camera_patch()
    install_follower_connect_patch()


def camera_black_fallback_enabled(cfg: ProjectConfig | None = None) -> bool:
    """True when connect/read failures should emit black frames instead of raising."""
    cfg = cfg or ProjectConfig.load()
    return not cfg.camera.fail_on_error


def install_follower_connect_patch() -> None:
    """Connect USB cameras before the servo bus (macOS warmup + stagger)."""
    global _FOLLOWER_CONNECT_PATCHED
    if _FOLLOWER_CONNECT_PATCHED:
        return

    from lerobot.robots.so_follower import so_follower as sf
    from lerobot.utils.errors import DeviceAlreadyConnectedError

    def _patched_connect(original_connect):
        def connect(self, calibrate: bool = True) -> None:
            if self.is_connected:
                raise DeviceAlreadyConnectedError(f"{self} already connected")
            if self.cameras:
                connect_follower_robot(self, calibrate=calibrate)
                return
            original_connect(self, calibrate=calibrate)
            from .robot import sync_follower_goals_to_present

            sync_follower_goals_to_present(self)

        return connect

    for cls in (sf.SO100Follower, sf.SO101Follower):
        cls.connect = _patched_connect(cls.connect)
    _FOLLOWER_CONNECT_PATCHED = True


def require_working_cameras(cfg: ProjectConfig) -> None:
    """Exit if any configured USB camera index cannot capture frames."""
    issues = _configured_index_mismatches(cfg)
    if not issues:
        return
    print("Camera config does not match working USB devices:", file=sys.stderr)
    for line in issues:
        print(f"  - {line}", file=sys.stderr)
    print(
        "\nRun:  sarm-hand list-cameras\n"
        "Update index_or_path in config/default.yaml to indices with captures: yes.\n"
        "Remove broken cameras from config until hardware is fixed.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def connect_usb_cameras(
    cameras: dict[str, Any],
    *,
    cfg: ProjectConfig | None = None,
    stagger_s: float | None = None,
    retries: int = 2,
    fallback_black: bool | None = None,
) -> None:
    """Open USB cameras with staggered connect (macOS multi-cam is flaky back-to-back)."""
    if not cameras:
        return

    cfg = cfg or ProjectConfig.load()
    if fallback_black is None:
        fallback_black = camera_black_fallback_enabled(cfg)

    prepare_opencv_platform()
    if fallback_black:
        install_all_camera_patches(cfg=cfg, resilient=True)
    else:
        install_udp_camera_patch()
        install_stream_camera_patch()
        install_usb_downscale_patch()

    gap = stagger_s if stagger_s is not None else (1.5 if platform.system() == "Darwin" else 0.25)
    names = list(cameras.keys())
    for i, name in enumerate(names):
        if i > 0 and gap > 0:
            time.sleep(gap)
        cam = cameras[name]
        from .esp32_udp_camera import is_esp32_udp_source

        is_udp = is_esp32_udp_source(str(getattr(cam, "index_or_path", "")))
        cam_retries = max(retries, 4) if is_udp else retries
        last_exc: Exception | None = None
        for attempt in range(max(1, cam_retries)):
            try:
                cam.connect()
                if getattr(cam, "_sarm_black_fallback", False):
                    if not fallback_black:
                        raise ConnectionError(
                            f"Camera '{name}' is not capturing (black-frame fallback active)"
                        )
                    print(f"  {name}: black frames (configured index unavailable)")
                last_exc = None
                break
            except (ConnectionError, TimeoutError, RuntimeError) as exc:
                last_exc = exc
                if cam.is_connected:
                    cam.disconnect()
                if attempt + 1 < cam_retries:
                    time.sleep(gap if not is_udp else max(gap, 1.0))
        if last_exc is not None:
            if fallback_black:
                _enable_black_fallback(cam, last_exc, label=name)
            else:
                raise ConnectionError(
                    f"Camera '{name}' failed to connect: {last_exc}\n"
                    "  Run: sarm-hand list-cameras  and  sarm-hand camera-test\n"
                    "  To allow black-frame fallback (not recommended for recording), "
                    "set camera.fail_on_error: false in config/default.yaml"
                ) from last_exc


def connect_follower_robot(
    robot: Any,
    *,
    calibrate: bool = False,
    cfg: ProjectConfig | None = None,
) -> None:
    """Connect follower arm: cameras first, then servo bus (avoids bus timeout during cam warmup)."""
    from .robot import ensure_bus_calibration, sync_follower_goals_to_present

    cfg = cfg or ProjectConfig.load()
    if robot.cameras:
        print(f"Connecting {len(robot.cameras)} camera(s)...")
        connect_usb_cameras(robot.cameras, cfg=cfg)
    robot.bus.connect()
    if not robot.is_calibrated and calibrate:
        robot.calibrate()
    elif not robot.is_calibrated:
        ensure_bus_calibration(robot, "follower")
    robot.configure()
    sync_follower_goals_to_present(robot)


def prepare_opencv_platform() -> None:
    """macOS: prefer AVFoundation and avoid FFmpeg device enumeration (conflicts with PyAV)."""
    if platform.system() != "Darwin":
        return
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_LIST", "AVFOUNDATION")


def build_robot_camera_configs(cfg: ProjectConfig) -> dict:
    """Build LeRobot camera configs keyed by name (for teleop / recording)."""
    from dataclasses import replace

    multi = len(cfg.cameras) > 1
    configs = {}
    for name, cam in cfg.cameras.items():
        config = build_lerobot_camera_config(cam)
        if multi and platform.system() == "Darwin" and config.warmup_s < 3:
            config = replace(config, warmup_s=3)
        configs[name] = config
    return configs


def cameras_lerobot_dict(cfg: ProjectConfig) -> dict[str, dict[str, Any]]:
    return {name: camera_to_lerobot_dict(cam) for name, cam in cfg.cameras.items()}


def describe_camera(name: str, cam: CameraSettings) -> str:
    source = resolve_camera_source(cam)
    if is_udp_camera(cam):
        kind = "esp32-udp"
    elif is_stream_camera(cam):
        kind = "stream"
    else:
        kind = "usb"
    parts = [f"{name}: {kind} ({cam.type})", f"source={source!r}"]
    cap = _explicit_capture_dims(cam)
    out_w, out_h = _output_dims(cam)
    if cap and _wants_downscale(cam):
        parts.append(f"capture {cap[0]}x{cap[1]} → output {out_w}x{out_h}")
    elif _wants_downscale(cam):
        parts.append(f"native → output {out_w}x{out_h} (auto_resolution)")
    elif cam.auto_resolution:
        parts.append("auto_resolution")
    elif cam.width and cam.height:
        parts.append(f"{cam.width}x{cam.height}")
    if cam.fps and not _wants_downscale(cam):
        parts.append(f"@{cam.fps}fps")
    if is_stream_camera(cam):
        parts.append("(native at connect)")
    if is_udp_camera(cam):
        host, port = resolve_udp_endpoint(cam)
        parts.append(f"udp→{host}:{port} (jpeg decode on host)")
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
                "  macOS: run  uv run sarm-hand camera-probe --name <camera>",
                "  macOS: set capture_width/capture_height from probe, width/height for output",
                "  macOS: close FaceTime/Zoom if the camera is in use elsewhere",
            ])
        elif not settings.auto_resolution and (settings.width or settings.fps):
            hints.append(
                "  Try auto_resolution: true if the camera rejects the requested size/fps"
            )
    return "\n".join(hints)


def _read_one_frame(settings: CameraSettings):
    """Connect and return one RGB frame."""
    import logging

    logging.getLogger("lerobot.cameras.opencv.camera_opencv").setLevel(logging.ERROR)

    camera = build_camera_from_config(settings)
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
    """Capture frames from a USB camera, HTTP stream, or ESP32 UDP JPEG."""
    import cv2

    try:
        settings = _settings_from_cli(
            name=name, index=index, url=url, width=width, height=height, fps=fps
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    install_all_camera_patches(cfg=ProjectConfig.load())
    camera = build_camera_from_config(settings)

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
        if is_udp_camera(settings):
            host, port = resolve_udp_endpoint(settings)
            print(
                f"\nESP32 UDP tips:",
                f"  - ESP32 must be running udp_stream firmware on {host}:{port}",
                "  - Mac listens on a random local UDP port and sends SUBSCRIBE packets",
                "  - Ensure Mac and ESP32 are on the same Wi‑Fi network",
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


def _configured_index_mismatches(cfg: ProjectConfig) -> list[str]:
    """Warn when a USB camera index is missing or does not capture frames."""
    working = working_usb_camera_ids()
    if not working:
        return []
    issues: list[str] = []
    working_ids = {str(i) for i in working}
    for name, cam in cfg.cameras.items():
        if is_stream_camera(cam) or is_udp_camera(cam):
            continue
        source = resolve_camera_source(cam)
        if isinstance(source, int):
            if str(source) not in working_ids:
                ids = ", ".join(str(i) for i in working)
                issues.append(
                    f"{name}: index {source} not usable — working indices: {ids}"
                )
    return issues


def _expected_output_size(settings: CameraSettings) -> tuple[int, int] | None:
    if settings.width and settings.height:
        return int(settings.width), int(settings.height)
    return None


def _validate_output_frame(settings: CameraSettings, frame) -> tuple[int, int]:
    h, w = frame.shape[:2]
    expected = _expected_output_size(settings)
    if expected and (w, h) != expected:
        exp_w, exp_h = expected
        raise RuntimeError(f"expected {exp_w}x{exp_h}, got {w}x{h}")
    return w, h


def build_configured_camera_instances(
    cfg: ProjectConfig,
    *,
    resilient: bool | None = None,
) -> dict[str, Any]:
    """Instantiate all configured cameras (same configs as record-leader)."""
    from lerobot.cameras.utils import make_cameras_from_configs

    if resilient is None:
        resilient = camera_black_fallback_enabled(cfg)
    install_udp_camera_patch()
    install_stream_camera_patch()
    install_usb_downscale_patch()
    if resilient:
        install_resilient_camera_patch()
    prepare_opencv_platform()
    return make_cameras_from_configs(build_robot_camera_configs(cfg))


def _test_configured_cameras_sequential(cfg: ProjectConfig) -> bool:
    """Connect each camera alone and grab one frame."""
    print(f"Testing {len(cfg.cameras)} configured camera(s) one at a time...\n")
    failed = False

    for name, settings in cfg.cameras.items():
        print(describe_camera(name, settings))
        attempts: list[tuple[str, CameraSettings]] = [("configured", settings)]
        if (
            not settings.auto_resolution
            and not is_stream_camera(settings)
            and not is_udp_camera(settings)
            and not _wants_downscale(settings)
        ):
            auto = replace(settings, auto_resolution=True, width=None, height=None, fps=None)
            attempts.append(("auto_resolution", effective_camera_settings(auto)))

        last_error: Exception | None = None
        for label, attempt_settings in attempts:
            try:
                frame = _read_one_frame(attempt_settings)
                w, h = _validate_output_frame(settings, frame)
                print(
                    f"  OK{f' ({label})' if label != 'configured' else ''} "
                    f"— frame {w}x{h}, {frame.dtype}\n"
                )
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
        print("Sequential camera test failed.", file=sys.stderr)
    else:
        print("All cameras passed sequential test.")
    return not failed


def _test_configured_cameras_together(cfg: ProjectConfig) -> bool:
    """Open every configured camera at once (same path as record-leader) and read one frame each."""
    cameras = build_configured_camera_instances(cfg, resilient=False)
    print(
        f"Testing {len(cameras)} configured camera(s) concurrently "
        f"(same connect path as record-leader)...\n"
    )
    for name, settings in cfg.cameras.items():
        print(f"  {describe_camera(name, settings)}")
    print()

    failed = False
    try:
        connect_usb_cameras(
            cameras,
            fallback_black=False,
        )
    except ConnectionError as exc:
        print(f"Concurrent connect failed: {exc}", file=sys.stderr)
        hints = (
            "  Lower USB capture load: set capture_width/capture_height from camera-probe --together\n"
            "  macOS: use different USB ports/hubs; close FaceTime/Zoom\n"
            "  Isolate: uv run sarm-hand camera-test --each"
        )
        print(hints, file=sys.stderr)
        return False

    try:
        for name, camera in cameras.items():
            settings = cfg.cameras[name]
            try:
                if getattr(camera, "_sarm_black_fallback", False):
                    raise RuntimeError("camera is using black-frame fallback (not capturing)")
                frame = camera.read()
                if getattr(camera, "_sarm_black_fallback", False):
                    raise RuntimeError("camera fell back to black frames during read")
                w, h = _validate_output_frame(settings, frame)
                print(f"  {name}: OK — frame {w}x{h}, {frame.dtype}")
            except Exception as exc:
                failed = True
                print(f"  {name}: FAIL — {exc}", file=sys.stderr)
                hints = _camera_failure_hints(settings)
                if hints:
                    print(hints, file=sys.stderr)
    finally:
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()

    if failed:
        print("\nConcurrent camera test failed.", file=sys.stderr)
        print(
            "  USB bandwidth/resolution may be too high for all streams at once.\n"
            "  Run: uv run sarm-hand camera-probe --together",
            file=sys.stderr,
        )
    else:
        print("\nAll cameras passed concurrent test.")
    return not failed


def test_configured_cameras(*, together: bool = False, each: bool = False) -> None:
    """Test configured cameras — concurrent by default when multiple are defined."""
    cfg = ProjectConfig.load()
    if not cfg.cameras:
        print("No cameras configured in config/default.yaml")
        sys.exit(1)

    mismatches = _configured_index_mismatches(cfg)
    if mismatches:
        print("Index mismatch (run list-cameras to verify):")
        for line in mismatches:
            print(f"  - {line}")
        print()

    multi = len(cfg.cameras) > 1
    if each and not together:
        run_together = False
    elif together:
        run_together = True
    else:
        run_together = multi

    run_each = each or (not multi and not together)
    if together and each:
        run_together = True
        run_each = True

    ok = True
    if run_together and multi:
        ok = _test_configured_cameras_together(cfg) and ok
        if run_each:
            print()
    elif run_together:
        ok = _test_configured_cameras_sequential(cfg) and ok

    if run_each:
        ok = _test_configured_cameras_sequential(cfg) and ok

    if not ok:
        sys.exit(1)
    if run_together and run_each and multi:
        print("\nAll sequential and concurrent camera tests passed.")
    elif run_together and multi:
        pass  # message already printed
    elif not run_together or not multi:
        pass  # sequential message already printed
