"""Rerun helpers for live teleop motor plots.

Rerun 0.26 requires ``SeriesLines`` + ``Scalars`` on a named timeline **and** a
``TimeSeriesView`` blueprint — otherwise the viewer opens with no graph panels.
"""

from __future__ import annotations

import numbers
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from .config import JOINT_NAMES

if TYPE_CHECKING:
    from lerobot.processor import RobotProcessorPipeline
    from lerobot.robots import Robot
    from lerobot.teleoperators import Teleoperator
    from lerobot.types import RobotAction, RobotObservation

TELEOP_TIMELINE = "step"
FOLLOWER_ROOT = "motors/follower"
LEADER_ROOT = "motors/leader"
CAMERA_ROOT = "cameras"


def _to_float(value: Any) -> float:
    if isinstance(value, np.ndarray):
        return float(value.item()) if value.ndim == 0 else float(value.flat[0])
    if hasattr(value, "item") and callable(value.item):
        return float(value.item())
    return float(value)


def _is_motor_pos_key(key: str) -> bool:
    return str(key).endswith(".pos")


def _motor_joint_name(key: str) -> str:
    return str(key).removesuffix(".pos")


def _leader_path(joint: str) -> str:
    return f"{LEADER_ROOT}/{joint}"


def smooth_action_targets(
    previous: dict[str, float] | None,
    target: dict[str, float],
    *,
    alpha: float,
) -> dict[str, float]:
    """EMA blend toward new leader targets (alpha=1 → no smoothing)."""
    if previous is None or alpha >= 1.0:
        return dict(target)
    blend = float(np.clip(alpha, 0.0, 1.0))
    out: dict[str, float] = {}
    for key, value in target.items():
        if key in previous:
            out[key] = previous[key] + blend * (float(value) - previous[key])
        else:
            out[key] = float(value)
    return out


def motor_observation(robot: Robot) -> RobotObservation:
    """Proprioception only — skips camera reads for fast control ticks."""
    positions = robot.bus.sync_read("Present_Position")
    return {f"{motor}.pos": val for motor, val in positions.items()}


def leader_record_loop(
    teleop: Teleoperator,
    robot: Robot,
    *,
    events: dict[str, bool],
    control_fps: int,
    control_time_s: float,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    dataset: Any | None = None,
    dataset_fps: int | None = None,
    single_task: str | None = None,
    remap_action: Callable[[dict[str, float]], dict[str, float]] | None = None,
    action_smoothing: float = 1.0,
    display_data: bool = False,
    camera_names: list[str] | None = None,
    show_countdown: bool = True,
    phase: Literal["record", "reset"] = "record",
) -> int:
    """High-rate leader→follower control; dataset frames at ``dataset_fps`` when set.

    Returns the number of dataset frames buffered this episode.
    """
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.robot_utils import precise_sleep

    from .recording_ui import clear_countdown_line, write_countdown_line

    if dataset is not None:
        if dataset_fps is None:
            raise ValueError("dataset_fps is required when dataset is set")
        if dataset.fps != dataset_fps:
            raise ValueError(
                f"Dataset fps ({dataset.fps}) must match dataset_fps ({dataset_fps})"
            )

    control_interval = 1.0 / control_fps
    record_interval = (1.0 / dataset_fps) if dataset_fps else None
    has_cameras = bool(getattr(robot, "cameras", None))

    smoothed_action: dict[str, float] | None = None
    start_t = time.perf_counter()
    next_record_t = start_t
    step = 0
    frames_recorded = 0
    last_countdown_sec = -1

    while time.perf_counter() - start_t < control_time_s:
        if events.get("exit_early"):
            events["exit_early"] = False
            break
        if events.get("stop_recording"):
            break

        elapsed = time.perf_counter() - start_t
        if show_countdown:
            sec_left = max(0, int(control_time_s - elapsed + 0.999))
            if sec_left != last_countdown_sec:
                last_countdown_sec = sec_left
                write_countdown_line(
                    seconds_left=sec_left,
                    frames_recorded=frames_recorded,
                    phase=phase,
                )

        loop_start = time.perf_counter()
        is_record_frame = False
        if dataset is not None and record_interval is not None:
            is_record_frame = loop_start >= next_record_t
            if is_record_frame:
                next_record_t += record_interval

        if is_record_frame or not has_cameras:
            obs = robot.get_observation()
        else:
            obs = motor_observation(robot)

        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))
        if remap_action is not None:
            teleop_action = remap_action(teleop_action)
        robot_action = robot_action_processor((teleop_action, obs))
        smoothed_action = smooth_action_targets(
            smoothed_action,
            robot_action,
            alpha=action_smoothing,
        )
        _ = robot.send_action(smoothed_action)

        if dataset is not None and is_record_frame:
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, teleop_action, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)
            frames_recorded += 1

        if display_data:
            obs_processed = robot_observation_processor(obs)
            # Joints only — streaming 3×640×480 RGB to Rerun exceeds its ~1.6 GiB cache.
            log_teleop_frame(
                step,
                observation=obs_processed,
                action=teleop_action,
                camera_frames=None,
            )
            step += 1

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(control_interval - dt_s, 0.0))

    if show_countdown:
        clear_countdown_line()

    return frames_recorded


def _follower_path(joint: str) -> str:
    return f"{FOLLOWER_ROOT}/{joint}"


def _camera_path(name: str) -> str:
    return f"/{CAMERA_ROOT}/{name}"


def _send_teleop_blueprint(*, with_cameras: bool, camera_names: list[str] | None = None) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    step_window = rrb.VisibleTimeRange(
        TELEOP_TIMELINE,
        start=rrb.TimeRangeBoundary.cursor_relative(seq=-600),
        end=rrb.TimeRangeBoundary.cursor_relative(),
    )
    follower_view = rrb.TimeSeriesView(
        origin=f"/{FOLLOWER_ROOT}",
        name="Follower joints",
        axis_y=rrb.ScalarAxis(range=(-110.0, 110.0)),
        time_ranges=[step_window],
    )
    leader_view = rrb.TimeSeriesView(
        origin=f"/{LEADER_ROOT}",
        name="Leader joints",
        axis_y=rrb.ScalarAxis(range=(-110.0, 110.0)),
        time_ranges=[step_window],
    )

    if with_cameras and camera_names:
        cam_row = rrb.Horizontal(
            *[
                rrb.Spatial2DView(
                    origin=_camera_path(name),
                    name=name,
                    time_ranges=[step_window],
                )
                for name in camera_names
            ],
            name="Cameras",
        )
        main = rrb.Vertical(
            rrb.Horizontal(follower_view, leader_view),
            cam_row,
            row_shares=[2, 1],
        )
    else:
        main = rrb.Horizontal(follower_view, leader_view)

    rr.send_blueprint(
        rrb.Blueprint(
            main,
            rrb.TimePanel(timeline=TELEOP_TIMELINE, fps=30.0),
            collapse_panels=True,
        )
    )


def init_leader_teleop_rerun(
    session_name: str = "teleoperation",
    *,
    with_cameras: bool = False,
    camera_names: list[str] | None = None,
) -> None:
    """Start Rerun, register joint series, and open follower/leader graph panels."""
    import rerun as rr
    from lerobot.utils.visualization_utils import init_rerun

    init_rerun(session_name=session_name)

    for joint in JOINT_NAMES:
        rr.log(
            _follower_path(joint),
            rr.SeriesLines(names=f"follower {joint}"),
            static=True,
        )
        rr.log(
            _leader_path(joint),
            rr.SeriesLines(names=f"leader {joint}"),
            static=True,
        )

    _send_teleop_blueprint(with_cameras=with_cameras, camera_names=camera_names)


def _log_camera_frame(key: str, frame: Any, *, compress_images: bool) -> None:
    import rerun as rr

    if frame is None or not isinstance(frame, np.ndarray):
        return
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim < 2:
        return
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(arr.max()) <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
    rr.log(_camera_path(key), entity=entity)


def log_teleop_frame(
    step: int,
    *,
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    camera_frames: dict[str, Any] | None = None,
    compress_images: bool = False,
) -> None:
    """Log one teleop frame on the ``step`` timeline."""
    import rerun as rr

    rr.set_time(TELEOP_TIMELINE, sequence=step)

    if camera_frames:
        for key, value in camera_frames.items():
            if value is not None:
                _log_camera_frame(str(key), value, compress_images=compress_images)

    if observation:
        for key, value in observation.items():
            if value is None:
                continue
            if _is_motor_pos_key(key):
                joint = _motor_joint_name(key)
                if isinstance(value, (float, int, numbers.Real, np.integer, np.floating)):
                    rr.log(_follower_path(joint), rr.Scalars(_to_float(value)))
                elif isinstance(value, np.ndarray) and value.ndim == 0:
                    rr.log(_follower_path(joint), rr.Scalars(_to_float(value)))
            elif isinstance(value, np.ndarray) and not _is_motor_pos_key(key):
                if camera_frames is None or key not in camera_frames:
                    _log_camera_frame(str(key), value, compress_images=compress_images)

    if action:
        for key, value in action.items():
            if value is None or not _is_motor_pos_key(key):
                continue
            joint = _motor_joint_name(key)
            if isinstance(value, (float, int, numbers.Real, np.integer, np.floating)):
                rr.log(_leader_path(joint), rr.Scalars(_to_float(value)))
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                rr.log(_leader_path(joint), rr.Scalars(_to_float(value)))


def leader_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    remap_action: Callable[[dict[str, float]], dict[str, float]] | None = None,
    camera_names: list[str] | None = None,
    action_smoothing: float = 1.0,
) -> None:
    """Leader-follower loop with Rerun motor time series."""
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import move_cursor_up

    has_cameras = bool(getattr(robot, "cameras", None))
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    step = 0
    smoothed_action: dict[str, float] | None = None
    while True:
        loop_start = time.perf_counter()

        if has_cameras and display_data and camera_names:
            obs = robot.get_observation()
        elif has_cameras:
            obs = motor_observation(robot)
        else:
            obs = robot.get_observation()
        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))
        if remap_action is not None:
            teleop_action = remap_action(teleop_action)
        robot_action = robot_action_processor((teleop_action, obs))
        smoothed_action = smooth_action_targets(
            smoothed_action,
            robot_action,
            alpha=action_smoothing,
        )
        _ = robot.send_action(smoothed_action)

        if display_data:
            obs_transition = robot_observation_processor(obs)
            camera_frames = None
            if camera_names:
                camera_frames = {
                    name: obs[name]
                    for name in camera_names
                    if name in obs and isinstance(obs[name], np.ndarray)
                }
            log_teleop_frame(
                step,
                observation=obs_transition,
                action=teleop_action,
                camera_frames=camera_frames,
                compress_images=display_compressed_images,
            )
            step += 1

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in smoothed_action.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(smoothed_action) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))

        if duration is not None and time.perf_counter() - start >= duration:
            return
