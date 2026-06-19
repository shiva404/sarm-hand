"""Rerun helpers for live teleop motor plots.

Rerun 0.26 requires ``SeriesLines`` + ``Scalars`` on a named timeline **and** a
``TimeSeriesView`` blueprint — otherwise the viewer opens with no graph panels.
"""

from __future__ import annotations

import numbers
import time
from typing import TYPE_CHECKING, Any

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


def _follower_path(joint: str) -> str:
    return f"{FOLLOWER_ROOT}/{joint}"


def _leader_path(joint: str) -> str:
    return f"{LEADER_ROOT}/{joint}"


def _send_teleop_blueprint(*, with_cameras: bool) -> None:
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

    if with_cameras:
        main = rrb.Vertical(
            rrb.Horizontal(follower_view, leader_view),
            rrb.Spatial2DView(origin=f"/{CAMERA_ROOT}", name="Camera"),
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

    _send_teleop_blueprint(with_cameras=with_cameras)


def _log_camera_frame(key: str, frame: Any, *, compress_images: bool) -> None:
    import rerun as rr

    if frame is None or not isinstance(frame, np.ndarray):
        return
    arr = frame
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim < 2:
        return
    entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
    rr.log(f"{CAMERA_ROOT}/{key}", entity=entity)


def log_teleop_frame(
    step: int,
    *,
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """Log one teleop frame on the ``step`` timeline."""
    import rerun as rr

    rr.set_time(TELEOP_TIMELINE, sequence=step)

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
            elif isinstance(value, np.ndarray):
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
) -> None:
    """Leader-follower loop with Rerun motor time series."""
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import move_cursor_up

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    step = 0
    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()
        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_teleop_frame(
                step,
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )
            step += 1

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return
