"""Tests for Rerun teleop visualization helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sarm_hand.config import JOINT_NAMES
from sarm_hand.rerun_viz import (
    FOLLOWER_ROOT,
    LEADER_ROOT,
    TELEOP_TIMELINE,
    init_leader_teleop_rerun,
    log_teleop_frame,
)


def test_init_leader_teleop_rerun_registers_joint_series_and_blueprint() -> None:
    mock_rr = MagicMock()
    mock_rrb = MagicMock()
    mock_rrb.Blueprint = MagicMock(side_effect=lambda *a, **k: MagicMock())
    mock_rrb.TimeSeriesView = MagicMock(side_effect=lambda **k: MagicMock())
    mock_rrb.Horizontal = MagicMock(side_effect=lambda *a: MagicMock())
    mock_rrb.TimePanel = MagicMock(side_effect=lambda **k: MagicMock())
    mock_rrb.ScalarAxis = MagicMock(side_effect=lambda **k: MagicMock())
    mock_rrb.VisibleTimeRange = MagicMock(side_effect=lambda *a, **k: MagicMock())
    mock_rrb.TimeRangeBoundary = MagicMock()
    mock_rrb.TimeRangeBoundary.cursor_relative = MagicMock(return_value="end")

    with (
        patch("lerobot.utils.visualization_utils.init_rerun"),
        patch.dict("sys.modules", {"rerun": mock_rr, "rerun.blueprint": mock_rrb}),
    ):
        init_leader_teleop_rerun(session_name="test")

    assert mock_rr.log.call_count == len(JOINT_NAMES) * 2
    paths = {call.args[0] for call in mock_rr.log.call_args_list}
    assert f"{FOLLOWER_ROOT}/shoulder_pan" in paths
    assert f"{LEADER_ROOT}/gripper" in paths
    mock_rr.send_blueprint.assert_called_once()


def test_log_teleop_frame_sets_step_timeline_and_motor_paths() -> None:
    mock_rr = MagicMock()
    obs = {"shoulder_pan.pos": 1.0, "gripper.pos": 50.0}
    action = {"shoulder_pan.pos": 2.0}
    with patch.dict("sys.modules", {"rerun": mock_rr}):
        log_teleop_frame(7, observation=obs, action=action)

    mock_rr.set_time.assert_called_once_with(TELEOP_TIMELINE, sequence=7)
    logged_paths = {call.args[0] for call in mock_rr.log.call_args_list}
    assert f"{FOLLOWER_ROOT}/shoulder_pan" in logged_paths
    assert f"{FOLLOWER_ROOT}/gripper" in logged_paths
    assert f"{LEADER_ROOT}/shoulder_pan" in logged_paths
