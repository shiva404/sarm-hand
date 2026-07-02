"""Tests for syncing servo goals to present position on connect."""

from unittest.mock import MagicMock

from sarm_hand.robot import sync_follower_goals_to_present


def test_sync_follower_goals_to_present():
    present = {
        "shoulder_pan": 12.0,
        "shoulder_lift": -3.5,
        "elbow_flex": 8.0,
        "wrist_flex": -1.0,
        "wrist_roll": 0.0,
        "gripper": 50.0,
    }
    bus = MagicMock()
    bus.sync_read.return_value = present
    robot = MagicMock()
    robot.bus = bus

    sync_follower_goals_to_present(robot)

    bus.sync_read.assert_called_once_with("Present_Position")
    bus.sync_write.assert_called_once_with("Goal_Position", present)
