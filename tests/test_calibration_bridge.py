"""Tests for leader ↔ follower calibration remapping."""

from __future__ import annotations

from sarm_hand.calibration_bridge import (
    calibration_mismatch_report,
    merge_calibration_ranges,
    remap_leader_action_to_follower,
)
from sarm_hand.genesis.calibration import (
    norm_from_travel_fraction,
    norm_to_raw,
    travel_fraction,
)


def _sample_leader_cal() -> dict:
    return {
        "shoulder_pan": {"range_min": 1144, "range_max": 2944, "drive_mode": 0},
        "shoulder_lift": {"range_min": 853, "range_max": 3078, "drive_mode": 0},
        "elbow_flex": {"range_min": 856, "range_max": 3068, "drive_mode": 0},
        "wrist_flex": {"range_min": 967, "range_max": 3194, "drive_mode": 0},
        "wrist_roll": {"range_min": 0, "range_max": 4095, "drive_mode": 0},
        "gripper": {"range_min": 1746, "range_max": 2333, "drive_mode": 0},
    }


def _broken_follower_cal() -> dict:
    return {
        "shoulder_pan": {"range_min": 852, "range_max": 3315, "drive_mode": 0},
        "shoulder_lift": {"range_min": 0, "range_max": 4095, "drive_mode": 0},
        "elbow_flex": {"range_min": 0, "range_max": 4095, "drive_mode": 0},
        "wrist_flex": {"range_min": 189, "range_max": 2500, "drive_mode": 0},
        "wrist_roll": {"range_min": 0, "range_max": 4095, "drive_mode": 0},
        "gripper": {"range_min": 1728, "range_max": 2540, "drive_mode": 0},
    }


def test_remap_preserves_travel_fraction():
    leader = _sample_leader_cal()
    follower = _broken_follower_cal()
    action = {"shoulder_lift.pos": 12.5, "gripper.pos": 55.0}
    remapped = remap_leader_action_to_follower(action, leader_cal=leader, follower_cal=follower)
    for joint in ("shoulder_lift", "gripper"):
        key = f"{joint}.pos"
        leader_raw = norm_to_raw(action[key], joint, leader)
        fraction = travel_fraction(leader_raw, joint, leader)
        expected = norm_from_travel_fraction(fraction, joint, follower)
        assert remapped[key] == expected


def test_merge_calibration_ranges_keeps_homing():
    leader = {
        "shoulder_pan": {
            "id": 1,
            "homing_offset": 100,
            "range_min": 1144,
            "range_max": 2944,
            "drive_mode": 0,
        },
        "shoulder_lift": {
            "id": 2,
            "homing_offset": 200,
            "range_min": 853,
            "range_max": 3078,
            "drive_mode": 0,
        },
    }
    follower = {
        "shoulder_pan": {
            "id": 1,
            "homing_offset": 999,
            "range_min": 852,
            "range_max": 3315,
            "drive_mode": 0,
        },
        "shoulder_lift": {
            "id": 2,
            "homing_offset": 888,
            "range_min": 0,
            "range_max": 4095,
            "drive_mode": 0,
        },
    }
    merged = merge_calibration_ranges(leader, follower)
    assert merged["shoulder_pan"]["homing_offset"] == 999
    assert merged["shoulder_pan"]["range_min"] == 1144
    assert merged["shoulder_lift"]["homing_offset"] == 888
    assert merged["shoulder_lift"]["range_max"] == 3078


def test_remap_differs_from_passthrough_when_cals_mismatch():
    leader = _sample_leader_cal()
    follower = _broken_follower_cal()
    action = {"elbow_flex.pos": 25.0}
    remapped = remap_leader_action_to_follower(action, leader_cal=leader, follower_cal=follower)
    assert remapped["elbow_flex.pos"] != action["elbow_flex.pos"]


def test_mismatch_report_flags_broken_follower():
    issues = calibration_mismatch_report(_sample_leader_cal(), _broken_follower_cal())
    assert any("shoulder_lift" in line for line in issues)
    assert any("elbow_flex" in line for line in issues)
