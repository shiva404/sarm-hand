"""Tests for leader→follower action smoothing."""

from __future__ import annotations

from sarm_hand.rerun_viz import smooth_action_targets


def test_smooth_action_targets_noop_at_one():
    target = {"shoulder_pan.pos": 10.0, "gripper.pos": 50.0}
    assert smooth_action_targets({"shoulder_pan.pos": 0.0}, target, alpha=1.0) == target


def test_smooth_action_targets_blends_toward_target():
    prev = {"shoulder_pan.pos": 0.0}
    target = {"shoulder_pan.pos": 10.0}
    out = smooth_action_targets(prev, target, alpha=0.5)
    assert out["shoulder_pan.pos"] == 5.0


def test_smooth_action_targets_first_frame_copies_target():
    target = {"elbow_flex.pos": -20.0}
    out = smooth_action_targets(None, target, alpha=0.3)
    assert out == target
