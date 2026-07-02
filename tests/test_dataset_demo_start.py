"""Tests for training demo start pose analysis."""

from pathlib import Path

from sarm_hand.data import dataset_demo_start_action_mean


def test_dataset_demo_start_action_mean_local_session():
    ds = Path("data/datasets/local/sarm101-dataset-20260627-101519-394717")
    if not (ds / "meta" / "info.json").is_file():
        return

    start = dataset_demo_start_action_mean(ds)
    assert start is not None
    assert "shoulder_lift.pos" in start
    # Demos in this session begin in the folded rest pose.
    assert start["shoulder_lift.pos"] < -90
    assert start["elbow_flex.pos"] > 90
