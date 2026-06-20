"""Tests for policy device handling."""

from __future__ import annotations

import torch

from sarm_hand.policy import _move_tree_to_device


def test_move_tree_to_device_nested():
    data = {
        "observation.state": torch.zeros(1, 6),
        "observation.images.front": torch.zeros(1, 3, 64, 64),
        "task": "pick",
    }
    moved = _move_tree_to_device(data, torch.device("mps"))
    assert moved["observation.state"].device.type == "mps"
    assert moved["observation.images.front"].device.type == "mps"
    assert moved["task"] == "pick"
