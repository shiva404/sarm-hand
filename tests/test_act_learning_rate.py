"""Tests for ACT learning-rate patching on resume."""

import json
from pathlib import Path

from sarm_hand.policy import _apply_act_learning_rate


def test_apply_act_learning_rate_patches_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "002000"
    (ckpt / "pretrained_model").mkdir(parents=True)
    (ckpt / "training_state").mkdir(parents=True)

    (ckpt / "pretrained_model" / "train_config.json").write_text(
        json.dumps(
            {
                "policy": {"optimizer_lr": 1e-05, "optimizer_lr_backbone": 1e-05},
                "optimizer": {"lr": 1e-05},
            }
        )
    )
    (ckpt / "training_state" / "optimizer_param_groups.json").write_text(
        json.dumps([{"lr": 1e-05, "params": [0]}, {"lr": 1e-05, "params": [1]}])
    )

    _apply_act_learning_rate(ckpt, 0.01)

    train_cfg = json.loads((ckpt / "pretrained_model" / "train_config.json").read_text())
    assert train_cfg["policy"]["optimizer_lr"] == 0.01
    assert train_cfg["policy"]["optimizer_lr_backbone"] == 0.01
    assert train_cfg["optimizer"]["lr"] == 0.01

    groups = json.loads((ckpt / "training_state" / "optimizer_param_groups.json").read_text())
    assert groups[0]["lr"] == 0.01
    assert groups[1]["lr"] == 0.01
