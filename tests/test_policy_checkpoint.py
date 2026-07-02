"""Tests for fine-tuned policy checkpoint path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from sarm_hand.policy import ensure_local_pretrained_dir, resolve_policy_checkpoint_path


def test_resolve_training_output_dir_to_last_checkpoint():
    root = Path("outputs/train/sarm101_smolvla")
    if not (root / "checkpoints" / "last" / "pretrained_model" / "config.json").is_file():
        return
    resolved = resolve_policy_checkpoint_path(str(root))
    assert resolved.endswith("pretrained_model")
    assert Path(resolved, "config.json").is_file()


def test_hub_model_id_unchanged():
    assert resolve_policy_checkpoint_path("lerobot/smolvla_base") == "lerobot/smolvla_base"


def test_pretrained_model_dir_unchanged():
    root = Path("outputs/train/sarm101_smolvla")
    direct = root / "checkpoints" / "last" / "pretrained_model"
    if not (direct / "config.json").is_file():
        return
    assert resolve_policy_checkpoint_path(str(direct)) == str(direct.resolve())


def test_resolve_numbered_checkpoint_when_last_missing(tmp_path: Path) -> None:
    train_root = tmp_path / "outputs" / "train" / "sarm101_act"
    model = train_root / "checkpoints" / "004000" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")

    broken_last = train_root / "checkpoints" / "last" / "pretrained_model"
    resolved = resolve_policy_checkpoint_path(str(broken_last))
    assert resolved == str(model.resolve())


def test_ensure_local_pretrained_dir_exits_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "outputs" / "train" / "missing" / "pretrained_model"
    with pytest.raises(SystemExit) as exc:
        ensure_local_pretrained_dir(str(missing))
    assert exc.value.code == 1
