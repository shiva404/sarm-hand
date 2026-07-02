"""Tests for ACT policy wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sarm_hand.config import ProjectConfig
from sarm_hand.policy import (
    _inference_blend_alpha,
    _require_act_cameras,
    apply_act_inference_overrides,
    resolve_act_training_batch_size,
)


def test_require_act_cameras_accepts_front_and_wrist():
    cfg = ProjectConfig.load()
    _require_act_cameras(cfg)
    assert "front" in cfg.cameras
    assert "wrist" in cfg.cameras


def test_resolve_act_training_batch_size_mps_cap():
    cfg = ProjectConfig.load()
    assert resolve_act_training_batch_size("mps", 16, cfg) == 8
    assert resolve_act_training_batch_size("cuda", 16, cfg) == 16


def test_act_inference_enables_step_clamp_by_default():
    cfg = ProjectConfig.load()
    assert cfg.policies.act.max_relative_target == 10.0


def test_apply_act_inference_overrides_temporal_ensemble():
    policy_cfg = SimpleNamespace(
        temporal_ensemble_coeff=None,
        n_action_steps=100,
    )
    act = SimpleNamespace(
        temporal_ensemble_coeff=0.01,
        inference_n_action_steps=10,
    )
    apply_act_inference_overrides(policy_cfg, act)
    assert policy_cfg.temporal_ensemble_coeff == 0.01
    assert policy_cfg.n_action_steps == 1


def test_apply_act_inference_overrides_chunk_queue():
    policy_cfg = SimpleNamespace(
        temporal_ensemble_coeff=None,
        n_action_steps=100,
    )
    act = SimpleNamespace(
        temporal_ensemble_coeff=None,
        inference_n_action_steps=30,
    )
    apply_act_inference_overrides(policy_cfg, act)
    assert policy_cfg.n_action_steps == 30


def test_read_checkpoint_training_dataset_from_act_checkpoint():
    from pathlib import Path

    from sarm_hand.policy import _read_checkpoint_training_dataset

    model = Path("outputs/train/sarm101_act/checkpoints/004000/pretrained_model")
    if not (model / "train_config.json").is_file():
        return
    trained = _read_checkpoint_training_dataset(str(model))
    assert trained is not None
    repo_id, path, info = trained
    assert "test-subsample" in repo_id
    assert path.is_dir()
    assert int(info.get("total_frames", 0)) > 0


def test_act_training_steps_from_dataset(tmp_path: Path) -> None:
    import json

    from sarm_hand.config import ActPolicySettings, ProjectConfig
    from sarm_hand.policy import resolve_act_training_steps

    cfg = ProjectConfig()
    cfg.dataset.num_episodes = 20
    cfg.dataset.episode_time_s = 25
    cfg.dataset.fps = 30
    act = ActPolicySettings(train_epochs=25, train_steps=None, train_batch_size=8)

    dataset_dir = tmp_path / "ds"
    (dataset_dir / "meta").mkdir(parents=True)
    # 20 eps, 14539 frames (matches sarm101-dataset-20260627)
    (dataset_dir / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 20, "total_frames": 14539, "fps": 30})
    )

    steps, per_epoch, frames = resolve_act_training_steps(act, cfg, dataset_dir, 8)
    assert frames == 14539
    assert per_epoch == 1818  # ceil(14539/8)
    assert steps == 25 * 1818


def test_act_training_steps_cli_override(tmp_path: Path) -> None:
    import json

    from sarm_hand.config import ActPolicySettings, ProjectConfig
    from sarm_hand.policy import resolve_act_training_steps

    cfg = ProjectConfig()
    act = ActPolicySettings(train_epochs=25, train_batch_size=8)
    dataset_dir = tmp_path / "ds"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 20, "total_frames": 14539, "fps": 30})
    )

    steps, per_epoch, frames = resolve_act_training_steps(
        act, cfg, dataset_dir, 8, steps_override=10000
    )
    assert steps == 10000
    assert per_epoch == 1818
    assert frames == 14539


def test_act_inference_defaults():
    cfg = ProjectConfig.load()
    assert cfg.policies.act.action_smoothing == 1.0
    assert cfg.policies.act.temporal_ensemble_coeff is None
    assert cfg.policies.act.inference_n_action_steps == 50
    assert cfg.policies.act.save_freq == 600
    assert cfg.policies.act.train_epochs == 25
    assert cfg.policies.act.train_steps is None
    assert cfg.dataset.num_episodes == 20
    assert cfg.dataset.episode_time_s == 25
    assert cfg.policies.act.episode_time_s == 25
    assert cfg.dataset.fps == 10
    assert cfg.policies.act.control_fps == 10


def test_inference_blend_alpha_startup_ramp():
    assert _inference_blend_alpha(
        0,
        inference_blend_steps=60,
        replan_blend_steps=8,
        n_action_steps=50,
        use_temporal_ensemble=False,
    ) == pytest.approx(1 / 60)
    assert _inference_blend_alpha(
        59,
        inference_blend_steps=60,
        replan_blend_steps=8,
        n_action_steps=50,
        use_temporal_ensemble=False,
    ) == pytest.approx(1.0)


def test_inference_blend_alpha_replan_boundary():
    # First chunk boundary after startup ease-in (step 100 % 50 == 0).
    assert _inference_blend_alpha(
        100,
        inference_blend_steps=60,
        replan_blend_steps=8,
        n_action_steps=50,
        use_temporal_ensemble=False,
    ) == pytest.approx(1 / 8)
    assert _inference_blend_alpha(
        107,
        inference_blend_steps=60,
        replan_blend_steps=8,
        n_action_steps=50,
        use_temporal_ensemble=False,
    ) == pytest.approx(1.0)
