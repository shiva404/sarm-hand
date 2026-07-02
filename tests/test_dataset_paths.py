"""Tests for LeRobot dataset path resolution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sarm_hand.config import ProjectConfig
from sarm_hand.dataset_session import recording_stamp, resolve_recording_paths


def test_resolve_dataset_path_includes_repo_id_segments() -> None:
    cfg = ProjectConfig.load()
    path = cfg.resolve_dataset_path("local/sarm101-dataset")
    assert path == cfg.resolve_dataset_root() / "local" / "sarm101-dataset"


def test_session_repo_id_appends_timestamp() -> None:
    when = datetime(2026, 6, 20, 13, 40, 43)
    assert (
        ProjectConfig.session_repo_id("local/sarm101-dataset", when=when)
        == "local/sarm101-dataset-20260620-134043-000000"
    )


def test_resolve_session_dataset_path() -> None:
    cfg = ProjectConfig.load()
    when = datetime(2026, 6, 20, 13, 40, 43)
    session_id, path = cfg.resolve_session_dataset_path(when=when)
    assert session_id == "local/sarm101-dataset-20260620-134043-000000"
    assert path == cfg.resolve_dataset_root() / "local/sarm101-dataset-20260620-134043-000000"


def test_recording_stamp_matches_record_sim() -> None:
    when = datetime(2026, 6, 20, 13, 40, 43)
    assert recording_stamp(when) == "20260620-134043-000000"


def test_require_local_dataset_exits_when_missing(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from sarm_hand import data as data_mod

    cfg = ProjectConfig.load()
    cfg.dataset.root = str(tmp_path / "datasets")
    cfg.dataset.repo_id = "local/sarm101-dataset"

    monkeypatch.setattr(data_mod, "ProjectConfig", type("C", (), {"load": staticmethod(lambda: cfg)}))

    with pytest.raises(SystemExit) as exc:
        data_mod._require_local_dataset(cfg, "local/sarm101-dataset")
    assert exc.value.code == 1


def test_require_local_dataset_returns_path_when_present(tmp_path: Path) -> None:
    from sarm_hand.data import _require_local_dataset

    cfg = ProjectConfig.load()
    cfg.dataset.root = str(tmp_path / "datasets")
    cfg.dataset.repo_id = "local/sarm101-dataset"
    ds_path = cfg.resolve_dataset_path()
    (ds_path / "meta").mkdir(parents=True)
    (ds_path / "meta" / "info.json").write_text("{}")

    assert _require_local_dataset(cfg, "local/sarm101-dataset") == ds_path


def test_record_leader_uses_timestamped_dataset_dir(monkeypatch, tmp_path: Path) -> None:
    from sarm_hand import record as record_mod

    cfg = ProjectConfig.load()
    cfg.dataset.root = str(tmp_path / "datasets")
    cfg.dataset.repo_id = "local/sarm101-dataset"
    when = datetime(2026, 6, 20, 13, 40, 43)
    captured: dict[str, Path | str] = {}

    def fake_session(_cfg, **kwargs):
        session_id, path = resolve_recording_paths(
            base_repo=cfg.dataset.repo_id,
            root=cfg.resolve_dataset_root(),
            repo_id=None,
            resume=False,
            timestamp=True,
            when=when,
        )
        captured["repo_id"] = session_id
        captured["path"] = path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(record_mod, "_run_hardware_record_session", fake_session)
    monkeypatch.setattr(record_mod, "ProjectConfig", type("C", (), {"load": staticmethod(lambda: cfg)}))
    monkeypatch.setattr(record_mod, "ensure_port", lambda p, _: p or "/dev/ttyUSB0")

    record_mod.record_leader(follower_port="/dev/f", leader_port="/dev/l", num_episodes=1)

    expected = cfg.resolve_dataset_root() / "local/sarm101-dataset-20260620-134043-000000"
    assert captured["repo_id"] == "local/sarm101-dataset-20260620-134043-000000"
    assert captured["path"] == expected


def test_resolve_dataset_lookup_skips_empty_latest_session(tmp_path: Path) -> None:
    import json

    from sarm_hand.data import resolve_dataset_lookup, write_latest_session_pointer

    cfg = ProjectConfig.load()
    cfg.dataset.root = str(tmp_path / "datasets")
    root = cfg.resolve_dataset_root()

    empty_id = "local/sarm101-dataset-empty"
    empty_path = root / "local" / "sarm101-dataset-empty"
    (empty_path / "meta").mkdir(parents=True)
    (empty_path / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 0, "total_frames": 0})
    )

    good_id = "local/sarm101-dataset-good"
    good_path = root / "local" / "sarm101-dataset-good"
    (good_path / "meta").mkdir(parents=True)
    (good_path / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 12})
    )

    write_latest_session_pointer(cfg, empty_id, empty_path)
    resolved_id, resolved_path = resolve_dataset_lookup(cfg, latest=True)
    assert resolved_id == good_id
    assert resolved_path == good_path


def test_resolve_training_dataset_skips_test_subsample_for_full_dataset(tmp_path: Path) -> None:
    import json

    from sarm_hand.data import resolve_training_dataset, write_latest_session_pointer

    cfg = ProjectConfig.load()
    cfg.dataset.root = str(tmp_path / "datasets")
    root = cfg.resolve_dataset_root()

    tiny_id = "local/test-subsample-ep0-fps10"
    tiny_path = root / "local" / "test-subsample-ep0-fps10"
    (tiny_path / "meta").mkdir(parents=True)
    (tiny_path / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 250, "fps": 10})
    )

    full_id = "local/sarm101-dataset-full"
    full_path = root / "local" / "sarm101-dataset-full"
    (full_path / "meta").mkdir(parents=True)
    (full_path / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 20, "total_frames": 14539, "fps": 30})
    )

    write_latest_session_pointer(cfg, tiny_id, tiny_path)
    resolved_id, resolved_path = resolve_training_dataset(cfg, require_frames=True)
    assert resolved_id == full_id
    assert resolved_path == full_path


def test_leader_record_loop_imports_build_dataset_frame() -> None:
    import inspect

    from sarm_hand import rerun_viz

    source = inspect.getsource(rerun_viz.leader_record_loop)
    assert "lerobot.datasets.feature_utils import build_dataset_frame" in source


def test_dataset_push_requires_hub_flag(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from sarm_hand import data as data_mod

    cfg = ProjectConfig.load()
    cfg.dataset.root = str(tmp_path / "datasets")
    cfg.dataset.repo_id = "local/sarm101-dataset"
    cfg.dataset.push_to_hub = False
    ds_path = cfg.resolve_dataset_path()
    (ds_path / "meta").mkdir(parents=True)
    (ds_path / "meta" / "info.json").write_text("{}")

    monkeypatch.setattr(data_mod, "ProjectConfig", type("C", (), {"load": staticmethod(lambda: cfg)}))

    with pytest.raises(SystemExit) as exc:
        data_mod.dataset_push("local/sarm101-dataset", push_to_hub=False)
    assert exc.value.code == 1
