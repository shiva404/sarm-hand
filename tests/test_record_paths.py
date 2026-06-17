"""Tests for timestamped Genesis dataset paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from sarm_hand.record_sim import resolve_recording_paths


def test_timestamped_path_is_unique(tmp_path: Path):
    repo1, path1 = resolve_recording_paths(
        base_repo="local/test-genesis",
        root=tmp_path,
        repo_id=None,
        resume=False,
        timestamp=True,
    )
    repo2, path2 = resolve_recording_paths(
        base_repo="local/test-genesis",
        root=tmp_path,
        repo_id=None,
        resume=False,
        timestamp=True,
    )
    assert repo1.startswith("local/test-genesis-")
    assert repo2.startswith("local/test-genesis-")
    assert repo1 != repo2
    assert path1 != path2


def test_resume_requires_existing_dataset(tmp_path: Path):
    with pytest.raises(SystemExit):
        resolve_recording_paths(
            base_repo="local/missing",
            root=tmp_path,
            repo_id="local/missing",
            resume=True,
            timestamp=True,
        )


def test_fixed_repo_fails_if_exists(tmp_path: Path):
    existing = tmp_path / "local/fixed"
    existing.mkdir(parents=True)
    with pytest.raises(SystemExit):
        resolve_recording_paths(
            base_repo="local/fixed",
            root=tmp_path,
            repo_id=None,
            resume=False,
            timestamp=False,
        )
