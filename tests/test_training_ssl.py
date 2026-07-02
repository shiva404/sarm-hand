"""Tests for training SSL / dataset path helpers."""

from __future__ import annotations

import os

from sarm_hand.config import ProjectConfig
from sarm_hand.data import configure_ssl_certificates, resolve_training_dataset


def test_configure_ssl_certificates_sets_bundle(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    configure_ssl_certificates()
    assert os.environ.get("SSL_CERT_FILE")
    assert os.environ.get("REQUESTS_CA_BUNDLE")


def test_resolve_training_dataset_strips_trailing_slash(tmp_path, monkeypatch):
    cfg = ProjectConfig()
    cfg.dataset.root = str(tmp_path)
    repo_id = "local/my-session"
    dataset_dir = tmp_path / "local" / "my-session"
    meta = dataset_dir / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text('{"total_frames": 10, "total_episodes": 1}')
    resolved_id, path = resolve_training_dataset(cfg, f"{repo_id}/", require_frames=False)
    assert resolved_id == repo_id
    assert path == dataset_dir
