"""Tests for config-driven camera fail-on-error behavior."""

from __future__ import annotations

from sarm_hand.cameras import camera_black_fallback_enabled, install_all_camera_patches
from sarm_hand.config import CameraBehaviorSettings, ProjectConfig


def test_camera_fail_on_error_default_true():
    cfg = ProjectConfig.load()
    assert cfg.camera.fail_on_error is True
    assert camera_black_fallback_enabled(cfg) is False


def test_camera_black_fallback_when_fail_on_error_disabled():
    cfg = ProjectConfig(
        camera=CameraBehaviorSettings(fail_on_error=False),
    )
    assert camera_black_fallback_enabled(cfg) is True


def test_install_all_camera_patches_respects_fail_on_error(monkeypatch):
    import sarm_hand.cameras as cameras_mod

    monkeypatch.setattr(cameras_mod, "_RESILIENT_CAMERA_PATCHED", False)
    calls: list[bool] = []

    def _track_resilient():
        calls.append(True)

    monkeypatch.setattr(cameras_mod, "install_resilient_camera_patch", _track_resilient)
    monkeypatch.setattr(cameras_mod, "install_udp_camera_patch", lambda: None)
    monkeypatch.setattr(cameras_mod, "install_stream_camera_patch", lambda: None)
    monkeypatch.setattr(cameras_mod, "install_usb_downscale_patch", lambda: None)
    monkeypatch.setattr(cameras_mod, "install_follower_connect_patch", lambda: None)

    strict_cfg = ProjectConfig(camera=CameraBehaviorSettings(fail_on_error=True))
    install_all_camera_patches(cfg=strict_cfg, resilient=None)
    assert calls == []

    monkeypatch.setattr(cameras_mod, "_RESILIENT_CAMERA_PATCHED", False)
    resilient_cfg = ProjectConfig(camera=CameraBehaviorSettings(fail_on_error=False))
    install_all_camera_patches(cfg=resilient_cfg, resilient=None)
    assert calls == [True]
