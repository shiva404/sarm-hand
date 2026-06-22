"""Tests for leader → sim mirror smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from sarm_hand.config import JOINT_NAMES, ProjectConfig
from sarm_hand.genesis.calibration import load_calibration
from sarm_hand.genesis.mirror import filter_raw_counts, smooth_mirror_radians
from sarm_hand.genesis.tensors import to_numpy


def test_to_numpy_from_list():
    arr = to_numpy([1.0, 2.0, 3.0])
    assert arr.dtype == np.float64
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])


class _FakeTensor:
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data


def test_to_numpy_from_device_tensor():
    arr = to_numpy(_FakeTensor([0.1, 0.2]))
    np.testing.assert_allclose(arr, [0.1, 0.2])


def test_to_numpy_from_torch_tensor():
    torch = pytest.importorskip("torch")
    t = torch.tensor([0.3, 0.4], device="cpu")
    np.testing.assert_allclose(to_numpy(t), [0.3, 0.4])


class _MpsStubTensor:
    """Mimics a tensor whose ``numpy()`` fails until ``cpu()`` is applied."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float64)
        self._on_device = True

    def detach(self):
        return self

    def cpu(self):
        return _FakeTensor(self._data)

    def numpy(self):
        if self._on_device:
            raise TypeError("can't convert mps:0 device type tensor to numpy")
        return self._data


def test_to_numpy_mps_like_tensor():
    arr = to_numpy(_MpsStubTensor([0.5, 0.6]))
    np.testing.assert_allclose(arr, [0.5, 0.6])


def test_mirror_settings_in_config():
    cfg = ProjectConfig.load()
    assert cfg.genesis.mirror_kinematic is True
    assert cfg.genesis.mirror_rate_hz == 30.0
    assert cfg.genesis.mirror_substeps == 1
    assert cfg.genesis.mirror_grasp_substeps == 2
    assert cfg.genesis.mirror_grasp_carry is True
    assert cfg.genesis.grasp_weld is False
    assert cfg.genesis.grasp_close_deg == 48.0
    assert cfg.genesis.grasp_open_deg == 42.0
    assert cfg.genesis.gripper_sim_extra_from_deg == 44.0
    assert cfg.genesis.mirror_raw_deadband == 2
    assert cfg.genesis.mirror_smoothing == 1.0
    assert cfg.genesis.joints["shoulder_lift"].mirror_raw_deadband == 4


def test_filter_raw_counts_ignores_jitter():
    last = {j: 2000 for j in JOINT_NAMES}
    raw = dict(last)
    raw["shoulder_lift"] = 2003
    filtered = filter_raw_counts(
        raw,
        last_raw=last,
        deadband_for_joint={"shoulder_lift": 6, "shoulder_pan": 3},
    )
    assert filtered["shoulder_lift"] == 2000
    raw["shoulder_lift"] = 2007
    filtered = filter_raw_counts(
        raw,
        last_raw=last,
        deadband_for_joint={"shoulder_lift": 6},
    )
    assert filtered["shoulder_lift"] == 2007


def test_smooth_mirror_radians_rate_limits():
    cfg = ProjectConfig.load()
    cal = load_calibration("leader", cfg)
    if cal is None:
        pytest.skip("leader calibration not present")
    current = [0.0] * len(JOINT_NAMES)
    target = [1.0] * len(JOINT_NAMES)
    cmd = smooth_mirror_radians(
        target,
        current_rad=current,
        previous_cmd_rad=current,
        cfg=cfg,
        calibration=cal,
        max_norm_step=5.0,
        smoothing=1.0,
        snap=False,
    )
    assert abs(float(cmd[0])) < abs(float(target[0]))
