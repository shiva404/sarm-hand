"""Wide-cal linear mapping mode (optional; config genesis.mapping: wide_cal)."""

from __future__ import annotations

import math

import pytest

from sarm_hand.config import GenesisJointSettings, ProjectConfig
from sarm_hand.genesis.calibration import load_calibration, raw_to_norm
from sarm_hand.genesis.units import _full_cal_home_offset_cache, norm_to_radians
from sarm_hand.genesis.urdf_limits import urdf_joint_limits


@pytest.fixture
def wide_cal_cfg() -> ProjectConfig:
    _full_cal_home_offset_cache.clear()
    cfg = ProjectConfig.load()
    cfg.genesis.urdf = "assets/robots/so101/so101_new_calib.urdf"
    cfg.genesis.mapping = "wide_cal"
    cfg.genesis.joints["shoulder_lift"] = GenesisJointSettings(
        sign=1, urdf_min=-3.31613, urdf_max=0.174533, frame_offset=1.5708
    )
    cfg.genesis.joints["elbow_flex"] = GenesisJointSettings(
        sign=1, urdf_min=-0.174533, urdf_max=3.14159, frame_offset=-1.5708
    )
    return cfg


@pytest.fixture
def leader_cal(wide_cal_cfg: ProjectConfig):
    cal = load_calibration("leader", wide_cal_cfg)
    if cal is None:
        pytest.skip("leader calibration not on this machine")
    return cal


def test_wide_cal_norm_gain_near_one_degree_per_unit(wide_cal_cfg, leader_cal):
    hard = urdf_joint_limits(wide_cal_cfg)
    for joint in ("shoulder_lift", "elbow_flex"):
        hn = raw_to_norm(wide_cal_cfg.genesis.home_raw[joint], joint, leader_cal)
        hr = norm_to_radians(hn, joint, wide_cal_cfg, calibration=leader_cal, urdf_limits=hard)
        for delta in (-30.0, -15.0, 15.0, 30.0):
            rad = norm_to_radians(
                hn + delta, joint, wide_cal_cfg, calibration=leader_cal, urdf_limits=hard
            )
            assert math.degrees(rad - hr) == pytest.approx(delta, abs=2.5)
