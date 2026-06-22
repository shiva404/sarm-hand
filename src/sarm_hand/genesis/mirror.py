"""Leader → sim mirror smoothing (encoder deadband, rate limit, EMA)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import JOINT_NAMES, ProjectConfig
from .units import norm_to_radians, radians_to_norm


def filter_raw_counts(
    raw: dict[str, int],
    *,
    last_raw: dict[str, int] | None,
    deadband_for_joint: dict[str, int],
) -> dict[str, int]:
    """Ignore encoder changes smaller than per-joint deadband."""
    if last_raw is None:
        return {j: int(raw[j]) for j in JOINT_NAMES}

    filtered: dict[str, int] = {}
    for joint in JOINT_NAMES:
        prev = int(last_raw[joint])
        cur = int(raw[joint])
        band = int(deadband_for_joint.get(joint, 0))
        filtered[joint] = prev if abs(cur - prev) <= band else cur
    return filtered


def smooth_mirror_radians(
    target_rad: Sequence[float],
    *,
    current_rad: Sequence[float],
    previous_cmd_rad: Sequence[float] | None,
    cfg: ProjectConfig,
    calibration: dict,
    max_norm_step: float,
    smoothing: float,
    snap: bool = False,
) -> np.ndarray:
    """Rate-limit in LeRobot norm space, then EMA-blend commanded radians."""
    target = np.asarray(target_rad, dtype=np.float64)
    if snap or previous_cmd_rad is None:
        return target.copy()

    alpha = float(np.clip(smoothing, 0.0, 1.0))
    current = np.asarray(current_rad, dtype=np.float64)
    prev_cmd = np.asarray(previous_cmd_rad, dtype=np.float64)
    cmd = np.empty(len(JOINT_NAMES), dtype=np.float64)

    for i, joint in enumerate(JOINT_NAMES):
        cur_n = radians_to_norm(
            float(current[i]),
            joint,
            cfg,
            calibration=calibration,
        )
        tgt_n = radians_to_norm(
            float(target[i]),
            joint,
            cfg,
            calibration=calibration,
        )
        step_n = float(np.clip(tgt_n - cur_n, -max_norm_step, max_norm_step))
        limited_rad = norm_to_radians(
            cur_n + step_n,
            joint,
            cfg,
            calibration=calibration,
        )
        cmd[i] = prev_cmd[i] + alpha * (float(limited_rad) - prev_cmd[i])
    return cmd
