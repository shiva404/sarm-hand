"""Shared Feetech ST-3215-C001 servo constants for hardware and simulators."""

from __future__ import annotations

from .config import ProjectConfig, ServoSettings


def servo_settings(cfg: ProjectConfig | None = None) -> ServoSettings:
    return (cfg or ProjectConfig.load()).servo


def servo_summary(cfg: ProjectConfig | None = None) -> str:
    """One-line description for startup logs."""
    s = servo_settings(cfg)
    return (
        f"{s.model} ({s.lerobot_type}) 1:{s.gear_ratio} "
        f"{s.resolution} counts/rev @ {s.voltage_nominal_v}V"
    )


def export_servo_dict(cfg: ProjectConfig | None = None) -> dict:
    """JSON/YAML-friendly servo block for browser sim and tooling."""
    s = servo_settings(cfg)
    return {
        "model": s.model,
        "lerobot_type": s.lerobot_type,
        "gear_ratio": s.gear_ratio,
        "resolution": s.resolution,
        "voltage_nominal_v": s.voltage_nominal_v,
        "stall_torque_kg_cm": s.stall_torque_kg_cm,
        "urdf_mechanical_reduction": s.urdf_mechanical_reduction,
        "mujoco_class": s.mujoco_class,
        "mujoco_forcerange_nm": s.mujoco_forcerange_nm,
    }
