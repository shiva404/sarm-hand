"""Log encoder pulses vs LeRobot norm vs Genesis sim angles to find mapping gaps."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import JOINT_NAMES, ProjectConfig
from .genesis.calibration import calibration_path, load_calibration, raw_to_norm, require_calibration
from .genesis.urdf_limits import urdf_joint_limits
from .genesis.units import (
    _is_wide_calibration,
    norm_to_radians,
    radians_to_norm,
    radians_to_raw,
)
from .robot import _make_setup_device, disable_arm_torque, resolve_role_port
from .servo import servo_summary


@dataclass
class JointSignalAnalysis:
    joint: str
    cal_min: int
    cal_max: int
    cal_span: int
    wide_cal: bool
    # ST-3215-C001: 4096 counts per output revolution (360°)
    encoder_pulses_per_90: float
    home_raw: int | None
    home_norm: float | None
    home_sim_deg: float | None
    # Pulses the current sim mapping uses for +target_deg from home
    sim_pulses_for_target_deg: float | None
    norm_for_target_deg: float | None
    # LeRobot: full cal span (-100..100 norm) in pulses
    pulses_per_100_norm: float
    pulses_per_norm_unit: float
    # Norm delta implied by physical 90° encoder rotation (no sim mapping)
    norm_for_encoder_90: float | None
    # sim_pulses / encoder_pulses — >1 means sim expects more pulses than physics
    sim_vs_encoder_pulse_ratio: float | None


@dataclass
class JointMovementSample:
    joint: str
    raw_start: int
    raw_end: int
    raw_delta: int
    norm_start: float
    norm_end: float
    norm_delta: float
    sim_deg_start: float
    sim_deg_end: float
    sim_deg_delta: float
    observed_pulses_per_degree: float | None
    sim_pulses_per_degree: float | None
    pulse_gap_ratio: float | None  # observed / expected from sim mapping


def _encoder_pulses_per_degree(resolution: int = 4096) -> float:
    return resolution / 360.0


def _encoder_pulses_for_angle(degrees: float, resolution: int = 4096) -> float:
    return abs(degrees) * resolution / 360.0


def analyze_joint_signals(
    cfg: ProjectConfig,
    *,
    role: str = "leader",
    target_degrees: float = 90.0,
) -> list[JointSignalAnalysis]:
    """Static table: expected pulses for target_deg sim travel vs encoder physics."""
    cal = load_calibration(role, cfg)
    if cal is None:
        print(f"No calibration at {calibration_path(role, cfg)}", file=sys.stderr)
        raise SystemExit(1)

    hard = urdf_joint_limits(cfg)
    resolution = cfg.servo.resolution
    rows: list[JointSignalAnalysis] = []

    for joint in JOINT_NAMES:
        lo = int(cal[joint]["range_min"])
        hi = int(cal[joint]["range_max"])
        span = hi - lo
        wide = _is_wide_calibration(joint, cal)
        home_raw = cfg.genesis.home_raw.get(joint)
        home_norm = raw_to_norm(home_raw, joint, cal) if home_raw is not None else None
        home_sim_deg = None
        sim_pulses_target = None
        norm_target = None

        if home_norm is not None:
            home_rad = norm_to_radians(home_norm, joint, cfg, calibration=cal, urdf_limits=hard)
            home_sim_deg = math.degrees(home_rad)
            target_rad = home_rad + math.radians(target_degrees)
            lo_h, hi_h = hard[joint]
            target_rad = max(lo_h, min(hi_h, target_rad))
            raw_target = radians_to_raw(target_rad, joint, cfg, cal, urdf_limits=hard)
            sim_pulses_target = abs(float(raw_target - home_raw))
            norm_target = radians_to_norm(target_rad, joint, cfg, calibration=cal, urdf_limits=hard)

        enc_90 = _encoder_pulses_for_angle(90.0, resolution)
        ppnu = span / 200.0
        norm_enc_90 = enc_90 / ppnu if ppnu > 0 else None
        sim_ratio = (
            sim_pulses_target / enc_90
            if sim_pulses_target is not None and enc_90 > 0
            else None
        )

        rows.append(
            JointSignalAnalysis(
                joint=joint,
                cal_min=lo,
                cal_max=hi,
                cal_span=span,
                wide_cal=wide,
                encoder_pulses_per_90=enc_90,
                home_raw=home_raw,
                home_norm=home_norm,
                home_sim_deg=home_sim_deg,
                sim_pulses_for_target_deg=sim_pulses_target,
                norm_for_target_deg=norm_target,
                pulses_per_100_norm=span / 2.0,
                pulses_per_norm_unit=ppnu,
                norm_for_encoder_90=norm_enc_90,
                sim_vs_encoder_pulse_ratio=sim_ratio,
            )
        )
    return rows


def print_signal_analysis(
    rows: list[JointSignalAnalysis],
    *,
    target_degrees: float = 90.0,
    role: str = "leader",
) -> None:
    cfg = ProjectConfig.load()
    print(f"Joint signal analysis ({role}) — {servo_summary(cfg)}")
    print(f"Target sim travel from home: +{target_degrees:.0f}°\n")
    headers = (
        "joint",
        "cal_span",
        "wide",
        "home_norm",
        "sim°",
        f"pulses/{target_degrees:.0f}°sim",
        f"norm@+{target_degrees:.0f}°",
        "norm@enc90°",
        "pulses/norm",
        "sim/enc",
    )
    print("  ".join(h.ljust(12) for h in headers))
    print("  ".join("-" * 12 for _ in headers))
    for r in rows:
        print(
            "  ".join(
                [
                    r.joint.ljust(12),
                    str(r.cal_span).ljust(12),
                    ("yes" if r.wide_cal else "no").ljust(12),
                    f"{r.home_norm:.1f}" if r.home_norm is not None else "-".ljust(12),
                    f"{r.home_sim_deg:.1f}" if r.home_sim_deg is not None else "-".ljust(12),
                    f"{r.sim_pulses_for_target_deg:.0f}"
                    if r.sim_pulses_for_target_deg is not None
                    else "-".ljust(12),
                    f"{r.norm_for_target_deg:.1f}" if r.norm_for_target_deg is not None else "-".ljust(12),
                    f"{r.norm_for_encoder_90:.1f}" if r.norm_for_encoder_90 is not None else "-".ljust(12),
                    f"{r.pulses_per_norm_unit:.1f}".ljust(12),
                    f"{r.sim_vs_encoder_pulse_ratio:.2f}"
                    if r.sim_vs_encoder_pulse_ratio is not None
                    else "-".ljust(12),
                ]
            )
        )
    print(
        "\nenc90° = 1024 raw counts for 90° output-shaft rotation (4096 counts/rev).\n"
        f"pulses/{target_degrees:.0f}°sim = raw delta Genesis mapping expects for +{target_degrees:.0f}° sim.\n"
        "norm@enc90° = LeRobot norm delta for a physical 90° shaft turn (training units).\n"
        "sim/enc = pulses/sim ÷ 1024 — values >>1 mean sim over-counts pulses vs physics.\n"
        "Run without --analyze-only to log live Δraw while you move each joint ~90°.\n"
    )


def _read_all_raw(bus) -> dict[str, int]:
    out: dict[str, int] = {}
    for joint in JOINT_NAMES:
        out[joint] = int(bus.read("Present_Position", joint, normalize=False))
    return out


def _sample_dict(
    raw: dict[str, int],
    cal: dict[str, dict[str, Any]],
    cfg: ProjectConfig,
    *,
    baseline_raw: dict[str, int] | None = None,
) -> dict[str, Any]:
    hard = urdf_joint_limits(cfg)
    joints: dict[str, Any] = {}
    for joint in JOINT_NAMES:
        r = raw[joint]
        norm = raw_to_norm(r, joint, cal)
        sim_rad = norm_to_radians(norm, joint, cfg, calibration=cal, urdf_limits=hard)
        entry: dict[str, Any] = {
            "raw": r,
            "norm": round(norm, 3),
            "sim_deg": round(math.degrees(sim_rad), 3),
        }
        if baseline_raw is not None:
            dr = r - baseline_raw[joint]
            dn = norm - raw_to_norm(baseline_raw[joint], joint, cal)
            sim0 = norm_to_radians(
                raw_to_norm(baseline_raw[joint], joint, cal),
                joint,
                cfg,
                calibration=cal,
                urdf_limits=hard,
            )
            entry["raw_delta"] = dr
            entry["norm_delta"] = round(dn, 3)
            entry["sim_deg_delta"] = round(math.degrees(sim_rad - sim0), 3)
        joints[joint] = entry
    return joints


def log_live_signal(
    *,
    role: str,
    port: str,
    duration_s: float = 30.0,
    rate_hz: float = 10.0,
    output: Path | None = None,
    target_degrees: float = 90.0,
) -> list[JointMovementSample]:
    """Stream leader/follower pulses vs sim angles; summarize max travel per joint."""
    cfg = ProjectConfig.load()
    cal = require_calibration(role, cfg)
    device = _make_setup_device(role, port)
    bus = device.bus
    if not bus.is_connected:
        bus._connect(handshake=False)
    disable_arm_torque(role, port)

    analysis = {r.joint: r for r in analyze_joint_signals(cfg, role=role, target_degrees=target_degrees)}
    interval = 1.0 / rate_hz
    deadline = time.perf_counter() + duration_s
    baseline_raw = _read_all_raw(bus)
    max_delta: dict[str, dict[str, float]] = {
        j: {"raw": 0.0, "norm": 0.0, "sim_deg": 0.0} for j in JOINT_NAMES
    }

    print(f"Live joint signal log ({role} on {port})")
    print(f"  Duration: {duration_s:.0f}s @ {rate_hz:.0f} Hz")
    print("  Torque off — move each joint ~90° in turn; deltas logged vs rest pose.\n")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Writing: {output}\n")

    fh = output.open("w") if output else None
    step = 0
    try:
        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()
            raw = _read_all_raw(bus)
            sample = {
                "t": round(time.perf_counter(), 3),
                "step": step,
                "joints": _sample_dict(raw, cal, cfg, baseline_raw=baseline_raw),
            }
            if fh:
                fh.write(json.dumps(sample) + "\n")
                fh.flush()

            for joint in JOINT_NAMES:
                jd = sample["joints"][joint]
                max_delta[joint]["raw"] = max(max_delta[joint]["raw"], abs(jd["raw_delta"]))
                max_delta[joint]["norm"] = max(max_delta[joint]["norm"], abs(jd["norm_delta"]))
                max_delta[joint]["sim_deg"] = max(max_delta[joint]["sim_deg"], abs(jd["sim_deg_delta"]))

            if step % int(rate_hz) == 0:
                parts = [f"{j}: Δraw={max_delta[j]['raw']:.0f}" for j in JOINT_NAMES[:3]]
                print(f"  [{step:4d}] max deltas (first 3 joints)  " + "  ".join(parts))
            step += 1
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(interval - elapsed, 0.0))
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        if fh:
            fh.close()
        try:
            bus.disable_torque()
            bus.port_handler.closePort()
        except Exception:
            pass

    summaries: list[JointMovementSample] = []
    print("\n=== Max observed travel from session start ===\n")
    headers = (
        "joint",
        "Δraw",
        "Δnorm",
        "Δsim°",
        "obs pulses/°",
        f"exp pulses/{target_degrees:.0f}°",
        "gap ratio",
    )
    print("  ".join(h.ljust(14) for h in headers))
    print("  ".join("-" * 14 for _ in headers))

    for joint in JOINT_NAMES:
        dr = max_delta[joint]["raw"]
        dn = max_delta[joint]["norm"]
        ds = max_delta[joint]["sim_deg"]
        obs_ppd = dr / ds if ds > 1.0 else None
        expected = analysis[joint].sim_pulses_for_target_deg
        gap = dr / expected if expected and expected > 0 and dr > 0 else None
        summaries.append(
            JointMovementSample(
                joint=joint,
                raw_start=baseline_raw[joint],
                raw_end=baseline_raw[joint] + int(dr),
                raw_delta=int(dr),
                norm_start=raw_to_norm(baseline_raw[joint], joint, cal),
                norm_end=raw_to_norm(baseline_raw[joint], joint, cal) + dn,
                norm_delta=dn,
                sim_deg_start=0.0,
                sim_deg_end=ds,
                sim_deg_delta=ds,
                observed_pulses_per_degree=obs_ppd,
                sim_pulses_per_degree=expected / target_degrees if expected else None,
                pulse_gap_ratio=gap,
            )
        )
        print(
            "  ".join(
                [
                    joint.ljust(14),
                    f"{dr:.0f}".ljust(14),
                    f"{dn:.1f}".ljust(14),
                    f"{ds:.1f}".ljust(14),
                    f"{obs_ppd:.1f}" if obs_ppd else "-".ljust(14),
                    f"{expected:.0f}" if expected else "-".ljust(14),
                    f"{gap:.2f}" if gap else "-".ljust(14),
                ]
            )
        )

    print(
        "\ngap ratio = observed Δraw / expected pulses for +90° sim (1.0 = training matches sim mapping).\n"
        "If gap ≈ 0.5, hardware travel uses ~half the pulses the old mapping assumed.\n"
    )
    return summaries


def run_joint_signal_log(
    *,
    role: str = "leader",
    port: str | None = None,
    analyze_only: bool = False,
    live: bool = True,
    duration_s: float = 45.0,
    rate_hz: float = 10.0,
    output: Path | None = None,
    target_degrees: float = 90.0,
) -> None:
    cfg = ProjectConfig.load()
    resolved_port = resolve_role_port(role, port)
    rows = analyze_joint_signals(cfg, role=role, target_degrees=target_degrees)
    print_signal_analysis(rows, target_degrees=target_degrees, role=role)

    if analyze_only:
        return

    if not live:
        print("Pass --live (default) to stream hardware pulses while you move the arm.")
        return

    out = output or (cfg.resolve_dataset_root().parent / "logs" / f"joint-signal-{role}.jsonl")
    log_live_signal(
        role=role,
        port=resolved_port,
        duration_s=duration_s,
        rate_hz=rate_hz,
        output=out,
        target_degrees=target_degrees,
    )
