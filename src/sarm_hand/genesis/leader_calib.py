"""Align Genesis sim with the USB leader: pulses, norm, and sim angles side by side."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG_PATH, JOINT_NAMES, ProjectConfig
from ..genesis.calibration import require_calibration
from ..genesis.deps import ensure_genesis
from ..genesis.shutdown import (
    check_shutdown,
    ensure_shutdown_handlers,
    exit_after_interrupt,
    install_shutdown_handlers,
    interruptible_sleep,
    shutdown_requested,
)
from ..genesis.tensors import to_numpy
from ..genesis.units import raw_to_radians
from ..genesis.urdf_limits import urdf_joint_limits
from ..joint_signal_log import (
    _read_all_raw,
    _sample_dict,
    analyze_joint_signals,
    print_signal_analysis,
)
from ..robot import _make_setup_device, disable_arm_torque, ensure_port, resolve_role_port
from .leader import so101_leader_config, sync_leader_to_scene
from .home_pose import format_home_pose_summary


@dataclass
class LeaderSimRow:
    joint: str
    raw: int
    norm: float
    enc_deg: float  # encoder shaft ° from genesis.home_raw
    map_deg: float  # leader → URDF mapped ° (commanded)
    sim_deg: float  # Genesis qpos °
    sim_track_deg: float  # map − sim (should ≈ 0)
    align_deg: float  # enc − sim (physical vs sim; constant ⇒ frame_offset)
    enc_deg_delta: float | None = None
    map_deg_delta: float | None = None
    sim_deg_delta: float | None = None
    # Legacy aliases used in travel summary
    raw_delta: int | None = None
    sim_deg_delta: float | None = None


def format_home_raw_yaml(home_raw: dict[str, int]) -> str:
    """YAML snippet for ``genesis.home_raw`` (copy into config/default.yaml)."""
    lines = ["  home_raw:"]
    for joint in JOINT_NAMES:
        value = home_raw.get(joint)
        if value is not None:
            lines.append(f"    {joint}: {int(value)}")
    return "\n".join(lines)


def patch_home_raw_in_yaml(path: Path, home_raw: dict[str, int]) -> None:
    """Replace the ``genesis.home_raw`` block in a YAML config file."""
    text = path.read_text()
    block = format_home_raw_yaml(home_raw)
    pattern = re.compile(
        r"^  home_raw:\n(?:    \w+: \d+\n)+",
        re.MULTILINE,
    )
    if not pattern.search(text):
        raise ValueError(f"No genesis.home_raw block found in {path}")
    updated = pattern.sub(block + "\n", text, count=1)
    path.write_text(updated)


def _encoder_deg_from_home(
    raw: int,
    joint: str,
    cfg: ProjectConfig,
    *,
    resolution: int,
) -> float:
    """Shaft degrees from ``genesis.home_raw`` — raw pulse direction, no sign flip."""
    home = cfg.genesis.home_raw.get(joint)
    if home is None:
        return 0.0
    delta_raw = int(raw) - int(home)
    return delta_raw * 360.0 / resolution


def _mapped_deg_from_raw(
    raw: int,
    joint: str,
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
) -> float:
    hard = urdf_joint_limits(cfg)
    rad = raw_to_radians(int(raw), joint, cfg, calibration, hard_limits=hard)
    return math.degrees(rad)


def _read_sim_deg(scene) -> dict[str, float]:
    qpos = to_numpy(scene.robot.get_dofs_position(scene.dof_indices))
    return {name: math.degrees(float(qpos[i])) for i, name in enumerate(JOINT_NAMES)}


def _deg_baselines(
    raw: dict[str, int],
    sim_deg: dict[str, float],
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    resolution = cfg.servo.resolution
    enc = {
        j: _encoder_deg_from_home(raw[j], j, cfg, resolution=resolution) for j in JOINT_NAMES
    }
    mapped = {j: _mapped_deg_from_raw(raw[j], j, cfg, calibration) for j in JOINT_NAMES}
    return enc, mapped, dict(sim_deg)


def build_leader_sim_rows(
    *,
    raw: dict[str, int],
    action: dict[str, float],
    sim_deg: dict[str, float],
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
    baseline_raw: dict[str, int] | None = None,
    baseline_map_deg: dict[str, float] | None = None,
    baseline_sim_deg: dict[str, float] | None = None,
    baseline_enc_deg: dict[str, float] | None = None,
) -> list[LeaderSimRow]:
    """Combine hardware pulses, encoder °, mapped URDF °, and Genesis qpos °."""
    resolution = cfg.servo.resolution
    samples = _sample_dict(raw, calibration, cfg, baseline_raw=baseline_raw)
    rows: list[LeaderSimRow] = []
    for joint in JOINT_NAMES:
        jd = samples[joint]
        enc = _encoder_deg_from_home(int(jd["raw"]), joint, cfg, resolution=resolution)
        mapped = _mapped_deg_from_raw(int(jd["raw"]), joint, cfg, calibration)
        sdeg = sim_deg[joint]
        rows.append(
            LeaderSimRow(
                joint=joint,
                raw=int(jd["raw"]),
                norm=float(jd["norm"]),
                enc_deg=round(enc, 2),
                map_deg=round(mapped, 2),
                sim_deg=round(sdeg, 2),
                sim_track_deg=round(mapped - sdeg, 2),
                align_deg=round(enc - sdeg, 2),
                raw_delta=jd.get("raw_delta"),
                enc_deg_delta=(
                    round(enc - baseline_enc_deg[joint], 2)
                    if baseline_enc_deg is not None
                    else None
                ),
                map_deg_delta=(
                    round(mapped - baseline_map_deg[joint], 2)
                    if baseline_map_deg is not None
                    else None
                ),
                sim_deg_delta=(
                    round(sdeg - baseline_sim_deg[joint], 2)
                    if baseline_sim_deg is not None
                    else None
                ),
            )
        )
    return rows


def format_leader_sim_table(
    rows: list[LeaderSimRow],
    *,
    show_deltas: bool = False,
) -> str:
    if show_deltas:
        headers = ("joint", "raw", "enc°", "map°", "sim°", "m−s", "e−s", "Δenc°", "Δmap°", "Δsim°")
    else:
        headers = ("joint", "raw", "norm", "enc°", "map°", "sim°", "m−s", "e−s")
    widths = [
        max(len(h), max(_cell(r, i, show_deltas) for r in rows))
        for i, h in enumerate(headers)
    ]

    def pad(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines = [pad(headers), pad(["-" * w for w in widths])]
    for r in rows:
        if show_deltas:
            lines.append(
                pad(
                    [
                        r.joint,
                        str(r.raw),
                        f"{r.enc_deg:.1f}",
                        f"{r.map_deg:.1f}",
                        f"{r.sim_deg:.1f}",
                        f"{r.sim_track_deg:+.1f}",
                        f"{r.align_deg:+.1f}",
                        f"{r.enc_deg_delta:+.1f}" if r.enc_deg_delta is not None else "-",
                        f"{r.map_deg_delta:+.1f}" if r.map_deg_delta is not None else "-",
                        f"{r.sim_deg_delta:+.1f}" if r.sim_deg_delta is not None else "-",
                    ]
                )
            )
        else:
            lines.append(
                pad(
                    [
                        r.joint,
                        str(r.raw),
                        f"{r.norm:.1f}",
                        f"{r.enc_deg:.1f}",
                        f"{r.map_deg:.1f}",
                        f"{r.sim_deg:.1f}",
                        f"{r.sim_track_deg:+.1f}",
                        f"{r.align_deg:+.1f}",
                    ]
                )
            )
    return "\n".join(lines)


def _cell(row: LeaderSimRow, col: int, show_deltas: bool) -> int:
    if show_deltas:
        vals = [
            row.joint,
            str(row.raw),
            f"{row.enc_deg:.1f}",
            f"{row.map_deg:.1f}",
            f"{row.sim_deg:.1f}",
            f"{row.sim_track_deg:+.1f}",
            f"{row.align_deg:+.1f}",
            f"{row.enc_deg_delta:+.1f}" if row.enc_deg_delta is not None else "-",
            f"{row.map_deg_delta:+.1f}" if row.map_deg_delta is not None else "-",
            f"{row.sim_deg_delta:+.1f}" if row.sim_deg_delta is not None else "-",
        ]
    else:
        vals = [
            row.joint,
            str(row.raw),
            f"{row.norm:.1f}",
            f"{row.enc_deg:.1f}",
            f"{row.map_deg:.1f}",
            f"{row.sim_deg:.1f}",
            f"{row.sim_track_deg:+.1f}",
            f"{row.align_deg:+.1f}",
        ]
    return len(vals[col])


def _print_angle_legend(cfg: ProjectConfig | None = None) -> None:
    mode = (cfg.genesis.mapping if cfg else "delta").lower()
    print(
        "  enc°  = (raw − home_raw) × 360/4096 — pulse travel from anchor, no sign\n"
        "  map°  = leader raw → URDF ° (sent to sim)\n"
        "  sim°  = Genesis joint qpos °\n"
        "  m−s   = map − sim (sim tracking; should ≈ 0)\n"
    )
    if mode == "delta":
        print(
            "  Delta mapping: at rest sim° == genesis.rest_pose; any move tracks\n"
            "  encoder 1:1 (Δenc° == Δsim°). If a joint moves the wrong way,\n"
            "  flip its genesis.joints.<joint>.sign in config/default.yaml.\n"
        )
    else:
        print("  e−s   = enc − sim (leader vs sim; constant offset ⇒ add frame_offset)\n")


def _print_rest_alignment(rows: list[LeaderSimRow], cfg: ProjectConfig | None = None) -> None:
    mode = (cfg.genesis.mapping if cfg else "delta").lower()
    print("=== Rest alignment (put leader in its folded rest pose) ===\n")
    print(format_leader_sim_table(rows, show_deltas=False))

    if mode == "delta" and cfg is not None:
        warns: list[str] = []
        for r in rows:
            target = cfg.genesis.rest_pose.get(r.joint)
            if target is None:
                continue
            if abs(r.sim_deg - float(target)) >= 5.0:
                warns.append(
                    f"  {r.joint}: sim°={r.sim_deg:+.1f} but rest_pose={float(target):+.1f} "
                    f"(enc°={r.enc_deg:+.1f}) — re-capture anchor with --capture-home --save-home"
                )
        if warns:
            print("\nLeader is not at the captured rest anchor:")
            print("\n".join(warns))
        else:
            print("\n✓ Leader at rest anchor — sim matches genesis.rest_pose.")
        print()
        return

    hints: list[str] = []
    for r in rows:
        if abs(r.align_deg) >= 15.0:
            offset_rad = -math.radians(r.align_deg)
            hints.append(
                f"  {r.joint}: e−s={r.align_deg:+.1f}° — try "
                f"frame_offset: {offset_rad:.4f}  (~{offset_rad:.4f} rad) on top of current value"
            )
    if hints:
        print("\nSuggested genesis.joints offsets (add to existing frame_offset):")
        print("\n".join(hints))
    print()


def _print_travel_summary(rows: list[LeaderSimRow], *, target_deg: float = 90.0) -> None:
    enc_pulses = target_deg * 4096.0 / 360.0
    print(f"\n=== Max travel from session baseline (target {target_deg:.0f}° ≈ {enc_pulses:.0f} raw) ===\n")
    headers = ("joint", "Δenc°", "Δmap°", "Δsim°", "Δenc−Δsim", "Δraw")
    print("  ".join(h.ljust(12) for h in headers))
    print("  ".join("-" * 12 for _ in headers))
    for r in rows:
        if r.enc_deg_delta is None or r.sim_deg_delta is None:
            continue
        de = r.enc_deg_delta
        dm = r.map_deg_delta if r.map_deg_delta is not None else 0.0
        ds = r.sim_deg_delta
        print(
            "  ".join(
                [
                    r.joint.ljust(12),
                    f"{de:+.1f}".ljust(12),
                    f"{dm:+.1f}".ljust(12),
                    f"{ds:+.1f}".ljust(12),
                    f"{de - ds:+.1f}".ljust(12),
                    str(r.raw_delta if r.raw_delta is not None else "-").ljust(12),
                ]
            )
        )
    print(
        "\nΔenc° ≈ Δsim° when gain matches. Constant e−s at rest ⇒ frame_offset on that joint.\n"
        "shoulder_pan often needs frame_offset ≈ ±1.5708 (90°) in genesis.joints.\n"
    )


def _interactive_measure_joints(
    *,
    bus,
    leader,
    scene,
    cfg: ProjectConfig,
    calibration: dict[str, dict[str, Any]],
) -> None:
    print("\n=== Per-joint travel measure ===")
    print("For each joint: rest → move ~90° on the leader → press Enter.\n")
    baseline_raw = _read_all_raw(bus)
    sync_leader_to_scene(scene, leader)
    scene.refresh_previews()

    for joint in JOINT_NAMES:
        input(f"  [{joint}] at rest — press Enter to capture baseline...")
        baseline_raw = _read_all_raw(bus)
        sync_leader_to_scene(scene, leader)
        scene.refresh_previews()
        baseline_sim_at_rest = _read_sim_deg(scene)
        b_enc, b_map, b_sim = _deg_baselines(
            baseline_raw, baseline_sim_at_rest, cfg, calibration
        )
        input(f"  [{joint}] move ~90° then press Enter...")
        raw = _read_all_raw(bus)
        action = sync_leader_to_scene(scene, leader)
        scene.refresh_previews()
        sim_deg = _read_sim_deg(scene)
        rows = build_leader_sim_rows(
            raw=raw,
            action=action,
            sim_deg=sim_deg,
            cfg=cfg,
            calibration=calibration,
            baseline_raw=baseline_raw,
            baseline_enc_deg=b_enc,
            baseline_map_deg=b_map,
            baseline_sim_deg=b_sim,
        )
        row = next(r for r in rows if r.joint == joint)
        print(
            f"    enc°={row.enc_deg:.1f}  map°={row.map_deg:.1f}  sim°={row.sim_deg:.1f}  "
            f"e−s={row.align_deg:+.1f}  "
            f"Δenc°={row.enc_deg_delta:+.1f}  Δsim°={row.sim_deg_delta:+.1f}"
        )
    print("\nMeasure pass complete.\n")


def run_genesis_leader_calib(
    *,
    leader_port: str | None = None,
    rate_hz: float | None = None,
    duration_s: float | None = None,
    capture_home: bool = False,
    save_home: bool = False,
    config_path: Path | None = None,
    print_analysis: bool = True,
    measure_joints: bool = False,
    headless: bool = False,
) -> None:
    """Mirror leader into Genesis and compare encoder pulses vs sim angles."""
    ensure_genesis()
    install_shutdown_handlers()
    cfg = ProjectConfig.load()
    if headless:
        cfg.genesis.headless = True

    port = ensure_port(resolve_role_port("leader", leader_port), "Leader")
    cal = require_calibration("leader", cfg)
    cal_role = cfg.genesis.calibration_role or "leader"

    if print_analysis:
        rows = analyze_joint_signals(cfg, role="leader", target_degrees=90.0)
        print_signal_analysis(rows, target_degrees=90.0, role="leader")
        print("Starting live leader ↔ Genesis calibration (torque off).\n")

    from lerobot.teleoperators.so_leader import SO101Leader

    from .scene import SO101GenesisScene

    leader_cfg = so101_leader_config(cfg, port)
    leader = SO101Leader(leader_cfg)
    device = _make_setup_device("leader", port)
    bus = device.bus

    disable_arm_torque("leader", port)
    leader.connect()
    if not bus.is_connected:
        bus._connect(handshake=False)

    if capture_home:
        home_raw = _read_all_raw(bus)
        print("Captured leader rest pose (Present_Position, torque off):\n")
        print(format_home_raw_yaml(home_raw))
        if save_home:
            path = config_path or DEFAULT_CONFIG_PATH
            patch_home_raw_in_yaml(path, home_raw)
            print(f"\nUpdated {path}")
        leader.disconnect()
        return

    scene = SO101GenesisScene.create(cfg, calibration_role=cal_role, apply_home=False)
    ensure_shutdown_handlers()

    home_raw = _read_all_raw(bus)
    rest_action = sync_leader_to_scene(scene, leader)
    scene.refresh_previews()
    print(format_home_pose_summary(cfg, calibration=cal))
    print()

    baseline_raw = dict(home_raw)
    baseline_sim = _read_sim_deg(scene)
    baseline_enc, baseline_map, baseline_sim = _deg_baselines(
        baseline_raw, baseline_sim, cfg, cal
    )
    rest_rows = build_leader_sim_rows(
        raw=baseline_raw,
        action=rest_action,
        sim_deg=baseline_sim,
        cfg=cfg,
        calibration=cal,
    )
    interval = 1.0 / (rate_hz if rate_hz is not None else cfg.genesis.mirror_rate_hz)
    deadline = time.perf_counter() + duration_s if duration_s else None
    steps = 0
    interrupted = False
    max_rows: list[LeaderSimRow] | None = None

    print("Genesis leader calibration")
    print(f"  Leader:  {port}")
    print(f"  Scene:   {cfg.genesis.scene}")
    hz = rate_hz if rate_hz is not None else cfg.genesis.mirror_rate_hz
    print(f"  Rate:    {hz} Hz")
    print("  Move the leader — sim should track. Ctrl+C to stop.\n")
    _print_angle_legend(cfg)
    _print_rest_alignment(rest_rows, cfg)

    if measure_joints:
        try:
            _interactive_measure_joints(
                bus=bus,
                leader=leader,
                scene=scene,
                cfg=cfg,
                calibration=cal,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\nStopped measure pass.")

    try:
        while (deadline is None or time.perf_counter() < deadline) and not shutdown_requested():
            check_shutdown()
            loop_start = time.perf_counter()
            raw = _read_all_raw(bus)
            action = sync_leader_to_scene(scene, leader)
            sim_deg = _read_sim_deg(scene)
            rows = build_leader_sim_rows(
                raw=raw,
                action=action,
                sim_deg=sim_deg,
                cfg=cfg,
                calibration=cal,
                baseline_raw=baseline_raw,
                baseline_enc_deg=baseline_enc,
                baseline_map_deg=baseline_map,
                baseline_sim_deg=baseline_sim,
            )
            if max_rows is None:
                max_rows = [LeaderSimRow(**{**r.__dict__}) for r in rows]
            else:
                for i, r in enumerate(rows):
                    prev = max_rows[i]
                    if r.enc_deg_delta is not None and prev.enc_deg_delta is not None:
                        if abs(r.enc_deg_delta) > abs(prev.enc_deg_delta):
                            max_rows[i] = r

            if steps == 0 or steps % int(hz * 2) == 0:
                print(format_leader_sim_table(rows, show_deltas=True))
                print()
            steps += 1
            elapsed = time.perf_counter() - loop_start
            interruptible_sleep(max(interval - elapsed, 0.0))
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped.")
    finally:
        if max_rows:
            _print_travel_summary(max_rows)
        print("\nRest pose at exit (paste into genesis.home_raw if sim rest ≠ leader rest):\n")
        print(format_home_raw_yaml(_read_all_raw(bus)))
        scene.close()
        if leader.is_connected:
            leader.disconnect()
        if interrupted or shutdown_requested():
            exit_after_interrupt()

    print(f"Done ({steps} steps).")
