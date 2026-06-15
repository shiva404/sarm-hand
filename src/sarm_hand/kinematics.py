"""Forward and inverse kinematics for the SO-ARM101 3D simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

KINEMATIC_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


@dataclass
class JointMap:
    """Maps kinematic angle <-> control value (normalized or degrees)."""

    zero: float = 0.0
    sign: float = 1.0
    min_val: float | None = None
    max_val: float | None = None

    def to_control(self, kin: float) -> float:
        return self.zero + self.sign * kin

    def to_kin(self, control: float) -> float:
        s = self.sign if self.sign != 0 else 1.0
        return (control - self.zero) / s

    def violation(self, control: float) -> float:
        if self.min_val is not None and control < self.min_val:
            return self.min_val - control
        if self.max_val is not None and control > self.max_val:
            return control - self.max_val
        return 0.0

    def clamp_control(self, control: float) -> float:
        lo = self.min_val if self.min_val is not None else -math.inf
        hi = self.max_val if self.max_val is not None else math.inf
        return max(lo, min(hi, control))


@dataclass
class ArmGeometry:
    shoulder_height: float
    upper_arm: float
    forearm: float
    hand: float
    wrist_rot_offset: float
    gripper_offset: float
    gripper_motor: float
    units: str
    elbow: str
    shoulder_pan_map: JointMap = field(default_factory=JointMap)
    shoulder_lift_map: JointMap = field(default_factory=JointMap)
    elbow_flex_map: JointMap = field(default_factory=JointMap)
    wrist_flex_map: JointMap = field(default_factory=JointMap)
    wrist_roll_map: JointMap = field(default_factory=JointMap)


@dataclass
class IKSolution:
    reachable: bool
    joint_values: dict[str, float]
    kin_angles: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    elbow: str | None = None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _planar_two_link(
    geom: ArmGeometry,
    wrist_r: float,
    wrist_z: float,
    elbow: str,
    warnings: list[str],
) -> tuple[float, float, float, bool]:
    l1, l2 = geom.upper_arm, geom.forearm
    dx = wrist_r
    dz = wrist_z - geom.shoulder_height
    dist = math.hypot(dx, dz)
    reachable = True
    reach_max = l1 + l2
    reach_min = abs(l1 - l2)
    if dist > reach_max:
        reachable = False
        warnings.append(
            f"target {dist:.1f}{geom.units} away exceeds max reach "
            f"{reach_max:.1f}{geom.units}; arm will extend toward it"
        )
        dist = reach_max - 1e-6
    elif dist < reach_min:
        reachable = False
        warnings.append(
            f"target {dist:.1f}{geom.units} is closer than min reach "
            f"{reach_min:.1f}{geom.units}; arm will fold toward it"
        )
        dist = reach_min + 1e-6

    cos_q2 = _clamp((dist * dist - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0)
    q2 = math.acos(cos_q2)
    if elbow == "up":
        q2 = -q2
    q1 = math.atan2(dz, dx) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    return q1, q2, dist, reachable


def _tip_xy_from_centerline(az: float, tip_r: float, gripper_offset: float) -> tuple[float, float]:
    return (
        tip_r * math.cos(az) - gripper_offset * math.sin(az),
        tip_r * math.sin(az) + gripper_offset * math.cos(az),
    )


def _centerline_from_tip_xy(x: float, y: float, gripper_offset: float) -> tuple[float, float]:
    az = math.atan2(y, x)
    cx = x + gripper_offset * math.sin(az)
    cy = y - gripper_offset * math.cos(az)
    return math.hypot(cx, cy), math.atan2(cy, cx)


def _wrist_rot_from_pitch(
    wrist_r: float, wrist_z: float, theta_arm: float, q3: float, offset: float
) -> tuple[float, float]:
    if offset == 0.0:
        return wrist_r, wrist_z
    perp = theta_arm + math.pi / 2 + q3
    return wrist_r + offset * math.cos(perp), wrist_z + offset * math.sin(perp)


def _joint_maps(geom: ArmGeometry) -> dict[str, JointMap]:
    return {
        "shoulder_pan": geom.shoulder_pan_map,
        "shoulder_lift": geom.shoulder_lift_map,
        "elbow_flex": geom.elbow_flex_map,
        "wrist_flex": geom.wrist_flex_map,
    }


def _solve_branch(
    geom: ArmGeometry,
    r: float,
    azimuth: float,
    z: float,
    pitch_deg: float | None,
    elbow: str,
) -> tuple[dict[str, float], dict[str, float], bool, list[str]]:
    warnings: list[str] = []
    off = geom.wrist_rot_offset
    reachable = True

    if pitch_deg is not None:
        p = math.radians(pitch_deg)
        rot_r = r - geom.hand * math.cos(p)
        rot_z = z - geom.hand * math.sin(p)
        wp_r, wp_z = rot_r, rot_z
        q1 = q2 = q3 = 0.0
        for _ in range(6):
            q1, q2, _, reach_ok = _planar_two_link(geom, wp_r, wp_z, elbow, warnings)
            reachable = reachable and reach_ok
            theta_arm = q1 + q2
            q3 = p - theta_arm
            perp = theta_arm + math.pi / 2 + q3
            wp_r = rot_r - off * math.cos(perp)
            wp_z = rot_z - off * math.sin(perp)
    else:
        q1, q2, _, reach_ok = _planar_two_link(geom, r, z, elbow, warnings)
        reachable = reachable and reach_ok
        q3 = 0.0

    kin = {
        "shoulder_pan": math.degrees(azimuth),
        "shoulder_lift": math.degrees(q1),
        "elbow_flex": math.degrees(q2),
    }
    control = {
        "shoulder_pan": geom.shoulder_pan_map.to_control(kin["shoulder_pan"]),
        "shoulder_lift": geom.shoulder_lift_map.to_control(kin["shoulder_lift"]),
        "elbow_flex": geom.elbow_flex_map.to_control(kin["elbow_flex"]),
    }
    if pitch_deg is not None:
        kin["wrist_flex"] = math.degrees(q3)
        control["wrist_flex"] = geom.wrist_flex_map.to_control(kin["wrist_flex"])
    return kin, control, reachable, warnings


def solve_ik(
    geom: ArmGeometry,
    x: float,
    y: float,
    z: float,
    pitch_deg: float | None = None,
    elbow: str | None = None,
) -> IKSolution:
    r, azimuth = _centerline_from_tip_xy(x, y, geom.gripper_offset)
    maps = _joint_maps(geom)
    preferred = (elbow or geom.elbow or "up").lower()
    branches = [preferred] if elbow is not None else [preferred] + [
        b for b in ("up", "down") if b != preferred
    ]

    def total_violation(values: dict[str, float]) -> float:
        return sum(maps[n].violation(v) for n, v in values.items() if n in maps)

    candidates = []
    for br in branches:
        kin, control, geom_ok, warns = _solve_branch(geom, r, azimuth, z, pitch_deg, br)
        candidates.append((br, kin, control, geom_ok, warns, total_violation(control)))

    candidates.sort(key=lambda c: (not c[3], c[5]))
    branch, kin, control, geom_ok, warnings, _violation = candidates[0]

    if elbow is None and branch != preferred:
        warnings.append(
            f"elbow '{preferred}' branch exceeded joint limits; using '{branch}' branch instead"
        )

    within_limits = True
    for name, value in list(control.items()):
        if name not in maps:
            continue
        jm = maps[name]
        over = jm.violation(value)
        if over > 1e-6:
            within_limits = False
            clamped = jm.clamp_control(value)
            warnings.append(
                f"{name} value {value:.1f} outside travel "
                f"{jm.min_val:.0f}..{jm.max_val:.0f}; clamped to {clamped:.1f}"
            )
            control[name] = clamped

    warnings = list(dict.fromkeys(warnings))
    return IKSolution(
        reachable=geom_ok and within_limits,
        joint_values=control,
        kin_angles=kin,
        warnings=warnings,
        elbow=branch,
    )


def forward_kinematics(geom: ArmGeometry, joint_values: dict[str, float]) -> dict:
    az = math.radians(
        geom.shoulder_pan_map.to_kin(
            joint_values.get("shoulder_pan", geom.shoulder_pan_map.zero)
        )
    )
    q1 = math.radians(
        geom.shoulder_lift_map.to_kin(
            joint_values.get("shoulder_lift", geom.shoulder_lift_map.zero)
        )
    )
    q2 = math.radians(
        geom.elbow_flex_map.to_kin(joint_values.get("elbow_flex", geom.elbow_flex_map.zero))
    )

    l1, l2 = geom.upper_arm, geom.forearm
    wrist_r = l1 * math.cos(q1) + l2 * math.cos(q1 + q2)
    wrist_z = geom.shoulder_height + l1 * math.sin(q1) + l2 * math.sin(q1 + q2)

    if "wrist_flex" in joint_values:
        q3 = math.radians(geom.wrist_flex_map.to_kin(joint_values["wrist_flex"]))
    else:
        q3 = 0.0
    theta_arm = q1 + q2
    pitch = theta_arm + q3
    rot_r, rot_z = _wrist_rot_from_pitch(wrist_r, wrist_z, theta_arm, q3, geom.wrist_rot_offset)
    tip_r = rot_r + geom.hand * math.cos(pitch)
    tip_z = rot_z + geom.hand * math.sin(pitch)

    az_deg = math.degrees(az)
    tip_x, tip_y = _tip_xy_from_centerline(az, tip_r, geom.gripper_offset)
    return {
        "x": tip_x,
        "y": tip_y,
        "z": tip_z,
        "wrist": (wrist_r * math.cos(az), wrist_r * math.sin(az), wrist_z),
        "pitch_deg": math.degrees(pitch),
        "azimuth_deg": az_deg,
        "reach_mm": tip_r,
        "kin_angles": {
            "shoulder_pan": az_deg,
            "shoulder_lift": math.degrees(q1),
            "elbow_flex": math.degrees(q2),
            "wrist_flex": math.degrees(q3),
        },
    }


REACH_GO_TOL_MM = 15.0
SUGGEST_PITCH_CANDIDATES: tuple[float | None, ...] = (-90.0, -60.0, -45.0, 0.0, None)


def reach_error_mm(
    geom: ArmGeometry,
    target: dict[str, float],
    joint_values: dict[str, float],
) -> float:
    tip = forward_kinematics(geom, joint_values)
    return math.hypot(
        tip["x"] - target["x"],
        tip["y"] - target["y"],
        tip["z"] - target["z"],
    )


def suggest_pitch(
    geom: ArmGeometry,
    x: float,
    y: float,
    z: float,
    *,
    tol_mm: float = REACH_GO_TOL_MM,
) -> dict[str, float | bool | None]:
    target = {"x": x, "y": y, "z": z}
    for pitch in SUGGEST_PITCH_CANDIDATES:
        sol = solve_ik(geom, x, y, z, pitch_deg=pitch)
        err = reach_error_mm(geom, target, sol.joint_values)
        if sol.reachable and err <= tol_mm:
            return {"found": True, "pitch_deg": pitch}
    return {"found": False, "pitch_deg": None}
