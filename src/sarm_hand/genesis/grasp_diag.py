"""Grasp latch diagnostics for leader teleop debugging."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GraspDiagnostic:
    leader_gripper_deg: float
    sim_gripper_deg: float
    latched: bool
    prop_name: str | None = None
    prop_dist_m: float | None = None
    pen_z_m: float | None = None
    block_reason: str = "idle"
    link_dist_m: dict[str, float] = field(default_factory=dict)
    latch_prop: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.prop_dist_m is not None:
            out["prop_dist_cm"] = round(self.prop_dist_m * 100.0, 2)
        if self.leader_gripper_deg is not None:
            out["leader_gripper_deg"] = round(self.leader_gripper_deg, 1)
        if self.sim_gripper_deg is not None:
            out["sim_gripper_deg"] = round(self.sim_gripper_deg, 1)
        if self.pen_z_m is not None:
            out["pen_z_cm"] = round(self.pen_z_m * 100.0, 2)
        return out


class GraspLogWriter:
    """Append JSONL grasp diagnostics (one line per mirror frame)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self._last_latched: bool | None = None

    def write(self, diag: GraspDiagnostic, *, episode: int | None = None, frame: int | None = None) -> None:
        row: dict[str, Any] = {
            "ts": time.time(),
            **diag.to_dict(),
        }
        if episode is not None:
            row["episode"] = episode
        if frame is not None:
            row["frame"] = frame
        self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._file.flush()

    def maybe_print_transition(self, diag: GraspDiagnostic) -> None:
        if self._last_latched == diag.latched:
            return
        self._last_latched = diag.latched
        if diag.latched:
            dist = f"{diag.prop_dist_m * 100:.1f}cm" if diag.prop_dist_m is not None else "?"
            print(
                f"  [grasp] LATCHED {diag.latch_prop or diag.prop_name} "
                f"(leader gripper {diag.leader_gripper_deg:.0f}°, gap {dist})"
            )
        else:
            print(f"  [grasp] released (leader gripper {diag.leader_gripper_deg:.0f}°)")

    def close(self) -> None:
        self._file.close()


def leader_gripper_deg(gripper_rad: float) -> float:
    return float(math.degrees(gripper_rad))
