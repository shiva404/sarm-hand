"""Tests for grasp diagnostic logging."""

from __future__ import annotations

import json
from pathlib import Path

from sarm_hand.genesis.grasp_diag import GraspDiagnostic, GraspLogWriter


def test_grasp_log_writer_roundtrip(tmp_path: Path):
    path = tmp_path / "grasp_log.jsonl"
    writer = GraspLogWriter(path)
    diag = GraspDiagnostic(
        leader_gripper_deg=47.0,
        sim_gripper_deg=81.0,
        latched=True,
        prop_name="pen",
        prop_dist_m=0.07,
        pen_z_m=0.029,
        block_reason="latched",
        link_dist_m={"gripper": 0.072, "jaw": 0.084},
        latch_prop="pen",
    )
    writer.write(diag, episode=1, frame=3)
    writer.close()
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["latched"] is True
    assert row["block_reason"] == "latched"
    assert row["prop_dist_cm"] == 7.0
    assert row["episode"] == 1
    assert row["frame"] == 3


def test_grasp_log_prints_transition(capsys):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        writer = GraspLogWriter(Path(tmp) / "g.jsonl")
        open_diag = GraspDiagnostic(42.0, 42.0, False, block_reason="too_far")
        latched = GraspDiagnostic(70.0, 100.0, True, block_reason="latched", latch_prop="pen")
        writer.maybe_print_transition(open_diag)
        writer.maybe_print_transition(latched)
        writer.maybe_print_transition(latched)
        writer.close()
    out = capsys.readouterr().out
    assert "LATCHED pen" in out
    assert out.count("LATCHED") == 1
