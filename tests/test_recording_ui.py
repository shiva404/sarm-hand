"""Tests for record-leader terminal UI."""

from __future__ import annotations

import re

from sarm_hand.recording_ui import print_episode_banner, print_session_ready


def test_session_ready_banner(capsys):
    print_session_ready(num_episodes=3, episode_time_s=60.0)
    out = capsys.readouterr().out
    assert out.count("#") >= 72
    assert "RECORD-LEADER READY" in out
    assert "3 episode" in out


def test_recording_episode_banner(capsys):
    print_episode_banner(
        episode_index=0,
        num_episodes=5,
        task="Pick cube",
        duration_s=60.0,
        phase="record",
    )
    out = capsys.readouterr().out
    assert "RECORDING STARTED" in out
    assert "MOVE THE LEADER ARM NOW" in out
    assert "Episode 1 of 5" in out
    assert "Pick cube" in out


def test_reset_banner(capsys):
    print_episode_banner(
        episode_index=1,
        num_episodes=5,
        task="Pick cube",
        duration_s=10.0,
        phase="reset",
    )
    out = capsys.readouterr().out
    assert "RESET WINDOW" in out
    assert re.search(r"10s", out)
