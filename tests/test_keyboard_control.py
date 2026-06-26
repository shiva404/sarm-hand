"""Tests for stdin episode controls."""

from __future__ import annotations

from sarm_hand.keyboard_control import _recording_events_template, start_stdin_episode_controls


def test_recording_events_template():
    events = _recording_events_template()
    assert events == {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }


def test_stdin_controls_noop_without_tty(monkeypatch):
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    events = _recording_events_template()
    thread = start_stdin_episode_controls(events)
    assert not thread.is_alive()
