"""Episode controls that work without macOS Accessibility (pynput)."""

from __future__ import annotations

import select
import sys
import termios
import threading
import tty
from typing import Any


def _recording_events_template() -> dict[str, bool]:
    return {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }


def start_stdin_episode_controls(events: dict[str, bool]) -> threading.Thread:
    """Read s/r/q from the terminal — no Accessibility permission required."""
    if not sys.stdin.isatty():
        return threading.Thread()  # non-tty: no-op thread

    def _reader() -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not events.get("stop_recording"):
                ready, _, _ = select.select([sys.stdin], [], [], 0.25)
                if not ready:
                    continue
                ch = sys.stdin.read(1).lower()
                if ch == "s":
                    print("\n[s] Save episode now.", flush=True)
                    events["exit_early"] = True
                elif ch == "r":
                    print("\n[r] Re-record episode.", flush=True)
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif ch == "q":
                    print("\n[q] Stop recording.", flush=True)
                    events["stop_recording"] = True
                    events["exit_early"] = True
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    thread = threading.Thread(target=_reader, name="stdin-episode-controls", daemon=True)
    thread.start()
    return thread


def init_recording_keyboard_listener(
    *,
    use_pynput: bool = True,
    stdin_fallback: bool = True,
) -> tuple[Any | None, dict[str, bool], threading.Thread | None]:
    """pynput arrows when allowed; always offer stdin s/r/q as fallback on a TTY."""
    events = _recording_events_template()
    pynput_listener = None
    stdin_thread: threading.Thread | None = None

    if use_pynput:
        try:
            from lerobot.utils.control_utils import init_keyboard_listener

            pynput_listener, pynput_events = init_keyboard_listener()
            events = pynput_events
        except Exception:
            pynput_listener = None

    if stdin_fallback and sys.stdin.isatty():
        stdin_thread = start_stdin_episode_controls(events)

    return pynput_listener, events, stdin_thread


def print_stdin_controls_hint() -> None:
    if not sys.stdin.isatty():
        return
    print(
        "Terminal controls (click this window; no Accessibility needed):\n"
        "  s   save episode now\n"
        "  r   discard and re-record\n"
        "  q   stop all recording\n"
        "  (Arrow keys need macOS Accessibility for Cursor/Terminal.)\n"
    )
