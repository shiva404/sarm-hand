"""Terminal banners and countdown for record-leader."""

from __future__ import annotations

import sys

BANNER_WIDTH = 72


def _rule(char: str = "#") -> str:
    return char * BANNER_WIDTH


def print_session_ready(*, num_episodes: int, episode_time_s: float) -> None:
    print()
    print(_rule())
    print("###  RECORD-LEADER READY")
    print(f"###  {num_episodes} episode(s)  |  {episode_time_s:.0f}s max per episode")
    print("###  Click this terminal — use s / r / q (or wait for timer)")
    print(_rule())
    print(flush=True)


def print_episode_banner(
    *,
    episode_index: int,
    num_episodes: int,
    task: str,
    duration_s: float,
    phase: str = "record",
) -> None:
    ep_label = f"Episode {episode_index + 1} of {num_episodes}"
    if phase == "reset":
        print()
        print(_rule())
        print("###  RESET WINDOW — reposition objects (not saved)")
        print(f"###  {ep_label}  |  {duration_s:.0f}s")
        print(_rule())
        print(flush=True)
        return

    print()
    print(_rule())
    print("###  RECORDING STARTED — MOVE THE LEADER ARM NOW")
    print(f"###  {ep_label}  |  Task: {task!r}")
    print(f"###  {duration_s:.0f}s max  |  s = save now  |  r = re-record  |  q = stop")
    print(_rule())
    print(flush=True)


def write_countdown_line(*, seconds_left: int, frames_recorded: int, phase: str = "record") -> None:
    label = "recording" if phase == "record" else "reset"
    sys.stderr.write(
        f"\r###  {seconds_left:3d}s left ({label})  |  dataset frames: {frames_recorded}     "
    )
    sys.stderr.flush()


def clear_countdown_line() -> None:
    sys.stderr.write("\n")
    sys.stderr.flush()
