"""Cooperative shutdown for long-running Genesis loops (Ctrl+C friendly)."""

from __future__ import annotations

import os
import signal
import threading
import time

_shutdown_requested = False
_watchdog_started = False


def shutdown_requested() -> bool:
    return _shutdown_requested


def _sigint_handler(signum: int, _frame) -> None:
    global _shutdown_requested
    if _shutdown_requested:
        print("\nForce quit.", flush=True)
        os._exit(128 + signum)
    _shutdown_requested = True
    print("\nStopping... (press Ctrl+C again to force quit)", flush=True)
    raise KeyboardInterrupt


def _start_shutdown_watchdog(timeout_s: float = 12.0) -> None:
    """Force-exit if graceful teardown stalls (e.g. Genesis kernel cache)."""
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True

    def _watch() -> None:
        while not _shutdown_requested:
            time.sleep(0.1)
        time.sleep(timeout_s)
        if _shutdown_requested:
            print("\nCleanup timed out; forcing exit.", flush=True)
            os._exit(130)

    threading.Thread(target=_watch, daemon=True, name="shutdown-watchdog").start()


def install_shutdown_handlers() -> None:
    """Register SIGINT/SIGTERM handlers and start the shutdown watchdog."""
    ensure_shutdown_handlers()
    _start_shutdown_watchdog()


def ensure_shutdown_handlers() -> None:
    """Re-assert our handlers if another library replaced them after scene init."""
    if signal.getsignal(signal.SIGINT) is not _sigint_handler:
        signal.signal(signal.SIGINT, _sigint_handler)
    if signal.getsignal(signal.SIGTERM) is not _sigint_handler:
        signal.signal(signal.SIGTERM, _sigint_handler)


def check_shutdown() -> None:
    """Raise ``KeyboardInterrupt`` once shutdown was requested."""
    if _shutdown_requested:
        raise KeyboardInterrupt


def interruptible_sleep(seconds: float) -> None:
    """Sleep in short slices so Ctrl+C is picked up promptly."""
    if seconds <= 0:
        return
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        check_shutdown()
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))


def exit_after_interrupt(code: int = 130) -> None:
    """Terminate immediately after interrupt cleanup (skip slow Genesis atexit)."""
    os._exit(code)


def reset_shutdown_state() -> None:
    """Clear shutdown flag (mainly for tests)."""
    global _shutdown_requested, _watchdog_started
    _shutdown_requested = False
    _watchdog_started = False
