"""Tests for cooperative Genesis shutdown helpers."""

from __future__ import annotations

import signal

import pytest

from sarm_hand.genesis.shutdown import (
    _sigint_handler,
    check_shutdown,
    ensure_shutdown_handlers,
    interruptible_sleep,
    reset_shutdown_state,
    shutdown_requested,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_shutdown_state()
    yield
    reset_shutdown_state()


def test_interruptible_sleep_returns():
    interruptible_sleep(0.01)
    assert not shutdown_requested()


def test_check_shutdown_raises_when_flag_set():
    import sarm_hand.genesis.shutdown as mod

    mod._shutdown_requested = True
    with pytest.raises(KeyboardInterrupt):
        check_shutdown()


def test_ensure_shutdown_handlers_installs_sigint():
    ensure_shutdown_handlers()
    assert signal.getsignal(signal.SIGINT) is _sigint_handler
    assert signal.getsignal(signal.SIGTERM) is _sigint_handler
