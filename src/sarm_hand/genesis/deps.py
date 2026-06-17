"""Optional Genesis dependency checks."""

from __future__ import annotations

import sys


def ensure_genesis() -> None:
    """Verify genesis-world is installed (`uv sync --extra genesis`)."""
    try:
        import genesis  # noqa: F401
    except ImportError:
        print(
            "Genesis World is not installed.\n"
            "Install with:  uv sync --extra genesis\n"
            "Then fetch SO-101 assets:  ./scripts/fetch_so101_assets.sh",
            file=sys.stderr,
        )
        sys.exit(1)


def ensure_lerobot_genesis() -> None:
    """Verify lerobot-genesis bridge is installed."""
    ensure_genesis()
    try:
        import lerobot_genesis  # noqa: F401
    except ImportError:
        print(
            "lerobot-genesis is not installed.\n"
            "Install with:  uv sync --extra genesis",
            file=sys.stderr,
        )
        sys.exit(1)
