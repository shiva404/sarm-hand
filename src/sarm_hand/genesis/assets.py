"""Resolve SO-101 URDF and mesh assets for Genesis."""

from __future__ import annotations

from pathlib import Path

from ..config import PROJECT_ROOT

ASSETS_DIR = PROJECT_ROOT / "assets" / "robots" / "so101"
DEFAULT_URDF = ASSETS_DIR / "so101_new_calib.urdf"


def resolve_urdf(path: str | Path | None = None) -> Path:
    """Return path to SO-101 URDF, raising if assets are missing."""
    urdf = Path(path) if path else DEFAULT_URDF
    if not urdf.is_absolute():
        urdf = PROJECT_ROOT / urdf
    if not urdf.is_file():
        raise FileNotFoundError(
            f"SO-101 URDF not found at {urdf}.\n"
            "Run:  ./scripts/fetch_so101_assets.sh"
        )
    return urdf
