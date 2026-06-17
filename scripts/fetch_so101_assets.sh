#!/usr/bin/env bash
# Download SO-101 URDF + meshes from TheRobotStudio/SO-ARM100 for Genesis sim.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/assets/robots/so101"
TMP="${TMPDIR:-/tmp}/so-arm101-sim-$$"
REPO="https://github.com/TheRobotStudio/SO-ARM100.git"
SPARSE_PATH="Simulation/SO101"

mkdir -p "$DEST"
echo "Fetching SO-101 simulation assets into $DEST ..."

if command -v git >/dev/null 2>&1; then
  rm -rf "$TMP"
  git clone --depth 1 --filter=blob:none --sparse "$REPO" "$TMP"
  git -C "$TMP" sparse-checkout set "$SPARSE_PATH"
  rsync -a "$TMP/$SPARSE_PATH/" "$DEST/"
  rm -rf "$TMP"
else
  echo "git is required. Install git or copy Simulation/SO101 manually to $DEST" >&2
  exit 1
fi

if [[ ! -f "$DEST/so101_new_calib.urdf" ]]; then
  echo "URDF not found after fetch." >&2
  exit 1
fi

echo "Done: $DEST/so101_new_calib.urdf"
echo "Run:  uv sync --extra genesis && sarm-hand genesis-spike"
