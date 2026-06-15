"""LeLab UI launcher and dataset visualization helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

from .config import PROJECT_ROOT, ProjectConfig

LELAB_DOCS = "https://huggingface.co/docs/lerobot/main/en/lelab"
LELAB_GITHUB = "git+https://github.com/huggingface/leLab.git"
LELAB_HF_SPACE = "https://huggingface.co/spaces/lerobot/LeLab"
DATASET_VIZ_SPACE = "https://huggingface.co/spaces/lerobot/visualize_dataset"


def lelab_env(config: ProjectConfig | None = None) -> dict[str, str]:
    """Environment variables so LeLab finds this project's datasets and calibration."""
    cfg = config or ProjectConfig.load()
    env = dict(os.environ)

    dataset_home = cfg.lelab.resolve_hf_lerobot_home(cfg)
    env["HF_LEROBOT_HOME"] = str(dataset_home)
    env["SARM_HAND_PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["SARM_HAND_DATASET_REPO_ID"] = cfg.dataset.repo_id

    if cfg.robot.port:
        env["SARM_HAND_FOLLOWER_PORT"] = cfg.robot.port
    if cfg.teleop.leader.port:
        env["SARM_HAND_LEADER_PORT"] = cfg.teleop.leader.port

    return env


def find_lelab_command() -> list[str] | None:
    """Return argv prefix to launch LeLab, or None if not installed."""
    if shutil.which("lelab"):
        return ["lelab"]

    try:
        import lelab.scripts.lelab  # noqa: F401
    except ImportError:
        return None

    return [sys.executable, "-m", "lelab.scripts.lelab"]


def install_lelab() -> None:
    """Install LeLab as a uv tool (recommended — avoids lerobot version conflicts)."""
    print("Installing LeLab via uv tool...")
    print(f"  source: {LELAB_GITHUB}\n")
    subprocess.run(["uv", "tool", "install", LELAB_GITHUB], check=True)
    print("\nLeLab installed. Launch with:")
    print("  uv run sarm-hand lelab")


def launch_lelab(*, dev: bool = False, open_browser: bool = True) -> None:
    """Start LeLab with project dataset paths configured."""
    cfg = ProjectConfig.load()
    cmd = find_lelab_command()
    if cmd is None:
        print(
            "LeLab is not installed.\n\n"
            "Install (recommended):\n"
            f"  uv run sarm-hand lelab --install\n\n"
            "Or manually:\n"
            f"  uv tool install {LELAB_GITHUB}\n\n"
            f"Docs: {LELAB_DOCS}",
            file=sys.stderr,
        )
        sys.exit(1)

    env = lelab_env(cfg)
    if dev:
        cmd.append("--dev")

    dataset_home = env["HF_LEROBOT_HOME"]
    print("Launching LeLab — LeRobot web UI for S-ARM101")
    print(f"  Datasets:  {dataset_home}")
    print(f"  Default:   {cfg.dataset.repo_id}")
    print(f"  Robot id:  {cfg.robot.id}")
    print()
    print("LeLab features:")
    print("  - Calibrate leader/follower with guided web flow")
    print("  - Teleoperate with live 3D arm rendering")
    print("  - Record, browse, upload datasets")
    print("  - Train policies and run inference")
    print("  - Replay episodes with embedded dataset visualizer")
    print()
    print("  App:    http://localhost:8000")
    print(f"  Docs:   {LELAB_DOCS}")
    print()

    if open_browser and cfg.lelab.open_browser and not dev:
        webbrowser.open(f"http://localhost:{cfg.lelab.port}/")

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nLeLab stopped.")


def open_dataset_viz_hub(repo_id: str | None = None) -> None:
    """Open the Hugging Face dataset visualizer (3D URDF + charts) in the browser."""
    cfg = ProjectConfig.load()
    resolved = repo_id or cfg.dataset.repo_id
    print(f"Opening dataset visualizer for: {resolved}")
    print("Paste the repo id into the Space if it is not pre-filled.")
    print(f"  Space: {DATASET_VIZ_SPACE}")
    webbrowser.open(DATASET_VIZ_SPACE)


def viz_dataset_local(
    repo_id: str | None = None,
    root: str | None = None,
    episode: int = 0,
) -> None:
    """Launch local lerobot-dataset-viz (Rerun) for a project dataset."""
    cfg = ProjectConfig.load()
    resolved_repo = repo_id or cfg.dataset.repo_id
    resolved_root = root or str(cfg.lelab.resolve_hf_lerobot_home(cfg))
    dataset_path = Path(resolved_root) / Path(*resolved_repo.split("/"))

    if not (dataset_path / "meta" / "info.json").exists():
        print(
            f"Dataset not found at {dataset_path}\n"
            f"Record data first or check repo-id / HF_LEROBOT_HOME ({resolved_root}).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Opening local dataset viewer (Rerun)")
    print(f"  repo_id:  {resolved_repo}")
    print(f"  root:     {resolved_root}")
    print(f"  episode:  {episode}")

    subprocess.run(
        [
            "lerobot-dataset-viz",
            f"--repo-id={resolved_repo}",
            f"--root={resolved_root}",
            "--mode=local",
            f"--episode-index={episode}",
        ],
        check=True,
        env=lelab_env(cfg),
    )


def lelab_info() -> None:
    """Print LeLab integration status and URLs."""
    cfg = ProjectConfig.load()
    cmd = find_lelab_command()
    dataset_home = cfg.lelab.resolve_hf_lerobot_home(cfg)

    print("LeLab integration")
    print(f"  Installed:     {'yes' if cmd else 'no — run: uv run sarm-hand lelab --install'}")
    print(f"  HF_LEROBOT_HOME: {dataset_home}")
    print(f"  Default dataset: {cfg.dataset.repo_id}")
    print(f"  LeLab app:     http://localhost:{cfg.lelab.port}")
    print(f"  HF Space:      {LELAB_HF_SPACE}")
    print(f"  Dataset viz:   {DATASET_VIZ_SPACE}")
    print(f"  Docs:          {LELAB_DOCS}")
