"""Dataset recording for S-ARM101 using LeRobot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import ProjectConfig
from .robot import ensure_port


def record_leader(
    follower_port: str | None = None,
    leader_port: str | None = None,
    repo_id: str | None = None,
    num_episodes: int | None = None,
    single_task: str | None = None,
    push_to_hub: bool | None = None,
) -> None:
    """Record demonstrations with leader-follower teleoperation."""
    cfg = ProjectConfig.load()
    follower_port = ensure_port(follower_port or cfg.robot.port, "Follower")
    leader_port = ensure_port(leader_port or cfg.teleop.leader.port, "Leader")

    resolved_repo_id = repo_id or cfg.dataset.repo_id
    resolved_episodes = num_episodes if num_episodes is not None else cfg.dataset.num_episodes
    resolved_task = single_task or cfg.dataset.single_task
    resolved_push = push_to_hub if push_to_hub is not None else cfg.dataset.push_to_hub

    dataset_root = cfg.resolve_dataset_root()
    dataset_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lerobot-record",
        f"--robot.type={cfg.robot.type}",
        f"--robot.port={follower_port}",
        f"--robot.id={cfg.robot.id}",
        f"--teleop.type={cfg.teleop.leader.type}",
        f"--teleop.port={leader_port}",
        f"--teleop.id={cfg.teleop.leader.id}",
        f"--dataset.repo_id={resolved_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.fps={cfg.dataset.fps}",
        f"--dataset.num_episodes={resolved_episodes}",
        f"--dataset.single_task={resolved_task}",
        f"--dataset.episode_time_s={cfg.dataset.episode_time_s}",
        f"--dataset.reset_time_s={cfg.dataset.reset_time_s}",
        f"--dataset.push_to_hub={'true' if resolved_push else 'false'}",
        "--display_data=true",
    ]

    cameras = cfg.cameras_lerobot_dict()
    if cameras:
        cmd.append(f"--robot.cameras={cameras!r}")

    print(f"Recording {resolved_episodes} episodes → {resolved_repo_id}")
    print(f"Dataset root: {dataset_root}")
    print("Use the leader arm to demonstrate each episode.\n")
    subprocess.run(cmd, check=True)


def record_quest(
    repo_id: str | None = None,
    push_to_hub: bool | None = None,
) -> None:
    """Guide user to record via Quest 2 phospho app (records through phosphobot)."""
    cfg = ProjectConfig.load()
    resolved_repo_id = repo_id or cfg.dataset.repo_id
    resolved_push = push_to_hub if push_to_hub is not None else cfg.dataset.push_to_hub

    print(
        f"""
Quest 2 Data Collection
=======================

Recording is done through the phospho Quest app while phosphobot is running.

Steps:
  1. Start the server:  sarm-hand teleop quest
  2. Connect Quest 2 to phosphobot
  3. Press B on the controller to start/stop recording each episode
  4. Press Y to discard a bad episode

Target dataset repo_id: {resolved_repo_id}
Push to Hugging Face Hub: {resolved_push}

After recording, inspect data with:
  sarm-hand data info --repo-id {resolved_repo_id}
"""
    )


def record_policy(
    follower_port: str | None = None,
    policy_path: str = "",
    repo_id: str | None = None,
    num_episodes: int = 10,
) -> None:
    """Record evaluation rollouts using a trained policy (no teleop device)."""
    if not policy_path:
        print("Error: --policy-path is required for policy recording.", file=sys.stderr)
        sys.exit(1)

    cfg = ProjectConfig.load()
    follower_port = ensure_port(follower_port or cfg.robot.port, "Follower")
    resolved_repo_id = repo_id or f"{cfg.dataset.repo_id}-eval"
    dataset_root = cfg.resolve_dataset_root()

    cmd = [
        "lerobot-record",
        f"--robot.type={cfg.robot.type}",
        f"--robot.port={follower_port}",
        f"--robot.id={cfg.robot.id}",
        f"--dataset.repo_id={resolved_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.fps={cfg.dataset.fps}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={cfg.dataset.single_task}",
        f"--policy.path={policy_path}",
        "--display_data=true",
    ]

    cameras = cfg.cameras_lerobot_dict()
    if cameras:
        cmd.append(f"--robot.cameras={cameras!r}")

    subprocess.run(cmd, check=True)
