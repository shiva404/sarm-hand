"""Record leader-arm demos and replay them on the follower by task name."""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import JOINT_NAMES, ProjectConfig
from .robot import (
    _motor_write_retries,
    build_robot,
    disable_arm_torque,
    ensure_port,
    require_all_motors,
)

TASK_MOTION_VERSION = 1
_DEMO_PREFIX = "demo_"


@dataclass
class TaskMotionFrame:
    t: float
    joints: dict[str, float]


@dataclass
class TaskMotionDemo:
    version: int
    task: str
    task_slug: str
    demo_id: str
    recorded_at: str
    fps: int
    source: str
    frames: list[TaskMotionFrame]

    @property
    def duration_s(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames[-1].t

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task": self.task,
            "task_slug": self.task_slug,
            "demo_id": self.demo_id,
            "recorded_at": self.recorded_at,
            "fps": self.fps,
            "source": self.source,
            "frames": [asdict(frame) for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskMotionDemo:
        frames = [
            TaskMotionFrame(
                t=float(item["t"]),
                joints={k: float(v) for k, v in item["joints"].items()},
            )
            for item in raw.get("frames", [])
        ]
        return cls(
            version=int(raw.get("version", 1)),
            task=str(raw["task"]),
            task_slug=str(raw["task_slug"]),
            demo_id=str(raw["demo_id"]),
            recorded_at=str(raw.get("recorded_at", "")),
            fps=int(raw.get("fps", 30)),
            source=str(raw.get("source", "leader")),
            frames=frames,
        )


def task_slug(task: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", task.strip().lower())
    slug = slug.strip("_")
    if not slug:
        raise ValueError("Task name must contain at least one letter or digit.")
    return slug


def _demo_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _action_from_joints(joints: dict[str, float]) -> dict[str, float]:
    return {f"{joint}.pos": float(joints[joint]) for joint in JOINT_NAMES}


def _joints_from_action(action: dict[str, float]) -> dict[str, float]:
    return {joint: float(action[f"{joint}.pos"]) for joint in JOINT_NAMES}


def resolve_tasks_root(cfg: ProjectConfig) -> Path:
    return cfg.resolve_tasks_root()


def task_dir(cfg: ProjectConfig, slug: str) -> Path:
    return resolve_tasks_root(cfg) / slug


def demo_path(cfg: ProjectConfig, slug: str, demo_id: str) -> Path:
    demo_name = demo_id if demo_id.startswith(_DEMO_PREFIX) else f"{_DEMO_PREFIX}{demo_id}"
    if not demo_name.endswith(".json"):
        demo_name = f"{demo_name}.json"
    return task_dir(cfg, slug) / demo_name


def save_demo(demo: TaskMotionDemo, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(demo.to_dict(), indent=2) + "\n")


def load_demo(path: Path) -> TaskMotionDemo:
    raw = json.loads(path.read_text())
    return TaskMotionDemo.from_dict(raw)


def list_demos(cfg: ProjectConfig, slug: str) -> list[Path]:
    directory = task_dir(cfg, slug)
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{_DEMO_PREFIX}*.json"))


def resolve_demo_path(
    cfg: ProjectConfig,
    *,
    task: str | None = None,
    task_slug: str | None = None,
    demo_id: str | None = None,
) -> Path:
    slug = task_slug or task_slug_from_task(task or "")
    demos = list_demos(cfg, slug)
    if not demos:
        print(f"No demos found for task '{slug}' under {task_dir(cfg, slug)}", file=sys.stderr)
        raise SystemExit(1)

    if demo_id in (None, "latest", "last"):
        return demos[-1]

    candidate = demo_path(cfg, slug, demo_id)
    if candidate.is_file():
        return candidate

    matches = [p for p in demos if p.stem == demo_id or p.stem == f"{_DEMO_PREFIX}{demo_id}"]
    if len(matches) == 1:
        return matches[0]
    if matches:
        print(f"Ambiguous demo id '{demo_id}' for task '{slug}'.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Demo '{demo_id}' not found for task '{slug}'.", file=sys.stderr)
    raise SystemExit(1)


def task_slug_from_task(task: str) -> str:
    try:
        return task_slug(task)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def list_task_motions(cfg: ProjectConfig | None = None) -> None:
    cfg = cfg or ProjectConfig.load()
    root = resolve_tasks_root(cfg)
    if not root.is_dir():
        print("No task demos yet. Record one with:\n  sarm-hand task record --task \"Your task\"")
        return

    rows: list[tuple[str, str, int, float, str]] = []
    for slug_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for demo_file in sorted(slug_dir.glob(f"{_DEMO_PREFIX}*.json")):
            demo = load_demo(demo_file)
            rows.append(
                (demo.task_slug, demo.demo_id, len(demo.frames), demo.duration_s, demo.task)
            )

    if not rows:
        print(f"No demos under {root}")
        return

    print(f"Task demos ({root}):\n")
    print(f"{'Slug':<24} {'Demo':<22} {'Frames':>7} {'Duration':>10}  Task")
    print("-" * 90)
    for slug, demo_id, frames, duration, task in rows:
        print(f"{slug:<24} {demo_id:<22} {frames:7d} {duration:8.1f}s  {task}")


def show_task_motion(
    *,
    task: str | None = None,
    task_slug: str | None = None,
    demo_id: str | None = None,
) -> None:
    cfg = ProjectConfig.load()
    path = resolve_demo_path(cfg, task=task, task_slug=task_slug, demo_id=demo_id)
    demo = load_demo(path)
    print(f"Task demo: {path}")
    print(f"  Task:       {demo.task}")
    print(f"  Slug:       {demo.task_slug}")
    print(f"  Demo id:    {demo.demo_id}")
    print(f"  Recorded:   {demo.recorded_at}")
    print(f"  FPS:        {demo.fps}")
    print(f"  Frames:     {len(demo.frames)}")
    print(f"  Duration:   {demo.duration_s:.1f}s")
    print(f"  Source:     {demo.source}")
    if demo.frames:
        first = demo.frames[0].joints
        last = demo.frames[-1].joints
        moved = [j for j in JOINT_NAMES if abs(last[j] - first[j]) > 1.0]
        print(f"  Moved joints: {', '.join(moved) if moved else '(minimal motion)'}")


def record_task_motion(
    *,
    task: str,
    leader_port: str | None = None,
    follower_port: str | None = None,
    fps: int | None = None,
    duration_s: float | None = None,
    mirror_follower: bool = True,
) -> Path:
    """Record a leader-arm demonstration tagged with a task name."""
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader import SO101Leader
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.utils.robot_utils import precise_sleep

    cfg = ProjectConfig.load()
    slug = task_slug_from_task(task)
    resolved_fps = fps if fps is not None else cfg.tasks.fps
    follower_port = ensure_port(follower_port or cfg.robot.port, "Follower")
    leader_port = ensure_port(leader_port or cfg.teleop.leader.port, "Leader")

    require_all_motors("leader", leader_port, context="task record")
    if mirror_follower:
        require_all_motors("follower", follower_port, context="task record")

    disable_arm_torque("leader", leader_port)

    leader_cfg = SO101LeaderConfig(
        id=cfg.teleop.leader.id,
        port=leader_port,
        use_degrees=cfg.robot.use_degrees,
    )
    leader = SO101Leader(leader_cfg)
    teleop_action_processor, robot_action_processor, _ = make_default_processors()

    robot: SO101Follower | None = None
    if mirror_follower:
        robot_cfg = SOFollowerRobotConfig(
            id=cfg.robot.id,
            port=follower_port,
            use_degrees=cfg.robot.use_degrees,
            max_relative_target=cfg.robot.max_relative_target,
            disable_torque_on_disconnect=cfg.robot.disable_torque_on_disconnect,
            cameras={},
        )
        robot = SO101Follower(robot_cfg)

    print("Task motion recording")
    print(f"  Task:     {task}")
    print(f"  Slug:     {slug}")
    print(f"  Leader:   {leader_port}")
    if mirror_follower:
        print(f"  Follower: {follower_port} (mirrors while recording)")
    print(f"  FPS:      {resolved_fps}")
    if duration_s:
        print(f"  Limit:    {duration_s:.0f}s")
    print("\nMove the leader arm through the task. Press Ctrl+C when done.\n")

    frames: list[TaskMotionFrame] = []
    interval = 1.0 / resolved_fps
    start = time.perf_counter()
    deadline = start + duration_s if duration_s else None

    try:
        with _motor_write_retries():
            leader.connect()
            if robot is not None:
                robot.connect()

        while deadline is None or time.perf_counter() < deadline:
            loop_start = time.perf_counter()
            raw_action = leader.get_action()
            teleop_action = teleop_action_processor((raw_action, {}))
            joints = _joints_from_action(teleop_action)
            frames.append(TaskMotionFrame(t=loop_start - start, joints=joints))

            if robot is not None:
                obs = robot.get_observation()
                sent = robot_action_processor((teleop_action, obs))
                robot.send_action(sent)

            if len(frames) == 1 or len(frames) % resolved_fps == 0:
                preview = ", ".join(f"{j}={joints[j]:.0f}" for j in JOINT_NAMES[:3])
                print(f"  [{len(frames):4d} frames] {preview}...")

            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(interval - elapsed, 0.0))
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        if leader.is_connected:
            leader.disconnect()
        if robot is not None and robot.is_connected:
            robot.disconnect()

    if not frames:
        print("No frames captured — nothing saved.", file=sys.stderr)
        raise SystemExit(1)

    demo_id = f"{_DEMO_PREFIX}{_demo_stamp()}"
    demo = TaskMotionDemo(
        version=TASK_MOTION_VERSION,
        task=task,
        task_slug=slug,
        demo_id=demo_id,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        fps=resolved_fps,
        source="leader",
        frames=frames,
    )
    path = demo_path(cfg, slug, demo_id)
    save_demo(demo, path)
    print(f"\nSaved {len(frames)} frames ({demo.duration_s:.1f}s) → {path}")
    print(f"Replay: sarm-hand task replay --task-slug {slug} --demo latest")
    return path


def replay_task_motion(
    *,
    task: str | None = None,
    task_slug: str | None = None,
    demo_id: str | None = None,
    follower_port: str | None = None,
    speed: float = 1.0,
    loop: bool = False,
    pause_s: float = 2.0,
) -> None:
    """Replay a recorded task demo on the follower arm."""
    from lerobot.utils.robot_utils import precise_sleep

    if speed <= 0:
        print("--speed must be > 0", file=sys.stderr)
        raise SystemExit(1)

    cfg = ProjectConfig.load()
    path = resolve_demo_path(cfg, task=task, task_slug=task_slug, demo_id=demo_id)
    demo = load_demo(path)
    port = ensure_port(follower_port or cfg.robot.port, "Follower")
    require_all_motors("follower", port, context="task replay")

    print("Task motion replay")
    print(f"  Task:     {demo.task}")
    print(f"  Demo:     {demo.demo_id} ({len(demo.frames)} frames, {demo.duration_s:.1f}s)")
    print(f"  File:     {path}")
    print(f"  Follower: {port}")
    print(f"  Speed:    {speed}x")
    if loop:
        print("  Loop:     yes")
    print(f"\nStarting in {pause_s:.0f}s — reset the scene, then stand clear. Ctrl+C to stop.\n")
    time.sleep(pause_s)

    robot = build_robot(port, cfg)
    robot.config.max_relative_target = None

    if not robot.is_calibrated:
        robot.disconnect()
        print("Follower is not calibrated. Run calibrate first.", file=sys.stderr)
        raise SystemExit(1)

    interval = 1.0 / (demo.fps * speed)
    run = 0
    try:
        while True:
            run += 1
            if loop and run > 1:
                print(f"\n--- Loop {run} ---")
            for index, frame in enumerate(demo.frames):
                loop_start = time.perf_counter()
                action = _action_from_joints(frame.joints)
                robot.send_action(action)
                if index == 0 or (index + 1) % demo.fps == 0:
                    print(f"  frame {index + 1}/{len(demo.frames)}")
                elapsed = time.perf_counter() - loop_start
                precise_sleep(max(interval - elapsed, 0.0))
            if not loop:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.disconnect()

    print("\nReplay finished.")
