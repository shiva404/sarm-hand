"""Hardware → Genesis digital twin loop."""

from __future__ import annotations

import sys
import time

from .backends.genesis_twin import GenesisTwin
from .config import ProjectConfig
from .genesis.deps import ensure_genesis
from .genesis.shutdown import (
    check_shutdown,
    ensure_shutdown_handlers,
    exit_after_interrupt,
    install_shutdown_handlers,
    interruptible_sleep,
    shutdown_requested,
)
from .robot import ensure_port, _motor_write_retries


def run_twin(
    *,
    follower_port: str | None = None,
    rate_hz: float | None = None,
    duration_s: float | None = None,
) -> None:
    """Mirror USB follower joint positions into a Genesis World scene."""
    ensure_genesis()
    install_shutdown_handlers()
    cfg = ProjectConfig.load()
    port = ensure_port(follower_port or cfg.robot.port, "Follower")
    hz = rate_hz if rate_hz is not None else cfg.twin.rate_hz
    duration = duration_s if duration_s is not None else cfg.twin.duration_s
    interval = 1.0 / hz

    print("Genesis digital twin (hardware → sim)")
    print(f"  Follower: {port}")
    print(f"  Rate:     {hz} Hz")
    print(f"  Backend:  {cfg.genesis.backend}")
    print(f"  Scene:    {cfg.genesis.scene}")
    if duration:
        print(f"  Duration: {duration:.0f}s")
    print("\nMove the follower arm — Genesis mirror should track.\n")

    twin = GenesisTwin(port, cfg)
    deadline = time.perf_counter() + duration if duration else None
    steps = 0
    interrupted = False

    try:
        with _motor_write_retries():
            twin.start()
        ensure_shutdown_handlers()
        while deadline is None or time.perf_counter() < deadline:
            check_shutdown()
            loop_start = time.perf_counter()
            twin.sync_hardware_to_sim()
            steps += 1
            if steps == 1 or steps % int(hz * 5) == 0:
                obs = twin.hardware.get_observation()
                joints = [f"{k}={v:.1f}" for k, v in obs.items() if k.endswith(".pos")]
                print(f"  step {steps}: {', '.join(joints[:3])}...")
            elapsed = time.perf_counter() - loop_start
            interruptible_sleep(max(interval - elapsed, 0.0))
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped.")
    except ConnectionError as exc:
        print(f"\nLost contact with follower: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        twin.stop()
        if interrupted or shutdown_requested():
            exit_after_interrupt()

    print(f"Done ({steps} steps).")


def run_genesis_spike(*, headless: bool = False) -> None:
    """Smoke-test Genesis: load SO-101 URDF, step physics, render one frame."""
    ensure_genesis()
    install_shutdown_handlers()
    from .genesis.scene import SO101GenesisScene

    cfg = ProjectConfig.load()
    if headless:
        cfg.genesis.headless = True
    print("Genesis spike: loading SO-101 scene...")
    scene = SO101GenesisScene.create(cfg)
    ensure_shutdown_handlers()
    interrupted = False
    try:
        scene.apply_home_pose()
        for _ in range(30):
            check_shutdown()
            scene.step()
        for name in scene.cameras:
            frame = scene.render_rgb(name)
            if frame is not None:
                print(f"  Camera {name}: {frame.shape[1]}x{frame.shape[0]} dtype={frame.dtype}")
        if not cfg.genesis.headless:
            print("  Preview: OpenCV windows 'sarm-hand: front|top|arm' (Ctrl+C to exit)")
            scene.refresh_previews()
            while not shutdown_requested():
                check_shutdown()
                scene.refresh_previews()
                interruptible_sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped.")
    finally:
        scene.close()
        if interrupted or shutdown_requested():
            exit_after_interrupt()
    print("Genesis spike passed.")
