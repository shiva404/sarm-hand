"""Teleoperation: leader-follower (LeRobot) and Meta Quest 2 (phospho)."""

from __future__ import annotations

import subprocess
import sys
import webbrowser

from .calibration_bridge import (
    calibration_mismatch_report,
    remap_leader_action_to_follower,
    require_teleop_calibrations,
)
from .cameras import build_robot_camera_configs, connect_follower_robot
from .config import ProjectConfig
from .rerun_viz import init_leader_teleop_rerun, leader_teleop_loop
from .robot import (
    _motor_write_retries,
    disable_arm_torque,
    ensure_port,
    require_all_motors,
)


def teleop_leader(
    follower_port: str | None = None,
    leader_port: str | None = None,
    display_data: bool = True,
    with_cameras: bool = False,
) -> None:
    """Teleoperate S-ARM101 follower with a matching leader arm via USB."""
    from .cameras import install_all_camera_patches

    install_all_camera_patches()
    import rerun as rr
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader import SO101Leader
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.utils.utils import init_logging

    cfg = ProjectConfig.load()
    follower_port = ensure_port(follower_port or cfg.robot.port, "Follower")
    leader_port = ensure_port(leader_port or cfg.teleop.leader.port, "Leader")
    leader_cal, follower_cal = require_teleop_calibrations(cfg)
    mismatch = calibration_mismatch_report(leader_cal, follower_cal)
    if mismatch:
        print("Calibration mismatch (teleop remaps via encoder counts):")
        for line in mismatch:
            print(f"  - {line}")
        print("  Fix permanently:  sarm-hand sync-calibration --from leader --write-motors\n")

    def _remap(action: dict[str, float]) -> dict[str, float]:
        return remap_leader_action_to_follower(
            action,
            leader_cal=leader_cal,
            follower_cal=follower_cal,
        )

    require_all_motors("leader", leader_port, context="teleoperate")
    require_all_motors("follower", follower_port, context="teleoperate")

    print("Leader-follower teleoperation")
    print(f"  Leader (move by hand):     {leader_port}")
    print(f"  Follower (mirrors leader): {follower_port}")
    print("  The FOLLOWER holds position and will feel stiff — only the LEADER is backdrivable.")
    print()

    disable_arm_torque("leader", leader_port)

    if with_cameras and not cfg.cameras:
        print("No cameras configured in config/default.yaml — continuing without.", file=sys.stderr)

    if with_cameras and cfg.cameras:
        from .cameras import prepare_opencv_platform

        prepare_opencv_platform()

    robot_cfg = SOFollowerRobotConfig(
        id=cfg.robot.id,
        port=follower_port,
        use_degrees=cfg.robot.use_degrees,
        max_relative_target=cfg.robot.max_relative_target,
        disable_torque_on_disconnect=cfg.robot.disable_torque_on_disconnect,
        cameras=build_robot_camera_configs(cfg) if with_cameras else {},
    )
    leader_cfg = SO101LeaderConfig(
        id=cfg.teleop.leader.id,
        port=leader_port,
        use_degrees=cfg.robot.use_degrees,
    )

    init_logging()
    if display_data:
        camera_names = list(cfg.cameras) if with_cameras and cfg.cameras else None
        init_leader_teleop_rerun(
            session_name="teleoperation",
            with_cameras=with_cameras,
            camera_names=camera_names,
        )
        print("Rerun: follower/leader joint graphs should open automatically.")
        if camera_names:
            print(f"  Cameras: {', '.join(camera_names)}  |  Timeline: step")
        else:
            print("  Timeline: step  |  Entities: motors/follower/*, motors/leader/*")

    robot = SO101Follower(robot_cfg)
    teleop = SO101Leader(leader_cfg)
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    print("Starting teleoperation (Ctrl+C to stop)...")
    try:
        with _motor_write_retries():
            if with_cameras and robot.cameras:
                connect_follower_robot(robot, calibrate=False)
            else:
                robot.connect(calibrate=False)
            teleop.connect()
        leader_teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.teleop.control_fps,
            display_data=display_data,
            duration=None,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            remap_action=_remap,
            camera_names=list(cfg.cameras) if with_cameras and cfg.cameras else None,
            action_smoothing=cfg.teleop.action_smoothing,
        )
    except (ConnectionError, TimeoutError) as exc:
        err = str(exc).lower()
        if "camera" in err or "opencv" in err or isinstance(exc, TimeoutError):
            print(
                "\nFailed to connect a USB camera.\n"
                "  sarm-hand list-cameras\n"
                "  sarm-hand camera-test\n"
                "\nTips:"
                "\n  - Close FaceTime/Zoom; allow Camera access for Terminal"
                "\n  - Unplug/replug cameras; verify index_or_path in config/default.yaml"
                "\n  - Try teleop without cameras first: sarm-hand teleop-leader",
                file=sys.stderr,
            )
        else:
            print(
                "\nLost contact with a servo while connecting the follower or leader.\n"
                "This is usually a loose daisy-chain cable or insufficient 12V power.\n"
                "\n  sarm-hand test-motors --role follower\n"
                "  sarm-hand test-motors --role leader\n"
                "\nReseat the 3-pin cable at the joint mentioned in the error, then retry.",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        pass
    finally:
        if display_data:
            rr.rerun_shutdown()
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


def teleop_quest(
    follower_port: str | None = None,
    open_dashboard: bool = True,
) -> None:
    """Start phosphobot server for Meta Quest 2 VR teleoperation."""
    try:
        import phosphobot  # noqa: F401
    except ImportError:
        print(
            "phosphobot is required for Quest 2 teleoperation.\n"
            "Install with: uv sync --extra quest",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = ProjectConfig.load()
    follower_port = ensure_port(follower_port or cfg.robot.port, "Follower")

    print("Starting phosphobot for Quest 2 teleoperation.")
    print(f"  Follower port: {follower_port}")
    print(f"  Server: http://localhost:{cfg.teleop.quest.port}")
    print()
    print("Quest 2 setup:")
    print("  1. Install the phospho teleoperation app from the Meta Store")
    print("  2. Connect Quest 2 and your computer to the same WiFi network")
    print("  3. Open the phospho app on Quest and connect to this server")
    print("  4. Press A to start/stop teleoperation, B to record, Y to discard")
    print()

    cmd = ["phosphobot", "run", "--port", str(cfg.teleop.quest.port)]
    if cfg.teleop.quest.simulation:
        cmd.append("--simulation=gui")

    # phosphobot reads robot config from its dashboard; export env hints for SO-101.
    env = {
        **dict(__import__("os").environ),
        "SARM101_PORT": follower_port,
        "SARM101_ROBOT_ID": cfg.robot.id,
    }

    dashboard_url = f"http://localhost:{cfg.teleop.quest.port}"
    if open_dashboard:
        webbrowser.open(dashboard_url)

    print(f"Dashboard: {dashboard_url}")
    print("Configure the SO-101 robot in the dashboard with your USB port.\n")
    subprocess.run(cmd, env=env, check=True)


def teleop_quest_instructions() -> None:
    """Print Quest 2 + phospho teleoperation instructions."""
    cfg = ProjectConfig.load()
    print(
        """
Meta Quest 2 Teleoperation (phospho)
====================================

Hardware:
  - S-ARM101 follower arm connected via USB + 12V power supply
  - Meta Quest 2 on the same WiFi as this computer
  - Optional: USB cameras for recording

Software:
  1. uv sync --extra quest
  2. sarm-hand calibrate --role follower --port <USB_PORT>
  3. sarm-hand teleop-quest --follower-port <USB_PORT>
  4. On Quest 2: open phospho teleop app → Connect → teleoperate

Controls (Quest 2):
  - A: start/stop teleoperation
  - Trigger: close gripper
  - B: start/stop recording
  - Y: discard last recording

Recordings are saved in LeRobot v2 format and can be uploaded to Hugging Face.
"""
    )
    print(f"Default phosphobot port: {cfg.teleop.quest.port}")
