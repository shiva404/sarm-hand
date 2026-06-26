"""S-ARM101 USB arm control helpers."""

from __future__ import annotations

import glob
import subprocess
import sys
from contextlib import contextmanager
from typing import Any

from .config import JOINT_NAMES, ProjectConfig


def find_usb_ports() -> list[str]:
    """List likely USB serial ports for the S-ARM101 control board."""
    patterns = [
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "/dev/tty.usbmodem*",
        "/dev/tty.usbserial*",
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
    ]
    ports: list[str] = []
    for pattern in patterns:
        ports.extend(sorted(glob.glob(pattern)))
    return ports


def find_port() -> None:
    """Run LeRobot port discovery and print local USB candidates."""
    print("S-ARM101 uses USB serial (Feetech ST-3215-C001 bus, 1:345 on all joints).")
    print("Connect power + USB, then unplug/replug when prompted.\n")

    candidates = find_usb_ports()
    if candidates:
        print("Detected serial ports:")
        for port in candidates:
            print(f"  - {port}")
    else:
        print("No serial ports detected yet.")

    print("\nRunning lerobot-find-port for interactive detection...\n")
    subprocess.run(["lerobot-find-port"], check=False)


def _make_setup_device(role: str, port: str):
    """Create a LeRobot device instance for motor setup (follower or leader)."""
    if role == "follower":
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        cfg = SOFollowerRobotConfig(id="setup", port=port)
        return SO101Follower(cfg)

    from lerobot.teleoperators.so_leader import SO101Leader
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    cfg = SO101LeaderConfig(id="setup", port=port)
    return SO101Leader(cfg)


def _print_motor_table(
    *,
    target_ids: dict[str, int],
    initial_ids: dict[str, int],
    detected: dict[int, int] | None,
) -> None:
    """Print a row per joint showing target ID, configured initial ID, and bus scan."""
    detected = detected or {}
    headers = ("Joint", "Target ID", "Initial ID", "On bus")
    rows: list[tuple[str, ...]] = []
    for joint in JOINT_NAMES:
        target = str(target_ids[joint])
        initial = str(initial_ids[joint]) if joint in initial_ids else "-"
        on_bus = "yes" if target_ids[joint] in detected else "no"
        rows.append((joint, target, initial, on_bus))

    widths = [max(len(row[i]) for row in ([headers] + rows)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))

    if detected:
        print(f"\nBus scan: {len(detected)} servo(s) at IDs {sorted(detected)}")
    else:
        print("\nBus scan: no servos detected")


def _expected_motor_ids(role: str) -> dict[str, int]:
    cfg = ProjectConfig.load()
    motor_map = cfg.motor_map(role)
    return {joint: motor_map.ids.get(joint, idx + 1) for idx, joint in enumerate(JOINT_NAMES)}


@contextmanager
def _motor_write_retries(min_retries: int = 5):
    """Raise default retry count for Feetech bus writes during connect/configure."""
    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import MotorsBus

    original_write = MotorsBus.write
    original_enable = FeetechMotorsBus.enable_torque
    original_disable = FeetechMotorsBus.disable_torque

    def write(self, data_name, motor, value, *, normalize=True, num_retry=0):
        return original_write(
            self,
            data_name,
            motor,
            value,
            normalize=normalize,
            num_retry=max(num_retry, min_retries),
        )

    def enable_torque(self, motors=None, num_retry=0):
        return original_enable(self, motors, num_retry=max(num_retry, min_retries))

    def disable_torque(self, motors=None, num_retry=0):
        return original_disable(self, motors, num_retry=max(num_retry, min_retries))

    MotorsBus.write = write  # type: ignore[method-assign]
    FeetechMotorsBus.enable_torque = enable_torque  # type: ignore[method-assign]
    FeetechMotorsBus.disable_torque = disable_torque  # type: ignore[method-assign]
    try:
        yield
    finally:
        MotorsBus.write = original_write  # type: ignore[method-assign]
        FeetechMotorsBus.enable_torque = original_enable  # type: ignore[method-assign]
        FeetechMotorsBus.disable_torque = original_disable  # type: ignore[method-assign]


def _scan_bus(role: str, port: str) -> dict[int, int]:
    """Return detected servo id -> model number on the bus (no handshake)."""
    device = _make_setup_device(role, port)
    bus = device.bus
    if not bus.is_connected:
        bus._connect(handshake=False)
    detected = bus.broadcast_ping() or {}
    bus.port_handler.closePort()
    return detected


def _print_motor_failure_help(
    role: str,
    failures: list[str],
    detected: dict[int, int] | None = None,
) -> None:
    """Print actionable fixes for missing or failed servos."""
    detected = detected or {}
    middle_joints = ("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    end_joints = ("shoulder_pan", "gripper")
    middle_ok = all(j not in failures for j in middle_joints)
    end_failed = [j for j in end_joints if j in failures]
    middle_ids = {2, 3, 4, 5}
    middle_on_bus = middle_ids.issubset(detected)

    print(f"\nFailed joints: {', '.join(failures)}")

    if middle_ok and end_failed and (middle_on_bus or not detected):
        print(
            "\nLikely cause: end-of-chain servos (shoulder_pan and/or gripper) still need "
            "their IDs programmed. IDs 2–5 are fine; 1 and 6 were skipped because they "
            "cannot be auto-detected when the full chain is connected.\n"
            "\nProgram each end motor alone — only that motor plugged into the USB controller "
            "(12V power on, daisy-chain cable to the next motor disconnected):\n"
            f"\n  # shoulder_pan → ID 1 (connect controller to shoulder_pan only)"
            f"\n  sarm-hand setup-motors --role {role} --only shoulder_pan\n"
            f"\n  # Reconnect chain through wrist_roll, then scan to confirm IDs 1–5"
            f"\n  sarm-hand setup-motors --role {role} --scan\n"
            f"\n  # gripper → ID 6 (connect controller to gripper only)"
            f"\n  sarm-hand setup-motors --role {role} --only gripper\n"
            f"\n  # Full chain connected — verify all six"
            f"\n  sarm-hand test-motors --role {role}\n"
            "\nIf a solo motor is still at factory ID 1, add e.g. --initial-ids shoulder_pan=1"
            f"\nFull factory-fresh arm: sarm-hand setup-motors --role {role} --one-at-a-time\n"
        )
        return

    found_ids = sorted(detected)
    print(
        "\nCommon fixes:"
        "\n  - Check 12V power supply is connected (USB alone is not enough)"
        "\n  - Reseat the 3-pin cable between the failed servo and its neighbor"
        "\n  - Try a different USB cable/port (avoid hubs if possible)"
        "\n  - On macOS, prefer /dev/tty.usbmodem* over /dev/cu.usbmodem*"
        f"\n  - Scan bus: sarm-hand setup-motors --role {role} --scan"
        f"\n  - Program IDs: sarm-hand setup-motors --role {role} --one-at-a-time"
    )
    if found_ids:
        print(f"\nServo IDs currently on bus: {found_ids}")


def require_all_motors(role: str, port: str, *, context: str) -> None:
    """Exit with actionable help if any configured servo IDs are missing on the bus."""
    target_ids = _expected_motor_ids(role)
    detected = _scan_bus(role, port)

    missing_joints = [joint for joint in JOINT_NAMES if target_ids[joint] not in detected]
    if not missing_joints:
        return

    print(f"\nCannot {context}: {len(missing_joints)} servo(s) missing on {port}\n")
    _print_motor_table(
        target_ids=target_ids,
        initial_ids={},
        detected=detected,
    )

    found_ids = sorted(detected)
    missing_ids = [target_ids[joint] for joint in missing_joints]
    print(f"\nMissing joints: {', '.join(missing_joints)} (IDs {missing_ids})")
    print(f"Found on bus:   IDs {found_ids or '(none)'}")
    _print_motor_failure_help(role, missing_joints, detected)
    sys.exit(1)


def _resolve_initial_id(
    joint: str,
    target_id: int,
    configured_initial: dict[str, int],
    detected: dict[int, int] | None,
) -> int | None:
    """Pick the current servo ID to use when programming one joint row."""
    if joint in configured_initial:
        return configured_initial[joint]

    if detected and target_id in detected:
        return target_id

    if detected and len(detected) == 1:
        return next(iter(detected))

    return None


def setup_motors(
    role: str,
    port: str,
    *,
    initial_ids: dict[str, int] | None = None,
    one_at_a_time: bool = False,
    scan_only: bool = False,
    only_joints: list[str] | None = None,
) -> None:
    """Assign servo IDs per joint row using configured or detected IDs.

    By default all motors can stay connected. Set target IDs in config/default.yaml
    under `motors.<role>` and optional `initial_id` per joint when reprogramming.
    """
    if role not in ("follower", "leader"):
        raise ValueError("role must be 'follower' or 'leader'")

    if only_joints:
        unknown = [j for j in only_joints if j not in JOINT_NAMES]
        if unknown:
            raise ValueError(f"unknown joint(s): {', '.join(unknown)}")

    cfg = ProjectConfig.load()
    motor_map = cfg.motor_map(role)
    target_ids = _expected_motor_ids(role)
    configured_initial = dict(motor_map.initial_ids)
    if initial_ids:
        configured_initial.update(initial_ids)

    print(f"Motor setup for {role} arm on {port}")
    print("Per-joint servo ID map:\n")

    device = _make_setup_device(role, port)
    bus = device.bus

    if not bus.is_connected:
        bus._connect(handshake=False)

    detected = bus.broadcast_ping()
    _print_motor_table(
        target_ids=target_ids,
        initial_ids=configured_initial,
        detected=detected,
    )

    if scan_only:
        bus.port_handler.closePort()
        return

    if one_at_a_time:
        print("\nOne-at-a-time mode: connect each motor individually when prompted.\n")
        bus.port_handler.closePort()
        device.setup_motors()
        return

    print("\nProgramming each joint row from configured/detected servo IDs...\n")
    missing: list[str] = []
    joints_to_program = (
        [j for j in reversed(JOINT_NAMES) if j in only_joints]
        if only_joints
        else list(reversed(JOINT_NAMES))
    )

    if only_joints:
        print(f"  Only programming: {', '.join(reversed(joints_to_program))}\n")

    for joint in joints_to_program:
        target_id = target_ids[joint]
        bus.motors[joint].id = target_id
        current_id = _resolve_initial_id(joint, target_id, configured_initial, detected)

        if current_id is None:
            missing.append(joint)
            print(f"  skip {joint}: no initial servo ID (set in config or --initial-ids)")
            continue

        print(f"  {joint}: servo {current_id} -> {target_id}")
        bus.setup_motor(joint, initial_id=current_id)
        detected = bus.broadcast_ping() or {}

    bus.port_handler.closePort()

    if missing:
        detected_after = _scan_bus(role, port)
        _print_motor_failure_help(role, missing, detected_after)
        print(
            "\nCould not program some joints. For end motors (shoulder_pan / gripper), "
            "connect only that motor to the controller and use --only, e.g.\n"
            f"  sarm-hand setup-motors --role {role} --only shoulder_pan\n"
            "Or set initial_id in config/default.yaml / --initial-ids, or use --one-at-a-time."
        )
        sys.exit(1)

    print("\nMotor setup complete.")


def test_motors(role: str, port: str, *, retries: int = 3) -> None:
    """Ping, read, and torque-test each servo individually to isolate bus issues."""
    cfg = ProjectConfig.load()
    motor_map = cfg.motor_map(role)
    device = _make_setup_device(role, port)
    bus = device.bus

    if not bus.is_connected:
        bus._connect(handshake=False)

    print(f"Testing {role} servos on {port} ({retries} retries per write)\n")
    headers = ("Joint", "ID", "Ping", "Raw pos", "Torque off")
    rows: list[tuple[str, ...]] = []
    failures: list[str] = []

    for joint in JOINT_NAMES:
        servo_id = motor_map.ids[joint]
        ping_ok = read_ok = torque_ok = "fail"
        raw_pos = "-"

        try:
            if bus.ping(joint, num_retry=retries, raise_on_error=False) is not None:
                ping_ok = "ok"
        except Exception:
            pass

        try:
            # Raw read — normalize=False because calibration may not exist yet
            raw_pos = str(bus.read("Present_Position", joint, normalize=False, num_retry=retries))
            read_ok = "ok"
        except Exception:
            pass

        try:
            bus.disable_torque(joint, num_retry=retries)
            bus.enable_torque(joint, num_retry=retries)
            torque_ok = "ok"
        except Exception:
            pass

        rows.append((joint, str(servo_id), ping_ok, raw_pos if read_ok == "ok" else "fail", torque_ok))
        if ping_ok != "ok" or read_ok != "ok" or torque_ok != "ok":
            failures.append(joint)

    # Leave servos in a safe state — especially the leader, which must be backdrivable by hand.
    try:
        bus.disable_torque(num_retry=retries)
    except Exception:
        pass

    widths = [max(len(row[i]) for row in ([headers] + rows)) for i in range(len(headers))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))

    detected = bus.broadcast_ping() or {}
    bus.port_handler.closePort()

    if failures:
        _print_motor_failure_help(role, failures, detected)
        sys.exit(1)

    print("\nAll servos passed ping, read, and torque tests.")
    if role == "leader":
        print("Torque disabled — leader arm should move freely by hand.")


def disable_arm_torque(role: str, port: str) -> None:
    """Disable servo torque so the arm can be moved by hand (required for leader teleop)."""
    if role not in ("follower", "leader"):
        raise ValueError("role must be 'follower' or 'leader'")

    device = _make_setup_device(role, port)
    bus = device.bus
    if not bus.is_connected:
        bus._connect(handshake=False)
    bus.disable_torque(num_retry=3)
    bus.port_handler.closePort()
    print(f"Torque disabled on {role} arm ({port}).")
    if role == "leader":
        print("The leader arm should now move freely by hand for teleoperation.")


def _print_calibration_failure_help(role: str, exc: BaseException) -> None:
    if not isinstance(exc, ConnectionError):
        return
    print(
        "\nCalibration lost contact with the servo bus during range recording (step 2).\n"
        "This is usually a transient USB/serial glitch, not a bad calibration pose.\n"
        f"\nTry again:\n  sarm-hand calibrate --role {role}"
        "\n\nTips for step 2:"
        "\n  - Keep 12V power connected; avoid USB hubs if possible"
        "\n  - Move one joint at a time, slowly — don't tug multiple joints at once"
        "\n  - Don't press ENTER until MIN and MAX differ for every joint (except wrist_roll)"
        f"\n  - If it keeps failing, run `sarm-hand test-motors --role {role}` and reseat cables",
        file=sys.stderr,
    )


@contextmanager
def _calibration_sync_read_retries(min_retries: int = 5):
    """Raise default sync_read retries during LeRobot calibration (avoids single-packet drops)."""
    from lerobot.motors.motors_bus import MotorsBus

    original = MotorsBus.sync_read

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        return original(
            self,
            data_name,
            motors,
            normalize=normalize,
            num_retry=max(num_retry, min_retries),
        )

    MotorsBus.sync_read = sync_read  # type: ignore[method-assign]
    try:
        yield
    finally:
        MotorsBus.sync_read = original


def _calibration_dict_to_motor_cal(
    cal: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from lerobot.motors import MotorCalibration

    return {
        joint: MotorCalibration(
            id=int(entry["id"]),
            drive_mode=int(entry.get("drive_mode", 0)),
            homing_offset=int(entry["homing_offset"]),
            range_min=int(entry["range_min"]),
            range_max=int(entry["range_max"]),
        )
        for joint, entry in cal.items()
    }


def ensure_bus_calibration(
    device: Any,
    role: str,
    *,
    cfg: ProjectConfig | None = None,
) -> None:
    """Load saved calibration onto servos without interactive prompts."""
    if device.is_calibrated:
        return

    if not device.calibration and device.calibration_fpath.is_file():
        device._load_calibration()

    if not device.calibration:
        from .genesis.calibration import calibration_path, load_calibration

        cfg = cfg or ProjectConfig.load()
        cal_dict = load_calibration(role, cfg)
        if cal_dict:
            device.calibration = _calibration_dict_to_motor_cal(cal_dict)

    if not device.calibration:
        from .genesis.calibration import calibration_path

        cfg = cfg or ProjectConfig.load()
        print(
            f"{role.capitalize()} has no calibration file at {calibration_path(role, cfg)}.\n"
            f"Run:  uv run sarm-hand calibrate --role {role}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not device.is_calibrated:
        device.bus.write_calibration(device.calibration)


def _make_calibrate_device(role: str, port: str, resolved_id: str, cfg: ProjectConfig) -> Any:
    if role == "follower":
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        device_cfg = SOFollowerRobotConfig(
            id=resolved_id,
            port=port,
            use_degrees=cfg.robot.use_degrees,
        )
        return SO101Follower(device_cfg)

    from lerobot.teleoperators.so_leader import SO101Leader
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    device_cfg = SO101LeaderConfig(
        id=resolved_id,
        port=port,
        use_degrees=cfg.robot.use_degrees,
    )
    return SO101Leader(device_cfg)


def calibrate(role: str, port: str, robot_id: str | None = None) -> None:
    """Calibrate follower or leader arm and save calibration file."""
    require_all_motors(role, port, context="calibrate")

    cfg = ProjectConfig.load()
    from .data import configure_local_lerobot_env

    configure_local_lerobot_env(cfg)
    resolved_id = robot_id or (
        cfg.robot.id if role == "follower" else cfg.teleop.leader.id
    )
    device_type = cfg.robot.type if role == "follower" else cfg.teleop.leader.type

    print(f"Calibrating {role} arm ({device_type}) on {port}\n")
    print(
        "Calibration steps (follow prompts from LeRobot):\n"
        "  1. Move ALL joints (including wrist_roll) to your desired CENTER pose, then ENTER\n"
        "     - wrist_roll: pick a neutral orientation (e.g. aligned with forearm)\n"
        "     - LeRobot records that pose as encoder center (~2047) for every joint\n"
        "  2. Move these joints through their FULL mechanical range (one at a time):\n"
        "       shoulder_pan → shoulder_lift → elbow_flex → wrist_flex → gripper\n"
        "     Watch MIN/MAX change in the live table — do NOT press ENTER until all differ\n"
        "     - Move slowly; a dropped USB packet can abort this step\n"
        "  3. wrist_roll is NOT moved in step 2 — it can spin 360° continuously\n"
        "     - min/max are set automatically to 0–4095 (full encoder range)\n"
        "     - its center comes only from step 1 (homing offset)\n"
        "  4. After calibration, verify wrist_roll raw position is ~2047 at your neutral pose:\n"
        f"       uv run sarm-hand test-motors --role {role}\n"
        "\n"
        "If all values stay at 2047 in step 2, the arm was not moved before pressing ENTER.\n"
        "Torque is disabled during calibration — move joints by hand.\n"
    )

    device = _make_calibrate_device(role, port, resolved_id, cfg)
    try:
        with _calibration_sync_read_retries():
            device.connect(calibrate=False)
            try:
                device.calibrate()
            finally:
                device.disconnect()
    except ConnectionError as exc:
        _print_calibration_failure_help(role, exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\nCalibration cancelled.", file=sys.stderr)
        raise SystemExit(130) from None


def build_robot(port: str, config: ProjectConfig | None = None, *, use_cameras: bool = True):
    """Create a connected robot backend for programmatic use."""
    from .backends import build_robot_backend

    cfg = config or ProjectConfig.load()
    robot = build_robot_backend(port, config=cfg, connect=True, use_cameras=use_cameras)
    return robot.inner if hasattr(robot, "inner") else robot


def resolve_role_port(role: str, port: str | None) -> str:
    """Resolve USB port for follower or leader from CLI arg or config/default.yaml."""
    if role not in ("follower", "leader"):
        raise ValueError("role must be 'follower' or 'leader'")
    cfg = ProjectConfig.load()
    if role == "follower":
        return ensure_port(port or cfg.robot.port, "Follower")
    return ensure_port(port or cfg.teleop.leader.port, "Leader")


def ensure_port(port: str | None, label: str) -> str:
    """Validate that a USB port was provided."""
    if port:
        return port
    ports = find_usb_ports()
    config_key = "robot.port" if label == "Follower" else "teleop.leader.port"
    msg = f"{label} USB port is required."
    if ports:
        msg += f" Detected: {', '.join(ports)}"
    msg += f" Run `sarm-hand find-port` or set {config_key} in config/default.yaml."
    print(msg, file=sys.stderr)
    sys.exit(1)
