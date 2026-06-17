"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

# S-ARM101 is the SO-ARM101 6-axis arm (Feetech ST-3215-C001, USB serial).
ROBOT_TYPE = "so101_follower"
LEADER_TYPE = "so101_leader"
DEFAULT_SERVO_MODEL = "ST-3215-C001"
DEFAULT_LEROBOT_SERVO_TYPE = "sts3215"
DEFAULT_SERVO_GEAR_RATIO = 345
DEFAULT_SERVO_RESOLUTION = 4096
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
DEFAULT_MOTOR_IDS: dict[str, int] = dict(zip(JOINT_NAMES, range(1, 7), strict=True))

DEFAULT_POSES: dict[str, dict[str, float]] = {
    "home": {
        "shoulder_pan": 0,
        "shoulder_lift": 0,
        "elbow_flex": 0,
        "wrist_flex": 0,
        "wrist_roll": 0,
        "gripper": 50,
    },
    "ready": {
        "shoulder_pan": 0,
        "shoulder_lift": -40,
        "elbow_flex": 40,
        "wrist_flex": -40,
        "wrist_roll": 0,
        "gripper": 50,
    },
    "park": {
        "shoulder_pan": 0,
        "shoulder_lift": 65,
        "elbow_flex": -55,
        "wrist_flex": 25,
        "wrist_roll": 0,
        "gripper": 15,
    },
}
DEFAULT_POSE_SEQUENCE = ("home", "ready", "park", "home")


@dataclass
class ServoSettings:
    """Shared Feetech servo hardware — identical on all six SO-ARM101 joints."""

    model: str = DEFAULT_SERVO_MODEL
    lerobot_type: str = DEFAULT_LEROBOT_SERVO_TYPE
    gear_ratio: int = DEFAULT_SERVO_GEAR_RATIO  # 1:345 gearbox; URDF/MJCF angle = output shaft
    resolution: int = DEFAULT_SERVO_RESOLUTION  # 12-bit counts per output revolution
    voltage_nominal_v: float = 7.4
    stall_torque_kg_cm: float = 19.5
    urdf_mechanical_reduction: float = 1.0  # joint axis is gearbox output, not rotor
    mujoco_class: str = "sts3215"
    mujoco_forcerange_nm: float = 3.35


@dataclass
class MotorMapSettings:
    """Per-joint servo ID mapping for motor setup."""

    ids: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MOTOR_IDS))
    initial_ids: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, raw: dict[str, Any] | None) -> MotorMapSettings:
        if not raw:
            return cls()

        ids = dict(DEFAULT_MOTOR_IDS)
        initial_ids: dict[str, int] = {}

        for joint, value in raw.items():
            if joint not in JOINT_NAMES:
                continue
            if isinstance(value, int):
                ids[joint] = value
            elif isinstance(value, dict):
                if "id" in value:
                    ids[joint] = int(value["id"])
                if "initial_id" in value:
                    initial_ids[joint] = int(value["initial_id"])

        flat_initial = raw.get("initial_ids", {})
        if isinstance(flat_initial, dict):
            for joint, servo_id in flat_initial.items():
                if joint in JOINT_NAMES:
                    initial_ids[joint] = int(servo_id)

        return cls(ids=ids, initial_ids=initial_ids)


@dataclass
class RobotSettings:
    type: str = ROBOT_TYPE
    id: str = "sarm101_follower"
    port: str | None = None
    backend: str = "hardware"  # hardware | genesis | twin
    use_degrees: bool = False
    max_relative_target: float = 10.0
    disable_torque_on_disconnect: bool = True


@dataclass
class GenesisCameraSettings:
    width: int = 640
    height: int = 480
    attach_link: str | None = None
    pos: list[float] | None = None
    lookat: list[float] | None = None
    fov: float | None = None


def _default_genesis_cameras() -> dict[str, GenesisCameraSettings]:
    from .genesis.cameras import default_genesis_cameras

    return {
        name: GenesisCameraSettings(**spec)
        for name, spec in default_genesis_cameras().items()
    }


def _default_genesis_joints() -> dict[str, "GenesisJointSettings"]:
    """URDF axis sign vs LeRobot positive direction (not the 2D FK geometry block)."""
    return {
        name: GenesisJointSettings(sign=sign)
        for name, sign in (
            ("shoulder_pan", -1.0),
            ("shoulder_lift", -1.0),
            ("elbow_flex", 1.0),
            ("wrist_flex", 1.0),
            ("wrist_roll", -1.0),
            ("gripper", 1.0),
        )
    }


@dataclass
class GenesisJointSettings:
    sign: float = 1.0
    urdf_offset: float = 0.0
    # Legacy so101_old_calib limits — used only to anchor home pose for wide cal (0..4095).
    urdf_min: float | None = None
    urdf_max: float | None = None
    # Radians added after legacy home mapping (old → new_calib URDF frame).
    frame_offset: float = 0.0


@dataclass
class GenesisSettings:
    urdf: str = "assets/robots/so101/so101_new_calib.urdf"
    backend: str = "auto"  # auto | metal | cuda | cpu | amdgpu
    dt: float = 0.01
    scene: str = "pick_place_desk"
    scene_file: str | None = None  # optional path override for config/scenes/*.yaml
    headless: bool = False
    # Servo Present_Position at rest (from `sarm-hand test-motors`); converted via calibration.
    home_raw: dict[str, int] = field(default_factory=dict)
    calibration_role: str = "leader"  # leader | follower — which cal file for home_raw
    joints: dict[str, GenesisJointSettings] = field(default_factory=_default_genesis_joints)
    cameras: dict[str, GenesisCameraSettings] = field(default_factory=_default_genesis_cameras)


@dataclass
class TwinSettings:
    sync_mode: str = "hardware_to_sim"
    rate_hz: float = 30.0
    duration_s: float | None = None  # None = run until Ctrl+C


@dataclass
class LeaderSettings:
    type: str = LEADER_TYPE
    id: str = "sarm101_leader"
    port: str | None = None


@dataclass
class QuestSettings:
    host: str = "0.0.0.0"
    port: int = 8020
    simulation: bool = False


@dataclass
class TeleopSettings:
    leader: LeaderSettings = field(default_factory=LeaderSettings)
    quest: QuestSettings = field(default_factory=QuestSettings)


@dataclass
class CameraSettings:
    """USB camera (opencv/usb) or network stream (http/rtsp).

    USB examples:
      type: opencv, index_or_path: 0
      type: opencv, index_or_path: /dev/video0

    HTTP/RTSP examples:
      type: http, url: http://192.168.1.100:8080/video
      type: rtsp, url: rtsp://192.168.1.100:554/stream

    For streams, omit width/height/fps (null) to use the source's native resolution.
    On macOS, set auto_resolution: true — built-in cameras often fail at 640x480@30.
    """

    type: str = "opencv"
    index_or_path: int | str = 0
    url: str | None = None
    auto_resolution: bool = False
    width: int | None = 640
    height: int | None = 480
    fps: int | None = 30
    warmup_s: int | None = None
    # HTTP/RTSP: max age for read_latest() (ms). Default scales with fps (~15 frame periods).
    max_frame_age_ms: int | None = None
    fourcc: str | None = None


@dataclass
class DatasetSettings:
    repo_id: str = "local/sarm101-dataset"
    root: str = "data/datasets"
    fps: int = 30
    single_task: str = "Pick and place the object"
    num_episodes: int = 50
    episode_time_s: int = 60
    reset_time_s: int = 10
    push_to_hub: bool = False


@dataclass
class PolicySettings:
    """SmolVLA / LeRobot policy settings."""

    path: str = "lerobot/smolvla_base"
    device: str | None = None  # cuda, mps, cpu — auto-detect if null
    episode_time_s: int = 60
    control_fps: int = 10  # inference/record loop rate; 5–10 for HTTP cameras
    # Map robot camera names → SmolVLA names (smolvla_base expects camera1/2/3)
    camera_map: dict[str, str] = field(default_factory=lambda: {"front": "camera1"})
    empty_cameras: int | None = 2  # pad missing camera2/3 when using one physical camera
    # smolvla_base stores stats under so100.buffer.* — remap for SO-101 inference
    stats_buffer: str = "so100.buffer"
    train_dataset: str | None = None
    train_steps: int = 20_000
    train_batch_size: int = 64
    output_dir: str = "outputs/train/sarm101_smolvla"


@dataclass
class LeLabSettings:
    """LeLab web UI settings (https://huggingface.co/docs/lerobot/main/en/lelab)."""

    port: int = 8000
    open_browser: bool = True
    hf_lerobot_home: str | None = None

    def resolve_hf_lerobot_home(self, config: ProjectConfig) -> Path:
        if self.hf_lerobot_home:
            path = Path(self.hf_lerobot_home)
            return path if path.is_absolute() else PROJECT_ROOT / path
        return config.resolve_dataset_root()


@dataclass
class SimSettings:
    """3D joint simulator (sim/arm3d.html) settings."""

    host: str = "127.0.0.1"
    port: int = 8763
    open_browser: bool = True
    brand_title: str = "sarm-hand"
    brand_subtitle: str = "3D joint simulator"
    value_suffix: str = ""
    reach_z_max: int = 350
    reach_z_tolerance: int = 10
    reach_go_tol_mm: float = 15.0
    reach_steps: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_pan": 16,
            "shoulder_lift": 12,
            "elbow_flex": 12,
            "wrist_flex": 8,
        }
    )
    visual: dict[str, float | int] = field(default_factory=dict)


@dataclass
class ProjectConfig:
    robot: RobotSettings = field(default_factory=RobotSettings)
    teleop: TeleopSettings = field(default_factory=TeleopSettings)
    cameras: dict[str, CameraSettings] = field(default_factory=dict)
    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    policy: PolicySettings = field(default_factory=PolicySettings)
    lelab: LeLabSettings = field(default_factory=LeLabSettings)
    sim: SimSettings = field(default_factory=SimSettings)
    genesis: GenesisSettings = field(default_factory=GenesisSettings)
    twin: TwinSettings = field(default_factory=TwinSettings)
    servo: ServoSettings = field(default_factory=ServoSettings)
    motors: dict[str, MotorMapSettings] = field(
        default_factory=lambda: {
            "follower": MotorMapSettings(),
            "leader": MotorMapSettings(),
        }
    )
    _geometry: dict[str, Any] = field(default_factory=dict)
    _joints: dict[str, Any] = field(default_factory=dict)
    _poses: dict[str, dict[str, float]] = field(default_factory=lambda: {
        name: dict(joints) for name, joints in DEFAULT_POSES.items()
    })
    _pose_sequence: tuple[str, ...] = DEFAULT_POSE_SEQUENCE

    def motor_map(self, role: str) -> MotorMapSettings:
        return self.motors.get(role, MotorMapSettings())

    @property
    def poses(self) -> dict[str, dict[str, float]]:
        return self._poses

    @property
    def pose_sequence(self) -> tuple[str, ...]:
        return self._pose_sequence

    def pose_names(self) -> list[str]:
        return list(self._poses.keys())

    def sim_geometry(self) -> dict[str, Any]:
        if not self._geometry:
            raise ValueError("geometry section missing in config — required for 3D sim")
        return self._geometry

    def sim_joint_limits(self) -> dict[str, tuple[float, float]]:
        if not self._joints:
            raise ValueError("joints section missing in config — required for 3D sim")
        limits: dict[str, tuple[float, float]] = {}
        for name in JOINT_NAMES:
            entry = self._joints.get(name)
            if not isinstance(entry, dict):
                raise ValueError(f"joints.{name} must be defined in config")
            limits[name] = (float(entry["min"]), float(entry["max"]))
        return limits

    def sim_joint_meta(self) -> dict[str, dict[str, Any]]:
        meta: dict[str, dict[str, Any]] = {}
        for name in JOINT_NAMES:
            entry = self._joints.get(name, {})
            if isinstance(entry, dict):
                meta[name] = entry
        return meta

    def sim_visual(self) -> dict[str, float | int]:
        return dict(self.sim.visual)

    def sim_reach_steps(self) -> dict[str, int]:
        return dict(self.sim.reach_steps)

    def sim_reach_z_max(self) -> int:
        return self.sim.reach_z_max

    def sim_reach_z_tolerance(self) -> int:
        return self.sim.reach_z_tolerance

    def sim_reach_go_tol_mm(self) -> float:
        return self.sim.reach_go_tol_mm

    def sim_brand_title(self) -> str:
        return self.sim.brand_title

    def sim_brand_subtitle(self) -> str:
        return self.sim.brand_subtitle

    def sim_value_suffix(self) -> str:
        return self.sim.value_suffix

    @classmethod
    def load(cls, path: Path | None = None) -> ProjectConfig:
        config_path = path or DEFAULT_CONFIG_PATH
        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        robot = RobotSettings(**raw.get("robot", {}))
        teleop_raw = raw.get("teleop", {})
        teleop = TeleopSettings(
            leader=LeaderSettings(**teleop_raw.get("leader", {})),
            quest=QuestSettings(**teleop_raw.get("quest", {})),
        )
        cameras = {
            name: CameraSettings(**cam_cfg)
            for name, cam_cfg in raw.get("cameras", {}).items()
        }
        dataset = DatasetSettings(**raw.get("dataset", {}))
        policy = PolicySettings(**raw.get("policy", {}))
        lelab = LeLabSettings(**raw.get("lelab", {}))
        sim = SimSettings(**raw.get("sim", {}))
        genesis_raw = raw.get("genesis", {})
        genesis_cameras = {
            name: GenesisCameraSettings(**cam)
            for name, cam in genesis_raw.get("cameras", {}).items()
        }
        genesis_joints = {
            name: GenesisJointSettings(**spec)
            for name, spec in genesis_raw.get("joints", {}).items()
            if isinstance(spec, dict)
        }
        genesis_kwargs = {
            k: v for k, v in genesis_raw.items() if k not in ("cameras", "joints")
        }
        if genesis_cameras:
            genesis_kwargs["cameras"] = genesis_cameras
        if genesis_joints:
            default_joints = _default_genesis_joints()
            for name, spec in genesis_joints.items():
                default_joints[name] = spec
            genesis_kwargs["joints"] = default_joints
        genesis = GenesisSettings(**genesis_kwargs)
        twin = TwinSettings(**raw.get("twin", {}))
        servo = ServoSettings(**raw.get("servo", {}))
        motors_raw = raw.get("motors", {})
        motors = {
            role: MotorMapSettings.from_yaml(motors_raw.get(role))
            for role in ("follower", "leader")
        }
        poses, pose_sequence = _load_poses(raw.get("poses", {}))
        geometry = raw.get("geometry") or {}
        joints = raw.get("joints") or {}
        return cls(
            robot=robot,
            teleop=teleop,
            cameras=cameras,
            dataset=dataset,
            policy=policy,
            lelab=lelab,
            sim=sim,
            genesis=genesis,
            twin=twin,
            servo=servo,
            motors=motors,
            _geometry=geometry,
            _joints=joints,
            _poses=poses,
            _pose_sequence=pose_sequence,
        )

    def resolve_dataset_root(self) -> Path:
        root = Path(self.dataset.root)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        return root

    def cameras_lerobot_dict(self) -> dict[str, dict[str, Any]]:
        from .cameras import camera_to_lerobot_dict

        return {name: camera_to_lerobot_dict(cam) for name, cam in self.cameras.items()}


def _load_poses(raw: dict[str, Any]) -> tuple[dict[str, dict[str, float]], tuple[str, ...]]:
    poses = {name: dict(joints) for name, joints in DEFAULT_POSES.items()}
    for name, joints in raw.items():
        if name == "sequence" or not isinstance(joints, dict):
            continue
        merged = dict(poses.get(name, DEFAULT_POSES.get(name, {})))
        for joint, value in joints.items():
            if joint in JOINT_NAMES:
                merged[joint] = float(value)
        poses[name] = merged

    sequence_raw = raw.get("sequence", DEFAULT_POSE_SEQUENCE)
    pose_sequence = tuple(sequence_raw) if sequence_raw else DEFAULT_POSE_SEQUENCE
    return poses, pose_sequence


def parse_initial_ids(spec: str | None) -> dict[str, int]:
    """Parse CLI overrides like 'shoulder_pan=1,gripper=6'."""
    if not spec:
        return {}

    parsed: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --initial-ids entry '{part}'. Use joint=id format.")
        joint, servo_id = part.split("=", 1)
        joint = joint.strip()
        if joint not in JOINT_NAMES:
            raise ValueError(f"Unknown joint '{joint}'. Valid: {', '.join(JOINT_NAMES)}")
        parsed[joint] = int(servo_id.strip())
    return parsed
