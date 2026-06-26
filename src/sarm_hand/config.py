"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
class GenesisViewerSettings:
    """Interactive Genesis 3D viewer (orbit / zoom) — separate from recording cameras."""

    pos: list[float] | None = None
    lookat: list[float] | None = None
    fov: float | None = None


@dataclass
class GenesisCameraSettings:
    width: int = 640
    height: int = 480
    attach_link: str | None = None
    pos: list[float] | None = None
    lookat: list[float] | None = None
    fov: float | None = None


def _default_genesis_viewer() -> GenesisViewerSettings:
    from .genesis.cameras import VIEWER_PRESET

    return GenesisViewerSettings(
        pos=list(VIEWER_PRESET["pos"]),  # type: ignore[arg-type]
        lookat=list(VIEWER_PRESET["lookat"]),  # type: ignore[arg-type]
        fov=float(VIEWER_PRESET["fov"]),  # type: ignore[arg-type]
    )


def _default_genesis_cameras() -> dict[str, GenesisCameraSettings]:
    from .genesis.cameras import default_genesis_cameras

    return {
        name: GenesisCameraSettings(**spec)
        for name, spec in default_genesis_cameras().items()
    }


def _default_genesis_joints() -> dict[str, GenesisJointSettings]:
    """URDF axis sign vs LeRobot positive direction (not the 2D FK geometry block)."""
    out = {
        name: GenesisJointSettings(sign=sign)
        for name, sign in (
            ("shoulder_pan", 1.0),
            ("shoulder_lift", 1.0),
            ("elbow_flex", 1.0),
            ("wrist_flex", 1.0),
            ("wrist_roll", 1.0),
            ("gripper", 1.0),
        )
    }
    out["gripper"] = GenesisJointSettings(sign=1.0, mirror_raw_deadband=0)
    return out


@dataclass
class GenesisJointSettings:
    sign: float = 1.0
    urdf_offset: float = 0.0
    # Legacy so101_old_calib limits — used only to anchor home pose for wide cal (0..4095).
    urdf_min: float | None = None
    urdf_max: float | None = None
    # Radians added after legacy home mapping (old → new_calib URDF frame).
    frame_offset: float = 0.0
    # Ignore encoder jitter below this many raw counts when mirroring leader → sim.
    mirror_raw_deadband: int | None = None


def _default_rest_pose_deg() -> dict[str, float]:
    """URDF degrees at ``home_raw`` for delta mapping (legacy uses cal → URDF instead)."""
    return {
        "shoulder_pan": 0.0,
        "shoulder_lift": -45.0,
        "elbow_flex": 72.0,
        "wrist_flex": 75.0,
        "wrist_roll": 0.0,
        "gripper": 42.0,
    }


@dataclass
class GenesisSettings:
    urdf: str = "assets/robots/so101/so101_old_calib.urdf"
    backend: str = "auto"  # auto | metal | cuda | cpu | amdgpu
    # legacy: cal min/max → old_calib URDF (folded physical rest at home_raw)
    # delta: rest_pose + encoder pulse delta from home_raw
    # wide_cal: linear norm gain for 0..4095 cals on new_calib URDF
    mapping: str = "legacy"
    dt: float = 0.01
    # Physics substeps per dt — higher = more stable contacts (less fly-off).
    substeps: int = 8
    # World gravity (m/s^2).
    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    scene: str = "pick_place_desk"
    scene_file: str | None = None  # optional path override for config/scenes/*.yaml
    headless: bool = False
    # Robot base orientation in the world (degrees, XYZ euler). Default yaws the
    # arm to face the desk/bench (+X), where the scene objects and cameras sit.
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 90.0])
    # Servo Present_Position at rest (from `sarm-hand test-motors`); converted via calibration.
    home_raw: dict[str, int] = field(default_factory=dict)
    # Sim URDF degrees when leader sits at home_raw (delta mapping anchor).
    rest_pose: dict[str, float] = field(default_factory=_default_rest_pose_deg)
    calibration_role: str = "leader"  # leader | follower — which cal file for home_raw
    joints: dict[str, GenesisJointSettings] = field(default_factory=_default_genesis_joints)
    cameras: dict[str, GenesisCameraSettings] = field(default_factory=_default_genesis_cameras)
    viewer: GenesisViewerSettings = field(default_factory=_default_genesis_viewer)
    # Leader → sim mirror (calibrate-genesis, record-sim --leader).
    mirror_kinematic: bool = True       # instant set_dofs_position (1:1 with leader)
    mirror_rate_hz: float = 30.0        # leader ↔ sim loop rate (calibrate-genesis)
    mirror_substeps: int = 1            # scene.step count per mirror (1 = fastest)
    mirror_grasp_substeps: int = 2      # extra physics steps so gripper PD builds squeeze force
    mirror_grasp_carry: bool = True     # latch prop to jaw when closed nearby
    grasp_weld: bool = False            # weld constraint; false = kinematic carry + desk clamp
    grasp_close_deg: float = 48.0       # leader gripper ° to latch (above mapped rest ~46)
    grasp_open_deg: float = 42.0        # leader gripper ° to release (below rest = clearly open)
    grasp_radius_m: float = 0.14        # max finger-to-prop surface distance to latch (m)
    grasp_tight_radius_m: float = 0.08  # sim fingers nearly shut can latch within this gap
    grasp_log: bool = True               # write grasp_log.jsonl during record-sim --leader
    grasp_link: str = "jaw"             # carry anchor if grasp_anchor_links unset
    grasp_anchor_links: list[str] = field(default_factory=lambda: ["gripper", "jaw"])
    gripper_sim_extra_close_deg: float = 34.0  # sim-only tighter close when squeezing
    gripper_sim_extra_from_deg: float = 44.0   # apply extra close once gripper moves off rest
    mirror_raw_deadband: int = 2        # ignore ±N encoder counts of jitter
    mirror_max_norm_step: float | None = None  # PD mode only; None → robot.max_relative_target
    mirror_smoothing: float = 1.0         # PD mode only; 1 = instant, lower = softer


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
    # Follower command rate for teleop-leader / record-leader (independent of dataset.fps).
    control_fps: int = 30
    # Leader→follower EMA blend per control tick (1.0 = instant, lower = smoother).
    action_smoothing: float = 1.0


@dataclass
class CameraSettings:
    """USB camera (opencv/usb), network stream (http/rtsp), or ESP32 UDP JPEG (udp).

    USB examples:
      type: opencv, index_or_path: 0

    HTTP/RTSP examples:
      type: http, url: http://192.168.1.100:8080/video

    ESP32-CAM raw UDP (chunked JPEG, decoded on host):
      type: udp, host: 192.168.0.58, port: 82

    On macOS USB cameras that reject 640x480, set auto_resolution: true and keep
    width/height as the output size — frames are captured natively then downscaled.
    """

    type: str = "opencv"
    index_or_path: int | str = 0
    url: str | None = None
    host: str | None = None
    port: int | None = 82
    auto_resolution: bool = False
    width: int | None = 640
    height: int | None = 480
    fps: int | None = 30
    warmup_s: int | None = None
    rotate_180: bool = True
    flip_horizontal: bool = True
    stale_sec: float = 0.6
    connect_grace_s: float = 10.0
    fps_window: int = 5
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
    # LeRobot dataset writer — keep video on for SmolVLA / training.
    video: bool = True
    streaming_encoding: bool = False
    vcodec: str = "auto"
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    encoder_threads: int | None = 2
    # Live Rerun preview during record-leader (off by default — 3 cameras exceed Rerun memory).
    display_rerun: bool = False


@dataclass
class TaskSettings:
    """Lightweight leader-demo storage for task replay (JSON under data/tasks/)."""

    root: str = "data/tasks"
    fps: int = 30


@dataclass
class PolicySettings:
    """SmolVLA / LeRobot policy settings."""

    path: str = "lerobot/smolvla_base"
    device: str | None = None  # cuda, mps, cpu — auto-detect if null
    episode_time_s: int = 60
    control_fps: int = 10  # inference/record loop rate; 5–10 for HTTP cameras
    # Optional rename robot camera → policy image key. Empty = use cameras: names as-is.
    camera_map: dict[str, str] = field(default_factory=dict)
    empty_cameras: int | None = None  # pad missing SmolVLA slots (auto when fewer than 3 cams)
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
    tasks: TaskSettings = field(default_factory=TaskSettings)
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
            control_fps=int(teleop_raw.get("control_fps", 30)),
            action_smoothing=float(teleop_raw.get("action_smoothing", 1.0)),
        )
        cameras = {
            name: CameraSettings(**cam_cfg)
            for name, cam_cfg in raw.get("cameras", {}).items()
        }
        dataset = DatasetSettings(**raw.get("dataset", {}))
        tasks = TaskSettings(**raw.get("tasks", {}))
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
            k: v for k, v in genesis_raw.items() if k not in ("cameras", "joints", "viewer")
        }
        if genesis_raw.get("viewer"):
            genesis_kwargs["viewer"] = GenesisViewerSettings(**genesis_raw["viewer"])
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
            tasks=tasks,
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

    def resolve_dataset_path(self, repo_id: str | None = None) -> Path:
        """Full on-disk path for a LeRobot dataset (``root/local/name``)."""
        rid = repo_id or self.dataset.repo_id
        return self.resolve_dataset_root() / Path(*rid.split("/"))

    @staticmethod
    def session_repo_id(base_repo_id: str, when: datetime | None = None) -> str:
        """Append record-sim style timestamp to the last repo_id segment."""
        from .dataset_session import recording_stamp

        stamp = recording_stamp(when)
        parts = base_repo_id.split("/")
        parts[-1] = f"{parts[-1]}-{stamp}"
        return "/".join(parts)

    def resolve_session_dataset_path(
        self,
        repo_id: str | None = None,
        when: datetime | None = None,
    ) -> tuple[str, Path]:
        """Unique repo_id and path (same layout as record-sim timestamped sessions)."""
        from .dataset_session import resolve_recording_paths

        base = repo_id or self.dataset.repo_id
        return resolve_recording_paths(
            base_repo=base,
            root=self.resolve_dataset_root(),
            repo_id=None,
            resume=False,
            timestamp=True,
            when=when,
        )

    def resolve_tasks_root(self) -> Path:
        root = Path(self.tasks.root)
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
