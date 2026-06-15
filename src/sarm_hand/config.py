"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

# S-ARM101 is the SO-ARM101 6-axis arm (Feetech STS3215, USB serial).
ROBOT_TYPE = "so101_follower"
LEADER_TYPE = "so101_leader"
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
    use_degrees: bool = False
    max_relative_target: float = 10.0
    disable_torque_on_disconnect: bool = True


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
    type: str = "opencv"
    index_or_path: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 30


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
    lelab: LeLabSettings = field(default_factory=LeLabSettings)
    sim: SimSettings = field(default_factory=SimSettings)
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
        lelab = LeLabSettings(**raw.get("lelab", {}))
        sim = SimSettings(**raw.get("sim", {}))
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
            lelab=lelab,
            sim=sim,
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
        return {
            name: {
                "type": cam.type,
                "index_or_path": cam.index_or_path,
                "width": cam.width,
                "height": cam.height,
                "fps": cam.fps,
            }
            for name, cam in self.cameras.items()
        }


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
