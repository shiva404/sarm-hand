"""Genesis scene: SO-101 arm, desk, and pick/place props."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import GenesisSettings, JOINT_NAMES, ProjectConfig
from .assets import resolve_urdf
from .calibration import calibration_summary, require_calibration
from .cameras import GRIPPER_CAMERA_OFFSET, resolve_camera_pose
from .deps import ensure_genesis
from .preview import CameraPreview
from .scene_loader import SceneDefinition, SceneProp, load_scene_definition, spawn_scene_objects
from .shutdown import ensure_shutdown_handlers
from .units import calibrated_urdf_limits, home_pose_radians, observation_to_radians


def resolve_genesis_backend(name: str):
    """Map config string to genesis backend constant."""
    ensure_genesis()
    import genesis as gs

    key = name.lower()
    if key == "auto":
        import platform

        import torch

        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return gs.metal
        if torch.cuda.is_available():
            return gs.gpu
        return gs.cpu
    mapping = {
        "metal": gs.metal,
        "mps": gs.metal,
        "cuda": gs.gpu,
        "gpu": gs.gpu,
        "cpu": gs.cpu,
        "amdgpu": getattr(gs, "amdgpu", gs.gpu),
    }
    if key not in mapping:
        raise ValueError(f"Unknown genesis backend {name!r}. Use: auto, metal, cuda, cpu")
    return mapping[key]


@dataclass
class SO101GenesisScene:
    """Genesis World scene with SO-101 and a simple pick/place desk."""

    cfg: ProjectConfig
    genesis_cfg: GenesisSettings
    scene: Any
    robot: Any
    plane: Any
    scene_def: SceneDefinition
    props: dict[str, SceneProp] = field(default_factory=dict)
    cameras: dict[str, Any] = field(default_factory=dict)
    camera_attach_links: dict[str, str] = field(default_factory=dict)
    preview: CameraPreview | None = None
    dof_indices: list[int] = field(default_factory=list)
    joint_names: tuple[str, ...] = JOINT_NAMES
    calibration_role: str = "leader"
    calibration: dict[str, dict] = field(default_factory=dict)

    @property
    def camera(self) -> Any | None:
        """Primary recording camera (front), for backward compatibility."""
        return self.cameras.get("front") or (next(iter(self.cameras.values())) if self.cameras else None)

    @classmethod
    def create(
        cls,
        cfg: ProjectConfig | None = None,
        *,
        calibration_role: str | None = None,
    ) -> SO101GenesisScene:
        ensure_genesis()
        import genesis as gs

        cfg = cfg or ProjectConfig.load()
        gcfg = cfg.genesis
        role = calibration_role or gcfg.calibration_role
        calibration = require_calibration(role, cfg)
        backend = resolve_genesis_backend(gcfg.backend)
        gs.init(backend=backend)
        import torch

        torch.set_default_device("cpu")

        show_viewer = not gcfg.headless
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.45, -0.55, 0.35),
                camera_lookat=(0.25, 0.0, 0.12),
                camera_fov=50,
            ),
            sim_options=gs.options.SimOptions(dt=gcfg.dt),
            show_viewer=show_viewer,
        )

        plane = scene.add_entity(gs.morphs.Plane())
        urdf = resolve_urdf(gcfg.urdf)
        robot = scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf),
                fixed=True,
            )
        )

        scene_def = load_scene_definition(gcfg)
        props = spawn_scene_objects(scene, gs, scene_def)

        cameras: dict[str, Any] = {}
        camera_attach_links: dict[str, str] = {}

        for name, cam_cfg in gcfg.cameras.items():
            pos, lookat, fov, attach_link = resolve_camera_pose(name, cam_cfg)
            res = (cam_cfg.width, cam_cfg.height)
            cameras[name] = scene.add_camera(
                res=res,
                pos=pos,
                lookat=lookat,
                fov=fov,
                GUI=False,
            )
            if attach_link:
                camera_attach_links[name] = attach_link

        scene.build()

        dof_indices: list[int] = []
        for joint_name in JOINT_NAMES:
            joint = robot.get_joint(joint_name)
            dof_indices.append(joint.dofs_idx_local[0])

        robot.set_dofs_kp(
            kp=np.array([2000.0] * len(dof_indices)),
            dofs_idx_local=dof_indices,
        )
        robot.set_dofs_kv(
            kv=np.array([200.0] * len(dof_indices)),
            dofs_idx_local=dof_indices,
        )

        cls._attach_cameras(robot, cameras, camera_attach_links)

        preview = CameraPreview(enabled=show_viewer)
        from ..servo import servo_summary

        print(f"Genesis servos: {servo_summary(cfg)}")
        print(f"Genesis calibration ({role}): {calibration_summary(calibration)}")
        instance = cls(
            cfg=cfg,
            genesis_cfg=gcfg,
            scene=scene,
            robot=robot,
            plane=plane,
            scene_def=scene_def,
            props=props,
            cameras=cameras,
            camera_attach_links=camera_attach_links,
            preview=preview,
            dof_indices=dof_indices,
            joint_names=JOINT_NAMES,
            calibration_role=role,
            calibration=calibration,
        )
        instance.apply_home_pose()
        instance.refresh_previews()
        ensure_shutdown_handlers()
        return instance

    @staticmethod
    def _attach_cameras(
        robot: Any,
        cameras: dict[str, Any],
        attach_links: dict[str, str],
    ) -> None:
        for name, link_name in attach_links.items():
            link = robot.get_link(link_name)
            cam = cameras.get(name)
            if cam is None:
                continue
            cam.attach(link, GRIPPER_CAMERA_OFFSET)
            cam.move_to_attach()

    def set_joint_positions_norm(self, obs: dict[str, float]) -> None:
        """Mirror LeRobot observation joint positions into Genesis."""
        radians = observation_to_radians(obs, self.cfg, calibration=self.calibration)
        self.robot.set_dofs_position(
            np.array(radians, dtype=np.float64),
            self.dof_indices,
            zero_velocity=True,
        )

    def set_joint_positions_rad(self, radians: list[float] | np.ndarray) -> None:
        self.robot.set_dofs_position(
            np.asarray(radians, dtype=np.float64),
            self.dof_indices,
            zero_velocity=True,
        )

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.scene.step()
        self.refresh_previews()

    def refresh_previews(self) -> None:
        if self.preview is not None and self.preview.enabled:
            from .shutdown import check_shutdown

            check_shutdown()
            self.preview.show(self.render_all_rgb())

    def close_previews(self) -> None:
        if self.preview is not None:
            self.preview.close()

    def close(self) -> None:
        """Close OpenCV previews and tear down the Genesis viewer/scene."""
        ensure_shutdown_handlers()
        self.close_previews()
        try:
            viewer = getattr(self.scene, "viewer", None)
            if viewer is not None:
                stop = getattr(viewer, "stop", None)
                if callable(stop):
                    stop()
        except Exception:
            pass
        try:
            destroy = getattr(self.scene, "destroy", None)
            if callable(destroy):
                destroy()
        except Exception:
            pass
        try:
            import genesis as gs

            if getattr(gs, "_initialized", False):
                print("Closing Genesis...", flush=True)
                gs.destroy()
        except Exception:
            pass

    def _camera_target_hw(self, camera_name: str) -> tuple[int, int]:
        cam_cfg = self.genesis_cfg.cameras.get(camera_name)
        if cam_cfg is None and self.genesis_cfg.cameras:
            cam_cfg = next(iter(self.genesis_cfg.cameras.values()))
        if cam_cfg is None:
            return 480, 640
        return cam_cfg.height, cam_cfg.width

    def render_rgb(self, camera_name: str = "front") -> np.ndarray | None:
        """Return HxWx3 uint8 RGB from a named recording camera."""
        camera = self.cameras.get(camera_name)
        if camera is None:
            return None
        rendered = camera.render(rgb=True)
        rgb = rendered[0] if isinstance(rendered, tuple) else rendered
        frame = np.asarray(rgb, dtype=np.uint8)
        if frame.dtype != np.uint8 and frame.max() <= 1.0:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        target_h, target_w = self._camera_target_hw(camera_name)
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            import cv2

            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return frame

    def render_all_rgb(self) -> dict[str, np.ndarray]:
        """Render all recording cameras."""
        frames: dict[str, np.ndarray] = {}
        for name in self.cameras:
            frame = self.render_rgb(name)
            if frame is not None:
                frames[name] = frame
        return frames

    def reset_props(self) -> None:
        """Reset movable scene objects to their configured spawn poses."""
        for prop in self.props.values():
            if prop.spec.fixed:
                continue
            prop.entity.set_pos(prop.spec.initial_pos)

    def home_pose_norm(self) -> dict[str, float]:
        pose = self.cfg.poses.get("home", {})
        return {f"{k}.pos": float(v) for k, v in pose.items() if k in JOINT_NAMES}

    def apply_home_pose(self, *, role: str | None = None) -> None:
        cal_role = role or self.calibration_role
        cal = (
            self.calibration
            if cal_role == self.calibration_role
            else require_calibration(cal_role, self.cfg)
        )
        radians = home_pose_radians(self.cfg, calibration=cal, role=cal_role)
        self.set_joint_positions_rad(radians)
        self.step(10)

    def calibrated_radian_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-DOF radian limits matching calibrated servo min..max."""
        limits = calibrated_urdf_limits(self.cfg, self.calibration)
        lo = [limits[name][0] for name in JOINT_NAMES]
        hi = [limits[name][1] for name in JOINT_NAMES]
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    def sync_norm_pose(self, obs: dict[str, float]) -> None:
        """Teleport the sim arm to a LeRobot normalized observation dict."""
        self.set_joint_positions_norm(obs)
        self.step(5)
