"""Genesis scene: SO-101 arm, desk, and pick/place props."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import GenesisSettings, JOINT_NAMES, ProjectConfig
from .assets import resolve_urdf
from .calibration import calibration_summary, require_calibration
from .cameras import GRIPPER_CAMERA_OFFSET, resolve_camera_pose, resolve_viewer_pose
from .deps import ensure_genesis
from .preview import CameraPreview
from .scene_loader import SceneDefinition, SceneProp, load_scene_definition, spawn_quat_wxyz, spawn_scene_objects
from .shutdown import ensure_shutdown_handlers
from .grasp import (
    GraspLatch,
    anchor_distance_to_prop,
    can_acquire_latch,
    carry_kinematic,
    latch_kinematic,
    latch_weld,
    link_world_pose,
    nearest_prop_to_links,
    release_latch,
    should_latch,
)
from .grasp_diag import GraspDiagnostic, leader_gripper_deg
from .home_pose import format_home_pose_summary
from .support import enforce_desk_support, resting_pen_center_z
from .mirror import filter_raw_counts, smooth_mirror_radians
from .tensors import to_numpy
from .units import (
    calibrated_urdf_limits,
    home_pose_radians,
    observation_to_radians,
    raw_to_radians,
)
from .urdf_limits import urdf_joint_limits


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
    _mirror_last_raw: dict[str, int] | None = field(default=None, repr=False)
    _mirror_cmd_rad: np.ndarray | None = field(default=None, repr=False)
    _gripper_limit_hi: float = field(default=1.74533, repr=False)
    _grasp_latch: GraspLatch | None = field(default=None, repr=False)
    _last_grasp_diag: GraspDiagnostic | None = field(default=None, repr=False)

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
        apply_home: bool = True,
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
        viewer_pos, viewer_lookat, viewer_fov = resolve_viewer_pose(gcfg.viewer)
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=viewer_pos,
                camera_lookat=viewer_lookat,
                camera_fov=viewer_fov,
            ),
            sim_options=gs.options.SimOptions(
                dt=gcfg.dt,
                substeps=gcfg.substeps,
                gravity=tuple(gcfg.gravity),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                use_gjk_collision=True,
                enable_multi_contact=True,
                contact_pruning_tolerance=0.001,
                noslip_iterations=5,
            ),
            show_viewer=show_viewer,
        )

        plane = scene.add_entity(gs.morphs.Plane())
        urdf = resolve_urdf(gcfg.urdf)
        robot = scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf),
                fixed=True,
                euler=tuple(gcfg.base_euler),
            ),
            material=gs.materials.Rigid(
                gravity_compensation=1.0,
                friction=2.5,
                coup_friction=0.5,
            ),
        )

        scene_def = load_scene_definition(gcfg)
        props = spawn_scene_objects(scene, gs, scene_def)

        cameras: dict[str, Any] = {}
        camera_attach_links: dict[str, str] = {}

        for name, cam_cfg in gcfg.cameras.items():
            pos, lookat, fov, attach_link = resolve_camera_pose(name, cam_cfg, urdf=gcfg.urdf)
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

        kp = np.array([5000.0] * len(dof_indices), dtype=np.float64)
        kv = np.array([400.0] * len(dof_indices), dtype=np.float64)
        if len(dof_indices) >= 6:
            kp[5] = 15000.0
            kv[5] = 900.0
        robot.set_dofs_kp(kp, dofs_idx_local=dof_indices)
        robot.set_dofs_kv(kv, dofs_idx_local=dof_indices)
        cls._configure_gripper_force(robot, dof_indices)

        hard_limits = urdf_joint_limits(cfg)
        gripper_limit_hi = float(hard_limits["gripper"][1])

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
            _gripper_limit_hi=gripper_limit_hi,
        )
        if apply_home:
            print(format_home_pose_summary(cfg, calibration=calibration))
            instance.apply_home_pose()
            instance.refresh_previews()
        ensure_shutdown_handlers()
        return instance

    @staticmethod
    def _configure_gripper_force(robot: Any, dof_indices: list[int]) -> None:
        """Allow stronger squeeze torque on the gripper DOF."""
        if len(dof_indices) < 6:
            return
        try:
            force_range = np.asarray(to_numpy(robot.get_dofs_force_range(dof_indices)), dtype=np.float64)
            if force_range.ndim == 2 and force_range.shape[0] >= 6:
                force_range[5] = [-15.0, 15.0]
                robot.set_dofs_force_range(force_range, dofs_idx_local=dof_indices)
            elif force_range.ndim == 1 and force_range.shape[0] >= 2:
                robot.set_dofs_force_range(
                    np.array([-15.0, 15.0], dtype=np.float64),
                    dofs_idx_local=[dof_indices[5]],
                )
        except Exception:
            pass

    def _gripper_dof_index(self) -> int:
        return self.dof_indices[-1]

    def _gripper_target_rad(self, gripper_rad: float) -> float:
        """Sim-only extra close when deliberately squeezing (meshes gappier than real gripper)."""
        extra = float(self.genesis_cfg.gripper_sim_extra_close_deg)
        squeeze_from = np.deg2rad(float(self.genesis_cfg.gripper_sim_extra_from_deg))
        if extra <= 0.0 or gripper_rad < squeeze_from:
            return float(gripper_rad)
        hi = self._gripper_limit_hi
        return float(min(gripper_rad + np.deg2rad(extra), hi))

    def _grasp_anchor_links(self) -> list[str]:
        links = getattr(self.genesis_cfg, "grasp_anchor_links", None)
        if links:
            return list(links)
        return [self.genesis_cfg.grasp_link]

    def _grasp_carry_link(self):
        if self._grasp_latch is None:
            raise RuntimeError("no grasp latch")
        return self.robot.get_link(self._grasp_latch.anchor_link_name)

    def _rigid_solver(self) -> Any:
        return self.scene.sim.rigid_solver

    def _desk_spec(self):
        desk = self.props.get("desk")
        return desk.spec if desk is not None else None

    def _enforce_desk_support(self) -> None:
        exclude: set[str] | None = None
        if self._grasp_latch is not None:
            exclude = {self._grasp_latch.prop_name}
        adjusted = enforce_desk_support(self.props, exclude=exclude)
        if adjusted and self._grasp_latch is not None and self._grasp_latch.mode == "weld":
            self._release_grasp_latch()

    def _release_grasp_latch(self) -> None:
        if self._grasp_latch is None:
            return
        try:
            release_latch(self._rigid_solver(), self._grasp_latch)
        except Exception:
            pass
        self._grasp_latch = None

    def _sim_latch_squeeze_rad(self) -> float:
        """Sim gripper angle that counts as 'shut enough' for proximity latch."""
        from_deg = float(self.genesis_cfg.gripper_sim_extra_from_deg)
        extra = float(self.genesis_cfg.gripper_sim_extra_close_deg)
        return float(np.deg2rad(from_deg + max(0.0, extra * 0.5)))

    def _probe_grasp_distances(self) -> tuple[str | None, float | None, dict[str, float]]:
        links = self._grasp_anchor_links()
        link_dist: dict[str, float] = {}
        best_name: str | None = None
        best_dist: float | None = None
        pen = self.props.get("pen")
        if pen is None:
            return None, None, link_dist
        for link_name in links:
            try:
                pos, _ = link_world_pose(self.robot.get_link(link_name))
                link_dist[link_name] = anchor_distance_to_prop(pen, pos)
            except Exception:
                continue
        found = nearest_prop_to_links(
            self.robot,
            self.props,
            links,
            radius=float(self.genesis_cfg.grasp_radius_m),
        )
        if found is not None:
            best_name, best_dist, _ = found
        return best_name, best_dist, link_dist

    def _set_grasp_diag(
        self,
        *,
        leader_gripper: float,
        sim_gripper: float,
        block_reason: str,
        prop_name: str | None = None,
        prop_dist: float | None = None,
    ) -> GraspDiagnostic:
        _, _, link_dist = self._probe_grasp_distances()
        pen_z: float | None = None
        pen = self.props.get("pen")
        if pen is not None:
            pen_z = float(to_numpy(pen.entity.get_pos()).reshape(-1)[2])
        diag = GraspDiagnostic(
            leader_gripper_deg=leader_gripper_deg(leader_gripper),
            sim_gripper_deg=leader_gripper_deg(sim_gripper),
            latched=self._grasp_latch is not None,
            prop_name=prop_name,
            prop_dist_m=prop_dist,
            pen_z_m=pen_z,
            block_reason=block_reason,
            link_dist_m=link_dist,
            latch_prop=self._grasp_latch.prop_name if self._grasp_latch else None,
        )
        self._last_grasp_diag = diag
        return diag

    def _update_grasp_carry(
        self,
        gripper_rad: float,
        *,
        sim_gripper_rad: float | None = None,
        allow_latch: bool = True,
    ) -> None:
        """Latch props to the gripper when closed nearby; kinematic carry follows the anchor link."""
        if not self.genesis_cfg.mirror_grasp_carry:
            return
        close_rad = np.deg2rad(float(self.genesis_cfg.grasp_close_deg))
        open_rad = np.deg2rad(float(self.genesis_cfg.grasp_open_deg))
        sim_gripper = float(sim_gripper_rad if sim_gripper_rad is not None else gripper_rad)
        tight_radius = float(getattr(self.genesis_cfg, "grasp_tight_radius_m", 0.07))
        sim_squeeze = self._sim_latch_squeeze_rad()

        if self._grasp_latch is not None:
            if not should_latch(
                gripper_rad,
                close_rad=close_rad,
                open_rad=open_rad,
                latched=True,
            ):
                self._release_grasp_latch()
                self._set_grasp_diag(
                    leader_gripper=gripper_rad,
                    sim_gripper=sim_gripper,
                    block_reason="released",
                )
                return
            if self._grasp_latch.mode == "kinematic":
                prop = self.props[self._grasp_latch.prop_name]
                carry_kinematic(self._grasp_latch, prop, self._grasp_carry_link())
            self._set_grasp_diag(
                leader_gripper=gripper_rad,
                sim_gripper=sim_gripper,
                block_reason="carry",
                prop_name=self._grasp_latch.prop_name,
            )
            return

        if not allow_latch:
            self._set_grasp_diag(
                leader_gripper=gripper_rad,
                sim_gripper=sim_gripper,
                block_reason="no_latch_retry",
            )
            return

        found = nearest_prop_to_links(
            self.robot,
            self.props,
            self._grasp_anchor_links(),
            radius=float(self.genesis_cfg.grasp_radius_m),
        )
        if found is None:
            self._set_grasp_diag(
                leader_gripper=gripper_rad,
                sim_gripper=sim_gripper,
                block_reason="too_far",
            )
            return
        name, dist, link_name = found
        if not can_acquire_latch(
            gripper_rad,
            sim_gripper,
            close_rad=close_rad,
            prop_dist=dist,
            tight_radius_m=tight_radius,
            sim_squeeze_rad=sim_squeeze,
        ):
            self._set_grasp_diag(
                leader_gripper=gripper_rad,
                sim_gripper=sim_gripper,
                block_reason="gripper_open",
                prop_name=name,
                prop_dist=dist,
            )
            return
        prop = self.props[name]
        link = self.robot.get_link(link_name)
        if self.genesis_cfg.grasp_weld:
            self._grasp_latch = latch_weld(
                self._rigid_solver(), name, prop.entity, link, anchor_link_name=link_name
            )
        else:
            self._grasp_latch = latch_kinematic(
                name,
                prop.entity,
                self.robot,
                self._grasp_anchor_links(),
                anchor_link_name=link_name,
            )
            carry_kinematic(self._grasp_latch, prop, link)
        self._set_grasp_diag(
            leader_gripper=gripper_rad,
            sim_gripper=sim_gripper,
            block_reason="latched",
            prop_name=name,
            prop_dist=dist,
        )

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

    def step(self, n: int = 1, *, enforce_desk: bool = True) -> None:
        for _ in range(n):
            self.scene.step()
        if enforce_desk:
            self._enforce_desk_support()
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
        self._release_grasp_latch()
        desk = self._desk_spec()
        for prop in self.props.values():
            if prop.spec.fixed:
                continue
            pos = np.asarray(prop.spec.initial_pos, dtype=np.float64).copy()
            if prop.spec.name == "pen" and desk is not None:
                pos[2] = resting_pen_center_z(desk, prop.spec)
            prop.entity.set_pos(pos, relative=False, zero_velocity=True)
            quat = spawn_quat_wxyz(prop.spec)
            if quat is not None:
                prop.entity.set_quat(quat, relative=False, zero_velocity=True)
        self.step(20)

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
        self._mirror_last_raw = None
        self._mirror_cmd_rad = None
        self.set_joint_positions_rad(radians)
        self.step(10)

    def calibrated_radian_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-DOF radian limits matching calibrated servo min..max."""
        limits = calibrated_urdf_limits(self.cfg, self.calibration)
        lo = [limits[name][0] for name in JOINT_NAMES]
        hi = [limits[name][1] for name in JOINT_NAMES]
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    def _mirror_deadband(self, joint: str) -> int:
        spec = self.genesis_cfg.joints.get(joint)
        if spec is not None and spec.mirror_raw_deadband is not None:
            return int(spec.mirror_raw_deadband)
        return int(self.genesis_cfg.mirror_raw_deadband)

    def _mirror_deadband_map(self) -> dict[str, int]:
        return {joint: self._mirror_deadband(joint) for joint in JOINT_NAMES}

    def _filter_raw_for_mirror(self, raw: dict[str, int]) -> dict[str, int]:
        filtered = filter_raw_counts(
            raw,
            last_raw=self._mirror_last_raw,
            deadband_for_joint=self._mirror_deadband_map(),
        )
        self._mirror_last_raw = dict(filtered)
        return filtered

    def _radians_from_raw(self, raw: dict[str, int]) -> np.ndarray:
        hard = urdf_joint_limits(self.cfg)
        return np.array(
            [
                raw_to_radians(
                    int(raw[name]),
                    name,
                    self.cfg,
                    self.calibration,
                    hard_limits=hard,
                )
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )

    def _mirror_max_norm_step(self) -> float:
        step = self.genesis_cfg.mirror_max_norm_step
        if step is not None:
            return float(step)
        return float(self.cfg.robot.max_relative_target)

    def _smooth_mirror_command(
        self,
        target_rad: np.ndarray,
        *,
        snap: bool = False,
    ) -> np.ndarray:
        current = to_numpy(self.robot.get_dofs_position(self.dof_indices))
        cmd = smooth_mirror_radians(
            target_rad,
            current_rad=current,
            previous_cmd_rad=self._mirror_cmd_rad,
            cfg=self.cfg,
            calibration=self.calibration,
            max_norm_step=self._mirror_max_norm_step(),
            smoothing=float(self.genesis_cfg.mirror_smoothing),
            snap=snap,
        )
        self._mirror_cmd_rad = cmd
        return cmd

    def _apply_mirror_command(self, cmd_rad: np.ndarray, *, snap: bool = False) -> None:
        if snap:
            self.robot.set_dofs_position(
                np.asarray(cmd_rad, dtype=np.float64),
                self.dof_indices,
                zero_velocity=True,
            )
        else:
            self.robot.control_dofs_position(
                np.asarray(cmd_rad, dtype=np.float64),
                self.dof_indices,
            )
        self.step(1)

    def _mirror_to_target(self, target_rad: np.ndarray, *, snap: bool = False) -> None:
        """Apply a leader target to the sim arm."""
        target = np.asarray(target_rad, dtype=np.float64)
        if self.genesis_cfg.mirror_kinematic:
            cmd = target.copy()
            cmd[-1] = self._gripper_target_rad(float(target[-1]))
            # Full pose copy — gripper must be in set_dofs_position or it won't move in sim.
            self.robot.set_dofs_position(cmd, self.dof_indices, zero_velocity=True)
            # Gripper PD on top for squeeze contact forces when closing on objects.
            self.robot.control_dofs_position(
                np.array([cmd[-1]], dtype=np.float64),
                [self._gripper_dof_index()],
            )
            self._mirror_cmd_rad = target.copy()
            leader_gripper = float(target[-1])
            sim_gripper = float(cmd[-1])
            n = max(1, int(self.genesis_cfg.mirror_substeps))
            n += max(0, int(self.genesis_cfg.mirror_grasp_substeps))
            # Latch/release uses leader gripper — not sim extra-close (or rest reads as "closed").
            self._update_grasp_carry(
                leader_gripper,
                sim_gripper_rad=sim_gripper,
                allow_latch=True,
            )
            if self._grasp_latch is not None and self._grasp_latch.mode == "kinematic":
                self._update_grasp_carry(
                    leader_gripper,
                    sim_gripper_rad=sim_gripper,
                    allow_latch=False,
                )
            self.step(n)
            if self._grasp_latch is not None and self._grasp_latch.mode == "kinematic":
                self._update_grasp_carry(
                    leader_gripper,
                    sim_gripper_rad=sim_gripper,
                    allow_latch=False,
                )
            elif self._grasp_latch is None:
                self._update_grasp_carry(
                    leader_gripper,
                    sim_gripper_rad=sim_gripper,
                    allow_latch=True,
                )
            return

        first = self._mirror_cmd_rad is None
        cmd = self._smooth_mirror_command(target, snap=snap or first)
        self._apply_mirror_command(cmd, snap=snap or first)

    def sync_raw_pose(self, raw: dict[str, int], *, snap: bool = False) -> None:
        """Mirror leader encoder counts into Genesis (deadband filter; kinematic by default)."""
        filtered = self._filter_raw_for_mirror(raw)
        target = self._radians_from_raw(filtered)
        self._mirror_to_target(target, snap=snap)

    def sync_norm_pose(self, obs: dict[str, float], *, snap: bool = False) -> None:
        """Mirror a LeRobot normalized observation into Genesis."""
        target = np.array(
            observation_to_radians(obs, self.cfg, calibration=self.calibration),
            dtype=np.float64,
        )
        self._mirror_to_target(target, snap=snap)
