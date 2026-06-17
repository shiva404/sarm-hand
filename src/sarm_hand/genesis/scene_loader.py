"""Load Genesis scene object definitions from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import PROJECT_ROOT, GenesisSettings


@dataclass
class SceneObjectSpec:
    """One spawnable object in a Genesis scene."""

    name: str
    shape: str  # box | cylinder | sphere
    pos: tuple[float, float, float]
    enabled: bool = True
    fixed: bool = False
    color: tuple[float, float, float] | None = None
    surface: str = "default"  # default | plastic | rough | glass | gold | aluminum
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None
    density: float | None = None

    @property
    def initial_pos(self) -> np.ndarray:
        return np.array(self.pos, dtype=np.float64)


@dataclass
class SceneDefinition:
    name: str
    task: str = ""
    description: str = ""
    objects: list[SceneObjectSpec] = field(default_factory=list)


@dataclass
class SceneProp:
    """Runtime handle for a spawned scene object."""

    spec: SceneObjectSpec
    entity: Any


def resolve_scene_path(genesis_cfg: GenesisSettings) -> Path:
    if getattr(genesis_cfg, "scene_file", None):
        path = Path(genesis_cfg.scene_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    return PROJECT_ROOT / "config" / "scenes" / f"{genesis_cfg.scene}.yaml"


def _vec3(raw: list | tuple | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if raw is None:
        return default
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _parse_color(raw) -> tuple[float, float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.startswith("#") and len(raw) == 7:
        return (
            int(raw[1:3], 16) / 255.0,
            int(raw[3:5], 16) / 255.0,
            int(raw[5:7], 16) / 255.0,
        )
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        values = [float(v) for v in raw[:3]]
        if max(values) > 1.0:
            values = [v / 255.0 for v in values]
        return (values[0], values[1], values[2])
    return None


def _parse_object(name: str, raw: dict[str, Any]) -> SceneObjectSpec:
    if raw.get("enabled", True) is False:
        return SceneObjectSpec(name=name, shape="box", pos=(0.0, 0.0, 0.0), enabled=False)

    shape = str(raw.get("shape", "box")).lower()
    pos = _vec3(raw.get("pos"), (0.0, 0.0, 0.0))
    color = _parse_color(raw.get("color"))
    size_raw = raw.get("size")
    size = _vec3(size_raw, (0.05, 0.05, 0.05)) if size_raw is not None else None

    return SceneObjectSpec(
        name=name,
        shape=shape,
        pos=pos,
        enabled=True,
        fixed=bool(raw.get("fixed", False)),
        color=color,
        surface=str(raw.get("surface", "default")).lower(),
        size=size,
        radius=float(raw["radius"]) if "radius" in raw else None,
        height=float(raw["height"]) if "height" in raw else None,
        density=float(raw["density"]) if "density" in raw else None,
    )


def load_scene_definition(genesis_cfg: GenesisSettings) -> SceneDefinition:
    """Load scene YAML; fall back to built-in pick/place layout if missing."""
    path = resolve_scene_path(genesis_cfg)
    if not path.is_file():
        return _default_pick_place_scene(genesis_cfg.scene)

    raw = yaml.safe_load(path.read_text()) or {}
    objects: list[SceneObjectSpec] = []

    if "objects" in raw:
        for name, obj in raw["objects"].items():
            objects.append(_parse_object(name, obj or {}))
    else:
        # Legacy flat layout (desk/pen/holder at top level).
        for key in ("desk", "pen", "holder", "cube", "target"):
            if key in raw:
                legacy = dict(raw[key])
                legacy.setdefault("shape", "cylinder" if key == "pen" else "box")
                legacy.setdefault("fixed", key in ("desk", "holder"))
                objects.append(_parse_object(key, legacy))

    return SceneDefinition(
        name=str(raw.get("name", genesis_cfg.scene)),
        task=str(raw.get("task", "")),
        description=str(raw.get("description", "")),
        objects=[obj for obj in objects if obj.enabled],
    )


def _default_pick_place_scene(name: str) -> SceneDefinition:
    return SceneDefinition(
        name=name,
        task="Pick up the pen and place it in the holder",
        objects=[
            SceneObjectSpec("desk", "box", (0.35, 0.0, 0.01), fixed=True, size=(0.5, 0.35, 0.02), color=(0.45, 0.32, 0.18)),
            SceneObjectSpec("pen", "cylinder", (0.28, 0.08, 0.08), radius=0.004, height=0.12, color=(0.15, 0.35, 0.85)),
            SceneObjectSpec("holder", "box", (0.42, -0.08, 0.05), fixed=True, size=(0.04, 0.04, 0.06), color=(0.25, 0.25, 0.28)),
        ],
    )


def build_surface(gs, spec: SceneObjectSpec):
    """Create a Genesis surface from object spec."""
    color = spec.color
    preset = spec.surface
    if preset == "plastic":
        return gs.surfaces.Plastic(color=color) if color else gs.surfaces.Plastic()
    if preset == "rough":
        return gs.surfaces.Rough(color=color) if color else gs.surfaces.Rough()
    if preset == "glass":
        return gs.surfaces.Glass(color=color) if color else gs.surfaces.Glass()
    if preset == "gold":
        return gs.surfaces.Gold()
    if preset == "aluminum":
        return gs.surfaces.Aluminium()
    return gs.surfaces.Default(color=color) if color else gs.surfaces.Default()


def build_morph(gs, spec: SceneObjectSpec):
    """Create a Genesis morph from object spec."""
    kwargs: dict[str, Any] = {"pos": spec.pos, "fixed": spec.fixed}
    if spec.shape == "box":
        if spec.size is None:
            raise ValueError(f"Object {spec.name!r}: box requires size [x, y, z]")
        return gs.morphs.Box(size=spec.size, **kwargs)
    if spec.shape == "cylinder":
        if spec.radius is None or spec.height is None:
            raise ValueError(f"Object {spec.name!r}: cylinder requires radius and height")
        return gs.morphs.Cylinder(radius=spec.radius, height=spec.height, **kwargs)
    if spec.shape == "sphere":
        if spec.radius is None:
            raise ValueError(f"Object {spec.name!r}: sphere requires radius")
        return gs.morphs.Sphere(radius=spec.radius, **kwargs)
    raise ValueError(f"Object {spec.name!r}: unsupported shape {spec.shape!r} (use box, cylinder, sphere)")


def spawn_scene_objects(scene, gs, definition: SceneDefinition) -> dict[str, SceneProp]:
    """Add all enabled objects from a scene definition."""
    props: dict[str, SceneProp] = {}
    for spec in definition.objects:
        if not spec.enabled:
            continue
        morph = build_morph(gs, spec)
        surface = build_surface(gs, spec)
        material = gs.materials.Rigid(rho=spec.density) if spec.density else gs.materials.Rigid()
        entity = scene.add_entity(morph, material=material, surface=surface, name=spec.name)
        props[spec.name] = SceneProp(spec=spec, entity=entity)
    return props
