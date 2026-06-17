"""FastAPI server for the SO-ARM101 3D joint simulator."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT, ProjectConfig
from .kinematics import (
    ArmGeometry,
    forward_kinematics,
    reach_error_mm,
    solve_ik,
    suggest_pitch,
)
from .sim_config import dump_robot_yaml, export_robot_yaml, geometry_from_config

SIM_DIR = PROJECT_ROOT / "sim"

_config: ProjectConfig | None = None
_geom: ArmGeometry | None = None


def _load() -> ProjectConfig:
    global _config, _geom
    _config = ProjectConfig.load()
    _geom = geometry_from_config(_config)
    return _config


def _require_geometry() -> ArmGeometry:
    if _geom is None:
        raise HTTPException(
            status_code=503,
            detail="No geometry in config/default.yaml — add a geometry: section",
        )
    return _geom


def _joint_limits() -> dict[str, tuple[float, float]]:
    cfg = _config or _load()
    return cfg.sim_joint_limits()


def _sample_step(lo: float, hi: float, index: int, count: int) -> float:
    if count <= 1:
        return (lo + hi) / 2
    return lo + (hi - lo) * index / (count - 1)


def _ik_payload(
    sol,
    *,
    target_x: float | None = None,
    target_y: float | None = None,
    target_z: float | None = None,
) -> dict[str, Any]:
    geom = _require_geometry()
    tip = forward_kinematics(geom, sol.joint_values)
    payload: dict[str, Any] = {
        "reachable": sol.reachable,
        "joint_values": sol.joint_values,
        "servo_angles": sol.joint_values,
        "kin_angles": sol.kin_angles,
        "warnings": sol.warnings,
        "elbow": sol.elbow,
        "tip": {
            "x": tip["x"],
            "y": tip["y"],
            "z": tip["z"],
            "pitch_deg": tip["pitch_deg"],
        },
    }
    if target_x is not None and target_y is not None and target_z is not None:
        payload["error_mm"] = reach_error_mm(
            geom,
            {"x": target_x, "y": target_y, "z": target_z},
            sol.joint_values,
        )
    return payload


def sample_reach_points(steps: dict[str, int] | None = None) -> list[dict[str, float]]:
    geom = _require_geometry()
    limits = _joint_limits()
    cfg = _config or _load()
    default_steps = dict(cfg.sim_reach_steps())
    steps = {**default_steps, **(steps or {})}
    points: list[dict[str, float]] = []

    for ip in range(steps["shoulder_pan"]):
        pan = _sample_step(*limits["shoulder_pan"], ip, steps["shoulder_pan"])
        for il in range(steps["shoulder_lift"]):
            lift = _sample_step(*limits["shoulder_lift"], il, steps["shoulder_lift"])
            for ie in range(steps["elbow_flex"]):
                elbow = _sample_step(*limits["elbow_flex"], ie, steps["elbow_flex"])
                for iw in range(steps["wrist_flex"]):
                    wrist = _sample_step(*limits["wrist_flex"], iw, steps["wrist_flex"])
                    tip = forward_kinematics(
                        geom,
                        {
                            "shoulder_pan": pan,
                            "shoulder_lift": lift,
                            "elbow_flex": elbow,
                            "wrist_flex": wrist,
                        },
                    )
                    points.append({"x": tip["x"], "y": tip["y"], "z": tip["z"]})
    return points


class IKRequest(BaseModel):
    x: float
    y: float
    z: float
    pitch_deg: float | None = None
    elbow: str | None = None


class FKRequest(BaseModel):
    joint_values: dict[str, float] = Field(default_factory=dict)
    servo_angles: dict[str, float] = Field(default_factory=dict)


class ReachTargetRequest(BaseModel):
    x: float
    y: float
    z: float


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _load()
    yield


app = FastAPI(title="sarm-hand 3D sim API", version="0.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    _require_geometry()
    return {"status": "ok"}


@app.post("/api/ik")
def api_solve_ik(req: IKRequest) -> dict[str, Any]:
    geom = _require_geometry()
    sol = solve_ik(geom, req.x, req.y, req.z, pitch_deg=req.pitch_deg, elbow=req.elbow)
    return _ik_payload(sol, target_x=req.x, target_y=req.y, target_z=req.z)


@app.post("/api/fk")
def api_forward_kinematics(req: FKRequest) -> dict[str, Any]:
    geom = _require_geometry()
    values = req.joint_values or req.servo_angles
    tip = forward_kinematics(geom, values)
    return {
        "x": tip["x"],
        "y": tip["y"],
        "z": tip["z"],
        "pitch_deg": tip["pitch_deg"],
        "reach_mm": tip.get("reach_mm"),
    }


@app.post("/api/ik/suggest-pitch")
def api_suggest_pitch(req: ReachTargetRequest) -> dict[str, Any]:
    return suggest_pitch(_require_geometry(), req.x, req.y, req.z)


@app.get("/api/reach/samples")
def api_reach_samples(
    shoulder_pan: int = Query(16, ge=2, le=32),
    shoulder_lift: int = Query(12, ge=2, le=32),
    elbow_flex: int = Query(12, ge=2, le=32),
    wrist_flex: int = Query(8, ge=2, le=32),
) -> dict[str, Any]:
    points = sample_reach_points(
        {
            "shoulder_pan": shoulder_pan,
            "shoulder_lift": shoulder_lift,
            "elbow_flex": elbow_flex,
            "wrist_flex": wrist_flex,
        }
    )
    return {"points": points, "count": len(points)}


@app.post("/api/config/reload")
def api_reload_config() -> dict[str, str]:
    _load()
    _require_geometry()
    return {"status": "reloaded"}


@app.get("/robot.yaml")
def serve_robot_yaml() -> PlainTextResponse:
    cfg = _config or _load()
    return PlainTextResponse(dump_robot_yaml(cfg), media_type="application/x-yaml")


if SIM_DIR.is_dir():
    app.mount("/sim", StaticFiles(directory=SIM_DIR, html=True), name="sim")


def launch_sim(
    *,
    host: str | None = None,
    port: int | None = None,
    open_browser: bool = True,
) -> None:
    """Start the 3D simulator API and optionally open the browser."""
    import uvicorn
    import webbrowser

    from .servo import servo_summary

    cfg = _load()
    bind_host = host or cfg.sim.host
    bind_port = port or cfg.sim.port
    url = f"http://{bind_host}:{bind_port}/sim/arm3d.html"

    if open_browser and cfg.sim.open_browser:
        webbrowser.open(url)

    print(f"sarm-hand 3D sim  {url}")
    print(f"servos           {servo_summary(cfg)}")
    print(f"config           {PROJECT_ROOT / 'config' / 'default.yaml'}")
    print(f"robot.yaml       http://{bind_host}:{bind_port}/robot.yaml")
    print("Ctrl+C to stop")

    uvicorn.run(
        "sarm_hand.sim_api:app",
        host=bind_host,
        port=bind_port,
        reload=False,
    )
