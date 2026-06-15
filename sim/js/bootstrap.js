/** Startup — load config, build scene, run render loop. */

import { state } from "./state.js";
import { loadRobotConfig, setJointValue } from "./config.js";
import { checkKinematicsApi } from "./api.js";
import { buildReachability } from "./reach.js";
import { buildScene, onResize } from "./scene.js";
import { buildUI } from "./ui.js";
import { buildCalLab } from "./cal-lab.js";
import { update } from "./arm-update.js";

export function animate() {
  requestAnimationFrame(animate);
  if (!state.renderer || !state.scene || !state.camera) return;
  state.controls?.update();
  state.renderer.render(state.scene, state.camera);
  state.labelRenderer?.render(state.scene, state.camera);
}

export async function bootstrap() {
  const src = document.getElementById("source");
  try {
    state.CONFIG = await loadRobotConfig();
    const apiOk = await checkKinematicsApi();
    let msg = "config: config/default.yaml (via /robot.yaml)";
    msg += apiOk ? " · IK via sarm-hand API" : " · API offline — run: uv run sarm-hand sim";
    src.textContent = msg;
    src.className = apiOk ? "source ok" : "source err";

    const brand = document.querySelector(".brand h1");
    if (brand) brand.textContent = state.CONFIG.brandTitle;
    const sub = document.querySelector(".brand p");
    if (sub) sub.textContent = state.CONFIG.brandSubtitle;
  } catch (err) {
    src.textContent = String(err.message || err);
    src.className = "source err";
    document.getElementById("loading").textContent =
      "Failed to load robot.yaml — run uv run sarm-hand sim from project root.";
    return;
  }
  document.getElementById("loading").style.display = "none";

  state.UNIT = state.CONFIG.unit;
  state.reachZTolerance = state.CONFIG.reachZTolerance;
  state.JOINTS = Object.keys(state.CONFIG.limits);
  const startPose = state.CONFIG.poses.park || state.CONFIG.poses.home || state.CONFIG.homeValues;
  state.activePose = state.CONFIG.poses.park ? "park" : state.CONFIG.poses.home ? "home" : null;
  state.joints = { ...state.CONFIG.homeValues, ...startPose };
  state.JOINTS.forEach((j) => setJointValue(j, state.joints[j] ?? state.CONFIG.homeValues[j] ?? 0));

  buildUI();
  buildCalLab();
  const reportSceneErr = (step, err) => {
    console.error(step + " failed:", err);
    document.getElementById("source").textContent +=
      " · 3D scene error (" + step + "): " + (err?.message || err);
    document.getElementById("source").className = "source err";
  };
  try {
    buildScene();
  } catch (err) {
    reportSceneErr("buildScene", err);
    return;
  }
  try {
    update();
  } catch (err) {
    reportSceneErr("update", err);
    return;
  }
  try {
    animate();
  } catch (err) {
    reportSceneErr("animate", err);
  }
  buildReachability().catch((err) => {
    console.warn("reach cloud:", err);
    const countEl = document.getElementById("reachZCount");
    if (countEl) countEl.textContent = "reach cloud failed — is sarm-hand sim running?";
  });
}
