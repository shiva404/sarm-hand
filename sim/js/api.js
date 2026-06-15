/** Kinematics API — same code as sarm-hand sim_api (Python). */

import { API_BASE } from "./constants.js";
import { state } from "./state.js";

export async function apiFetch(path, options = {}) {
  const res = await fetch(API_BASE + path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

export async function checkKinematicsApi() {
  try {
    await apiFetch("/api/health");
    state.apiReady = true;
    return true;
  } catch (_) {
    state.apiReady = false;
    return false;
  }
}

export async function apiSolveIk(x, y, z, pitchDeg = null, elbow = null) {
  const body = { x, y, z };
  if (pitchDeg != null) body.pitch_deg = pitchDeg;
  if (elbow != null) body.elbow = elbow;
  const sol = await apiFetch("/api/ik", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return {
    reachable: sol.reachable,
    jointValues: sol.joint_values || sol.servo_angles,
    kinAngles: sol.kin_angles,
    warnings: sol.warnings || [],
    elbow: sol.elbow,
    tip: sol.tip,
    errorMm: sol.error_mm,
  };
}

export async function apiSuggestPitch(x, y, z) {
  try {
    const res = await apiFetch("/api/ik/suggest-pitch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, z }),
    });
    if (res && typeof res.found === "boolean") return res;
  } catch (_) { /* fall through */ }
  return { found: false, pitch_deg: null };
}
