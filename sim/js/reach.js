/** Reach cloud, Z-slice filter, hover/click picker, and IK Go-to. */

import * as THREE from "three";
import { state } from "./state.js";
import { kinToThree, formatReachCoords } from "./kinematics.js";
import { apiFetch, apiSolveIk, apiSuggestPitch } from "./api.js";
import { setJointValue } from "./config.js";
import { update } from "./arm-update.js";
import { refreshJointUI, setActivePose } from "./ui.js";
import { resolvePitchForIk } from "./pitch-viz.js";

function parseXyzInput(text) {
  const parts = text.trim().split(/\s+/).filter(Boolean);
  if (parts.length !== 3) return null;
  const nums = parts.map(Number);
  if (nums.some((n) => !Number.isFinite(n))) return null;
  return { x: nums[0], y: nums[1], z: nums[2] };
}

function setReachGoStatus(msg, kind = "") {
  const el = document.getElementById("reachGoStatus");
  if (!el) return;
  el.textContent = msg;
  el.className = "status" + (kind ? " " + kind : "");
}

function animateToJoints(targetValues, durationMs = 500) {
  if (state.reachAnimFrame) cancelAnimationFrame(state.reachAnimFrame);
  const start = { ...state.joints };
  const keys = Object.keys(targetValues);
  const t0 = performance.now();
  function step(now) {
    const t = Math.min(1, (now - t0) / durationMs);
    for (const j of keys) {
      if (targetValues[j] == null) continue;
      const v = start[j] + (targetValues[j] - start[j]) * t;
      const clamped = setJointValue(j, v);
      const s = document.getElementById("s_" + j);
      if (s) s.value = clamped;
      if (state.valEls[j]) state.valEls[j].textContent = Math.round(clamped);
      refreshJointUI(j);
    }
    setActivePose(null);
    update();
    if (t < 1) state.reachAnimFrame = requestAnimationFrame(step);
    else state.reachAnimFrame = null;
  }
  state.reachAnimFrame = requestAnimationFrame(step);
}

export async function goToReachCoords() {
  if (!state.apiReady) {
    setReachGoStatus("start API: uv run sarm-hand sim", "err");
    return;
  }
  const tol = state.CONFIG?.reachGoTolMm ?? 15;
  try {
    const input = document.getElementById("reachGoInput");
    const parsed = parseXyzInput(input?.value ?? "");
    if (!parsed) {
      setReachGoStatus("enter three numbers: x y z", "err");
      return;
    }
    const resolved = resolvePitchForIk();
    if (resolved.error) {
      setReachGoStatus(resolved.error, "err");
      return;
    }

    const pitchEl = document.getElementById("reachGoPitch");
    let pitch = resolved.pitch;
    let autoPitch = false;
    const wristMode = resolved.wristMode;

    if (!wristMode && resolved.auto && pitch === undefined) {
      const sugg = await apiSuggestPitch(parsed.x, parsed.y, parsed.z);
      if (sugg?.found) {
        pitch = sugg.pitch_deg;
        autoPitch = true;
        if (pitchEl && pitch != null) pitchEl.value = String(pitch);
      } else {
        setReachGoStatus("no auto pitch found — try manual pitch or wrist mode", "warn");
        return;
      }
    }

    const sol = await apiSolveIk(parsed.x, parsed.y, parsed.z, pitch);
    if (!sol?.tip) {
      setReachGoStatus("IK failed — is sarm-hand sim running?", "err");
      return;
    }
    const tip = sol.tip;
    const err = sol.errorMm ?? tol + 1;

    if (!sol.reachable || err > tol) {
      const sugg = await apiSuggestPitch(parsed.x, parsed.y, parsed.z);
      let msg = `cannot reach (${parsed.x}, ${parsed.y}, ${parsed.z})`;
      if (wristMode) msg += " (wrist target)";
      else if (pitch != null) msg += ` with pitch ${pitch}°`;
      msg += ` — FK would land (${Math.round(tip.x)}, ${Math.round(tip.y)}, ${Math.round(tip.z)}) mm, ${err.toFixed(0)} mm away`;
      if (sugg?.found) {
        msg += sugg.pitch_deg === null ? ". Try clearing pitch" : `. Try pitch ${sugg.pitch_deg}°`;
      } else {
        msg += ". Pick a closer point from the reach cloud";
      }
      if (sol.warnings.length) msg += " · " + sol.warnings[0];
      setReachGoStatus(msg, "err");
      return;
    }

    let status = `moving to (${parsed.x}, ${parsed.y}, ${parsed.z}) mm`;
    if (wristMode) status += " · wrist target (no pitch IK)";
    else if (autoPitch && pitch != null) status += ` · pitch ${pitch}° (auto)`;
    else if (pitch != null) status += ` · pitch ${pitch}°`;
    setReachGoStatus(status, "ok");
    animateToJoints(sol.jointValues);
    if (sol.warnings.length) {
      setReachGoStatus(sol.warnings.join(" · "), "warn");
    }
  } catch (err) {
    console.error("reach Go failed:", err);
    setReachGoStatus(String(err.message || err), "err");
  }
}

function fillReachGoInput(text) {
  const input = document.getElementById("reachGoInput");
  if (input) input.value = text;
}

function initReachGo() {
  const btn = document.getElementById("reachGoBtn");
  const input = document.getElementById("reachGoInput");
  btn?.addEventListener("click", goToReachCoords);
  input?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") goToReachCoords();
  });
}

export function disposeReachability() {
  for (const obj of [state.reachCloud, state.reachFloor, state.reachBounds]) {
    if (!obj) continue;
    state.scene.remove(obj);
    obj.geometry?.dispose();
    obj.material?.dispose();
  }
  state.reachCloud = state.reachFloor = state.reachBounds = null;
  state.reachKinCoords = [];
  state.reachAllSamples = [];
  hideReachTooltip();
}

function configureReachZSlider() {
  const slider = document.getElementById("reachZSlider");
  const valEl = document.getElementById("reachZVal");
  if (!slider || !state.reachAllSamples.length) return;
  const zs = state.reachAllSamples.map((s) => s.z);
  const zMin = Math.min(...zs);
  const zMax = Math.max(...zs);
  state.reachZTolerance = state.CONFIG?.reachZTolerance ?? Math.max(8, (zMax - zMin) / 24);
  slider.min = String(Math.floor(zMin / 5) * 5);
  slider.max = String(Math.ceil(zMax / 5) * 5);
  slider.step = "5";
  if (state.reachFilterZ < Number(slider.min)) state.reachFilterZ = Number(slider.min);
  if (state.reachFilterZ > Number(slider.max)) state.reachFilterZ = Number(slider.max);
  slider.value = String(state.reachFilterZ);
  if (valEl) valEl.textContent = state.reachFilterZ;
}

export function applyReachZFilter() {
  if (!state.reachCloud || !state.reachFloor) return;
  const filtered = state.reachAllSamples.filter(
    (s) => Math.abs(s.z - state.reachFilterZ) <= state.reachZTolerance
  );
  state.reachKinCoords = filtered.map((s) => ({ x: s.x, y: s.y, z: s.z }));
  state.reachCloud.userData.kinCoords = state.reachKinCoords;
  state.reachFloor.userData.kinCoords = state.reachKinCoords;

  const vol = [];
  const floor = [];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const s of filtered) {
    const p = kinToThree(s.x, s.y, s.z);
    vol.push(p.x, p.y, p.z);
    floor.push(s.x, 0.8, s.y);
    minX = Math.min(minX, s.x); maxX = Math.max(maxX, s.x);
    minY = Math.min(minY, s.y); maxY = Math.max(maxY, s.y);
  }

  state.reachCloud.geometry.setAttribute("position", new THREE.Float32BufferAttribute(vol, 3));
  state.reachCloud.geometry.computeBoundingSphere();
  state.reachFloor.geometry.setAttribute("position", new THREE.Float32BufferAttribute(floor, 3));
  state.reachFloor.geometry.computeBoundingSphere();

  if (state.reachBounds) {
    state.reachBounds.visible = state.showReach && filtered.length > 0;
    if (filtered.length > 0) {
      const pad = 8;
      const w = maxX - minX + pad * 2;
      const d = maxY - minY + pad * 2;
      state.reachBounds.geometry.dispose();
      state.reachBounds.geometry = new THREE.EdgesGeometry(new THREE.PlaneGeometry(w, d));
      state.reachBounds.position.set((minX + maxX) / 2, 0.4, (minY + maxY) / 2);
    }
  }

  const countEl = document.getElementById("reachZCount");
  if (countEl) {
    countEl.textContent = filtered.length
      ? filtered.length + " points at z ≈ " + state.reachFilterZ + " mm (±" + Math.round(state.reachZTolerance) + ")"
      : "no points at z ≈ " + state.reachFilterZ + " mm — try another slice";
  }
  hideReachTooltip();
}

export async function buildReachability() {
  disposeReachability();
  if (!state.apiReady) {
    const countEl = document.getElementById("reachZCount");
    if (countEl) countEl.textContent = "reach cloud needs API — uv run sarm-hand sim";
    return;
  }
  const steps = state.CONFIG.reachSteps || {};
  const qs = new URLSearchParams(steps).toString();
  const data = await apiFetch("/api/reach/samples" + (qs ? "?" + qs : ""));
  const vol = [];
  const floor = [];
  state.reachAllSamples = [];
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;

  for (const tip of data.points) {
    state.reachAllSamples.push({ x: tip.x, y: tip.y, z: tip.z });
    const p = kinToThree(tip.x, tip.y, tip.z);
    vol.push(p.x, p.y, p.z);
    floor.push(tip.x, 0.8, tip.y);
    minX = Math.min(minX, tip.x); maxX = Math.max(maxX, tip.x);
    minZ = Math.min(minZ, tip.y); maxZ = Math.max(maxZ, tip.y);
  }

  const volGeom = new THREE.BufferGeometry();
  volGeom.setAttribute("position", new THREE.Float32BufferAttribute(vol, 3));
  state.reachCloud = new THREE.Points(volGeom, new THREE.PointsMaterial({
    color: 0x4ea1ff,
    size: 3.2,
    transparent: true,
    opacity: 0.32,
    depthWrite: false,
    sizeAttenuation: true,
  }));
  state.reachCloud.renderOrder = 0;
  state.reachCloud.visible = state.showReach;
  state.scene.add(state.reachCloud);

  const floorGeom = new THREE.BufferGeometry();
  floorGeom.setAttribute("position", new THREE.Float32BufferAttribute(floor, 3));
  state.reachFloor = new THREE.Points(floorGeom, new THREE.PointsMaterial({
    color: 0x3ecf8e,
    size: 2.4,
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
    sizeAttenuation: true,
  }));
  state.reachFloor.renderOrder = 0;
  state.reachFloor.visible = state.showReach;
  state.scene.add(state.reachFloor);

  if (Number.isFinite(minX)) {
    const pad = 8;
    const w = maxX - minX + pad * 2;
    const d = maxZ - minZ + pad * 2;
    const cx = (minX + maxX) / 2;
    const cz = (minZ + maxZ) / 2;
    state.reachBounds = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(w, d)),
      new THREE.LineBasicMaterial({ color: 0x3ecf8e, transparent: true, opacity: 0.45 })
    );
    state.reachBounds.rotation.x = -Math.PI / 2;
    state.reachBounds.position.set(cx, 0.4, cz);
    state.reachBounds.visible = state.showReach;
    state.reachBounds.renderOrder = 0;
    state.scene.add(state.reachBounds);
  }
  configureReachZSlider();
  applyReachZFilter();
}

function pickReachPoint(clientX, clientY) {
  if (!state.showReach || !state.reachKinCoords.length || !state.renderer || !state.camera) return null;
  const rect = state.renderer.domElement.getBoundingClientRect();
  state.reachMouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  state.reachMouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  state.reachRaycaster.setFromCamera(state.reachMouse, state.camera);
  const targets = [state.reachCloud, state.reachFloor].filter(Boolean);
  const hits = state.reachRaycaster.intersectObjects(targets, false);
  if (!hits.length || hits[0].index == null) return null;
  return state.reachKinCoords[hits[0].index] ?? null;
}

export function hideReachTooltip() {
  state.reachHoverPt = null;
  const tip = document.getElementById("reachTooltip");
  if (tip) {
    tip.classList.remove("visible", "copied");
    tip.setAttribute("aria-hidden", "true");
  }
  state.renderer?.domElement.classList.remove("reach-hover");
}

function showReachTooltip(clientX, clientY, pt, copied = false) {
  const tip = document.getElementById("reachTooltip");
  if (!tip || !pt) return;
  const fmt = formatReachCoords(pt);
  tip.innerHTML = copied
    ? '<span class="xyz">Copied</span> <span class="action">' + fmt.copy + '</span>'
    : '<span class="xyz">' + fmt.display + '</span> <span class="action">mm · click to copy</span>';
  tip.style.left = clientX + "px";
  tip.style.top = clientY + "px";
  tip.classList.add("visible");
  tip.classList.toggle("copied", copied);
  tip.setAttribute("aria-hidden", "false");
}

export function updateReachViewHint() {
  const el = document.getElementById("viewHint");
  if (!el) return;
  el.textContent = state.showReach
    ? "drag orbit · hover reach cloud · click to copy x y z · or type coords below and Go"
    : "drag to orbit · scroll to zoom · type x y z below and Go";
}

export function initReachPicker() {
  if (!state.reachRaycaster) {
    state.reachRaycaster = new THREE.Raycaster();
    state.reachRaycaster.params.Points.threshold = 18;
    state.reachMouse = new THREE.Vector2();
  }
  const canvas = state.renderer.domElement;
  canvas.addEventListener("pointerdown", (e) => {
    state.reachPointerDown = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener("pointermove", (e) => {
    state.reachLastPointer = { x: e.clientX, y: e.clientY };
    if (!state.showReach) {
      hideReachTooltip();
      return;
    }
    const pt = pickReachPoint(e.clientX, e.clientY);
    if (!pt) {
      hideReachTooltip();
      return;
    }
    state.reachHoverPt = pt;
    showReachTooltip(e.clientX, e.clientY, pt);
    canvas.classList.add("reach-hover");
  });
  canvas.addEventListener("pointerleave", hideReachTooltip);
  canvas.addEventListener("click", (e) => {
    const dx = e.clientX - state.reachPointerDown.x;
    const dy = e.clientY - state.reachPointerDown.y;
    if (dx * dx + dy * dy > 36) return;
    const pt = pickReachPoint(e.clientX, e.clientY);
    if (!pt) return;
    const text = formatReachCoords(pt).copy;
    const copied = () => {
      showReachTooltip(e.clientX, e.clientY, pt, true);
      clearTimeout(state.reachCopiedTimer);
      state.reachCopiedTimer = setTimeout(() => {
        if (state.reachHoverPt) showReachTooltip(state.reachLastPointer.x, state.reachLastPointer.y, state.reachHoverPt);
        else hideReachTooltip();
      }, 1200);
    };
    fillReachGoInput(text);
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(copied).catch(() => {});
    } else {
      copied();
    }
  });
  updateReachViewHint();
  initReachGo();
  const reachZSlider = document.getElementById("reachZSlider");
  if (reachZSlider) {
    reachZSlider.addEventListener("input", () => {
      state.reachFilterZ = Number(reachZSlider.value);
      const valEl = document.getElementById("reachZVal");
      if (valEl) valEl.textContent = state.reachFilterZ;
      applyReachZFilter();
    });
  }
}
