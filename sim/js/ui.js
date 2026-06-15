/** Joint sliders, pose buttons, and panel readouts. */

import { SIM_MIN, SIM_MAX, KIN_JOINTS } from "./constants.js";
import { state } from "./state.js";
import { setJointValue } from "./config.js";
import { toKin } from "./kinematics.js";

function scheduleUpdate() {
  import("./arm-update.js").then((m) => m.update());
}

export function sliderBounds(j) {
  return state.freeSwing ? [SIM_MIN, SIM_MAX] : state.CONFIG.limits[j];
}

export function valueLabel() {
  const suffix = state.CONFIG?.valueSuffix ?? "";
  return suffix ? " " + suffix : "";
}

export function refreshPanelTitle() {
  const el = document.getElementById("sliderTitle");
  if (!el) return;
  el.textContent = state.freeSwing
    ? "Joint values (free −180…180)"
    : "Joint values (config limits)";
}

export function refreshKinReadouts(M) {
  for (const j of KIN_JOINTS) {
    const kinEl = document.getElementById("k_" + j);
    const jc = state.CONFIG.joints[j];
    if (!jc || state.joints[j] == null) continue;
    if (kinEl && M[j]) kinEl.textContent = "→ kin " + toKin(state.joints[j], M[j]).toFixed(0) + "°";
  }
}

export function refreshGripperReadout() {
  const el = document.getElementById("p_gripper");
  if (!el || state.joints.gripper == null) return;
  el.textContent = "open " + Math.round(state.joints.gripper) + valueLabel();
}

export function refreshJointUI(j) {
  const [limMin, limMax] = state.CONFIG.limits[j];
  const v = state.joints[j];
  const inRange = v >= limMin && v <= limMax;
  const wrap = document.getElementById("joint_" + j);
  if (wrap) wrap.classList.toggle("out-of-range", state.freeSwing && !inRange);
  const limEl = document.getElementById("lim_" + j);
  if (limEl) {
    if (state.freeSwing) {
      limEl.textContent = "config limit " + limMin + "–" + limMax + (inRange ? " ✓" : " (outside)");
      limEl.className = "limits" + (inRange ? " in-range" : "");
    } else {
      limEl.textContent = "slider = config limit " + limMin + "–" + limMax;
      limEl.className = "limits in-range";
    }
  }
}

export function setActivePose(name) {
  state.activePose = name;
  document.querySelectorAll("#poses button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.pose === name);
  });
}

export function applyPose(name) {
  const p = state.CONFIG.poses[name];
  if (!p) return;
  state.JOINTS.forEach((j) => {
    if (p[j] == null) return;
    const clamped = setJointValue(j, p[j]);
    const s = document.getElementById("s_" + j);
    if (s) s.value = clamped;
    if (state.valEls[j]) state.valEls[j].textContent = clamped;
    refreshJointUI(j);
  });
  setActivePose(name);
  scheduleUpdate();
}

export function buildUI() {
  const slidersEl = document.getElementById("sliders");
  slidersEl.innerHTML = "";
  document.getElementById("poses").innerHTML = "";
  state.valEls = {};
  refreshPanelTitle();
  const suffix = valueLabel();
  state.JOINTS.forEach((j) => {
    if (state.joints[j] == null) state.joints[j] = state.CONFIG.homeValues[j] ?? 0;
    const [sMin, sMax] = sliderBounds(j);
    const v = Math.max(sMin, Math.min(sMax, state.joints[j]));
    state.joints[j] = v;
    const kinPart = KIN_JOINTS.has(j) ? ' <span class="kin" id="k_' + j + '"></span>' : "";
    const extraPart = j === "gripper" ? '<div class="limits pulse" id="p_gripper"></div>' : "";
    const wrap = document.createElement("div");
    wrap.className = "joint";
    wrap.id = "joint_" + j;
    wrap.innerHTML =
      '<div class="row"><span class="name"><span class="dot"></span>' + j + '</span>' +
      '<span class="val"><b id="v_' + j + '">' + state.joints[j] + '</b>' + suffix + '</span></div>' +
      '<input type="range" min="' + sMin + '" max="' + sMax + '" step="1" value="' + state.joints[j] + '" id="s_' + j + '">' +
      '<div class="limits" id="lim_' + j + '"></div>' + kinPart + extraPart;
    slidersEl.appendChild(wrap);
    state.valEls[j] = document.getElementById("v_" + j);
    document.getElementById("s_" + j).addEventListener("input", (e) => {
      const clamped = setJointValue(j, Number(e.target.value));
      document.getElementById("s_" + j).value = clamped;
      state.valEls[j].textContent = clamped;
      refreshJointUI(j);
      scheduleUpdate();
    });
    refreshJointUI(j);
  });

  const posesEl = document.getElementById("poses");
  Object.keys(state.CONFIG.poses).forEach((name) => {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.pose = name;
    b.textContent = name;
    b.classList.toggle("active", name === state.activePose);
    b.addEventListener("click", () => applyPose(name));
    posesEl.appendChild(b);
  });
}
