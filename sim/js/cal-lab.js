/** Angle mapping lab — limits vs zero/sign tuning. */

import { CAL_JOINTS } from "./constants.js";
import { state } from "./state.js";
import { clampAngle, setJointValue } from "./config.js";
import { toKin } from "./kinematics.js";
import { update } from "./arm-update.js";
import { buildUI, refreshJointUI } from "./ui.js";

export function applyJointLimits(j, min, max) {
  min = Math.round(min);
  max = Math.round(max);
  if (min >= max) return;
  state.CONFIG.joints[j].min = min;
  state.CONFIG.joints[j].max = max;
  state.CONFIG.limits[j] = [min, max];
  const s = document.getElementById("s_" + j);
  if (s && !state.freeSwing) {
    s.min = min;
    s.max = max;
  }
  const clamped = setJointValue(j, state.joints[j]);
  if (s) s.value = clamped;
  if (state.valEls[j]) state.valEls[j].textContent = clamped;
  refreshJointUI(j);
  update();
}

export function applyJointMap(j, zero, sign) {
  state.CONFIG.maps[j].zero = Math.round(zero);
  state.CONFIG.maps[j].sign = sign >= 0 ? 1 : -1;
  update();
}

function resetCalJoint(j) {
  const snap = state.CONFIG.yamlSnapshot;
  if (!snap || !snap.maps[j]) return;
  const [lo, hi] = snap.limits[j];
  state.CONFIG.joints[j].min = snap.joints[j].min;
  state.CONFIG.joints[j].max = snap.joints[j].max;
  state.CONFIG.limits[j] = [lo, hi];
  state.CONFIG.maps[j].zero = snap.maps[j].zero;
  state.CONFIG.maps[j].sign = snap.maps[j].sign;
}

export function resetCalAll() {
  CAL_JOINTS.forEach(resetCalJoint);
  buildCalControls();
  if (!state.freeSwing) buildUI();
  else update();
}

export function buildCalControls() {
  const j = state.calJoint;
  const jc = state.CONFIG.joints[j];
  const mp = state.CONFIG.maps[j];
  const el = document.getElementById("calControls");
  el.innerHTML =
    '<div class="cal-row"><div class="lbl"><span>joints.min</span><b id="cv_min">' + jc.min + '</b></div>' +
    '<input type="range" id="cal_min" min="-180" max="180" step="1" value="' + jc.min + '">' +
    '<div class="hint">Travel limit — does not change 3D bend by itself.</div></div>' +
    '<div class="cal-row"><div class="lbl"><span>joints.max</span><b id="cv_max">' + jc.max + '</b></div>' +
    '<input type="range" id="cal_max" min="-180" max="180" step="1" value="' + jc.max + '"></div>' +
    '<div class="cal-row"><div class="lbl"><span>geometry.zero</span><b id="cv_zero">' + mp.zero + '</b></div>' +
    '<input type="range" id="cal_zero" min="-180" max="180" step="1" value="' + mp.zero + '">' +
    '<div class="hint">Start offset — <b>this</b> rotates the link in the 3D world.</div></div>' +
    '<div class="cal-row"><div class="lbl"><span>geometry.sign</span><b id="cv_sign">' + mp.sign + '</b></div>' +
    '<div class="sign-btns"><button type="button" data-sign="1" class="' + (mp.sign === 1 ? "active" : "") + '">+1</button>' +
    '<button type="button" data-sign="-1" class="' + (mp.sign === -1 ? "active" : "") + '">−1</button></div>' +
    '<div class="hint">Flip if the arm bends the mirror way.</div></div>';

  document.getElementById("cal_min").addEventListener("input", (e) => {
    document.getElementById("cv_min").textContent = e.target.value;
    applyJointLimits(j, Number(e.target.value), state.CONFIG.joints[j].max);
  });
  document.getElementById("cal_max").addEventListener("input", (e) => {
    document.getElementById("cv_max").textContent = e.target.value;
    applyJointLimits(j, state.CONFIG.joints[j].min, Number(e.target.value));
  });
  document.getElementById("cal_zero").addEventListener("input", (e) => {
    document.getElementById("cv_zero").textContent = e.target.value;
    applyJointMap(j, Number(e.target.value), state.CONFIG.maps[j].sign);
  });
  el.querySelectorAll(".sign-btns button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const s = Number(btn.dataset.sign);
      el.querySelectorAll(".sign-btns button").forEach((b) => b.classList.toggle("active", b === btn));
      document.getElementById("cv_sign").textContent = s;
      applyJointMap(j, state.CONFIG.maps[j].zero, s);
    });
  });
}

export function refreshCalLab() {
  if (!state.calLabOpen || !state.CONFIG?.maps?.[state.calJoint]) return;
  const j = state.calJoint;
  const commanded = Number(document.getElementById("s_" + j)?.value ?? state.joints[j]);
  const clamped = clampAngle(commanded, state.CONFIG.joints[j]);
  const kin = toKin(clamped, state.CONFIG.maps[j]);
  const m = state.CONFIG.maps[j];
  const snap = state.CONFIG.yamlSnapshot;

  document.getElementById("calPipe").innerHTML =
    '<div class="pipe-step servo"><div class="tag">1 · commanded (slider)</div><div class="big">' + commanded + '</div></div>' +
    '<div class="pipe-arrow">↓ clamp joints.min / max</div>' +
    '<div class="pipe-step clamp"><div class="tag">2 · after clamp</div><div class="big">' + clamped + '</div>' +
    '<div class="hint">min=' + state.CONFIG.joints[j].min + ' max=' + state.CONFIG.joints[j].max +
    (commanded !== clamped ? ' <b style="color:var(--warn)">clamped!</b>' : '') + '</div></div>' +
    '<div class="pipe-arrow">↓ (value − zero) / sign</div>' +
    '<div class="pipe-step kin"><div class="tag">3 · kinematic° (FK + orange arc)</div><div class="big">' + kin.toFixed(0) + '°</div>' +
    '<div class="hint">zero=' + m.zero + ' sign=' + m.sign + '</div></div>' +
    '<div class="pipe-arrow">↓ rotates 3D link</div>' +
    '<div class="pipe-step mesh"><div class="tag">4 · physical arm in scene</div><div class="big">' + kin.toFixed(0) + '° bend</div></div>';

  if (snap) {
    const yl = snap.limits[j];
    const ym = snap.maps[j];
    const yKin = toKin(clamped, ym);
    const delta = kin - yKin;
    document.getElementById("calYamlRef").innerHTML =
      'config file: min=' + yl[0] + ' max=' + yl[1] + ', zero=' + ym.zero + ' sign=' + ym.sign +
      ' → kin <b>' + yKin.toFixed(0) + '°</b>' +
      (Math.abs(delta) > 0.5 ? ' &nbsp;|&nbsp; lab Δ <b style="color:var(--warn)">' + (delta > 0 ? '+' : '') + delta.toFixed(0) + '°</b>' : '');
  }
}

export function buildCalLab() {
  const sel = document.getElementById("calJoint");
  sel.innerHTML = "";
  CAL_JOINTS.forEach((j) => {
    const o = document.createElement("option");
    o.value = j;
    o.textContent = j;
    if (j === state.calJoint) o.selected = true;
    sel.appendChild(o);
  });
  sel.addEventListener("change", () => {
    state.calJoint = sel.value;
    buildCalControls();
    refreshCalLab();
  });
  document.getElementById("calResetJoint").addEventListener("click", () => {
    resetCalJoint(state.calJoint);
    buildCalControls();
    if (!state.freeSwing) buildUI();
    else update();
  });
  document.getElementById("calResetAll").addEventListener("click", resetCalAll);
  buildCalControls();
}
