/** Hand pitch diagram, IK mode (auto / manual / wrist), and live readout. */

import { state } from "./state.js";

const PIVOT = { x: 28, y: 36 };
const ARM_LEN = 44;

function pitchToSvgRad(pitchDeg) {
  return (-pitchDeg * Math.PI) / 180;
}

function handLine(pitchDeg, len = ARM_LEN) {
  const a = pitchToSvgRad(pitchDeg);
  return {
    x1: PIVOT.x,
    y1: PIVOT.y,
    x2: PIVOT.x + len * Math.cos(a),
    y2: PIVOT.y + len * Math.sin(a),
  };
}

function pitchLabel(pitchDeg) {
  if (pitchDeg >= -5 && pitchDeg <= 5) return `${Math.round(pitchDeg)}° level`;
  if (pitchDeg <= -75) return `${Math.round(pitchDeg)}° straight down`;
  if (pitchDeg < 0) return `${Math.round(pitchDeg)}° angled down`;
  return `${Math.round(pitchDeg)}° angled up`;
}

export function getPitchMode() {
  return document.querySelector('input[name="pitchMode"]:checked')?.value ?? "auto";
}

/** Pitch for /api/ik: number, null (wrist target), or undefined (auto-suggest). */
export function resolvePitchForIk() {
  const mode = getPitchMode();
  const pitchEl = document.getElementById("reachGoPitch");
  const raw = pitchEl?.value?.trim() ?? "";

  if (mode === "wrist") return { pitch: null, wristMode: true };
  if (mode === "manual") {
    if (raw === "") return { error: "set pitch ° (or switch to auto)" };
    const n = Number(raw);
    if (!Number.isFinite(n)) return { error: "pitch must be a number" };
    return { pitch: n, wristMode: false };
  }
  // auto — tip target; blank field triggers suggest-pitch on Go
  if (raw === "") return { pitch: undefined, auto: true, wristMode: false };
  const n = Number(raw);
  if (!Number.isFinite(n)) return { error: "pitch must be a number" };
  return { pitch: n, wristMode: false };
}

function setNeedle(id, pitchDeg, visible = true) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!visible || pitchDeg == null || !Number.isFinite(pitchDeg)) {
    el.style.display = "none";
    return;
  }
  const ln = handLine(pitchDeg, id === "pitchTarget" ? ARM_LEN + 6 : ARM_LEN + 2);
  el.setAttribute("x1", ln.x1);
  el.setAttribute("y1", ln.y1);
  el.setAttribute("x2", ln.x2);
  el.setAttribute("y2", ln.y2);
  el.style.display = "";
}

function highlightPresets(activeDeg) {
  document.querySelectorAll(".pitch-preset").forEach((g) => {
    const deg = Number(g.dataset.deg);
    const on = activeDeg != null && Math.abs(activeDeg - deg) < 8;
    g.classList.toggle("active", on);
  });
}

export function updatePitchViz(livePitchDeg) {
  const liveEl = document.getElementById("pitchLiveDesc");
  const rpEl = document.getElementById("rp");
  if (liveEl && livePitchDeg != null) {
    liveEl.textContent = "live: " + pitchLabel(livePitchDeg);
  }
  if (rpEl && livePitchDeg != null) {
    rpEl.textContent = Math.round(livePitchDeg);
  }
  setNeedle("pitchNeedle", livePitchDeg, true);
  highlightPresets(livePitchDeg);
  updateTargetPitchPreview();
}

function readTargetPitchDeg() {
  const mode = getPitchMode();
  if (mode === "wrist") return null;
  const raw = document.getElementById("reachGoPitch")?.value?.trim() ?? "";
  if (raw === "") return mode === "auto" ? undefined : null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function updateTargetPitchPreview() {
  const mode = getPitchMode();
  const target = readTargetPitchDeg();
  const show = mode !== "wrist" && target != null && Number.isFinite(target);
  setNeedle("pitchTarget", show ? target : null, show);
}

function onPitchModeChange() {
  const mode = getPitchMode();
  state.pitchMode = mode;
  const pitchEl = document.getElementById("reachGoPitch");
  const hint = document.getElementById("pitchHint");
  const fieldLbl = document.getElementById("pitchFieldLabel");
  const goStatus = document.getElementById("reachGoStatus");

  if (fieldLbl) {
    fieldLbl.textContent =
      mode === "manual" ? "pitch ° (required)" : mode === "wrist" ? "pitch ° (off)" : "pitch ° (optional)";
  }
  if (mode === "wrist") {
    pitchEl.disabled = true;
    pitchEl.value = "";
    pitchEl.placeholder = "—";
    if (hint) {
      hint.textContent =
        "Wrist mode: x y z is the wrist joint, not the gripper tip. Pitch is not solved — hand keeps its current angle.";
    }
    if (goStatus && !goStatus.classList.contains("ok") && !goStatus.classList.contains("err")) {
      goStatus.textContent = "wrist target · reach cloud shows tips, not wrist points";
    }
  } else if (mode === "manual") {
    pitchEl.disabled = false;
    pitchEl.placeholder = "-30";
    if (hint) {
      hint.textContent =
        "Tip + pitch: x y z is where the gripper tip must land. Pitch sets hand direction (0° level, -90° down).";
    }
  } else {
    pitchEl.disabled = false;
    pitchEl.placeholder = "auto";
    if (hint) {
      hint.textContent =
        "Tip + auto pitch: x y z is gripper tip. Leave pitch blank — Go picks the best angle (often -90° near table height).";
    }
  }
  updateTargetPitchPreview();
}

export function initPitchViz() {
  document.querySelectorAll('input[name="pitchMode"]').forEach((r) => {
    r.addEventListener("change", onPitchModeChange);
  });
  document.getElementById("reachGoPitch")?.addEventListener("input", updateTargetPitchPreview);
  document.querySelectorAll(".pitch-preset").forEach((g) => {
    g.addEventListener("click", () => {
      const deg = g.dataset.deg;
      const pitchEl = document.getElementById("reachGoPitch");
      const manual = document.querySelector('input[name="pitchMode"][value="manual"]');
      const auto = document.querySelector('input[name="pitchMode"][value="auto"]');
      if (manual) manual.checked = true;
      if (pitchEl) {
        pitchEl.disabled = false;
        pitchEl.value = deg;
      }
      onPitchModeChange();
      updateTargetPitchPreview();
    });
  });
  onPitchModeChange();
}
