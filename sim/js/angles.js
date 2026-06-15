/** Orange arc overlays for shoulder_lift / elbow_flex / wrist_flex kinematic angles. */

import * as THREE from "three";
import { state } from "./state.js";
import { deg } from "./kinematics.js";
import { makeLabel, setCss2dVisible, setLine, arcPoints, mkAngleLine } from "./three-utils.js";

const ANGLE_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"];

export function initAngleViz() {
  state.angleGroup = new THREE.Group();
  state.armRotateGroup.add(state.angleGroup);
  state.angleLines = {};
  state.angleLabels = {};
  for (const j of ANGLE_JOINTS) {
    state.angleLines[j] = {
      ref: mkAngleLine(0x6a7a8a, true),
      arm: mkAngleLine(0x4ea1ff),
      arc: mkAngleLine(0xf0a040),
    };
    for (const k of ["ref", "arm", "arc"]) state.angleGroup.add(state.angleLines[j][k]);
    state.angleLabels[j] = makeLabel("", "angle-label");
    state.armRotateGroup.add(state.angleLabels[j]);
  }
}

export function updateAngleViz(q1, q2, q3, M) {
  if (!state.angleGroup) return;
  state.angleGroup.visible = state.showAngles;
  for (const lbl of Object.values(state.angleLabels)) setCss2dVisible(lbl, state.showAngles);
  if (!state.showAngles) return;

  const L = state.CONFIG.lengths;
  const LEN = 50;
  const ARC = 38;
  const z = 3;
  const sh = new THREE.Vector3(0, 0, z);

  setLine(state.angleLines.shoulder_lift.ref, [sh, new THREE.Vector3(sh.x + LEN, sh.y, z)]);
  setLine(state.angleLines.shoulder_lift.arm, [
    sh,
    new THREE.Vector3(sh.x + LEN * Math.cos(q1), sh.y + LEN * Math.sin(q1), z),
  ]);
  setLine(state.angleLines.shoulder_lift.arc, Math.abs(q1) > 0.01 ? arcPoints(sh.x, sh.y, z, ARC, 0, q1) : []);
  state.angleLabels.shoulder_lift.element.innerHTML =
    'shoulder_lift <span class="dim">above horiz</span> <b>' + deg(q1).toFixed(0) + '°</b>' +
    ' <span class="dim">(val ' + state.joints.shoulder_lift + ')</span>';
  state.angleLabels.shoulder_lift.position.set(
    sh.x + (ARC + 14) * Math.cos(q1 / 2),
    sh.y + (ARC + 14) * Math.sin(q1 / 2),
    0
  );

  const el = new THREE.Vector3(
    sh.x + L.upperArm * Math.cos(q1),
    sh.y + L.upperArm * Math.sin(q1),
    z
  );
  const uDir = new THREE.Vector3(Math.cos(q1), Math.sin(q1), 0);
  const fDir = new THREE.Vector3(Math.cos(q1 + q2), Math.sin(q1 + q2), 0);
  setLine(state.angleLines.elbow_flex.ref, [el, el.clone().add(uDir.clone().multiplyScalar(LEN))]);
  setLine(state.angleLines.elbow_flex.arm, [el, el.clone().add(fDir.clone().multiplyScalar(LEN))]);
  setLine(
    state.angleLines.elbow_flex.arc,
    Math.abs(q2) > 0.01 ? arcPoints(el.x, el.y, z, ARC * 0.9, q1, q1 + q2) : []
  );
  const elMid = q1 + q2 / 2;
  state.angleLabels.elbow_flex.element.innerHTML =
    'elbow_flex <span class="dim">bend</span> <b>' + deg(q2).toFixed(0) + '°</b>' +
    ' <span class="dim">(val ' + state.joints.elbow_flex + ', zero ' + M.elbow_flex.zero + ')</span>';
  state.angleLabels.elbow_flex.position.set(
    el.x + (ARC + 10) * Math.cos(elMid),
    el.y + (ARC + 10) * Math.sin(elMid),
    0
  );

  const wr = new THREE.Vector3(
    el.x + L.forearm * Math.cos(q1 + q2),
    el.y + L.forearm * Math.sin(q1 + q2),
    z
  );
  setLine(state.angleLines.wrist_flex.ref, [wr, wr.clone().add(fDir.clone().multiplyScalar(LEN * 0.85))]);
  const hDir = new THREE.Vector3(Math.cos(q1 + q2 + q3), Math.sin(q1 + q2 + q3), 0);
  setLine(state.angleLines.wrist_flex.arm, [wr, wr.clone().add(hDir.clone().multiplyScalar(LEN * 0.85))]);
  setLine(
    state.angleLines.wrist_flex.arc,
    Math.abs(q3) > 0.01 ? arcPoints(wr.x, wr.y, z, ARC * 0.75, q1 + q2, q1 + q2 + q3) : []
  );
  const wrMid = q1 + q2 + q3 / 2;
  state.angleLabels.wrist_flex.element.innerHTML =
    'wrist_flex <span class="dim">tilt</span> <b>' + deg(q3).toFixed(0) + '°</b>' +
    ' <span class="dim">(val ' + state.joints.wrist_flex + ', zero ' + M.wrist_flex.zero + ')</span>';
  state.angleLabels.wrist_flex.position.set(
    wr.x + (ARC + 8) * Math.cos(wrMid),
    wr.y + (ARC + 8) * Math.sin(wrMid),
    0
  );
}
