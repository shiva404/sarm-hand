/** Apply joint values to the 3D arm and refresh gripper readout. */

import * as THREE from "three";
import { state } from "./state.js";
import { toKin, rad, fk } from "./kinematics.js";
import { setLine } from "./three-utils.js";
import { refreshKinReadouts, refreshGripperReadout } from "./ui.js";
import { refreshCalLab } from "./cal-lab.js";
import { updateAngleViz } from "./angles.js";
import { updatePitchViz } from "./pitch-viz.js";

export function update() {
  if (!state.armRotateGroup || !state.renderer || !state.scene) return;

  const M = state.CONFIG.maps;
  const az = rad(toKin(state.joints.shoulder_pan, M.shoulder_pan));
  const q1 = rad(toKin(state.joints.shoulder_lift, M.shoulder_lift));
  const q2 = rad(toKin(state.joints.elbow_flex, M.elbow_flex));
  const q3 = rad(toKin(state.joints.wrist_flex, M.wrist_flex));
  const roll = rad(toKin(state.joints.wrist_roll, M.wrist_roll));

  state.armRotateGroup.rotation.y = -az;
  state.shoulderGroup.rotation.z = q1;
  state.elbowGroup.rotation.z = q2;
  state.wristGroup.rotation.z = q3;
  state.wristRotGroup.rotation.x = roll;

  if (state.fingerL && state.fingerR) {
    const gl = state.CONFIG.limits.gripper;
    const open = 1 - (state.joints.gripper - gl[0]) / (gl[1] - gl[0]);
    const sep = state.fingerRestZ + open * (state.fingerRestZ * 1.35);
    state.fingerL.position.z = sep;
    state.fingerR.position.z = -sep;
  }

  const f = fk();
  document.getElementById("rx").textContent = f.x.toFixed(0);
  document.getElementById("ry").textContent = f.y.toFixed(0);
  document.getElementById("rz").textContent = f.z.toFixed(0);
  document.getElementById("reach").textContent = Math.hypot(f.x, f.y, f.z).toFixed(0);
  updatePitchViz(f.pitchDeg);

  if (!state.tipWorld) state.tipWorld = new THREE.Vector3();
  if (!state.wristWorld) state.wristWorld = new THREE.Vector3();
  state.tipMarker.getWorldPosition(state.tipWorld);
  if (state.wristRotGroup) {
    state.wristRotGroup.getWorldPosition(state.wristWorld);
    if (state.pitchArrow) {
      setLine(state.pitchArrow, [state.wristWorld, state.tipWorld]);
    }
  }
  const O = new THREE.Vector3(0, 0, 0);
  const onFloor = new THREE.Vector3(state.tipWorld.x, 0, state.tipWorld.z);
  setLine(state.dimX, [O, new THREE.Vector3(state.tipWorld.x, 0, 0)]);
  setLine(state.dimY, [new THREE.Vector3(state.tipWorld.x, 0, 0), onFloor]);
  setLine(state.dimZ, [onFloor, state.tipWorld]);
  state.tipCoordLabel.position.copy(state.tipWorld).add(new THREE.Vector3(0, 18, 0));
  state.tipCoordLabel.element.textContent =
    "(" + f.x.toFixed(0) + ", " + f.y.toFixed(0) + ", " + f.z.toFixed(0) + ") " + state.UNIT;

  refreshKinReadouts(M);
  refreshGripperReadout();
  refreshCalLab();
  updateAngleViz(q1, q2, q3, M);
}
