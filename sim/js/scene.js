/** Three.js scene — ground, arm meshes, toolbar wiring. */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { state } from "./state.js";
import { makeLabel, setCss2dVisible } from "./three-utils.js";
import { initAngleViz } from "./angles.js";
import { initReachPicker, applyReachZFilter, hideReachTooltip, updateReachViewHint } from "./reach.js";
import { initPitchViz } from "./pitch-viz.js";

import { buildUI, refreshPanelTitle } from "./ui.js";
import { update as armUpdate } from "./arm-update.js";

export function setLengthLabelsVisible(visible) {
  state.showLengthLabels = visible;
  for (const lbl of state.lengthLabels) setCss2dVisible(lbl, visible);
}

function trackLengthLabel(label) {
  state.lengthLabels.push(label);
  setCss2dVisible(label, state.showLengthLabels);
  return label;
}

export function buildScene() {
  state.lengthLabels = [];
  const VIS = state.CONFIG.visual;
  if (!VIS?.motor) {
    throw new Error("robot.yaml visual section is required (sim.visual in config/default.yaml)");
  }

const app = document.getElementById("app");
state.renderer = new THREE.WebGLRenderer({ antialias: true });
state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
state.renderer.setSize(window.innerWidth, window.innerHeight);
state.renderer.shadowMap.enabled = true;
state.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
app.appendChild(state.renderer.domElement);

state.labelRenderer = new CSS2DRenderer();
state.labelRenderer.setSize(window.innerWidth, window.innerHeight);
state.labelRenderer.domElement.style.position = "absolute";
state.labelRenderer.domElement.style.top = "0";
state.labelRenderer.domElement.style.pointerEvents = "none";
app.appendChild(state.labelRenderer.domElement);

state.scene = new THREE.Scene();
state.scene.background = new THREE.Color(0x0a0e16);
state.scene.fog = new THREE.Fog(0x0a0e16, 900, 2200);

state.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 5000);
state.camera.position.set(420, 360, 560);

state.controls = new OrbitControls(state.camera, state.renderer.domElement);
state.controls.enableDamping = true;
state.controls.dampingFactor = 0.08;
state.controls.target.set(60, 130, 0);
state.controls.minDistance = 200;
state.controls.maxDistance = 1600;

// Lighting
state.scene.add(new THREE.HemisphereLight(0x9fb4cc, 0x10141c, 0.55));
const ambient = new THREE.AmbientLight(0x4a5360, 0.5);
state.scene.add(ambient);
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(300, 600, 400);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 100;
key.shadow.camera.far = 1600;
key.shadow.camera.left = -500;
key.shadow.camera.right = 500;
key.shadow.camera.top = 500;
key.shadow.camera.bottom = -500;
key.shadow.bias = -0.0004;
state.scene.add(key);
const rim = new THREE.DirectionalLight(0x6f9fff, 0.4);
rim.position.set(-300, 200, -300);
state.scene.add(rim);

// Ground
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(4000, 4000),
  new THREE.MeshStandardMaterial({ color: 0x0e131c, roughness: 0.95, metalness: 0.0 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
state.scene.add(ground);

// Grid: 500 mm span, 50 mm per cell (1 world unit = 1 mm)
state.grid = new THREE.GridHelper(500, 10, 0x223049, 0x161d28);
state.grid.position.y = 0.2;
state.scene.add(state.grid);

// Ground rulers every 50 mm
for (let i = -200; i <= 400; i += 50) {
  if (i === 0) continue;
  const tick = new THREE.Mesh(
    new THREE.BoxGeometry(i % 100 === 0 ? 2 : 1, 0.5, 6),
    new THREE.MeshBasicMaterial({ color: 0x3a4a5c })
  );
  tick.position.set(i, 0.4, 0);
  state.scene.add(tick);
  if (i % 100 === 0) {
    const lbl = makeLabel(i + " " + state.UNIT, "axis-label");
    lbl.position.set(i, 8, 14);
    state.scene.add(lbl);
  }
}
for (let i = 50; i <= 400; i += 50) {
  const tick = new THREE.Mesh(
    new THREE.BoxGeometry(6, 0.5, i % 100 === 0 ? 2 : 1),
    new THREE.MeshBasicMaterial({ color: 0x3a4a5c })
  );
  tick.position.set(0, 0.4, i);
  state.scene.add(tick);
}

// Coordinate axes at origin (1 unit = 1 mm)
const axes = new THREE.Group();
const axLen = 120;
const mkArrow = (dir, color, label) => {
  const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), axLen, color, 14, 8);
  axes.add(arrow);
  const lbl = makeLabel(label, "axis-label");
  lbl.position.copy(dir.clone().multiplyScalar(axLen + 18));
  axes.add(lbl);
};
mkArrow(new THREE.Vector3(1, 0, 0), 0xe85d5d, "+X forward (" + state.UNIT + ")");
mkArrow(new THREE.Vector3(0, 1, 0), 0x5de88a, "+Z up (" + state.UNIT + ")");
mkArrow(new THREE.Vector3(0, 0, 1), 0x5d9ae8, "+Y left (" + state.UNIT + ")");
state.scene.add(axes);

// Origin marker
const originDot = new THREE.Mesh(
  new THREE.SphereGeometry(4, 12, 12),
  new THREE.MeshBasicMaterial({ color: 0xf0a040 })
);
originDot.position.y = 0.5;
state.scene.add(originDot);
const originLbl = makeLabel("origin (0,0,0)", "axis-label");
originLbl.position.set(0, 14, 0);
state.scene.add(originLbl);

// Dimension lines from origin to tip (updated each frame)
const dimMat = new THREE.LineDashedMaterial({ color: 0x4ea1ff, dashSize: 8, gapSize: 5 });
state.dimX = new THREE.Line(new THREE.BufferGeometry(), dimMat);
state.dimY = new THREE.Line(new THREE.BufferGeometry(), dimMat);
state.dimZ = new THREE.Line(new THREE.BufferGeometry(), dimMat);
state.scene.add(state.dimX, state.dimY, state.dimZ);
state.tipCoordLabel = makeLabel("", "coord-label");
state.scene.add(state.tipCoordLabel);

const pitchArrowMat = new THREE.LineBasicMaterial({ color: 0xffc266, linewidth: 2 });
state.pitchArrow = new THREE.Line(new THREE.BufferGeometry(), pitchArrowMat);
state.pitchArrow.renderOrder = 2;
state.scene.add(state.pitchArrow);

// ---------------------------------------------------------------------------
// Arm meshes — a kinematic chain of nested groups.
// Convention: build in X (forward) / Y (up) plane; rotate about Z to lift,
// rotate whole base group about Y for azimuth. Matches the planar FK above.
// ---------------------------------------------------------------------------
const L = state.CONFIG.lengths;
const metal = (color, m = 0.55, r = 0.45) =>
  new THREE.MeshStandardMaterial({ color, metalness: m, roughness: r });
const matLink = metal(0xb7c0cb);
const matDark = metal(0x2a2f38, 0.6, 0.5);
const matBase = metal(0x363c46, 0.5, 0.6);
const matGrip = metal(0xc7d0da, 0.5, 0.4);
const matAccent = metal(0x4ea1ff, 0.3, 0.5);

function addShadow(mesh) { mesh.castShadow = true; mesh.receiveShadow = true; return mesh; }

function beamMesh(length, mat) {
  const m = VIS.motor;
  const beamH = m * 0.75;
  const beam = addShadow(new THREE.Mesh(new THREE.BoxGeometry(length, beamH, beamH * 0.7), mat));
  beam.position.x = length / 2;
  return beam;
}

// Link along +X with servo block at the joint pivot.
function makeLink(length, mat) {
  const g = new THREE.Group();
  g.add(beamMesh(length, mat));
  const m = VIS.motor;
  g.add(addShadow(new THREE.Mesh(new THREE.BoxGeometry(m, m, m * 0.85), matDark)));
  return g;
}

function makeBeam(length, mat) {
  const g = new THREE.Group();
  g.add(beamMesh(length, mat));
  return g;
}

function makeMotorBlock(mat) {
  const m = VIS.motor;
  return addShadow(new THREE.Mesh(new THREE.BoxGeometry(m, m, m * 0.85), mat));
}

state.baseGroup = new THREE.Group();
state.scene.add(state.baseGroup);

// Base boxes: bottom 85×85×30 mm + top 55×55×70 mm = 100 mm to shoulder
const baseBottom = addShadow(new THREE.Mesh(
  new THREE.BoxGeometry(VIS.baseBottomW, VIS.baseBottomH, VIS.baseBottomW),
  matBase
));
baseBottom.position.y = VIS.baseBottomH / 2;
state.baseGroup.add(baseBottom);
const baseTop = addShadow(new THREE.Mesh(
  new THREE.BoxGeometry(VIS.baseTopW, VIS.baseTopH, VIS.baseTopW),
  matDark
));
baseTop.position.y = VIS.baseBottomH + VIS.baseTopH / 2;
state.baseGroup.add(baseTop);
const baseHLabel = trackLengthLabel(makeLabel("shoulder " + L.shoulderHeight + " " + state.UNIT));
baseHLabel.position.set(VIS.baseBottomW / 2 + 8, L.shoulderHeight / 2, 0);
state.baseGroup.add(baseHLabel);

// Turntable: only the arm rotates with the base joint; the pedestal stays fixed.
state.armRotateGroup = new THREE.Group();
state.armRotateGroup.position.y = L.shoulderHeight;
state.baseGroup.add(state.armRotateGroup);

// Shoulder pivot at top of base column
state.shoulderGroup = new THREE.Group();
state.armRotateGroup.add(state.shoulderGroup);
state.shoulderGroup.add(makeLink(L.upperArm, matLink));
const upperArmLabel = trackLengthLabel(makeLabel("upper arm " + L.upperArm + " " + state.UNIT));
upperArmLabel.position.set(L.upperArm / 2, VIS.motor * 0.6, 0);
state.shoulderGroup.add(upperArmLabel);

state.elbowGroup = new THREE.Group();
state.elbowGroup.position.x = L.upperArm;
state.shoulderGroup.add(state.elbowGroup);
state.elbowGroup.add(makeLink(L.forearm, matLink));
const forearmLabel = trackLengthLabel(makeLabel("forearm " + L.forearm + " " + state.UNIT));
forearmLabel.position.set(L.forearm / 2, VIS.motor * 0.55, 0);
state.elbowGroup.add(forearmLabel);

state.wristGroup = new THREE.Group();
state.wristGroup.position.x = L.forearm;
state.elbowGroup.add(state.wristGroup);
// Wrist pitch motor at forearm end (hand segment starts after wrist_rot stack).
const wristPitchMotor = makeLink(Math.min(18, L.forearm * 0.12), matLink);
state.wristGroup.add(wristPitchMotor);
const wristPitchLabel = trackLengthLabel(makeLabel("wrist pitch"));
wristPitchLabel.position.set(0, VIS.motor * 0.8, 0);
state.wristGroup.add(wristPitchLabel);

state.wristRotGroup = new THREE.Group();
state.wristRotGroup.position.y = L.wristRotOffset;
state.wristGroup.add(state.wristRotGroup);
const wristRotMotor = makeLink(Math.min(14, L.wristRotOffset * 0.25), matDark);
state.wristRotGroup.add(wristRotMotor);
const wristRotLabel = trackLengthLabel(makeLabel("wrist_roll +" + L.wristRotOffset + " " + state.UNIT));
wristRotLabel.position.set(0, VIS.motor * 0.5, 0);
state.wristRotGroup.add(wristRotLabel);

// hand: beam to motor, big box at gripper_motor, jaws run motor → tip (robot.yaml).
const gripAt = Math.max(0, Math.min(L.gripperMotor, L.hand));
const armLen = Math.max(0, L.hand - gripAt);
const gOff = L.gripperOffset || 0;
const handGroup = new THREE.Group();
if (gripAt > 0) {
  handGroup.add(makeBeam(gripAt, matLink));
}
// Whole gripper (motor + jaws + tip) shares gripper_offset — stays one unit.
state.gripperGroup = new THREE.Group();
state.gripperGroup.position.set(gripAt, 0, gOff);
handGroup.add(state.gripperGroup);
const motorSize = VIS.motor * 1.8;
state.gripperGroup.add(addShadow(new THREE.Mesh(
  new THREE.BoxGeometry(motorSize, motorSize, motorSize * 0.85), matAccent
)));
const gripperMotorLabel = trackLengthLabel(makeLabel("gripper @" + gripAt + " " + state.UNIT));
gripperMotorLabel.position.set(0, motorSize * 0.55, 0);
state.gripperGroup.add(gripperMotorLabel);
if (armLen > 0) {
  state.fingerRestZ = VIS.motor * 0.55;
  state.fingerL = addShadow(new THREE.Mesh(
    new THREE.BoxGeometry(armLen, VIS.motor * 0.5, VIS.motor * 0.45), matGrip
  ));
  state.fingerR = addShadow(new THREE.Mesh(
    new THREE.BoxGeometry(armLen, VIS.motor * 0.5, VIS.motor * 0.45), matGrip
  ));
  state.fingerL.position.set(armLen / 2, 0, state.fingerRestZ);
  state.fingerR.position.set(armLen / 2, 0, -state.fingerRestZ);
  state.gripperGroup.add(state.fingerL, state.fingerR);
}
state.tipMarker = new THREE.Mesh(new THREE.SphereGeometry(6, 16, 16), matAccent);
state.tipMarker.position.set(armLen, 0, 0);
state.gripperGroup.add(state.tipMarker);
if (gOff !== 0) {
  const offMat = new THREE.LineDashedMaterial({ color: 0x4ea1ff, dashSize: 4, gapSize: 3 });
  const offLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(gripAt, 0, 0),
      new THREE.Vector3(gripAt, 0, gOff),
    ]),
    offMat
  );
  offLine.computeLineDistances();
  handGroup.add(offLine);
  const offLabel = trackLengthLabel(makeLabel("offset " + gOff + " " + state.UNIT));
  offLabel.position.set(gripAt + 6, 8, gOff / 2);
  handGroup.add(offLabel);
}
const handLabel = trackLengthLabel(makeLabel("hand " + L.hand + " " + state.UNIT));
handLabel.position.set(L.hand / 2, VIS.motor * 0.45, 0);
handGroup.add(handLabel);
state.wristRotGroup.add(handGroup);

initAngleViz();

  state.homeCam = state.camera.position.clone();
  document.getElementById("resetView").addEventListener("click", () => {
    state.camera.position.copy(state.homeCam);
    state.controls.target.set(60, 130, 0);
  });
  document.getElementById("toggleGrid").addEventListener("click", () => {
    state.grid.visible = !state.grid.visible;
  });
  document.getElementById("toggleAngles").addEventListener("click", () => {
    state.showAngles = !state.showAngles;
    document.getElementById("toggleAngles").classList.toggle("active", state.showAngles);
    armUpdate();
  });
  document.getElementById("toggleFreeSwing").addEventListener("click", () => {
    state.freeSwing = !state.freeSwing;
    document.getElementById("toggleFreeSwing").classList.toggle("active", state.freeSwing);
    refreshPanelTitle();
    buildUI();
    armUpdate();
  });
  document.getElementById("toggleCalLab").addEventListener("click", () => {
    state.calLabOpen = !state.calLabOpen;
    document.getElementById("calPanel").classList.toggle("open", state.calLabOpen);
    document.getElementById("toggleCalLab").classList.toggle("active", state.calLabOpen);
  });
  const lengthsBtn = document.getElementById("toggleLengths");
  const onToggleLengths = () => {
    setLengthLabelsVisible(!state.showLengthLabels);
    lengthsBtn.classList.toggle("active", state.showLengthLabels);
  };
  lengthsBtn.closest(".toolbar-row")?.addEventListener("click", onToggleLengths);
  document.getElementById("toggleReach").addEventListener("click", () => {
    state.showReach = !state.showReach;
    if (state.reachCloud) state.reachCloud.visible = state.showReach;
    if (state.reachFloor) state.reachFloor.visible = state.showReach;
    document.getElementById("toggleReach").classList.toggle("reach-on", state.showReach);
    if (state.showReach) applyReachZFilter();
    else {
      if (state.reachBounds) state.reachBounds.visible = false;
      hideReachTooltip();
    }
    updateReachViewHint();
  });
  initReachPicker();
  initPitchViz();
  window.addEventListener("resize", onResize);
}

export function onResize() {
  state.camera.aspect = window.innerWidth / window.innerHeight;
  state.camera.updateProjectionMatrix();
  state.renderer.setSize(window.innerWidth, window.innerHeight);
  state.labelRenderer.setSize(window.innerWidth, window.innerHeight);
}
