/** Three.js helpers — labels, lines, arcs. */

import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { deg } from "./kinematics.js";

export function makeLabel(text, cls = "dim-label") {
  const el = document.createElement("div");
  el.className = cls;
  el.textContent = text;
  return new CSS2DObject(el);
}

/** CSS2DRenderer skips invisible objects without hiding their DOM — set display explicitly. */
export function setCss2dVisible(label, visible) {
  if (!label) return;
  label.visible = visible;
  if (label.element) label.element.style.display = visible ? "" : "none";
}

export function setLine(line, pts) {
  line.geometry.dispose();
  if (!pts.length) {
    line.geometry = new THREE.BufferGeometry();
    line.visible = false;
    return;
  }
  line.visible = true;
  line.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  if (line.material.dashSize != null) line.computeLineDistances();
}

export function arcPoints(cx, cy, cz, radius, a0, a1, segs = 22) {
  const pts = [];
  const n = Math.max(4, segs);
  for (let i = 0; i <= n; i++) {
    const t = a0 + ((a1 - a0) * i) / n;
    pts.push(new THREE.Vector3(cx + radius * Math.cos(t), cy + radius * Math.sin(t), cz));
  }
  return pts;
}

export function mkAngleLine(color, dashed = false) {
  const mat = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 6, gapSize: 4 })
    : new THREE.LineBasicMaterial({ color });
  return new THREE.Line(new THREE.BufferGeometry(), mat);
}

export { deg };
