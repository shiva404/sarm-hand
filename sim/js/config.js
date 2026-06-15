/** Load robot.yaml; parse into simulator CONFIG. */

import yaml from "js-yaml";
import {
  YAML_PATH,
  GEOMETRY_SCALAR_KEYS,
  GEOMETRY_JOINT_KEYS,
} from "./constants.js";
import { state } from "./state.js";

function requireGeometry(data) {
  const g = data.geometry;
  if (!g) throw new Error("robot.yaml must define geometry: (single source for FK/3D sim).");
  for (const k of GEOMETRY_SCALAR_KEYS) {
    if (g[k] == null) throw new Error("robot.yaml geometry." + k + " is required.");
  }
  const gj = g.joints;
  if (!gj) throw new Error("robot.yaml geometry.joints is required.");
  for (const jn of GEOMETRY_JOINT_KEYS) {
    const m = gj[jn];
    const zero = m?.zero ?? m?.zero_deg;
    if (!m || zero == null) {
      throw new Error("robot.yaml geometry.joints." + jn + " with zero is required.");
    }
  }
  return g;
}

function jointMap(gj, name) {
  const m = gj[name];
  const zero = Number(m.zero ?? m.zero_deg);
  return { zero, sign: m.sign != null ? Number(m.sign) : 1 };
}

export function clampAngle(angle, j) {
  return Math.max(j.min, Math.min(j.max, angle));
}

export function setJointValue(name, angle) {
  const j = state.CONFIG.joints[name];
  if (!j) return angle;
  const clamped = clampAngle(angle, j);
  state.joints[name] = clamped;
  return clamped;
}

function configFromYaml(data) {
  const g = requireGeometry(data);
  const gj = g.joints;
  const limits = {};
  const homeValues = data.home || {};
  const jointCfg = {};
  const simMeta = data.sim || {};
  const visual = data.visual || {};

  for (const j of data.joints || []) {
    const min = Number(j.min);
    const max = Number(j.max);
    limits[j.name] = [min, max];
    const home =
      homeValues[j.name] != null
        ? Number(homeValues[j.name])
        : Number(j.home ?? j.resting ?? (min + max) / 2);
    homeValues[j.name] = home;
    jointCfg[j.name] = {
      name: j.name,
      min,
      max,
      home,
      minPulse: Number(j.min_pulse_us ?? 500),
      maxPulse: Number(j.max_pulse_us ?? 2500),
      invert: Boolean(j.invert),
      hasCal: j.min_pulse_us != null || j.max_pulse_us != null,
    };
  }

  const defaults = { ...homeValues };
  const poses = {};
  for (const [name, angles] of Object.entries(data.poses || {})) {
    poses[name] = { ...defaults };
    for (const [jn, ang] of Object.entries(angles || {})) {
      poses[name][jn] = Number(ang);
    }
  }

  return {
    unit: String(g.units),
    valueSuffix: simMeta.value_suffix ?? "",
    brandTitle: simMeta.brand_title ?? "sarm-hand",
    brandSubtitle: simMeta.brand_subtitle ?? "3D joint simulator",
    reachSteps: simMeta.reach_steps ?? {},
    reachZMax: Number(simMeta.reach_z_max ?? 350),
    reachZTolerance: Number(simMeta.reach_z_tolerance ?? 10),
    reachGoTolMm: Number(simMeta.reach_go_tol_mm ?? 15),
    visual: {
      baseBottomW: Number(visual.base_bottom_w),
      baseBottomH: Number(visual.base_bottom_h),
      baseTopW: Number(visual.base_top_w),
      baseTopH: Number(visual.base_top_h),
      motor: Number(visual.motor),
    },
    lengths: {
      shoulderHeight: Number(g.shoulder_height),
      upperArm: Number(g.upper_arm),
      forearm: Number(g.forearm),
      wristRotOffset: Number(g.wrist_rot_offset),
      hand: Number(g.hand),
      gripperOffset: Number(g.gripper_offset),
      gripperMotor: Number(g.gripper_motor),
      elbowBranch: String(g.elbow),
    },
    maps: {
      shoulder_pan: jointMap(gj, "shoulder_pan"),
      shoulder_lift: jointMap(gj, "shoulder_lift"),
      elbow_flex: jointMap(gj, "elbow_flex"),
      wrist_flex: jointMap(gj, "wrist_flex"),
      wrist_roll: jointMap(gj, "wrist_roll"),
    },
    limits,
    poses,
    homeValues,
    joints: jointCfg,
  };
}

function snapshotConfig(cfg) {
  const snap = { limits: {}, joints: {}, maps: {} };
  for (const [k, [lo, hi]] of Object.entries(cfg.limits)) {
    snap.limits[k] = [lo, hi];
    snap.joints[k] = { min: cfg.joints[k].min, max: cfg.joints[k].max, home: cfg.joints[k].home };
  }
  for (const [k, m] of Object.entries(cfg.maps)) {
    snap.maps[k] = { zero: m.zero, sign: m.sign };
  }
  return snap;
}

export async function loadRobotConfig() {
  const res = await fetch(YAML_PATH);
  if (!res.ok) throw new Error("Could not fetch " + YAML_PATH + " (" + res.status + ")");
  const data = yaml.load(await res.text());
  const cfg = configFromYaml(data);
  cfg.yamlSnapshot = snapshotConfig(cfg);
  return cfg;
}
