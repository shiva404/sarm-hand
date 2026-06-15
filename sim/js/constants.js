/** Shared constants — paths, joint sets. All numeric values come from robot.yaml. */

export const YAML_PATH = "/robot.yaml";
export const API_BASE = "";

export const GEOMETRY_SCALAR_KEYS = [
  "units", "shoulder_height", "upper_arm", "forearm", "wrist_rot_offset", "hand",
  "gripper_offset", "gripper_motor", "elbow",
];
export const GEOMETRY_JOINT_KEYS = [
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
];
export const KIN_JOINTS = new Set(GEOMETRY_JOINT_KEYS);
export const CAL_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"];

export const SIM_MIN = -180;
export const SIM_MAX = 180;
