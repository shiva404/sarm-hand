# S-ARM101 Hand

LeRobot-based project for the **S-ARM101** (SO-ARM101) 6-axis robotic arm with USB control, Meta Quest 2 teleoperation via [phospho](https://docs.phospho.ai/), dataset recording, and data access utilities.

## Features

| Feature | Description |
|---------|-------------|
| **USB arm control** | S-ARM101 via Feetech STS3215 bus (LeRobot `so101_follower`) |
| **Quest 2 teleop** | VR control through phosphobot + Meta Quest phospho app |
| **Data collection** | Leader-follower recording or Quest 2 recording (LeRobot v2/v3 format) |
| **Data access** | Inspect, sample, export, and push datasets to Hugging Face Hub |
| **LeLab UI** | Web UI for calibration, 3D teleop, datasets, training via [LeLab](https://huggingface.co/docs/lerobot/main/en/lelab) |
| **3D simulator** | Config-driven joint sim with FK/IK, reach cloud, and pose presets |
| **Genesis twin** | Optional physics sim: hardware→Genesis mirror, sim recording, sim-to-real datasets |

## Requirements

- Python 3.12+ (required by LeRobot 0.5.x)
- [uv](https://docs.astral.sh/uv/) package manager
- S-ARM101 arm with USB cable + 12V power supply
- (Optional) SO-ARM101 leader arm for leader-follower teleop
- (Optional) Meta Quest 2 + phospho teleoperation app for VR teleop
- (Optional) USB cameras for vision during recording

## Quick Start

```bash
# Install core dependencies (USB control, leader teleop, data collection)
uv sync

# For Meta Quest 2 teleoperation, also install phosphobot:
uv sync --extra quest

# For Genesis World physics sim (digital twin, sim recording):
uv sync --extra genesis
./scripts/fetch_so101_assets.sh

# Find USB port (connect power + USB first)
uv run sarm-hand find-port

# First-time motor setup (all motors connected; IDs from config/default.yaml)
uv run sarm-hand setup-motors --role follower --port /dev/ttyACM0

# Scan bus only — show joint rows vs detected servo IDs
uv run sarm-hand setup-motors --role follower --port /dev/ttyACM0 --scan

# Override current servo ID for specific joints when reprogramming
uv run sarm-hand setup-motors --role follower --port /dev/ttyACM0 \
  --initial-ids shoulder_pan=1,shoulder_lift=1

# Factory-fresh motors (one at a time, legacy LeRobot flow)
uv run sarm-hand setup-motors --role follower --port /dev/ttyACM0 --one-at-a-time

# Calibrate the arm
uv run sarm-hand calibrate --role follower --port /dev/ttyACM0

# Validate calibration + motion with preset poses
uv run sarm-hand test-poses --port /dev/ttyACM0
uv run sarm-hand test-poses --list          # show pose targets
uv run sarm-hand test-poses --pose home     # single pose

# Show current config
uv run sarm-hand config-show
```

Edit `config/default.yaml` to set default ports, cameras, and dataset settings.

### Cameras (USB or HTTP stream)

Cameras are used during dataset recording (`record-leader`, `record-policy`) and optionally during teleop (`--with-cameras`).

**USB camera** — set device index or path in `config/default.yaml`:

```yaml
cameras:
  front:
    type: opencv
    index_or_path: 0          # or /dev/video0 on Linux
    width: 640
    height: 480
    fps: 30
```

**macOS built-in / FaceTime camera** — use native resolution (`auto_resolution: true`). Forced 640×480@30 often opens but fails to capture frames:

```yaml
cameras:
  front:
    type: opencv
    index_or_path: 0
    auto_resolution: true
    width: null
    height: null
    fps: null
    warmup_s: 3
```

Run `sarm-hand list-cameras` to see the native profile (e.g. 1920×1080 @ 15 fps).

**HTTP/MJPEG or RTSP stream** — use `type: http` (or `rtsp`) with a URL. Leave resolution/fps as `null` to auto-detect from the stream:

```yaml
cameras:
  overhead:
    type: http
    url: http://192.168.1.100:8080/video
    width: null
    height: null
    fps: null
```

```bash
# List USB cameras on this machine
uv run sarm-hand list-cameras

# Preview USB index 0 or a configured camera
uv run sarm-hand camera-preview --index 0
uv run sarm-hand camera-preview --name front

# Preview an HTTP stream (save snapshot headless)
uv run sarm-hand camera-preview --url http://192.168.1.100:8080/video --no-window --output /tmp/frame.jpg

# Test all cameras in config/default.yaml
uv run sarm-hand camera-test
```

### SO-ARM101 leader settings

If you have a matching **SO-ARM101 leader arm** (LeRobot `so101_leader`) for leader-follower teleop, configure it under `teleop.leader` in `config/default.yaml`:

```yaml
teleop:
  leader:
    type: so101_leader      # LeRobot teleoperator type
    id: sarm101_leader      # calibration file id (saved under ~/.cache/huggingface/lerobot/calibration/)
    port: /dev/ttyACM1      # leader USB port (null = must pass --leader-port on CLI)

motors:
  leader:
    shoulder_pan: 1
    shoulder_lift: 2
    elbow_flex: 3
    wrist_flex: 4
    wrist_roll: 5
    gripper: 6
```

The leader uses the same Feetech STS3215 servo ID map as the follower. Run `setup-motors --role leader` on the leader port before first use. Leader and follower must each be on their **own USB port** with separate 12V power supplies.

## LeLab Web UI

[LeLab](https://huggingface.co/docs/lerobot/main/en/lelab) is Hugging Face's official LeRobot GUI — calibration, live **3D arm rendering** during teleop, dataset recording/browsing, training, and replay. It supports SO-ARM101 natively.

```bash
# One-time install (uv tool — keeps lerobot versions separate)
uv run sarm-hand lelab --install

# Launch with project dataset paths configured
uv run sarm-hand lelab

# Check integration / paths
uv run sarm-hand lelab --info
uv run sarm-hand config-show
```

LeLab opens at **http://localhost:8000** and scans datasets under `data/datasets/` (via `HF_LEROBOT_HOME`).

## 3D Joint Simulator

Browser-based SO-ARM101 simulator (similar to raspi-roboarm): live FK readout, reach cloud, IK go-to-point, pose buttons, and calibration lab. **All link lengths, joint limits, poses, and visual scale come from `config/default.yaml`** — nothing is hardcoded in the sim code.

```bash
uv run sarm-hand sim
# opens http://127.0.0.1:8763/sim/arm3d.html
```

Edit these sections in `config/default.yaml`:

- `geometry` — SO-ARM101 link lengths (from `so101_new_calib.urdf`) and per-joint `zero`/`sign` mapping
- `joints` — slider min/max (LeRobot normalized units by default)
- `poses` — preset buttons (same values as `test-poses`)
- `sim` — port, reach cloud sampling, and visual mesh scale

### Dataset visualization

```bash
# Local viewer (Rerun) for a recorded episode
uv run sarm-hand viz-dataset --repo-id local/sarm101-dataset --episode 0

# Hugging Face Space — 3D URDF robot rendering + charts (paste repo id in UI)
uv run sarm-hand viz-dataset --hub --repo-id your-username/my-dataset
```

Online visualizer: [lerobot/visualize_dataset](https://huggingface.co/spaces/lerobot/visualize_dataset)

## Teleoperation

### Leader → follower (USB)

Leader-follower teleop lets you move the **follower** arm by hand on a matching **SO-ARM101 leader** arm. Both arms connect via USB; the leader reads joint positions and the follower mirrors them in real time.

**Hardware**

- SO-ARM101 **follower** arm (motorized) — USB + 12V power
- SO-ARM101 **leader** arm (position-sensing only) — USB + 12V power on a **separate** port
- Two free USB ports on your computer (or a powered USB hub)

**Setup (first time)**

1. **Find both USB ports** — connect power and USB to each arm, then:

   ```bash
   uv run sarm-hand find-port
   ```

   Note which port is follower vs leader (e.g. `/dev/ttyACM0` and `/dev/ttyACM1`). On macOS use the `tty.usbmodem*` variant.

2. **Set ports in config** — edit `config/default.yaml`:

   ```yaml
   robot:
     port: /dev/ttyACM0          # follower

   teleop:
     leader:
       port: /dev/ttyACM1        # leader
   ```

3. **Assign servo IDs on both arms** (once per arm, all motors connected):

   ```bash
   uv run sarm-hand setup-motors --role follower --port /dev/ttyACM0
   uv run sarm-hand setup-motors --role leader   --port /dev/ttyACM1
   ```

   Use `--scan` to verify IDs without programming. See [SO-ARM101 leader settings](#so-arm101-leader-settings) for the motor map.

4. **Calibrate both arms** — move each arm through the LeRobot calibration prompts (center pose, then full range per joint):

   ```bash
   uv run sarm-hand calibrate --role follower --port /dev/ttyACM0
   uv run sarm-hand calibrate --role leader   --port /dev/ttyACM1
   ```

5. **Verify servos** (optional but recommended):

   ```bash
   uv run sarm-hand test-motors --role follower --port /dev/ttyACM0
   uv run sarm-hand test-motors --role leader   --port /dev/ttyACM1
   uv run sarm-hand test-poses --port /dev/ttyACM0
   ```

**Run teleoperation**

```bash
uv run sarm-hand teleop-leader \
  --follower-port /dev/ttyACM0 \
  --leader-port /dev/ttyACM1
```

If ports are set in `config/default.yaml`, you can omit the flags:

```bash
uv run sarm-hand teleop-leader
```

Move the leader arm by hand — the follower tracks it. Press **Ctrl+C** to stop. A live Rerun window opens by default (`--no-display-data` to disable).

**Troubleshooting**

| Issue | What to try |
|-------|-------------|
| Wrong arm moves / ports swapped | Re-run `find-port`, swap `robot.port` and `teleop.leader.port` in config |
| Follower jerky or laggy | Reduce `robot.max_relative_target` in config (default `10.0`) |
| Joints misaligned | Re-calibrate **both** arms; leader and follower must use the same mechanical zero |
| Port not found | macOS: use `/dev/tty.usbmodem*`, not `cu.*`; check USB cable and 12V power |

### Meta Quest 2 (phospho)

```bash
# Start phosphobot server (opens dashboard in browser)
uv run sarm-hand teleop-quest --follower-port /dev/ttyACM0
```

On Quest 2:
1. Install the **phospho teleoperation** app from the Meta Store
2. Connect Quest and computer to the same WiFi
3. Open the app → Connect to your server
4. **A** = start/stop teleop, **Trigger** = gripper, **B** = record, **Y** = discard

See `uv run sarm-hand teleop-quest-help` for full instructions.

## Data Collection

### Leader-follower recording

```bash
uv run sarm-hand record-leader \
  --follower-port /dev/ttyACM0 \
  --leader-port /dev/ttyACM1 \
  --repo-id your-username/sarm101-pick-place \
  --num-episodes 50
```

### Quest 2 recording

Record episodes with the **B** button while `teleop-quest` is running. Then inspect:

```bash
uv run sarm-hand record-quest --repo-id your-username/sarm101-vr-demos
```

### SmolVLA (language → robot actions)

[SmolVLA](https://huggingface.co/docs/lerobot/smolvla) is a vision-language-action model: you give a **text task query**, it watches your camera(s), and drives the follower arm.

**Setup**

```bash
uv sync --extra smolvla
```

Configure cameras in `config/default.yaml` (required). For a **single camera** with `lerobot/smolvla_base`, map it to `camera1` and pad the other slots:

```yaml
policy:
  camera_map:
    front: camera1
  empty_cameras: 2
```

Optionally set `policy.path` to a fine-tuned checkpoint.

**Workflow**

1. Record demos: `sarm-hand record-leader ...`
2. Fine-tune: `sarm-hand train-smolvla --dataset-repo-id local/sarm101-dataset`
3. Run a task:

```bash
# Single task query
uv run sarm-hand run-smolvla --task "Pick up the cube and place it in the box"

# Interactive — prompt for each task
uv run sarm-hand run-smolvla --interactive

# Use a fine-tuned checkpoint
uv run sarm-hand run-smolvla \
  --task "Grasp the lego block" \
  --policy-path outputs/train/sarm101_smolvla
```

The base model `lerobot/smolvla_base` is pretrained on community data. For reliable SO-101 tasks, fine-tune on your own demonstrations first.

### Policy evaluation recording

```bash
uv run sarm-hand record-policy \
  --follower-port /dev/ttyACM0 \
  --policy-path your-username/my-policy \
  --num-episodes 10
```

## Data Access

```bash
# Dataset summary (local or Hub)
uv run sarm-hand data-info --repo-id local/sarm101-dataset

# Inspect one frame
uv run sarm-hand data-sample --repo-id local/sarm101-dataset --index 0

# Export episode to CSV
uv run sarm-hand data-export --repo-id local/sarm101-dataset --episode 0

# Upload to Hugging Face Hub
uv run sarm-hand data-push --repo-id local/sarm101-dataset
```

### Python API

```python
from sarm_hand.data import load_dataset

dataset = load_dataset(repo_id="local/sarm101-dataset")
frame = dataset[0]
print(frame.keys())
```

## Genesis World (physics sim)

Optional [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world) integration adds rigid-body physics, contact, and rendered cameras for **sim-to-real** workflows. The lightweight browser sim (`sarm-hand sim`) remains for FK/IK calibration; Genesis is the physics layer.

### Install

```bash
uv sync --extra genesis
./scripts/fetch_so101_assets.sh   # SO-101 URDF + meshes from SO-ARM100
```

Genesis is an optional extra so hardware-only installs stay lean.

### Configuration

`config/default.yaml`:

```yaml
robot:
  backend: hardware          # hardware | genesis | twin

genesis:
  urdf: assets/robots/so101/so101_new_calib.urdf
  backend: auto              # auto → Metal on Apple Silicon, CUDA on Linux
  scene: pick_place_desk
  headless: false            # false → main viewer + front/top/arm camera windows
  cameras:
    front: { width: 640, height: 480, pos: [0.35, -0.55, 0.35], lookat: [0.35, 0.0, 0.12] }
    top:   { width: 640, height: 480, pos: [0.35, 0.0, 1.15], lookat: [0.35, 0.0, 0.05] }
    arm:   { width: 640, height: 480, attach_link: gripper_link }  # gripper-mounted

twin:
  sync_mode: hardware_to_sim
  rate_hz: 30
```

### Scene objects (colors, shapes, placement)

Genesis props are defined in `config/scenes/<name>.yaml`. Set `genesis.scene` in `config/default.yaml` to choose a scene (`pick_place_desk`, `minimal`, or your own file).

```yaml
# config/scenes/pick_place_desk.yaml
objects:
  desk:
    shape: box
    size: [0.5, 0.35, 0.02]
    pos: [0.35, 0.0, 0.01]
    fixed: true
    color: [0.45, 0.32, 0.18]    # RGB 0–1, or hex e.g. "#e63946"
    surface: rough               # default | plastic | rough | glass | gold | aluminum

  pen:
    shape: cylinder
    radius: 0.004
    height: 0.12
    pos: [0.28, 0.08, 0.08]
    color: [0.15, 0.35, 0.85]

  red_cube:
    shape: box
    size: [0.03, 0.03, 0.03]
    pos: [0.30, -0.06, 0.05]
    color: "#e63946"
    enabled: false               # set true to spawn; false removes from scene
```

| Field | Purpose |
|-------|---------|
| `shape` | `box`, `cylinder`, or `sphere` |
| `pos` | World position `[x, y, z]` in meters |
| `size` / `radius` / `height` | Shape dimensions |
| `color` | RGB list (0–1) or `#rrggbb` hex |
| `surface` | Material preset (see above) |
| `fixed` | `true` = static (desk); `false` = movable (pen) |
| `enabled` | `false` = omit object from scene |
| `density` | Optional physics density (kg/m³) |

Use `genesis.scene: minimal` for an empty desk, or copy a scene file and add objects. Override path with `genesis.scene_file: config/scenes/my_scene.yaml`.

### Mac (native Metal) — interactive twin

Best for daily sim-to-real development on Apple Silicon:

```bash
# Smoke test: load URDF, step physics, render all cameras
uv run sarm-hand genesis-spike
# Opens main 3D viewer + separate front / top / arm camera windows (when headless: false)

# Digital twin: USB follower joints mirrored in Genesis at 30 Hz
uv run sarm-hand twin --follower-port /dev/tty.usbmodem...

# Record sim dataset with USB leader arm driving Genesis (no follower needed)
uv run sarm-hand record-sim --leader
# Each run writes a new timestamped dataset, e.g. local/sarm101-dataset-genesis-20260616-231500-123456

# Record sim dataset (viewer + OpenCV camera windows when headless: false)
uv run sarm-hand record-sim --num-episodes 10
# Records front, top, and arm as observation.images.* videos in the dataset
uv run sarm-hand record-sim --headless --num-episodes 10   # no viewer / preview windows

# Append more episodes to a specific dataset from an earlier run
uv run sarm-hand record-sim --leader --resume --repo-id local/sarm101-dataset-genesis-20260616-231500

# Record hardware joints + Genesis camera (sim2real dataset)
uv run sarm-hand record-twin --follower-port /dev/tty.usbmodem...
```

### Linux (Docker + NVIDIA) — batch recording

For overnight dataset generation on a GPU server:

```bash
# Build image (CUDA 12.8 + genesis-world + sarm-hand)
docker build -t sarm-hand-genesis -f docker/genesis/Dockerfile .

# Headless batch recording
docker run --gpus all -v $(pwd)/data:/workspace/hand/data sarm-hand-genesis \
  record-sim --headless --num-episodes 100

# Or use compose
docker compose -f docker/genesis/compose.yaml run --rm genesis-record
```

Do **not** rely on Docker for Mac GPU sim — use native Metal instead.

### Architecture

| Command | Purpose |
|---------|---------|
| `genesis-spike` | Verify Genesis + SO-101 URDF loads |
| `twin` | Hardware → Genesis mirror loop |
| `record-sim` | Pure sim LeRobot dataset |
| `record-twin` | Hardware state + sim-rendered images |

Set `robot.backend: genesis` to use the sim robot with existing teleop/policy paths (no USB). The twin backend is CLI-only (`sarm-hand twin` / `record-twin`).

## Project Structure

```
.
├── config/
│   ├── default.yaml         # Robot, genesis, twin, cameras, dataset
│   └── scenes/              # Scene metadata (pick_place_desk)
├── assets/robots/so101/     # SO-101 URDF (fetch via scripts/fetch_so101_assets.sh)
├── docker/genesis/          # Linux NVIDIA batch sim image
├── scripts/fetch_so101_assets.sh
├── sim/                     # 3D kinematic simulator (arm3d.html + Three.js)
├── src/sarm_hand/
│   ├── backends/            # hardware | genesis | twin robot backends
│   ├── genesis/             # Scene, units, SO101SceneDriver
│   ├── cli.py               # sarm-hand CLI entry point
│   ├── config.py            # Configuration loading
│   ├── robot.py             # USB arm helpers + build_robot()
│   ├── twin.py              # Hardware → Genesis twin loop
│   ├── record_sim.py        # Genesis + twin dataset recording
│   ├── cameras.py           # USB + HTTP/RTSP camera integration
│   ├── policy.py            # SmolVLA inference and training
│   ├── teleop.py            # Leader + Quest 2 teleoperation
│   ├── record.py            # Hardware data collection
│   └── data.py              # Dataset access utilities
└── pyproject.toml           # uv / hatchling project ([genesis] extra)
```

## macOS USB Notes

On macOS, ports typically appear as `/dev/tty.usbmodem*` or `/dev/cu.usbmodem*`. Use the `tty` variant with LeRobot.

## Training (optional)

Install training extras and use LeRobot directly:

```bash
uv sync --extra training
uv run lerobot-train --help
```

## References

- [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
- [phospho Quest teleop](https://docs.phospho.ai/examples/teleop)
- [SO-ARM101 hardware](https://github.com/TheRobotStudio/SO-ARM100)
