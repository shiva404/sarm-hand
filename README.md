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

## Project Structure

```
.
├── config/default.yaml      # Robot, geometry, joints, poses, sim settings
├── sim/                     # 3D joint simulator (arm3d.html + Three.js)
├── src/sarm_hand/
│   ├── cli.py               # sarm-hand CLI entry point
│   ├── config.py            # Configuration loading
│   ├── robot.py             # USB arm helpers
│   ├── teleop.py            # Leader + Quest 2 teleoperation
│   ├── record.py            # Data collection
│   └── data.py              # Dataset access utilities
└── pyproject.toml           # uv / hatchling project
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
