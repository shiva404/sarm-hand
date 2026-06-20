# sarm-hand cheatsheet

Copy-paste commands for this project. Ports and defaults match `config/default.yaml`.

Follower port: `/dev/tty.usbmodem5B141140091`  
Leader port: `/dev/tty.usbmodem5B140289771`

---

## Install

```bash
uv sync
uv sync --extra quest
uv sync --extra smolvla          # hardware policy (transformers, etc.)
uv sync --extra genesis          # Genesis sim only
uv sync --extra sim-policy       # Genesis + SmolVLA (run-smolvla --genesis)
# equivalent: uv sync --extra genesis --extra smolvla
./scripts/fetch_so101_assets.sh
```

---

## Config & ports

```bash
uv run sarm-hand config-show
uv run sarm-hand find-port
```

---

## Hardware setup

```bash
# Assign servo IDs (uses motor map in config)
uv run sarm-hand setup-motors --role follower --port /dev/tty.usbmodem5B141140091
uv run sarm-hand setup-motors --role leader --port /dev/tty.usbmodem5B140289771

# Bus scan only — no programming
uv run sarm-hand setup-motors --role follower --port /dev/tty.usbmodem5B141140091 --scan
uv run sarm-hand setup-motors --role leader --port /dev/tty.usbmodem5B140289771 --scan

# Factory-fresh motors — one joint at a time
uv run sarm-hand setup-motors --role follower --port /dev/tty.usbmodem5B141140091 --one-at-a-time

# Program a single joint (only that motor on the bus)
uv run sarm-hand setup-motors --role follower --port /dev/tty.usbmodem5B141140091 --only shoulder_pan

# Reprogram when servos still have wrong IDs
uv run sarm-hand setup-motors --role follower --port /dev/tty.usbmodem5B141140091 --initial-ids shoulder_pan=1,gripper=6

# Interactive calibration
uv run sarm-hand calibrate --role follower --port /dev/tty.usbmodem5B141140091
uv run sarm-hand calibrate --role leader --port /dev/tty.usbmodem5B140289771

# Ping / read / torque test each joint
uv run sarm-hand test-motors --role follower --port /dev/tty.usbmodem5B141140091
uv run sarm-hand test-motors --role leader --port /dev/tty.usbmodem5B140289771 --retries 5

# Disable leader torque so it moves freely by hand
uv run sarm-hand leader-free --port /dev/tty.usbmodem5B140289771
```

---

## Poses & validation

```bash
# Show configured poses (home, ready, park)
uv run sarm-hand test-poses --list

# Run full pose sequence on follower
uv run sarm-hand test-poses --port /dev/tty.usbmodem5B141140091

# Single pose
uv run sarm-hand test-poses --port /dev/tty.usbmodem5B141140091 --pose home
uv run sarm-hand test-poses --port /dev/tty.usbmodem5B141140091 --pose ready
uv run sarm-hand test-poses --port /dev/tty.usbmodem5B141140091 --pose park

# Custom sequence
uv run sarm-hand test-poses --port /dev/tty.usbmodem5B141140091 --sequence home ready park home

# Looser tolerance (default 8 norm units)
uv run sarm-hand test-poses --port /dev/tty.usbmodem5B141140091 --tolerance 10
```

---

## Cameras

```bash
uv run sarm-hand list-cameras
uv run sarm-hand camera-test

# USB camera index 0
uv run sarm-hand camera-preview --index 0 --seconds 10

# Config camera named "front" (HTTP stream in default.yaml)
uv run sarm-hand camera-preview --name front --seconds 10

# Direct stream URL
uv run sarm-hand camera-preview --url http://192.168.0.58:81/stream --seconds 10

# Headless snapshot
uv run sarm-hand camera-preview --url http://192.168.0.58:81/stream --no-window --output /tmp/front.jpg
```

---

## Teleoperation

```bash
# Leader drives follower — Rerun joint graphs open by default
uv run sarm-hand teleop-leader \
  --follower-port /dev/tty.usbmodem5B141140091 \
  --leader-port /dev/tty.usbmodem5B140289771

# Same, using ports from config/default.yaml
uv run sarm-hand teleop-leader

# Without Rerun
uv run sarm-hand teleop-leader --no-display-data

# Include cameras from config during teleop
uv run sarm-hand teleop-leader --with-cameras

# Meta Quest 2 via phosphobot
uv run sarm-hand teleop-quest --follower-port /dev/tty.usbmodem5B141140091
uv run sarm-hand teleop-quest --no-open-dashboard

# Print Quest 2 setup instructions
uv run sarm-hand teleop-quest-help
```

---

## Task motion (JSON demos in data/tasks/)

Lightweight record/replay — not a full LeRobot dataset.

```bash
# Record leader demo — Ctrl+C when done; follower mirrors live
uv run sarm-hand task record --task "Pick and place the object"

# Record with explicit ports
uv run sarm-hand task record \
  --task "Pick and place the object" \
  --leader-port /dev/tty.usbmodem5B140289771 \
  --follower-port /dev/tty.usbmodem5B141140091

# Record leader only (no follower mirror)
uv run sarm-hand task record --task "Pick and place the object" --no-mirror

# Fixed sample rate and max duration
uv run sarm-hand task record --task "Pick and place the object" --fps 30 --duration 45

# Replay latest demo for a task
uv run sarm-hand task replay --task "Pick and place the object"

# Replay by slug
uv run sarm-hand task replay --task-slug pick_and_place_the_object --demo latest

# Slower playback, loop until Ctrl+C
uv run sarm-hand task replay --task-slug pick_and_place_the_object --speed 0.75 --loop

# List and inspect saved demos
uv run sarm-hand task list
uv run sarm-hand task info --task-slug pick_and_place_the_object --demo latest
```

---

## Data collection (LeRobot datasets)

```bash
# Leader-follower recording (uses config ports, cameras, dataset settings)
uv run sarm-hand record-leader

# Explicit ports and dataset
uv run sarm-hand record-leader \
  --follower-port /dev/tty.usbmodem5B141140091 \
  --leader-port /dev/tty.usbmodem5B140289771 \
  --repo-id local/sarm101-dataset \
  --num-episodes 50 \
  --single-task "Pick and place the object"

# Upload to Hugging Face after recording
uv run sarm-hand record-leader \
  --repo-id local/sarm101-dataset \
  --num-episodes 50 \
  --push-to-hub

# Quest 2 recording instructions
uv run sarm-hand record-quest --repo-id local/sarm101-quest-demos

# Record policy evaluation rollouts
uv run sarm-hand record-policy \
  --follower-port /dev/tty.usbmodem5B141140091 \
  --policy-path outputs/train/sarm101_smolvla \
  --task "Pick and place the object" \
  --repo-id local/sarm101-dataset-eval \
  --num-episodes 10
```

---

## SmolVLA policy

```bash
# Run base model with language task
uv run sarm-hand run-smolvla --task "Pick and place the object"

# Interactive — prompt for each task
uv run sarm-hand run-smolvla --interactive

# Fine-tuned checkpoint
uv run sarm-hand run-smolvla \
  --task "Pick and place the object" \
  --policy-path outputs/train/sarm101_smolvla

# Genesis sim — policy drives sim arm; genesis.cameras feed vision (no USB)
uv run sarm-hand run-smolvla --genesis --task "Pick and place the object"
uv run sarm-hand run-smolvla --genesis \
  --task "Pick and place the object" \
  --policy-path outputs/train/sarm101_smolvla \
  --episode-time 60

# Record eval episodes while running policy
uv run sarm-hand run-smolvla \
  --task "Pick and place the object" \
  --policy-path outputs/train/sarm101_smolvla \
  --record \
  --repo-id local/sarm101-smolvla-eval \
  --num-episodes 5

# Fine-tune on recorded dataset
uv run sarm-hand train-smolvla --dataset-repo-id local/sarm101-dataset

# Training with explicit settings (Apple Silicon)
uv run sarm-hand train-smolvla \
  --dataset-repo-id local/sarm101-dataset \
  --policy-path lerobot/smolvla_base \
  --output-dir outputs/train/sarm101_smolvla \
  --steps 20000 \
  --batch-size 64 \
  --device mps
```

---

## Dataset tools

```bash
uv run sarm-hand data-info --repo-id local/sarm101-dataset
uv run sarm-hand data-sample --repo-id local/sarm101-dataset --index 0
uv run sarm-hand data-export --repo-id local/sarm101-dataset --episode 0 --output-dir data/exports
uv run sarm-hand data-push --repo-id shiva404/sarm101-dataset

# Local Rerun viewer for episode 0
uv run sarm-hand viz-dataset --repo-id local/sarm101-dataset --episode 0

# Hugging Face Space visualizer (browser)
uv run sarm-hand viz-dataset --hub --repo-id shiva404/sarm101-dataset
```

---

## 3D joint simulator (browser)

Opens http://127.0.0.1:8763/sim/arm3d.html

```bash
uv run sarm-hand sim
uv run sarm-hand sim --port 8763 --no-browser
```

---

## Genesis World (physics sim)

Requires `uv sync --extra genesis` and `./scripts/fetch_so101_assets.sh`.

```bash
uv run sarm-hand genesis-spike
uv run sarm-hand genesis-spike --headless

# Digital twin — follower hardware mirrored in Genesis
uv run sarm-hand twin --follower-port /dev/tty.usbmodem5B141140091
uv run sarm-hand twin --follower-port /dev/tty.usbmodem5B141140091 --rate 30 --duration 60

# Record sim dataset driven by USB leader arm
uv run sarm-hand record-sim --leader

# Record sim dataset with explicit settings
uv run sarm-hand record-sim \
  --leader \
  --leader-port /dev/tty.usbmodem5B140289771 \
  --repo-id local/sarm101-dataset-genesis \
  --num-episodes 10 \
  --episode-time 60 \
  --task "Pick and place the object"

# Append to existing sim dataset
uv run sarm-hand record-sim \
  --leader \
  --resume \
  --repo-id local/sarm101-dataset-genesis-20260619-120000

# Random actions wiring test
uv run sarm-hand record-sim --random-actions --num-episodes 1

# Hardware joints + Genesis camera frames → LeRobot dataset
uv run sarm-hand record-twin \
  --follower-port /dev/tty.usbmodem5B141140091 \
  --repo-id local/sarm101-dataset-twin \
  --num-episodes 5 \
  --task "Pick and place the object"

# Joint signal analysis (encoder vs norm vs sim)
uv run sarm-hand log-joint-signal --role leader --port /dev/tty.usbmodem5B140289771
uv run sarm-hand log-joint-signal --analyze-only
uv run sarm-hand log-joint-signal \
  --role leader \
  --port /dev/tty.usbmodem5B140289771 \
  --duration 45 \
  --rate 10 \
  --output data/logs/joint-signal-leader.jsonl

# Leader ↔ Genesis alignment (live mirror + pulse/norm/angle table)
uv run sarm-hand calibrate-genesis --leader-port /dev/tty.usbmodem5B140289771
uv run sarm-hand calibrate-genesis --leader-port /dev/tty.usbmodem5B140289771 --measure
uv run sarm-hand calibrate-genesis --leader-port /dev/tty.usbmodem5B140289771 --capture-home
uv run sarm-hand calibrate-genesis --leader-port /dev/tty.usbmodem5B140289771 --capture-home --save-home
```

---

## LeLab web UI

```bash
uv run sarm-hand lelab --install
uv run sarm-hand lelab
uv run sarm-hand lelab --info
uv run sarm-hand lelab --dev
uv run sarm-hand lelab --no-browser
```

Opens http://localhost:8000

---

## First-time setup (full sequence)

```bash
uv sync
uv run sarm-hand find-port
uv run sarm-hand setup-motors --role follower --port /dev/tty.usbmodem5B141140091 --scan
uv run sarm-hand setup-motors --role follower --port /dev/tty.usbmodem5B141140091
uv run sarm-hand calibrate --role follower --port /dev/tty.usbmodem5B141140091
uv run sarm-hand test-motors --role follower --port /dev/tty.usbmodem5B141140091
uv run sarm-hand test-poses --port /dev/tty.usbmodem5B141140091
uv run sarm-hand setup-motors --role leader --port /dev/tty.usbmodem5B140289771
uv run sarm-hand calibrate --role leader --port /dev/tty.usbmodem5B140289771
uv run sarm-hand leader-free --port /dev/tty.usbmodem5B140289771
uv run sarm-hand teleop-leader
```

---

## Config reference

| What | Where in config/default.yaml |
|------|------------------------------|
| Follower port | `robot.port` |
| Leader port | `teleop.leader.port` |
| Camera stream | `cameras.front.url` |
| Dataset | `dataset.repo_id`, `dataset.root` |
| Task demos | `tasks.root` → `data/tasks/` |
| Genesis scene | `genesis.scene` → `pick_place_desk` |
| Preset poses | `poses.home`, `poses.ready`, `poses.park` |
| Motor IDs | `motors.follower`, `motors.leader` |
