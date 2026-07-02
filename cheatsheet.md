# sarm-hand cheatsheet

Copy-paste commands for this project. Ports and defaults match `config/default.yaml`.

Follower port: `/dev/tty.usbmodem5B141140091`  
Leader port: `/dev/tty.usbmodem5B140289771`

---

## Install

```bash
uv sync
uv sync --extra quest
uv sync --extra training          # ACT / LeRobot train (lerobot-train)
uv sync --extra smolvla           # SmolVLA inference + fine-tune (transformers, etc.)
uv sync --extra genesis           # Genesis sim only
uv sync --extra sim-policy        # Genesis + SmolVLA (run-smolvla --genesis)
# ACT on hardware:  uv sync --extra training
# SmolVLA full:     uv sync --extra smolvla --extra training
# Genesis + both:   uv sync --extra genesis --extra smolvla --extra training
./scripts/fetch_so101_assets.sh   # Genesis / sim URDF assets
```

---

## Config & ports

```bash
uv run sarm-hand config-show      # robot, cameras, policies.act / policies.smolvla, dataset
uv run sarm-hand find-port
```

---

## Quick tests & debug (pre-flight)

Run these before teleop, recording, or policy inference. Uses ports from config when `--port` is omitted.

### One-shot “is everything ready?”

```bash
uv run sarm-hand find-port
uv run sarm-hand config-show
uv run sarm-hand test-motors --role follower
uv run sarm-hand test-motors --role leader
uv run sarm-hand test-poses --pose ready
uv run sarm-hand leader-free
uv run sarm-hand list-cameras
uv run sarm-hand camera-test
```

### Motors & USB

```bash
# Bus scan — which servo IDs respond (no programming)
uv run sarm-hand setup-motors --role follower --scan
uv run sarm-hand setup-motors --role leader --scan

# Ping each joint; extra retries if flaky cable
uv run sarm-hand test-motors --role follower
uv run sarm-hand test-motors --role leader --retries 5

# Lost a servo mid-session? Re-seat daisy-chain, then re-test
uv run sarm-hand test-motors --role follower --port /dev/tty.usbmodem5B141140091
```

### Poses — home / ready / park

```bash
uv run sarm-hand test-poses --list
uv run sarm-hand test-poses --pose ready          # recording start pose
uv run sarm-hand test-poses --pose home
uv run sarm-hand test-poses --pose park           # safe shutdown pose
uv run sarm-hand test-poses                       # full sequence + tolerance check
```

### Calibration quick fixes

```bash
# Leader stiff or follower won't mirror range — copy leader limits to follower
uv run sarm-hand sync-calibration --from leader --to follower
uv run sarm-hand sync-calibration --from leader --write-motors

# Leader won't move by hand before teleop
uv run sarm-hand leader-free

# Full interactive cal (only when offsets are wrong)
uv run sarm-hand calibrate --role follower
uv run sarm-hand calibrate --role leader
```

### Cameras

```bash
uv run sarm-hand list-cameras                     # USB indices for config
uv run sarm-hand camera-probe --name front        # find capture_width/height
uv run sarm-hand camera-probe --name wrist
uv run sarm-hand camera-probe --together          # all configured cams at once
uv run sarm-hand camera-test                      # concurrent (same as record-leader)
uv run sarm-hand camera-preview --name front --seconds 5
uv run sarm-hand camera-preview --name wrist --seconds 5
```

### Teleop smoke test

```bash
uv run sarm-hand leader-free
uv run sarm-hand teleop-leader --no-display-data  # quick mirror check, no Rerun
# Ctrl+C when done
```

### After recording — did data land?

```bash
uv run sarm-hand data-list
uv run sarm-hand data-info --latest               # shows session repo-id + frame count
uv run sarm-hand data-sample --index 0
uv run sarm-hand viz-dataset --repo-id <session-repo-id> --episode 0
```

### Common failures → try this


| Symptom                    | Command                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| Wrong USB port             | `find-port` then update `robot.port` / `teleop.leader.port`        |
| Servo not responding       | `test-motors --role follower --retries 5`                          |
| Leader stiff in teleop     | `leader-free`                                                      |
| Follower range mismatch    | `sync-calibration --from leader --write-motors`                    |
| Camera black / wrong index | `list-cameras` → `camera-probe --name front` → fix `index_or_path` |
| Record fails on cameras    | `camera-test` (must pass concurrent test)                          |
| Arm not at demo start pose | `test-poses --pose ready`                                          |


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

# Copy leader travel ranges onto follower (keeps each arm's homing offsets)
uv run sarm-hand sync-calibration --from leader --to follower
uv run sarm-hand sync-calibration --from leader --write-motors   # flash to follower EEPROM

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

Default setup: **front + wrist** USB cameras (`cameras.front`, `cameras.wrist` in config).  
Run probe + concurrent test before recording — same code path as `record-leader`.

```bash
uv run sarm-hand list-cameras

# Probe supported capture resolutions (set capture_width/height in config)
uv run sarm-hand camera-probe --name front
uv run sarm-hand camera-probe --name wrist
uv run sarm-hand camera-probe --all-usb
uv run sarm-hand camera-probe --together          # all configured USB cams open at once

# Test configured cameras (concurrent when 2+ in config — matches record-leader)
uv run sarm-hand camera-test
uv run sarm-hand camera-test --each               # sequential only
uv run sarm-hand camera-test --together           # force concurrent

# Live preview
uv run sarm-hand camera-preview --name front --seconds 10
uv run sarm-hand camera-preview --name wrist --seconds 10
uv run sarm-hand camera-preview --index 0 --seconds 10

# HTTP / RTSP stream (if configured)
uv run sarm-hand camera-preview --url http://192.168.0.58:81/stream --seconds 10

# Headless snapshot
uv run sarm-hand camera-preview --name front --no-window --output /tmp/front.jpg
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

During `record-leader`: **→ Right arrow** or **s** = save episode early; **r** = discard and re-record; **Esc** = quit session.

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

# Episode timing overrides
uv run sarm-hand record-leader \
  --episode-time-s 45 \
  --reset-time-s 15

# Live Rerun preview during recording (off by default — saves memory)
uv run sarm-hand record-leader --rerun

# Upload to Hugging Face after recording
uv run sarm-hand record-leader \
  --repo-id local/sarm101-dataset \
  --num-episodes 50 \
  --push-to-hub

# Quest 2 recording instructions
uv run sarm-hand record-quest --repo-id local/sarm101-quest-demos

# Record policy evaluation rollouts (SmolVLA — needs --task)
uv run sarm-hand record-policy \
  --follower-port /dev/tty.usbmodem5B141140091 \
  --policy-path outputs/train/sarm101_smolvla \
  --task "Pick and place the object" \
  --repo-id local/sarm101-dataset-eval \
  --num-episodes 10
```

---

## ACT policy (front + wrist — recommended)

Config: `policies.act` in `config/default.yaml`. Training needs `uv sync --extra training`.

```bash
# 1. Verify cameras, then record demos
uv run sarm-hand camera-test
uv run sarm-hand test-poses --pose ready

# Record N episodes (default N=50 from dataset.num_episodes in config)
uv run sarm-hand record-leader --num-episodes 50

# Or set count only on CLI; other defaults from config:
uv run sarm-hand record-leader \
  --num-episodes 20 \
  --single-task "Pick and place the object" \
  --episode-time-s 25 \
  --reset-time-s 10

# Omit --num-episodes to use dataset.num_episodes in config/default.yaml (50)
uv run sarm-hand record-leader

# During recording: → or s = save episode early; r = discard & re-record; Esc = stop
# After session: verify count
uv run sarm-hand data-info --latest
```

# 2. Train (defaults: outputs/train/sarm101_act, 50k steps)

uv run sarm-hand train-act --device mps
uv run sarm-hand train-act   
  --dataset-repo-id local/sarm101-dataset   
  --output-dir outputs/train/sarm101_act   
  --steps 5000   
  --batch-size 8   
  --device mps

# Resume interrupted training

uv run sarm-hand train-act --resume --device mps

# 3. Run on hardware

uv run sarm-hand run-act
uv run sarm-hand run-act   
  --policy-path outputs/train/sarm101_act   
  --episode-time 60   
  --device mps

# Without Rerun

uv run sarm-hand run-act --no-display-data

```

---

## SmolVLA policy

Config: `policies.smolvla` in `config/default.yaml`. Needs `uv sync --extra smolvla` (and `--extra training` to fine-tune).

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

# Training with explicit settings (Apple Silicon — batch 4 avoids MPS OOM)
uv run sarm-hand train-smolvla \
  --dataset-repo-id local/sarm101-dataset \
  --policy-path lerobot/smolvla_base \
  --output-dir outputs/train/sarm101_smolvla \
  --steps 20000 \
  --batch-size 4 \
  --device mps

# Resume interrupted training
uv run sarm-hand train-smolvla --resume --device mps
```

---

## Dataset tools

```bash
# List timestamped recording sessions under dataset.root
uv run sarm-hand data-list

# Metadata (omit --repo-id to use latest session)
uv run sarm-hand data-info --latest
uv run sarm-hand data-info --repo-id local/sarm101-dataset

uv run sarm-hand data-sample --repo-id local/sarm101-dataset --index 0
uv run sarm-hand data-export --repo-id local/sarm101-dataset --episode 0 --output-dir data/exports

# Downsample 30 fps recordings to 10 fps (no re-record; works on external datasets via --root)
uv run sarm-hand data-subsample --latest --target-fps 10
uv run sarm-hand data-subsample --repo-id local/sarm101-dataset --target-fps 10 --dry-run
uv run sarm-hand data-subsample --root /path/to/external/dataset --repo-id user/dataset --target-fps 10

uv run sarm-hand data-push --repo-id shiva404/sarm101-dataset

# Local Rerun viewer for episode 0
uv run sarm-hand viz-dataset --repo-id local/sarm101-dataset --episode 0

# Hugging Face Space visualizer (browser)
uv run sarm-hand viz-dataset --hub --repo-id shiva404/sarm101-dataset
```

---

## 3D joint simulator (browser)

Opens [http://127.0.0.1:8763/sim/arm3d.html](http://127.0.0.1:8763/sim/arm3d.html)

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

Opens [http://localhost:8000](http://localhost:8000)

---

## First-time setup (full sequence)

```bash
uv sync --extra training
uv run sarm-hand find-port
uv run sarm-hand setup-motors --role follower --port /dev/tty.usbmodem5B141140091 --scan
uv run sarm-hand setup-motors --role follower --port /dev/tty.usbmodem5B141140091
uv run sarm-hand calibrate --role follower --port /dev/tty.usbmodem5B141140091
uv run sarm-hand test-motors --role follower --port /dev/tty.usbmodem5B141140091
uv run sarm-hand test-poses --port /dev/tty.usbmodem5B141140091
uv run sarm-hand setup-motors --role leader --port /dev/tty.usbmodem5B140289771
uv run sarm-hand calibrate --role leader --port /dev/tty.usbmodem5B140289771
uv run sarm-hand sync-calibration --from leader --write-motors
uv run sarm-hand leader-free --port /dev/tty.usbmodem5B140289771
uv run sarm-hand list-cameras
uv run sarm-hand camera-probe --together
uv run sarm-hand camera-test
uv run sarm-hand teleop-leader
```

---

## ACT training workflow (quick)

```bash
uv sync --extra training
uv run sarm-hand camera-test
uv run sarm-hand record-leader --num-episodes 50
uv run sarm-hand data-info --latest
uv run sarm-hand train-act --device mps --steps 5000
uv run sarm-hand run-act --policy-path outputs/train/sarm101_act
```

---

## Config reference


| What                 | Where in config/default.yaml                                                      |
| -------------------- | --------------------------------------------------------------------------------- |
| Follower port        | `robot.port`                                                                      |
| Leader port          | `teleop.leader.port`                                                              |
| Cameras (ACT)        | `cameras.front`, `cameras.wrist` (`index_or_path`, `capture_*`, `width`/`height`) |
| ACT policy           | `policies.act` → `output_dir`, `train_steps`, `control_fps`                       |
| SmolVLA policy       | `policies.smolvla` → `path`, `output_dir`, `camera_map`, `stats_buffer`           |
| Shared train dataset | `policies.train_dataset`                                                          |
| Dataset              | `dataset.repo_id`, `dataset.root`, `dataset.fps`                                  |
| Task demos           | `tasks.root` → `data/tasks/`                                                      |
| Genesis scene        | `genesis.scene` → `pick_place_desk`                                               |
| Preset poses         | `poses.home`, `poses.ready`, `poses.park`                                         |
| Motor IDs            | `motors.follower`, `motors.leader`                                                |


