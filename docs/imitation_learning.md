# Imitation learning with LeRobot

Code: [`so101_mujoco_demo/`](../so101_mujoco_demo) (its README maps out the files).

Teleoperate a simulated [SO-101 arm](https://github.com/TheRobotStudio/SO-ARM100) in MuJoCo with the keyboard, record demonstrations as LeRobot datasets, visualize them with rerun, train a policy, and roll it back out in the sim.
This is the full LeRobot workflow, no hardware required.

The scene holds a **red** and a **green** cube.
Every episode drops one cube in the left slot and one in the right slot, with the colors swapped at random so a policy cannot memorize a side.
The *target color* decides which cube counts as a successful pick, and each recorded episode stores the matching instruction ("Pick up the green cube and lift it.").
That lets the same scene teach two lessons:

- **ACT**, a single-task policy, trained on red-only demonstrations.
- **SmolVLA**, a language-conditioned policy, fine-tuned on demonstrations of both colors so the instruction steers which cube it picks.

The sim is integrated the same way you would integrate real hardware ([Bring Your Own Hardware](https://huggingface.co/docs/lerobot/main/en/integrate_hardware)): a `Robot` plugin and a `Teleoperator` plugin that lerobot's CLIs auto-discover.
Datasets recorded here have the same schema as real SO-101 datasets (joint positions in degrees, `<joint>.pos` names, wrist/front cameras), and EE teleop runs through lerobot's own placo IK + SO-follower processor pipeline.

## Setup

Everything runs through the `lerobot` pixi environment defined in the repository's [`pixi.toml`](../pixi.toml): the LeRobot checkout (`external/lerobot`), placo, this package and both plugins are installed editable.
The first `pixi run` builds it.
The `so101-*` tasks run inside `so101_mujoco_demo/`, so the relative `data/` and `outputs/` paths below land there.
Extra arguments are passed through to the underlying command.

The `lerobot` environment sets `MUJOCO_GL=egl`, so the sim renders its policy cameras offscreen in every task and in `pixi run -e lerobot ...` commands; the MuJoCo viewer window still opens alongside it.
Without it, the cameras render through the viewer's OpenGL context, and recording with the viewer open slowed from 25 Hz to about 8 Hz.
To watch them in the MuJoCo viewer, add `--show-viewer` (`so101-collect`) or `--show` (`so101-eval`).

### Viewing over SSH

The native MuJoCo viewer needs a display.
On a remote machine, add `--viewer viser` to any `so101-*` task (or `--robot.viewer=viser` to `lerobot-rollout`) and the scene is served to a browser by [mjviser](https://github.com/mujocolab/mjviser) instead; nothing on the remote side needs OpenGL or an X server (the policy cameras still render through EGL).
Forward the port and open http://localhost:8080:

```bash
ssh -L 8080:localhost:8080 <remote>
pixi run so101-collect --repo-id you/so101_pick_red --root data/pick_red --episodes 100 --show-viewer --viewer viser
```

`--viser-port` changes the port.
The viewer is display-only (orbit with the mouse, toggle contacts in the *Visualization* tab); it does not forward key presses, so keyboard teleop (`so101-teleoperate`, `so101-record` without `--teleop leader`) still needs the keyboard listener to see a local display.
The leader arm, the scripted expert, `so101-eval` and `lerobot-rollout` have no such requirement and work fully over SSH.

Every script takes `--colors`: the target cube(s), with episodes alternating through the list.
The default is `red`; `--colors red green` is the language-conditioned setting.

## 1. Practice teleoperating

```bash
pixi run so101-teleoperate --colors red green
```

| key | action |
|---|---|
| ↑ / ↓ | end-effector forward / back |
| ← / → | end-effector left / right |
| w / s | end-effector up / down |
| q / e | pitch gripper up / down |
| a / d | roll jaws (align with a yawed cube) |
| space | toggle gripper open/close |
| x | reset scene (next target color, new cube positions) |
| ESC | quit |

The terminal names the target cube after every reset.

**Grasping technique**: the beak-style gripper can't pick off the floor at its default 45°.
Pitch down with `q` until the gripper is near-vertical, descend with the open jaws *behind* the cube, nudge forward so the cube sits between the jaw tips, close, lift.
Same technique as the real arm.

## 2. Record datasets

Two datasets: red-only for ACT, and both colors for SmolVLA, 100 episodes per color.
There are three ways to produce them, and all three write the same schema (joint actions in degrees, wrist/front cameras, one task string per episode), so they train with the same `so101-train` commands.

Every recorder re-randomizes the scene before each episode: the arm re-homes, both cubes respawn, and the next target color comes from `--colors`.
The terminal tells you which cube to pick, and the episode stores the matching task string.

### 2a. Keyboard

```bash
pixi run so101-record --repo-id you/so101_pick_red --root data/pick_red --episodes 100 --display-data
pixi run so101-record --repo-id you/so101_pick_two --root data/pick_two --episodes 200 --colors red green
```

Motion keys as in section 1, plus: **n** = save episode, **r** = discard & re-record, **ESC** = stop and finalize.
**ESC** discards the episode in progress, so press **n** first to keep it.
`--display-data` opens a live rerun view of both cameras + joint streams while you record.
LeRobot's `Record loop is running slower ... than the target FPS` warning is hidden by default because it repeats on every slow tick; add `--fps-warnings` to see it.
Frames are stored at a fixed 1/25 s spacing, so if the loop is consistently slow (for example with `--display-data`, or with `MUJOCO_GL` set to something other than `egl`), recorded episodes play back faster than you drove them.

### 2b. SO-101 leader arm

With a physical SO-101 leader arm (see [Hardware setup](#hardware-setup-optional)), add `--teleop leader`:

```bash
pixi run so101-record --teleop leader --repo-id you/so101_pick_two_leader --root data/pick_two_leader --episodes 200 --colors red green
```

`--port` defaults to `/dev/SO101Leader` and `--teleop-id` to `leader_arm`; pass them if yours differ.
The episode keys are the same as for the keyboard (**n** / **r** / **ESC**).

- **Syncing to the leader**: after each scene reset, the sim arm eases from home to the leader's pose over one second, unrecorded, before the episode starts.
  Without this, the first frame would snap the arm to wherever you are holding the leader and could knock the cubes.
  Between episodes, hold the leader near the sim's home pose, above and clear of the cubes.
- **No reset phase**: the next episode starts as soon as you save one.
- **Gripper range**: the leader's gripper reads about 0 when closed and up to 100 when open, while the keyboard recorder clips the gripper to −8 (firm close) to 74.

### 2c. Scripted expert

Teleoperating 100 episodes is a chore, so a scripted expert can produce the same datasets headlessly (failed attempts are discarded).
This is how the results below were produced; each run takes a few minutes:

```bash
pixi run so101-collect --repo-id you/so101_pick_red --root data/pick_red --episodes 100
pixi run so101-collect --repo-id you/so101_pick_two --root data/pick_two --episodes 200 --colors red green
```

To watch it work, add `--show-viewer` (e.g., `pixi run so101-collect --repo-id you/so101_pick_red --root data/pick_red --episodes 100 --show-viewer`); the loop then paces itself to real time.

## 3. Visualize with rerun

```bash
pixi run so101-viz --repo-id you/so101_pick_two --root data/pick_two --episode-index 1
```

## 4. Train a policy

Both options use the same `lerobot-train` CLI.
`--dataset.repo_id` and `--dataset.root` must point at a dataset you actually recorded.
If the local root is missing or the dataset wasn't finalized (recorder killed mid-run), lerobot falls back to fetching the repo_id from the HF Hub and fails with a 404.
`--batch_size=16` fits an 8 GB laptop GPU for both policies.

### 4a. ACT (from scratch, red cube only)

The fast, small baseline: the best first policy, and it trains on a laptop-class GPU.

```bash
pixi run so101-train \
    --dataset.repo_id=you/so101_pick_red --dataset.root=data/pick_red \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/train/so101_act --job_name=so101_act \
    --batch_size=16 --steps=25000 --save_freq=5000 --wandb.enable=true
```

On a laptop RTX 5070 this runs at about 5 steps/s, so 25k steps take about 80 minutes and checkpoints land every 5k steps.
More steps beyond that point do not help; more demonstrations do.

### 4b. Fine-tune SmolVLA (both cubes, language-conditioned)

[SmolVLA](https://huggingface.co/docs/lerobot/smolvla) is LERobot's 450M language-conditioned VLA.
Instead of training from scratch (`--policy.type=...`), you fine-tune the pretrained base with `--policy.path=lerobot/smolvla_base`, downloaded from the HF Hub on first run.
Each episode's task string ("Pick up the red/green cube and lift it.") is the language input, so the two-color dataset is what makes the instruction meaningful.

```bash
pixi run so101-train \
    --dataset.repo_id=you/so101_pick_two --dataset.root=data/pick_two \
    --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
    --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --output_dir=outputs/train/so101_smolvla --job_name=so101_smolvla \
    --batch_size=16 --steps=25000 --save_freq=5000 --wandb.enable=true
```

Actually, this model requires more data, you may want to train for about 60k steps as shown below.
This requires modifying the learning rate scheduler, which by default tapers down after 30k steps.

```bash
pixi run so101-train \
    --dataset.repo_id=you/so101_pick_two --dataset.root=data/pick_two \
    --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
    --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --output_dir=outputs/train/so101_smolvla_60k --job_name=so101_smolvla_60k \
    --batch_size=16 --steps=60000 --policy.scheduler_decay_steps=60000 --save_freq=10000 --wandb.enable=true
```

`--rename_map` is required: the pretrained base names its camera inputs `camera1/2/3`, and without the mapping lerobot aborts with a feature-mismatch error on this demo's `front`/`wrist` keys (two of three cameras is fine).
The mapping is saved into the checkpoint's preprocessor, so `so101-eval` needs no extra flags; `lerobot-rollout` checks camera names itself and needs the same `--rename_map` again.

At batch 16 the fine-tune uses about 5 GB of VRAM and runs at roughly 1.7 steps/s on a laptop RTX 5070, so 60k steps take about ten hours (run it overnight).

By default only the action expert is trained, and that is enough here.
The usual next lever on a real-robot dataset is unfreezing the VLM (`--policy.freeze_vision_encoder=false --policy.train_expert_only=false`), but note that `train_expert_only=false` makes the whole 450M-parameter VLM trainable, not just the vision encoder; its optimizer state alone needs about 7 GB, so it does not fit an 8 GB GPU at any batch size.

## 5. You have a trained policy. Now what?

Checkpoints live in `outputs/train/<job>/checkpoints/<step>/pretrained_model/`, with `last` pointing at the newest.
Each one is a self-contained folder (weights, config, and the pre/post-processors), so it can be loaded, shared, or pushed to the Hub as is.

To see what you actually trained, print the model's module tree and per-block parameter counts (`--no-tree` for just the counts, `--depth N` to expand nested blocks):

```bash
pixi run so101-inspect --policy-path outputs/train/so101_act/checkpoints/last/pretrained_model
```

For ACT this shows the ResNet-18 backbone, the CVAE encoder, and the transformer encoder/decoder (about 52M parameters, all trainable). For SmolVLA it shows the frozen 350M-parameter VLM next to the 100M-parameter action expert that fine-tuning actually updates.

There are two ways to run a checkpoint in the sim:

- `scripts/eval_policy.py` (the `so101-eval` task): seeded episodes with a success count.
  Success means the *target* cube was lifted; lifting the other one is reported separately as "wrong cube", which is the number to watch for language conditioning.
- `lerobot-rollout`: LeRobot's deployment CLI, the same command you would use on a real arm.
  `--strategy.type=base` means "just run the policy, record nothing"; `--fps=25` matches the rate the datasets were recorded at.

The `so101-eval` task runs inside `so101_mujoco_demo/` by itself; the direct `python` and `lerobot-rollout` commands below must be run from that directory.
Every command is a single line so that the JSON in `--rename_map` survives copy-paste.

### 5a. ACT

Headless success rate over 50 seeded episodes (about a minute; the seeds are fixed, so checkpoints are compared on identical scenes):

```bash
pixi run so101-eval --policy-path outputs/train/so101_act/checkpoints/last/pretrained_model --episodes 50
```

Same thing in the MuJoCo viewer (from `so101_mujoco_demo/`):

```bash
pixi run so101-eval --policy-path outputs/train/so101_act/checkpoints/last/pretrained_model --episodes 10 --show
```

ACT has no language input, so it can only ever go for the cube it was trained on.
Asking it for the green cube shows that; expect failures and "wrong cube" picks:

```bash
pixi run so101-eval --policy-path outputs/train/so101_act/checkpoints/last/pretrained_model --episodes 10 --colors green
```

Deploy-style rollout for 30 s in the viewer (from `so101_mujoco_demo/`):

```bash
pixi run -e lerobot lerobot-rollout --strategy.type=base --fps=25 --duration=30 --policy.path=outputs/train/so101_act/checkpoints/last/pretrained_model --robot.type=so101_sim
```

### 5b. SmolVLA

Headless success rate, alternating red and green instructions over 50 episodes:

```bash
pixi run so101-eval --policy-path outputs/train/so101_smolvla/checkpoints/last/pretrained_model --episodes 50 --colors red green
```

The instruction defaults to the target color's recorded task string; `--task "..."` overrides it, which is how you would test rephrasings.
In the viewer:

```bash
pixi run so101-eval --policy-path outputs/train/so101_smolvla/checkpoints/last/pretrained_model --episodes 10 --colors red green --show
```

Deploy-style rollout (from `so101_mujoco_demo/`).
`lerobot-rollout` compares camera names before loading the checkpoint's preprocessor, so the `--rename_map` from training is needed again, and `--robot.target_color` only affects which pick the sim reports as a success:

```bash
pixi run -e lerobot lerobot-rollout --strategy.type=base --fps=25 --duration=30 --policy.path=outputs/train/so101_smolvla/checkpoints/last/pretrained_model --robot.type=so101_sim --robot.target_color=green --task="Pick up the green cube and lift it." --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

Same for the red cube:

```bash
pixi run -e lerobot lerobot-rollout --strategy.type=base --fps=25 --duration=30 --policy.path=outputs/train/so101_smolvla/checkpoints/last/pretrained_model --robot.type=so101_sim --robot.target_color=red --task="Pick up the red cube and lift it." --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

### 5c. Comparing and improving

Compare checkpoints by pointing `--policy-path` at `checkpoints/005000`, `checkpoints/010000`, ... rather than only `last`.
The usual levers, in order of payoff: more (and more varied) demonstrations, more training steps, and for SmolVLA unfreezing the vision encoder.
`--resume=true --config_path=outputs/train/<job>/checkpoints/last/pretrained_model/train_config.json` continues a run instead of starting over.
Swapping `--robot.type=so101_follower` in the rollout commands points the same policy at a real arm.

## Sim notes (things that were required to make grasping work)

- **Jaw collision pads**: MuJoCo collides meshes via convex hulls, which fill the jaws' concave inner faces, so flat pinches are impossible and the cube squirts out.
  The two jaw meshes have box collision pads on their inner faces instead (`fixed_jaw_pad` / `moving_jaw_pad` in `so101_new_calib.xml`).
  The pads cover only the fingertip pinch zone at moderate friction; full-length high-friction pads make every graze stick and grasping trivially forgiving.
- **`cone="elliptic" impratio="10"`** in the scene `<option>`: the standard MuJoCo settings for stable pinch grasps.
  With the default pyramidal cone the angled jaws eject the cube.
- **Gripper close speed**: the position servo (kp≈1000) slams the jaws if you command open→closed in one step and smacks the cube away.
  Teleop closes via integrated velocity (`GripperVelocityToJoint`); the scripted expert ramps the target over ~1 s.
- **Units**: MuJoCo works in radians internally; the Robot plugin boundary is degrees (real SO-101 convention), so datasets and policies match real-robot data.
  The wrist camera is mounted on the gripper body in the MJCF (`camera name="wrist"`), placed so jaws + cube + workspace are all visible.
- **Cube slots**: both cubes spawn at `x∈[0.20,0.28]`; the right slot is `y∈[-0.09,-0.05]` and the left slot `y∈[0.0,0.04]`, with random yaw.
  The gap keeps the open jaws clear of the other cube, and the scripted expert picks either slot 100/100 without touching the distractor.
  The left slot is narrower because the moving jaw is offset toward +y, which makes grasps there harder.

## Pointing at the real arm

The `keyboard_pose` teleoperator and the processor pipeline in `scripts/sim_pipelines.py` are robot-agnostic.
Swap `SO101Sim` for lerobot's `SO101Follower` (`--robot.type=so101_follower`, `use_degrees=true`) and the same keyboard EE teleop drives the physical arm.

## Hardware setup (optional)

Only needed to record with a physical SO-101 leader arm ([2b](#2b-so-101-leader-arm)).

- **Motor SDK**: the `lerobot` environment installs LeRobot with the `feetech` extra (the equivalent of `pip install 'lerobot[feetech]'`), which provides the servo SDK the leader needs.
- **Serial port**: `so101-record --teleop leader` defaults to `/dev/SO101Leader`, a udev symlink to the leader's `/dev/ttyACM*` device.
  Without one, pass the `/dev/ttyACM*` path with `--port`.
  If you get a permission error, add yourself to the `dialout` group and log in again.
- **Calibration**: the first `so101-record --teleop leader` run walks you through calibrating the leader and saves it as `<teleop-id>.json` (default `leader_arm`) under `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/`.
  Later runs with the same `--teleop-id` reuse it.
