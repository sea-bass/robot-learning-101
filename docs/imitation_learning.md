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

The headless tasks (`so101-collect`, `so101-eval`) set `MUJOCO_GL=egl` for you.
To watch a viewer instead, invoke the script directly from `so101_mujoco_demo/` (`pixi run -e lerobot python scripts/... --show`) so MuJoCo uses its windowed GLFW backend.

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

Two datasets: red-only for ACT, and both colors for SmolVLA.

```bash
pixi run so101-record --repo-id you/so101_pick_red --root data/pick_red --episodes 30 --display-data
pixi run so101-record --repo-id you/so101_pick_two --root data/pick_two --episodes 60 --colors red green
```

Motion keys as above, plus: **ENTER** = save episode, **x** = discard & re-record, **ESC** = stop and finalize.
The terminal tells you which cube to pick before each episode.
`--display-data` opens a live rerun view of both cameras + joint streams while you record.

Teleoperating 90 episodes is a chore, so a scripted expert can produce the same datasets headlessly (failed attempts are discarded).
This is how the results below were produced:

```bash
pixi run so101-collect --repo-id you/so101_pick_red --root data/pick_red --episodes 30
pixi run so101-collect --repo-id you/so101_pick_two --root data/pick_two --episodes 60 --colors red green
```

To watch it work, run `pixi run -e lerobot python scripts/collect_scripted.py --show-viewer ...` instead; the loop then paces itself to real time.
(The interactive viewer needs the GLFW backend that the `so101-collect` task's `MUJOCO_GL=egl` would disable.)

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
    --batch_size=16 --steps=20000 --save_freq=5000 --wandb.enable=true
```

On a laptop RTX 5070 this runs at about 5 steps/s, so 20k steps take under an hour and checkpoints land every 5k steps.
About 50k steps is a reasonable budget for ACT on 20–50 episodes if the 20k result is not good enough.

### 4b. Fine-tune SmolVLA (both cubes, language-conditioned)

[SmolVLA](https://huggingface.co/docs/lerobot/smolvla) is lerobot's 450M language-conditioned VLA.
Instead of training from scratch (`--policy.type=...`), you fine-tune the pretrained base with `--policy.path=lerobot/smolvla_base`, downloaded from the HF Hub on first run.
Each episode's task string ("Pick up the red/green cube and lift it.") is the language input, so the two-color dataset is what makes the instruction meaningful.

```bash
pixi run so101-train \
    --dataset.repo_id=you/so101_pick_two --dataset.root=data/pick_two \
    --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
    --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --output_dir=outputs/train/so101_smolvla --job_name=so101_smolvla \
    --batch_size=16 --steps=20000 --save_freq=5000 --wandb.enable=true
```

`--rename_map` is required: the pretrained base names its camera inputs `camera1/2/3`, and without the mapping lerobot aborts with a feature-mismatch error on this demo's `front`/`wrist` keys (two of three cameras is fine).
The mapping is saved into the checkpoint's preprocessor, so `so101-eval` needs no extra flags; `lerobot-rollout` checks camera names itself and needs the same `--rename_map` again.

At batch 16 the fine-tune uses about 6 GB of VRAM and runs at roughly 1.4 steps/s on a laptop RTX 5070, so 10k steps take about two hours.
lerobot's recommended starting point is 20k steps at batch 64, roughly 4 h on an A100.
SmolVLA wants more data than ACT: the 60 scripted episodes above are a floor, not a target.

By default only the action expert is trained.
Unfreezing the vision encoder usually improves results substantially on a specialized task like this, at the cost of VRAM and step time:

```bash
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false
```

## 5. You have a trained policy. Now what?

Checkpoints live in `outputs/train/<job>/checkpoints/<step>/pretrained_model/`, with `last` pointing at the newest.
Each one is a self-contained folder (weights, config, and the pre/post-processors), so it can be loaded, shared, or pushed to the Hub as is.

There are two ways to run a checkpoint in the sim:

- `scripts/eval_policy.py` (the `so101-eval` task): seeded episodes with a success count.
  Success means the *target* cube was lifted; lifting the other one is reported separately as "wrong cube", which is the number to watch for language conditioning.
- `lerobot-rollout`: LeRobot's deployment CLI, the same command you would use on a real arm.
  `--strategy.type=base` means "just run the policy, record nothing"; `--fps=25` matches the rate the datasets were recorded at.

The `so101-eval` task runs inside `so101_mujoco_demo/` by itself; the direct `python` and `lerobot-rollout` commands below must be run from that directory.
Every command is a single line so that the JSON in `--rename_map` survives copy-paste.

### 5a. ACT

Headless success rate over 20 seeded episodes:

```bash
pixi run so101-eval --policy-path outputs/train/so101_act/checkpoints/last/pretrained_model --episodes 20
```

Same thing in the MuJoCo viewer (from `so101_mujoco_demo/`):

```bash
pixi run -e lerobot so101-eval --policy-path outputs/train/so101_act/checkpoints/last/pretrained_model --episodes 10 --show
```

ACT has no language input, so it can only ever go for the cube it was trained on.
Asking it for the green cube shows that; expect failures and "wrong cube" picks:

```bash
pixi run -e lerobot so101-eval --policy-path outputs/train/so101_act/checkpoints/last/pretrained_model --episodes 10 --colors green
```

Deploy-style rollout for 30 s in the viewer (from `so101_mujoco_demo/`):

```bash
pixi run -e lerobot lerobot-rollout --strategy.type=base --fps=25 --duration=30 --policy.path=outputs/train/so101_act/checkpoints/last/pretrained_model --robot.type=so101_sim
```

### 5b. SmolVLA

Headless success rate, alternating red and green instructions over 20 episodes:

```bash
pixi run so101-eval --policy-path outputs/train/so101_smolvla/checkpoints/last/pretrained_model --episodes 20 --colors red green
```

The instruction defaults to the target color's recorded task string; `--task "..."` overrides it, which is how you would test rephrasings.
In the viewer (from `so101_mujoco_demo/`):

```bash
pixi run -e lerobot python scripts/eval_policy.py --policy-path outputs/train/so101_smolvla/checkpoints/last/pretrained_model --episodes 10 --colors red green --show
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
