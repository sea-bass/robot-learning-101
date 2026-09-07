# Imitation learning with LeRobot

Code: [`so101_mujoco_demo/`](../so101_mujoco_demo) (its README maps out the files).

Teleoperate a simulated [SO-101 arm](https://github.com/TheRobotStudio/SO-ARM100) in MuJoCo with the keyboard, record pick-the-cube demonstrations as LeRobot datasets, visualize them with rerun, train a policy (ACT from scratch, or fine-tune SmolVLA), and roll it back out in the sim.
This is the full LeRobot workflow, no hardware required.

The sim is integrated the same way you would integrate real hardware ([Bring Your Own Hardware](https://huggingface.co/docs/lerobot/main/en/integrate_hardware)): a `Robot` plugin and a `Teleoperator` plugin that lerobot's CLIs auto-discover.
Datasets recorded here have the same schema as real SO-101 datasets (joint positions in degrees, `<joint>.pos` names, wrist/front cameras), and EE teleop runs through lerobot's own placo IK + SO-follower processor pipeline.

## Setup

Everything runs through the `lerobot` pixi environment defined in the repository's [`pixi.toml`](../pixi.toml): the LeRobot checkout (`external/lerobot`), placo, this package and both plugins are installed editable.
The first `pixi run` builds it.
The `so101-*` tasks run inside `so101_mujoco_demo/`, so the relative `data/` and `outputs/` paths below land there.
Extra arguments are passed through to the underlying command.

The headless tasks (`so101-collect`, `so101-eval`) set `MUJOCO_GL=egl` for you.
To watch a viewer instead, invoke the script directly from `so101_mujoco_demo/` (`pixi run -e lerobot python scripts/... --show`) so MuJoCo uses its windowed GLFW backend.

## 1. Practice teleoperating

```bash
pixi run so101-teleoperate
```

| key | action |
|---|---|
| ↑ / ↓ | end-effector forward / back |
| ← / → | end-effector left / right |
| w / s | end-effector up / down |
| q / e | pitch gripper up / down |
| a / d | roll jaws (align with a yawed cube) |
| space | toggle gripper open/close |
| x | reset scene (new cube position) |
| ESC | quit |

**Grasping technique**: the beak-style gripper can't pick off the floor at its default 45°.
Pitch down with `q` until the gripper is near-vertical, descend with the open jaws *behind* the cube, nudge forward so the cube sits between the jaw tips, close, lift.
Same technique as the real arm.

## 2. Record a dataset

```bash
pixi run so101-record --repo-id you/so101_pick_cube --root data/pick_cube \
    --episodes 20 --display-data
```

Motion keys as above, plus: **ENTER** = save episode, **x** = discard & re-record, **ESC** = stop and finalize.
`--display-data` opens a live rerun view of both cameras + joint streams while you record.

Optionally bootstrap more data without teleoperating with the scripted expert (~90 % success, failed attempts are discarded):

```bash
pixi run so101-collect \
    --repo-id you/so101_pick_cube_scripted --root data/pick_cube_scripted --episodes 25
```

To watch it work, run `pixi run -e lerobot python scripts/collect_scripted.py --show-viewer ...` instead; the loop then paces itself to real time.
(The interactive viewer needs the GLFW backend that the `so101-collect` task's `MUJOCO_GL=egl` would disable.)

## 3. Visualize with rerun

```bash
pixi run so101-viz --repo-id you/so101_pick_cube --root data/pick_cube --episode-index 0
```

## 4. Train a policy

Both options use the same `lerobot-train` CLI.
In either case `--dataset.repo_id` and `--dataset.root` must point at a dataset you actually recorded (e.g. the scripted one: `--dataset.repo_id=you/so101_pick_cube_scripted --dataset.root=data/pick_cube_scripted`).
If the local root is missing or the dataset wasn't finalized (recorder killed mid-run), lerobot falls back to fetching the repo_id from the HF Hub and fails with a 404.

### 4a. ACT (from scratch)

The fast, small baseline: the best first policy, and it trains on a laptop-class GPU.

```bash
pixi run so101-train \
    --dataset.repo_id=you/so101_pick_cube --dataset.root=data/pick_cube \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/train/so101_act --job_name=so101_act \
    --batch_size=16 --steps=50000 --save_freq=10000 --wandb.enable=true
```

About 50k steps is a reasonable starting point for ACT on 20–50 episodes.

### 4b. Fine-tune SmolVLA

[SmolVLA](https://huggingface.co/docs/lerobot/smolvla) is lerobot's 450M language-conditioned VLA.
Instead of training from scratch (`--policy.type=...`), you fine-tune the pretrained base with `--policy.path=lerobot/smolvla_base`, downloaded from the HF Hub on first run.
The dataset's task string ("Pick up the red cube and lift it.") becomes the language instruction.

```bash
pixi run so101-train \
    --dataset.repo_id=you/so101_pick_cube --dataset.root=data/pick_cube \
    --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
    --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --output_dir=outputs/train/so101_smolvla --job_name=so101_smolvla \
    --batch_size=16 --steps=20000 --save_freq=5000 --wandb.enable=true
```

`--rename_map` is required: the pretrained base names its camera inputs `camera1/2/3`, and without the mapping lerobot aborts with a feature-mismatch error on this demo's `front`/`wrist` keys (two of three cameras is fine).
The mapping is saved into the checkpoint's preprocessor, so eval/rollout below need no extra flags.

20k steps at batch 64 is the lerobot-recommended starting point, roughly 4 h on an A100; expect much longer on a desktop GPU.
Drop `--batch_size` if you hit OOM.
SmolVLA wants more data than ACT: ~50 episodes covering the spawn region is a realistic minimum.

By default only the action expert is trained.
Unfreezing the vision encoder usually improves results substantially on a specialized task like this, at the cost of VRAM and step time:

```bash
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false
```

## 5. Roll out the policy

Success rate over seeded episodes, for any checkpoint (ACT or SmolVLA).
For language-conditioned policies the recorded task string is the default instruction; override it with `--task "..."`.

```bash
pixi run so101-eval \
    --policy-path outputs/train/so101_act/checkpoints/last/pretrained_model --episodes 10
# SmolVLA: --policy-path outputs/train/so101_smolvla/checkpoints/last/pretrained_model
# to watch: pixi run -e lerobot python scripts/eval_policy.py ... --show
```

Because the robot is a lerobot plugin, the official deployment CLI works too:

```bash
pixi run -e lerobot lerobot-rollout --strategy.type=base \
    --policy.path=outputs/train/so101_act/checkpoints/last/pretrained_model \
    --robot.type=so101_sim --task="Pick up the red cube and lift it." --duration=30
```

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
- The cube spawn region (`x∈[0.20,0.28], y∈[-0.07,0.04]`) is where grasps are reliable; +y is tighter because the moving jaw is offset toward +y.

## Pointing at the real arm

The `keyboard_pose` teleoperator and the processor pipeline in `scripts/sim_pipelines.py` are robot-agnostic.
Swap `SO101Sim` for lerobot's `SO101Follower` (`--robot.type=so101_follower`, `use_degrees=true`) and the same keyboard EE teleop drives the physical arm.
