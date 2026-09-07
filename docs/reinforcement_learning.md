# Reinforcement learning with mjlab

[mjlab](https://github.com/mujocolab/mjlab) simulates thousands of environments in parallel with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) and trains PPO policies on them with [rsl_rl](https://github.com/leggedrobotics/rsl_rl).
Environments are built Isaac Lab-style from *managers* (observations, rewards, terminations, commands, ...), so a task is a config file, not a loop.
Two of its built-in tasks are wired up here.

Run the commands from the repository root; training logs land in `logs/rsl_rl/<experiment>/<timestamp>/`.
mjlab logs to [Weights & Biases](https://wandb.ai) by default; pass `--agent.logger tensorboard` if you don't have an account.

## Quick check: the packaged demo

Downloads a pretrained G1 policy and opens the viewer, no training needed:

```bash
pixi run mjlab-demo
```

## 1. Humanoid walking (Unitree G1)

`Mjlab-Velocity-Flat-Unitree-G1`: the G1 learns to follow random linear/angular velocity commands on flat ground.
The task is defined in `external/mjlab/src/mjlab/tasks/velocity/`.

```bash
pixi run g1-walk-train                       # 4096 envs, 30k iterations by default
pixi run g1-walk-train --agent.max-iterations 2000 --env.scene.num-envs 2048
```

A gait emerges within a few hundred iterations.
Watch a checkpoint:

```bash
pixi run g1-walk-play --checkpoint-file logs/rsl_rl/g1_velocity/<timestamp>/model_2000.pt
pixi run g1-walk-play --wandb-run-path <entity>/mjlab/<run-id>   # if logging to W&B
```

`play` opens the native MuJoCo viewer, or a browser viewer at http://localhost:8080 with `--viewer viser`.
Dummy agents are handy for sanity-checking a task before training: `--agent zero` / `--agent random`.

## 2. Arm manipulation from pixels (i2rt YAM)

`Mjlab-Lift-Cube-Yam-Rgb`: a YAM arm lifts a cube using an RGB camera observation plus proprioception.
`Mjlab-Lift-Cube-Yam` is the state-based variant, which trains faster.
The task lives in `external/mjlab/src/mjlab/tasks/manipulation/`.

```bash
pixi run yam-lift-train                      # 4096 envs, 3k iterations by default
pixi run yam-lift-play --checkpoint-file logs/rsl_rl/yam_lift_cube_vision/<timestamp>/model_3000.pt
```

Rendering 4096 cameras is memory hungry; drop `--env.scene.num-envs` (e.g. 1024) on an 8 GB GPU.

## Any other task

`pixi run mjlab-list-envs` prints every registered task.
The generic entry points take a task id as the first argument:

```bash
pixi run mjlab-train Mjlab-Velocity-Flat-Unitree-Go1 --env.scene.num-envs 4096
pixi run mjlab-play  Mjlab-Velocity-Flat-Unitree-Go1 --checkpoint-file ...
```

## Where to look next

- `external/mjlab/src/mjlab/tasks/velocity/velocity_env_cfg.py`: every reward term, observation and termination of the walking task in one file.
- `external/mjlab/scripts/demos/`: small standalone scripts (differential IK, raycast sensors, terrain) that show the API without RL.
- The [mjlab docs](https://mujocolab.github.io/mjlab/) for the full picture.
