# Robot Learning 101

Hands-on examples of robot learning, runnable end to end in simulation:

- **Reinforcement learning** with [mjlab](https://github.com/mujocolab/mjlab): a humanoid learns to walk, an arm learns to lift a cube.
- **Imitation learning** with [LeRobot](https://github.com/huggingface/lerobot): teleoperate an SO-101 arm, record demonstrations, train a policy.

## Install

1. Install [pixi](https://pixi.sh/latest/#installation).
2. Clone with submodules (LeRobot and mjlab live under `external/`):

   ```bash
   git clone --recurse-submodules https://github.com/sea-bass/robot-learning-101.git
   cd robot-learning-101
   ```

3. Build the environments, or let the first `pixi run` do it:

   ```bash
   pixi install --all
   ```

You need Linux and an NVIDIA GPU for training.
`pixi task list` shows every available command.

## Guides

- [Reinforcement learning](docs/reinforcement_learning.md)
- [Imitation learning](docs/imitation_learning.md)
