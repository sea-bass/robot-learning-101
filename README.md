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

4. Log in to [Weights & Biases](https://docs.coreweave.com/products/wandb/quickstart) for experiment tracking (optional).
   W&B accounts now sign in through [CoreWeave Forge](https://forge.coreweave.com); existing W&B credentials and API keys still work.
   To get a key, sign in to Forge, open your profile icon > **User settings** > **API keys** > **New key**, and copy it (it is shown only once).
   The `wandb` package is already in both environments, and the key is saved to `~/.netrc`, so one login covers both:

   ```bash
   pixi run -e lerobot wandb login   # paste the API key when prompted
   ```

   Alternatively, set `WANDB_API_KEY` in your shell.
   Without an account, skip `--wandb.enable=true` for LeRobot and pass `--agent.logger tensorboard` to mjlab.

You need Linux and an NVIDIA GPU for training.
`pixi task list` shows every available command.

## Guides

- [Reinforcement learning](docs/reinforcement_learning.md)
- [Imitation learning](docs/imitation_learning.md)
