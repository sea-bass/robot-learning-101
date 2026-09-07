from gymnasium.envs.registration import register

from so101_sim.env import SO101PickCubeEnv  # noqa: F401

register(
    id="SO101PickCube-v0",
    entry_point="so101_sim.env:SO101PickCubeEnv",
    max_episode_steps=500,
)
