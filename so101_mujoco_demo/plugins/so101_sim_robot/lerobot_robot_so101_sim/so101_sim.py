"""LeRobot Robot plugin wrapping the MuJoCo SO-101 pick-cube simulation.

Joint positions cross this boundary in degrees (the LeRobot / real-SO-101
convention); the MuJoCo env underneath works in radians.
"""

from functools import cached_property
from typing import Any

import numpy as np

from lerobot.robots import Robot

from so101_sim.env import JOINT_NAMES, SO101PickCubeEnv

from .config_so101_sim import SO101SimConfig


class SO101Sim(Robot):
    config_class = SO101SimConfig
    name = "so101_sim"

    def __init__(self, config: SO101SimConfig):
        super().__init__(config)
        self.config = config
        self.env: SO101PickCubeEnv | None = None
        self.viewer = None
        self.cameras: dict = {}  # no physical cameras; images come from the renderer

    # ------------------------------------------------------------- features

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in JOINT_NAMES}

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        h, w = self.config.image_height, self.config.image_width
        return {"wrist": (h, w, 3), "front": (h, w, 3)}

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    # ----------------------------------------------------------- connection

    @property
    def is_connected(self) -> bool:
        return self.env is not None

    def connect(self, calibrate: bool = True) -> None:
        self.env = SO101PickCubeEnv(
            image_size=(self.config.image_height, self.config.image_width),
        )
        self.env.reset(seed=self.config.seed)
        if self.config.show_viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        self.configure()

    def disconnect(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------ I/O

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        joint_pos_deg = np.degrees(self.env.joint_pos)
        obs: dict[str, Any] = {
            f"{joint}.pos": float(pos) for joint, pos in zip(JOINT_NAMES, joint_pos_deg)
        }
        obs["wrist"] = self.env.render_camera("wrist")
        obs["front"] = self.env.render_camera("front")
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        target_deg = np.array([action[f"{joint}.pos"] for joint in JOINT_NAMES], dtype=np.float64)
        self.env.step(np.radians(target_deg).astype(np.float32))
        if self.viewer is not None:
            self.viewer.sync()
        return action

    # -------------------------------------------------------- sim utilities

    def reset_scene(self, seed: int | None = None) -> None:
        """Re-home the arm and respawn the cube (between episodes)."""
        self.env.reset(seed=seed)
        if self.viewer is not None:
            self.viewer.sync()

    @property
    def is_success(self) -> bool:
        return self.env._is_success()
