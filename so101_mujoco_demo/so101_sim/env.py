"""Gymnasium environment: SO-101 arm picking a cube in MuJoCo.

Observations: (6,) joint positions in radians (5 arm joints + gripper).
Actions: (6,) absolute joint position targets in radians, same order as
JOINT_NAMES — the same interface LeRobot records from the real SO-101.
Camera images are rendered on demand via ``render_camera()`` (the LeRobot
robot plugin attaches them to its own observations).
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

MODEL_XML = Path(__file__).parent / "model" / "scene.xml"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EE_SITE = "gripperframe"

GRIPPER_OPEN = 1.3
GRIPPER_CLOSED = -0.1


class SO101PickCubeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        image_size: tuple[int, int] = (240, 320),
        control_dt: float = 0.04,
        # spawn region where grasps are reliable; +y is slightly tighter because
        # the moving jaw is offset toward +y and grasps there are harder
        cube_spawn_x: tuple[float, float] = (0.20, 0.28),
        cube_spawn_y: tuple[float, float] = (-0.07, 0.04),
        lift_height: float = 0.08,
    ):
        super().__init__()
        self.image_size = image_size
        self.lift_height = lift_height
        self.cube_spawn_x = cube_spawn_x
        self.cube_spawn_y = cube_spawn_y

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = round(control_dt / self.model.opt.timestep)
        self.control_dt = self.n_substeps * self.model.opt.timestep

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES
        ]
        self.qpos_ids = [self.model.jnt_qposadr[j] for j in self.joint_ids]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        cube_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_qpos_adr = self.model.jnt_qposadr[cube_joint]
        self.home_keyframe = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        ctrl_range = self.model.actuator_ctrlrange.copy()
        self.action_space = spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self._renderer = None

    # ------------------------------------------------------------------ helpers

    @property
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            h, w = self.image_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        return self._renderer

    def render_camera(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].astype(np.float32)

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3].copy()

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _is_success(self) -> bool:
        cube = self.cube_pos
        lifted = cube[2] > self.lift_height
        held = np.linalg.norm(cube - self.ee_pos) < 0.08
        return bool(lifted and held)

    # ------------------------------------------------------------------ gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_keyframe)
        # randomize cube spawn position and yaw
        x = self.np_random.uniform(*self.cube_spawn_x)
        y = self.np_random.uniform(*self.cube_spawn_y)
        yaw = self.np_random.uniform(-np.pi / 4, np.pi / 4)
        quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = [x, y, 0.0125]
        self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = quat
        mujoco.mj_forward(self.model, self.data)
        return self.joint_pos, {"cube_pos": self.cube_pos}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        success = self._is_success()
        reward = 1.0 if success else 0.0
        info = {"is_success": success, "cube_pos": self.cube_pos, "ee_pos": self.ee_pos}
        return self.joint_pos, reward, success, False, info

    def render(self):
        return self.render_camera("front")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
