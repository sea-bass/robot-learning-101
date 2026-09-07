"""End-effector target tracking on top of LeRobot's placo IK.

The controller maintains a virtual EE target (position + pitch offset). The
orientation target is kept consistent with base pan — it is the home grasp
orientation yawed toward the target position — so the 5-DOF arm can satisfy
position and orientation simultaneously (a fixed world-frame orientation target
would conflict with lateral motion and leave centimeter-level position error).
"""

import numpy as np

from so101_sim.env import GRIPPER_CLOSED, GRIPPER_OPEN, SO101PickCubeEnv
from so101_sim.ik import SO101Kinematics

WORKSPACE_LOW = np.array([0.12, -0.25, 0.005])
WORKSPACE_HIGH = np.array([0.38, 0.25, 0.35])

PITCH_RANGE = (-1.0, 1.0)  # rad, offset from the home grasp pitch
ROLL_RANGE = (-0.9, 0.9)  # rad, jaw-plane roll about the tool axis


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class EETargetController:
    """Tracks a virtual EE target pose and converts it to joint-target actions."""

    def __init__(self, env: SO101PickCubeEnv, ori_weight: float = 0.02):
        self.env = env
        self.kin = SO101Kinematics(ori_weight=ori_weight)
        self.target_pos = np.zeros(3)
        self.pitch = 0.0
        self.roll = 0.0
        self.gripper_open = True
        self._r_nominal = np.eye(3)
        self.reset()

    @property
    def q_arm(self) -> np.ndarray:
        return self.env.joint_pos[:5].astype(np.float64)

    def reset(self):
        """Re-anchor the target at the current EE pose (call after env.reset())."""
        pose = self.kin.fk(self.q_arm)
        self.target_pos = pose[:3, 3].copy()
        yaw = np.arctan2(self.target_pos[1], self.target_pos[0])
        self._r_nominal = _rot_z(-yaw) @ pose[:3, :3]
        self.pitch = 0.0
        self.roll = 0.0
        self.gripper_open = True

    @property
    def target_rot(self) -> np.ndarray:
        yaw = np.arctan2(self.target_pos[1], self.target_pos[0])
        return _rot_z(yaw) @ self._r_nominal @ _rot_y(self.pitch) @ _rot_z(self.roll)

    def move(self, delta_pos: np.ndarray, delta_pitch: float = 0.0, delta_roll: float = 0.0):
        """Displace the EE target, clamped to the workspace."""
        self.target_pos = np.clip(self.target_pos + delta_pos, WORKSPACE_LOW, WORKSPACE_HIGH)
        self.pitch = np.clip(self.pitch + delta_pitch, *PITCH_RANGE)
        self.roll = np.clip(self.roll + delta_roll, *ROLL_RANGE)

    def action(self) -> np.ndarray:
        """Solve IK toward the current target and return the (6,) joint-target action."""
        pose = np.eye(4)
        pose[:3, :3] = self.target_rot
        pose[:3, 3] = self.target_pos
        q_arm = self.kin.ik(self.q_arm, pose)
        gripper = GRIPPER_OPEN if self.gripper_open else GRIPPER_CLOSED
        action = np.append(q_arm, gripper).astype(np.float32)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def ee_error(self) -> float:
        """Distance between the actual EE position and the target."""
        return float(np.linalg.norm(self.kin.fk(self.q_arm)[:3, 3] - self.target_pos))
