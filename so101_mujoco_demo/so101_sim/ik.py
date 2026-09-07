"""Kinematics for the SO-101 arm, backed by LeRobot's placo-based solver.

This reuses ``lerobot.model.kinematics.RobotKinematics`` — the same IK stack
LeRobot uses for end-effector teleop on the real SO-100/SO-101 — pointed at the
SO-101 URDF that ships alongside the MJCF. The URDF's ``gripper_frame_link``
coincides with the MJCF ``gripperframe`` site, so IK targets computed here are
directly valid in the MuJoCo scene.

LeRobot's API works in degrees (real-robot convention); this adapter exposes
radians to match the simulation.
"""

from pathlib import Path

import numpy as np
from lerobot.model.kinematics import RobotKinematics

from so101_sim.env import JOINT_NAMES

ARM_JOINTS = JOINT_NAMES[:5]
URDF_PATH = Path(__file__).parent / "model" / "so101_new_calib.urdf"
EE_FRAME = "gripper_frame_link"


class SO101Kinematics:
    def __init__(self, ori_weight: float = 0.02):
        self.kin = RobotKinematics(str(URDF_PATH), EE_FRAME, ARM_JOINTS)
        self.ori_weight = ori_weight

    def fk(self, q_arm: np.ndarray) -> np.ndarray:
        """4x4 EE pose in world frame from 5 arm joint positions (radians)."""
        return self.kin.forward_kinematics(np.rad2deg(q_arm))

    def ik(self, q_arm: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        """Solve for 5 arm joint positions (radians) reaching the target 4x4 pose.

        Warm-started at ``q_arm``; position is weighted much higher than
        orientation since the 5-DOF arm cannot track full 6D poses.
        """
        q_deg = self.kin.inverse_kinematics(
            np.rad2deg(q_arm),
            target_pose,
            position_weight=1.0,
            orientation_weight=self.ori_weight,
        )
        return np.deg2rad(q_deg)
