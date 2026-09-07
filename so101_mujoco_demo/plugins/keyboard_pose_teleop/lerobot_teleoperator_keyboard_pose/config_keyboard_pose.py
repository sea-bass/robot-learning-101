from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard_pose")
@dataclass
class KeyboardPoseConfig(TeleoperatorConfig):
    """Keyboard EE-pose teleoperator.

    Emits the cumulative pose-offset action expected by lerobot's
    ``EEReferenceAndDelta`` processor step (with ``use_latched_reference=True``
    and unit step sizes): target_x/y/z in meters, target_wx/wy/wz as a
    rotation vector in the latched EE frame, plus a gripper velocity.
    """

    move_speed: float = 0.15    # m/s
    pitch_speed: float = 1.2    # rad/s
    gripper_vel: float = 1.0    # magnitude passed to GripperVelocityToJoint
