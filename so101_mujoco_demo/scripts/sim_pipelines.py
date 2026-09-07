"""Shared processor-pipeline construction for the SO-101 sim demo scripts.

The teleop action pipeline converts keyboard EE-pose offsets into joint-space
actions using lerobot's own SO-follower processor steps + placo kinematics:

    keyboard {target_*}  ->  EEReferenceAndDelta  ->  EEBoundsAndSafety
        ->  GripperVelocityToJoint  ->  InverseKinematicsEEToJoints  ->  {joint}.pos

Because the full chain lives in the *teleop* pipeline, the dataset records
joint-space actions in degrees — the same schema as real SO-101 datasets.
"""

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)

from so101_sim.control import WORKSPACE_HIGH, WORKSPACE_LOW
from so101_sim.env import CUBE_COLORS, JOINT_NAMES, TASKS
from so101_sim.ik import ARM_JOINTS, EE_FRAME, URDF_PATH

FPS = 25
TASK = TASKS["red"]  # single-task default; language-conditioned runs use TASKS[color]
WORKSPACE_BOUNDS = {"min": WORKSPACE_LOW.tolist(), "max": WORKSPACE_HIGH.tolist()}

# gripper joint units are degrees: -8 = firm close, 74 = open (wider openings
# make the moving finger protrude far enough to bulldoze the cube on approach)
GRIPPER_CLIP = (-8.0, 74.0)
GRIPPER_SPEED_FACTOR = 2.4  # deg per tick at |gripper_vel| = 1 (~60 deg/s at 25 fps)


def add_colors_arg(parser, default=("red",)):
    """--colors red green: which cube(s) to target; episodes cycle through the list."""
    parser.add_argument(
        "--colors", nargs="+", choices=CUBE_COLORS, default=list(default), metavar="COLOR",
        help=f"target cube color(s), episodes alternate through them (choices: {', '.join(CUBE_COLORS)}; "
        f"default: {' '.join(default)})",
    )


def make_kinematics() -> RobotKinematics:
    return RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=EE_FRAME,
        joint_names=list(ARM_JOINTS),
    )


def make_teleop_action_pipeline() -> RobotProcessorPipeline:
    """Keyboard EE-pose offsets -> joint-space action (degrees)."""
    kinematics = make_kinematics()
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEReferenceAndDelta(
                kinematics=kinematics,
                end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
                motor_names=list(ARM_JOINTS),
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(end_effector_bounds=WORKSPACE_BOUNDS, max_ee_step_m=0.05),
            GripperVelocityToJoint(
                speed_factor=GRIPPER_SPEED_FACTOR,
                clip_min=GRIPPER_CLIP[0],
                clip_max=GRIPPER_CLIP[1],
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=list(JOINT_NAMES),
                initial_guess_current_joints=True,
                orientation_weight=0.02,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def identity_action_pipeline() -> RobotProcessorPipeline:
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def identity_observation_pipeline() -> RobotProcessorPipeline:
    return RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
