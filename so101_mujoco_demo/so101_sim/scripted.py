"""Scripted expert for the pick-cube task.

``scripted_pick`` is a generator yielding one (6,) action per control tick; the
caller steps the environment with each action, which lets the recording loop
capture observation/action pairs exactly like a human teleop session would.

Grasp strategy (found empirically — see README):
    1. pitch the gripper steep (near-vertical beak) and hover above the cube,
       offset slightly *behind* it so the fixed jaw clears the cube on descent
    2. descend so the jaw tips are at floor level, still behind the cube
    3. slide forward so the cube seats in the jaw pocket
    4. ramp the gripper closed slowly (a fast close smacks the cube away)
    5. lift; retry from step 1 at the cube's new position if the grasp slipped
"""

from collections.abc import Iterator

import numpy as np

from so101_sim.control import WORKSPACE_HIGH, WORKSPACE_LOW, EETargetController
from so101_sim.env import CUBE_HALF, GRIPPER_OPEN, SO101PickCubeEnv

GRASP_PITCH = 0.6      # rad beyond home pitch: near-vertical approach
POCKET_PERP = 0.01     # jaw pocket offset from EE frame toward the moving jaw
BEHIND_OFFSET = 0.008  # descend clearance behind the cube
SEAT_OFFSET = 0.003    # remaining clearance once the cube is seated
CLOSE_TICKS = 22       # gripper close ramp duration



def _moving_jaw_dir(ctl: EETargetController) -> np.ndarray:
    """Unit vector from the fixed jaw toward the moving jaw (rolls with the wrist)."""
    return -(ctl.target_rot @ np.array([1.0, 0.0, 0.0]))


def _pocket_offset(ctl: EETargetController) -> np.ndarray:
    return POCKET_PERP * _moving_jaw_dir(ctl)


def _horiz(ctl: EETargetController) -> np.ndarray:
    """Horizontal (unit) projection of the fixed-jaw-to-moving-jaw direction."""
    h = _moving_jaw_dir(ctl)
    h[2] = 0.0
    return h / np.linalg.norm(h)


def _cube_relative_roll(env: SO101PickCubeEnv) -> float:
    """Roll command that aligns the jaw plane with the cube's nearest face pair.

    The cube has 4-fold symmetry, so the relative yaw is wrapped to [-45, 45]
    degrees; world jaw yaw responds as -roll (the tool axis points down).
    """
    adr = env.cube_qpos_adr
    qw, qx, qy, qz = env.data.qpos[adr + 3 : adr + 7]
    cube_yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    cube = env.cube_pos
    approach_yaw = np.arctan2(cube[1], cube[0])
    rel = cube_yaw - approach_yaw
    rel = (rel + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(-rel)


def _goto(env, ctl, pocket_goal, pitch_goal=GRASP_PITCH, roll_goal=0.0, max_ticks=250, step=0.005):
    best_err = np.inf
    stalled = 0
    for _ in range(max_ticks):
        dpitch = np.clip(pitch_goal - ctl.pitch, -0.02, 0.02)
        droll = np.clip(roll_goal - ctl.roll, -0.03, 0.03)
        # clamp to the workspace box: the controller clamps its target the same
        # way, and an out-of-workspace goal would otherwise never register as
        # "reached", stalling the phase for its full tick budget
        goal_ee = np.clip(pocket_goal - _pocket_offset(ctl), WORKSPACE_LOW, WORKSPACE_HIGH)
        delta = goal_ee - ctl.target_pos
        dist = np.linalg.norm(delta)
        ctl.move(delta / max(dist, 1e-9) * min(step, dist), dpitch, droll)
        yield ctl.action()
        if dist < 1e-4 and abs(pitch_goal - ctl.pitch) < 1e-3 and abs(roll_goal - ctl.roll) < 1e-3:
            # settle until the EE converges — or stops improving (the soft-
            # orientation IK can leave a steady-state bias above any fixed
            # tolerance, which must not stall the phase)
            err = ctl.ee_error()
            if err < 0.005:
                return
            if err < best_err - 3e-4:
                best_err = err
                stalled = 0
            else:
                stalled += 1
                if stalled >= 8:
                    return


def scripted_pick(
    env: SO101PickCubeEnv, ctl: EETargetController, retries: int = 2
) -> Iterator[np.ndarray]:
    """Yield actions that pick up the cube; stops shortly after success or gives up."""
    for _ in range(1 + retries):
        cube = env.cube_pos.copy()
        cube[2] = CUBE_HALF
        ctl.gripper_open = True
        roll = _cube_relative_roll(env)  # align jaw plane with the cube's faces
        yield from _goto(env, ctl, cube + [0, 0, 0.09] - BEHIND_OFFSET * _horiz(ctl), roll_goal=roll)
        yield from _goto(env, ctl, cube + [0, 0, -0.012] - BEHIND_OFFSET * _horiz(ctl), roll_goal=roll)
        yield from _goto(env, ctl, cube + [0, 0, -0.012] - SEAT_OFFSET * _horiz(ctl), roll_goal=roll, step=0.002)
        close_target = env.action_space.low[5] + 0.075  # slight squeeze margin
        for i in range(CLOSE_TICKS):
            action = ctl.action()
            action[5] = GRIPPER_OPEN + (close_target - GRIPPER_OPEN) * (i + 1) / CLOSE_TICKS
            yield action
        ctl.gripper_open = False
        yield from _goto(env, ctl, cube + [0, 0, 0.13], roll_goal=roll, max_ticks=120, step=0.004)
        for _ in range(10):
            yield ctl.action()
        if env._is_success():
            return
