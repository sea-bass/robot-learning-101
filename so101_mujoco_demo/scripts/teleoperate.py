"""Keyboard-teleoperate the SO-101 in MuJoCo (no recording) - for practice.

    pixi run so101-teleoperate
    pixi run so101-teleoperate --colors red green   # x cycles the target cube

arrows = EE forward/back/left/right, w/s = up/down, q/e = pitch,
a/d = roll jaws, space = toggle gripper, x = reset scene, ESC = quit.
"""

import argparse
import time

from pynput import keyboard as pynput_keyboard

from lerobot_robot_so101_sim import SO101Sim, SO101SimConfig
from lerobot_teleoperator_keyboard_pose import KeyboardPose, KeyboardPoseConfig
from sim_pipelines import FPS, add_colors_arg, make_teleop_action_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_colors_arg(parser)
    args = parser.parse_args()

    robot = SO101Sim(SO101SimConfig(show_viewer=True, target_color=args.colors[0]))
    teleop = KeyboardPose(KeyboardPoseConfig())
    pipeline = make_teleop_action_pipeline()

    events = {"reset": False, "quit": False}

    def on_press(key):
        if key == pynput_keyboard.Key.esc:
            events["quit"] = True
        elif isinstance(key, pynput_keyboard.KeyCode) and key.char and key.char.lower() == "x":
            events["reset"] = True

    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.start()

    robot.connect()
    teleop.connect()
    print("arrows/w/s = move, q/e = pitch, a/d = roll, space = gripper, x = reset, ESC = quit")
    print(f"target: the {robot.target_color.upper()} cube")

    was_success = False
    resets = 0
    try:
        while not events["quit"] and (robot.viewer is None or robot.viewer.is_running()):
            t0 = time.perf_counter()
            if events["reset"]:
                events["reset"] = False
                resets += 1
                robot.reset_scene(target_color=args.colors[resets % len(args.colors)])
                teleop.reset_state()
                pipeline.reset()
                was_success = False
                print(f"target: the {robot.target_color.upper()} cube")

            obs = robot.get_observation()
            action = pipeline((teleop.get_action(), obs))
            robot.send_action(action)

            if robot.is_success and not was_success:
                print(f"SUCCESS - {robot.target_color} cube lifted! (x to reset)")
            was_success = robot.is_success

            leftover = 1 / FPS - (time.perf_counter() - t0)
            if leftover > 0:
                time.sleep(leftover)
    finally:
        robot.disconnect()
        teleop.disconnect()
        listener.stop()


if __name__ == "__main__":
    main()
