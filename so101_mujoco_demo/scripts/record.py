"""Record a LeRobotDataset by teleoperating the SO-101 in MuJoCo.

    pixi run so101-record --repo-id you/so101_pick_cube --episodes 100 --display-data
    pixi run so101-record --repo-id you/so101_pick_two --episodes 200 --colors red green
    pixi run so101-record --repo-id you/so101_pick_red --teleop leader --port /dev/SO101Leader
    pixi run so101-record --repo-id you/so101_pick_red --teleop leader --viewer viser   # browser viewer, e.g. over SSH

--teleop keyboard (default) drives the end-effector with the keyboard_pose
teleoperator; --teleop leader mirrors a physical SO-101 leader arm joint for
joint. Either way the scene is re-randomized before every episode, and with the
leader the sim arm then eases from home to the leader's pose before recording.

The scene has a red and a green cube. Episodes alternate through --colors and
the terminal tells you which cube to pick; that color's task string is stored
with the episode (this is what language-conditioned policies train on).

Motion keys (--teleop keyboard):
    arrows = EE forward/back/left/right, w/s = up/down, q/e = pitch,
    a/d = roll jaws, space = toggle gripper open/close

Episode controls (the letter keys of lerobot-record; its Right/Left/q
equivalents are motion keys here, so they are not bound):
    n   = end episode & save        r = end episode & re-record
    ESC = stop recording (finalizes the dataset)

Uses lerobot's own record_loop + SO-follower EE processor pipeline, so the
recorded schema (joint actions in degrees, wrist/front cameras) matches real
SO-101 datasets.
"""

import argparse
import logging
import time

import numpy as np
from pynput import keyboard as pynput_keyboard

from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.feature_utils import combine_feature_dicts

from lerobot_robot_so101_sim import SO101Sim, SO101SimConfig
from lerobot_teleoperator_keyboard_pose import KeyboardPose, KeyboardPoseConfig
from so101_sim.env import JOINT_NAMES, TASKS
from sim_pipelines import (
    FPS,
    add_colors_arg,
    add_viewer_args,
    identity_action_pipeline,
    identity_observation_pipeline,
    make_teleop_action_pipeline,
    viewer_kwargs,
)

EPISODE_TIME_S = 120
SYNC_TIME_S = 1.0  # leader only: ease the re-homed sim arm onto the leader's pose


def episode_control_listener(events: dict) -> pynput_keyboard.Listener:
    """n = save, r = re-record, ESC = stop. (Arrows and q stay free for driving.)"""

    def on_press(key):
        char = key.char.lower() if isinstance(key, pynput_keyboard.KeyCode) and key.char else None
        if char == "n":
            events["exit_early"] = True
        elif char == "r":
            events["rerecord_episode"] = True
            events["exit_early"] = True
        elif key == pynput_keyboard.Key.esc:
            events["stop_recording"] = True
            events["exit_early"] = True

    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


class _HideSlowLoopWarnings(logging.Filter):
    """Drop record_loop's per-tick "running slower than the target FPS" warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("Record loop is running slower")


def sync_to_leader(robot: SO101Sim, teleop: SO101Leader) -> None:
    """Ease the sim arm from its (just re-homed) pose onto the leader's, unrecorded.

    Without this the first recorded action would snap the arm from home to
    wherever the leader is held, which can knock the freshly spawned cubes.
    """
    obs = robot.get_observation()
    start = np.array([obs[f"{joint}.pos"] for joint in JOINT_NAMES])
    ticks = max(1, round(SYNC_TIME_S * FPS))
    for tick in range(1, ticks + 1):
        t0 = time.perf_counter()
        leader = teleop.get_action()
        goal = np.array([leader[f"{joint}.pos"] for joint in JOINT_NAMES])
        target = start + (tick / ticks) * (goal - start)
        robot.send_action({f"{joint}.pos": float(pos) for joint, pos in zip(JOINT_NAMES, target)})
        leftover = 1 / FPS - (time.perf_counter() - t0)
        if leftover > 0:
            time.sleep(leftover)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="dataset repo id, e.g. you/so101_pick_cube")
    parser.add_argument("--root", default=None, help="local dataset root (default: HF cache)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--display-data", action="store_true", help="live rerun view")
    parser.add_argument(
        "--teleop", choices=["keyboard", "leader"], default="keyboard",
        help="keyboard EE teleop, or a physical SO-101 leader arm (default: keyboard)",
    )
    parser.add_argument("--port", default="/dev/SO101Leader", help="leader arm serial port (--teleop leader)")
    parser.add_argument(
        "--teleop-id", default="leader_arm",
        help="leader calibration id, i.e. <id>.json in the so_leader calibration folder (--teleop leader)",
    )
    parser.add_argument(
        "--fps-warnings", action="store_true",
        help="show record_loop's warning on every tick that runs slower than the target FPS",
    )
    add_viewer_args(parser)
    add_colors_arg(parser)
    args = parser.parse_args()

    if not args.fps_warnings:
        # record_loop logs it through the root logger, once per slow tick.
        logging.getLogger().addFilter(_HideSlowLoopWarnings())

    robot = SO101Sim(SO101SimConfig(show_viewer=True, **viewer_kwargs(args)))
    if args.teleop == "leader":
        # The leader already outputs {joint}.pos in degrees, so it needs no IK pipeline.
        teleop = SO101Leader(SO101LeaderConfig(port=args.port, id=args.teleop_id))
        teleop_action_processor = identity_action_pipeline()
    else:
        teleop = KeyboardPose(KeyboardPoseConfig())
        teleop_action_processor = make_teleop_action_pipeline()
    robot_action_processor = identity_action_pipeline()
    robot_observation_processor = identity_observation_pipeline()

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        root=args.root,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=teleop.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=8,
    )

    teleop.connect()  # leader: calibrates interactively if <teleop-id>.json is missing
    robot.connect()

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    listener = episode_control_listener(events)

    if args.display_data:
        from lerobot.utils.rerun_visualization import init_rerun

        init_rerun(session_name="so101_sim_record")

    if args.teleop == "keyboard":
        print("arrows/w/s = move, q/e = pitch, space = gripper")
    else:
        print("leader arm: hold it near the home pose between episodes")
    print("n = save episode | r = re-record | ESC = stop")

    episode_idx = 0
    try:
        while episode_idx < args.episodes and not events["stop_recording"]:
            # fresh scene (target cube for this episode) + re-anchored teleop reference
            color = args.colors[episode_idx % len(args.colors)]
            robot.reset_scene(target_color=color)
            if args.teleop == "leader":
                sync_to_leader(robot, teleop)
            else:
                teleop.reset_state()
            teleop_action_processor.reset()
            print(f"recording episode {episode_idx + 1}/{args.episodes} - lift the {color.upper()} cube!")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=EPISODE_TIME_S,
                single_task=TASKS[color],
                display_data=args.display_data,
            )

            if events["rerecord_episode"]:
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                print("re-recording episode")
            elif not events["stop_recording"]:
                dataset.save_episode()
                episode_idx += 1
                print(f"saved ({episode_idx}/{args.episodes})")
    finally:
        robot.disconnect()
        teleop.disconnect()
        listener.stop()
        dataset.finalize()

    print(f"Done: {episode_idx} episodes -> {dataset.root}")
    print(f"  lerobot-dataset-viz --repo-id {args.repo_id} --root {dataset.root} --episode-index 0")


if __name__ == "__main__":
    main()
