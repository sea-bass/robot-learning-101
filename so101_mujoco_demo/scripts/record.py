"""Record a LeRobotDataset by keyboard-teleoperating the SO-101 in MuJoCo.

    pixi run so101-record --repo-id you/so101_pick_cube --episodes 100 --display-data
    pixi run so101-record --repo-id you/so101_pick_two --episodes 200 --colors red green

The scene has a red and a green cube. Episodes alternate through --colors and
the terminal tells you which cube to pick; that color's task string is stored
with the episode (this is what language-conditioned policies train on).

Motion keys (keyboard_pose teleoperator):
    arrows = EE forward/back/left/right, w/s = up/down, q/e = pitch,
    a/d = roll jaws, space = toggle gripper open/close

Episode controls:
    ENTER = end episode & save        x = end episode & re-record
    ESC   = stop recording (finalizes the dataset)

Uses lerobot's own record_loop + SO-follower EE processor pipeline, so the
recorded schema (joint actions in degrees, wrist/front cameras) matches real
SO-101 datasets.
"""

import argparse

from pynput import keyboard as pynput_keyboard

from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.feature_utils import combine_feature_dicts

from lerobot_robot_so101_sim import SO101Sim, SO101SimConfig
from lerobot_teleoperator_keyboard_pose import KeyboardPose, KeyboardPoseConfig
from so101_sim.env import TASKS
from sim_pipelines import (
    FPS,
    add_colors_arg,
    identity_action_pipeline,
    identity_observation_pipeline,
    make_teleop_action_pipeline,
)

EPISODE_TIME_S = 120


def episode_control_listener(events: dict) -> pynput_keyboard.Listener:
    """ENTER = save, x = re-record, ESC = stop. (Arrows stay free for driving.)"""

    def on_press(key):
        if key == pynput_keyboard.Key.enter:
            events["exit_early"] = True
        elif isinstance(key, pynput_keyboard.KeyCode) and key.char and key.char.lower() == "x":
            events["rerecord_episode"] = True
            events["exit_early"] = True
        elif key == pynput_keyboard.Key.esc:
            events["stop_recording"] = True
            events["exit_early"] = True

    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="dataset repo id, e.g. you/so101_pick_cube")
    parser.add_argument("--root", default=None, help="local dataset root (default: HF cache)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--display-data", action="store_true", help="live rerun view")
    add_colors_arg(parser)
    args = parser.parse_args()

    robot = SO101Sim(SO101SimConfig(show_viewer=True))
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

    robot.connect()
    teleop.connect()

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    listener = episode_control_listener(events)

    if args.display_data:
        from lerobot.utils.rerun_visualization import init_rerun

        init_rerun(session_name="so101_sim_record")

    print("arrows/w/s = move, q/e = pitch, space = gripper")
    print("ENTER = save episode | x = re-record | ESC = stop")

    episode_idx = 0
    try:
        while episode_idx < args.episodes and not events["stop_recording"]:
            # fresh scene (target cube for this episode) + re-anchored teleop reference
            color = args.colors[episode_idx % len(args.colors)]
            robot.reset_scene(target_color=color)
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
