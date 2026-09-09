"""Collect pick-cube episodes with the scripted expert (headless, no display).

Bootstraps a dataset or smoke-tests the pipeline without teleoperating.
Records through the SO101Sim robot interface, so the dataset schema (joint
actions in degrees, wrist/front cameras) is identical to teleop recordings.

    pixi run so101-collect --repo-id you/so101_pick_cube --episodes 100
    pixi run so101-collect --repo-id you/so101_pick_two --episodes 200 --colors red green

With several --colors the episodes alternate between them and each episode's
task string names its target ("Pick up the green cube and lift it."), which
is what a language-conditioned policy such as SmolVLA trains on.
"""

import argparse
import time

import numpy as np

from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.datasets import LeRobotDataset
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features

from lerobot_robot_so101_sim import SO101Sim, SO101SimConfig
from so101_sim.control import EETargetController
from so101_sim.env import JOINT_NAMES, TASKS
from so101_sim.scripted import scripted_pick
from sim_pipelines import FPS, add_colors_arg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-viewer", action="store_true")
    add_colors_arg(parser)
    args = parser.parse_args()

    robot = SO101Sim(SO101SimConfig(show_viewer=args.show_viewer))
    robot.connect()

    features = combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, use_video=True),
        hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True),
    )
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        root=args.root,
        features=features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=8,
    )

    ctl = EETargetController(robot.env)
    rng = np.random.default_rng(args.seed)

    saved = attempts = 0
    try:
        while saved < args.episodes:
            attempts += 1
            color = args.colors[saved % len(args.colors)]
            robot.reset_scene(seed=int(rng.integers(0, 2**31)), target_color=color)
            ctl.reset()
            success = False
            for action_rad in scripted_pick(robot.env, ctl):
                t0 = time.perf_counter()
                obs = robot.get_observation()
                action = {
                    f"{j}.pos": float(np.degrees(a)) for j, a in zip(JOINT_NAMES, action_rad)
                }
                frame = {
                    **build_dataset_frame(features, obs, prefix=OBS_STR),
                    **build_dataset_frame(features, action, prefix=ACTION),
                    "task": TASKS[color],
                }
                dataset.add_frame(frame)
                robot.send_action(action)
                if args.show_viewer:  # watchable real-time pacing; headless runs flat out
                    leftover = 1 / FPS - (time.perf_counter() - t0)
                    if leftover > 0:
                        time.sleep(leftover)
                if robot.is_success:
                    success = True
                    break
            if success:
                dataset.save_episode()
                saved += 1
                print(f"episode {saved}/{args.episodes} saved: {color} cube ({attempts} attempts total)")
            else:
                dataset.clear_episode_buffer()
                print("pick failed, episode discarded")
    except KeyboardInterrupt:
        print(f"\ninterrupted - keeping the {saved} episodes saved so far")
        dataset.clear_episode_buffer()
    finally:
        robot.disconnect()
        # always finalize: an unfinalized dataset cannot be loaded back (lerobot
        # would fall back to fetching the repo_id from the HF Hub and 404)
        dataset.finalize()
        print(f"Done: {saved} episodes -> {dataset.root}")
        print(f"  lerobot-dataset-viz --repo-id {args.repo_id} --root {dataset.root} --episode-index 0")


if __name__ == "__main__":
    main()
