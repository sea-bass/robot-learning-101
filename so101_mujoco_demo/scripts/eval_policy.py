"""Roll out a trained policy in the SO-101 MuJoCo sim and report success rate.

    pixi run so101-eval \
        --policy-path outputs/train/so101_act/checkpoints/last/pretrained_model \
        --episodes 10
    pixi run so101-eval --policy-path ... --episodes 20 --colors red green   # language-conditioned

Episodes alternate through --colors; each one's default instruction is that
color's task string and success means *that* cube was lifted (lifting the
other one is counted separately, as "wrong cube").

To watch in the MuJoCo viewer:  pixi run so101-eval ... --show
In a browser (e.g. over SSH):   pixi run so101-eval ... --show --viewer viser
For deployment-style rollouts the official CLI also works:
    pixi run -e lerobot lerobot-rollout --robot.type=so101_sim ...
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from lerobot_robot_so101_sim import SO101Sim, SO101SimConfig
from so101_sim.env import JOINT_NAMES, TASKS
from sim_pipelines import FPS, add_colors_arg, add_viewer_args, viewer_kwargs

MAX_TICKS = 250  # per-episode budget in sim ticks (250 = 10 s simulated; wall time
                 # is longer when rendering + inference exceed the 40 ms tick budget)


def obs_to_batch(obs: dict, device: str, task: str) -> dict:
    state = torch.tensor(
        [obs[f"{j}.pos"] for j in JOINT_NAMES], dtype=torch.float32, device=device
    ).unsqueeze(0)
    batch = {"observation.state": state, "task": [task]}
    for cam in ("wrist", "front"):
        img = torch.from_numpy(obs[cam]).permute(2, 0, 1).float().div_(255.0)
        batch[f"observation.images.{cam}"] = img.unsqueeze(0).to(device)
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="checkpoint pretrained_model dir")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--show", action="store_true", help="display in the viewer (--viewer), paced to real time")
    add_viewer_args(parser)
    parser.add_argument("--max-ticks", type=int, default=MAX_TICKS, help="sim ticks per episode (25/s)")
    add_colors_arg(parser)
    parser.add_argument(
        "--task", default=None,
        help="override the language instruction (default: the target color's recorded task string)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="real-time factor for --show playback (0.5 = half speed); no effect headless",
    )
    args = parser.parse_args()

    # a missing local path would be misinterpreted as a HF Hub repo id downstream
    if not Path(args.policy_path).is_dir():
        parser.error(
            f"--policy-path is not a local directory: {args.policy_path}\n"
            "expected something like outputs/train/so101_act/checkpoints/last/pretrained_model "
            "(relative to the directory you run from)"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy = get_policy_class(policy_cfg.type).from_pretrained(args.policy_path)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    robot = SO101Sim(SO101SimConfig(show_viewer=args.show, seed=args.seed, **viewer_kwargs(args)))
    robot.connect()

    successes = {c: 0 for c in args.colors}
    wrong = {c: 0 for c in args.colors}
    counts = {c: 0 for c in args.colors}
    for ep in range(args.episodes):
        color = args.colors[ep % len(args.colors)]
        task = args.task or TASKS[color]
        counts[color] += 1
        robot.reset_scene(seed=args.seed + ep, target_color=color)
        policy.reset()
        success = False
        wrong_cube = False  # the policy lifted the other cube at some point
        ep_start = time.perf_counter()
        ticks = 0
        for _ in range(args.max_ticks):
            ticks += 1
            t0 = time.perf_counter()
            obs = robot.get_observation()
            batch = preprocessor(obs_to_batch(obs, device, task))
            with torch.inference_mode():
                action_t = policy.select_action(batch)
            action_t = postprocessor(action_t)
            action_vec = action_t.squeeze(0).cpu().numpy().astype(np.float64)
            robot.send_action({f"{j}.pos": float(a) for j, a in zip(JOINT_NAMES, action_vec)})
            if robot.is_success:
                success = True
                break
            wrong_cube |= robot.wrong_cube_lifted
            if args.show:
                leftover = 1 / (FPS * args.speed) - (time.perf_counter() - t0)
                if leftover > 0:
                    time.sleep(leftover)
        successes[color] += success
        wrong[color] += wrong_cube
        wall = time.perf_counter() - ep_start
        outcome = "success" if success else ("failure (wrong cube)" if wrong_cube else "failure")
        print(
            f"episode {ep + 1}/{args.episodes} [{color}]: {outcome} "
            f"({ticks} ticks = {ticks / 25:.1f} s sim in {wall:.1f} s wall, {ticks / wall:.0f} Hz)"
        )

    print()
    for color in args.colors:
        n = counts[color]
        print(f"{color:>6}: {successes[color]}/{n} ({100 * successes[color] / max(n, 1):.0f}%), wrong cube {wrong[color]}")
    total = sum(successes.values())
    print(f"success rate: {total}/{args.episodes} ({100 * total / args.episodes:.0f}%)")
    robot.disconnect()


if __name__ == "__main__":
    main()
