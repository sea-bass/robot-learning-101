# SO-101 MuJoCo demo

Keyboard-teleoperate a simulated SO-101 arm in a scene with a red and a green cube, record LeRobot datasets, train ACT (single task) or fine-tune SmolVLA (language picks the cube), and roll the policy back out in the sim.
The walkthrough is in [`docs/imitation_learning.md`](../docs/imitation_learning.md).

```
so101_sim/                    gym env (SO101PickCube-v0, two cubes + target color), MJCF scene,
                              placo IK wrapper, EE-target controller, scripted pick expert
plugins/so101_sim_robot/      LeRobot Robot plugin        -> --robot.type=so101_sim
plugins/keyboard_pose_teleop/ LeRobot Teleoperator plugin -> --teleop.type=keyboard_pose
scripts/                      teleoperate / record / collect_scripted / eval_policy
```
