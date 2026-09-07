from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("so101_sim")
@dataclass
class SO101SimConfig(RobotConfig):
    """SO-101 arm simulated in MuJoCo, with a cube-picking scene.

    The robot exposes the same interface as the real SO-101 follower: joint
    positions in degrees plus two RGB cameras ("wrist" and "front").
    """

    image_height: int = 240
    image_width: int = 320
    show_viewer: bool = True
    seed: int | None = None
