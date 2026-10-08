from dataclasses import dataclass
from typing import Literal

from lerobot.robots import RobotConfig

VIEWERS = ("native", "viser")


@RobotConfig.register_subclass("so101_sim")
@dataclass
class SO101SimConfig(RobotConfig):
    """SO-101 arm simulated in MuJoCo, with a cube-picking scene.

    The robot exposes the same interface as the real SO-101 follower: joint
    positions in degrees plus two RGB cameras ("wrist" and "front"). The scene
    has a red and a green cube; ``target_color`` is the one that counts as a
    successful pick (``reset_scene`` can change it per episode).

    ``viewer`` picks how ``show_viewer`` displays the scene: ``native`` is the
    MuJoCo window (needs a display), ``viser`` serves it to a browser at
    ``http://localhost:<viser_port>`` and works over SSH with port forwarding.
    """

    image_height: int = 240
    image_width: int = 320
    show_viewer: bool = True
    viewer: Literal["native", "viser"] = "native"
    viser_port: int = 8080
    seed: int | None = None
    target_color: str = "red"
