"""Keyboard end-effector pose teleoperator.

Bindings (active while connected; keys come from the local display via
pynput, or from the viser viewer's browser tab, see config.key_source):
    arrows      EE forward/back (up/down) and left/right
    w / s       EE up / down
    q / e       pitch the gripper up / down
    a / d       roll the jaws about the tool axis
    space       toggle gripper open/close

Episode-control keys (n/r/ESC) are intentionally not handled here — the
recording script owns those so bindings never conflict.
"""

import threading
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_keyboard_pose import KeyboardPoseConfig
from .keys import DOWN, LEFT, RIGHT, SPACE, UP, make_listener

MOVE_KEYS = {
    UP: np.array([1.0, 0.0, 0.0]),
    DOWN: np.array([-1.0, 0.0, 0.0]),
    LEFT: np.array([0.0, 1.0, 0.0]),
    RIGHT: np.array([0.0, -1.0, 0.0]),
    "w": np.array([0.0, 0.0, 1.0]),
    "s": np.array([0.0, 0.0, -1.0]),
}
PITCH_KEYS = {"q": 1.0, "e": -1.0}
ROLL_KEYS = {"a": 1.0, "d": -1.0}
PITCH_RANGE = (-1.2, 1.2)
ROLL_RANGE = (-0.9, 0.9)


class KeyboardPose(Teleoperator):
    config_class = KeyboardPoseConfig
    name = "keyboard_pose"

    def __init__(self, config: KeyboardPoseConfig):
        super().__init__(config)
        self.config = config
        self._held: set = set()
        self._lock = threading.Lock()
        self._listener = None  # pynput Listener or BrowserKeys subscription
        self._last_t: float | None = None
        self.gripper_open = True
        self._offset = np.zeros(3)
        self._pitch = 0.0
        self._roll = 0.0

    # ------------------------------------------------------------- features

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "enabled": float,
            "target_x": float,
            "target_y": float,
            "target_z": float,
            "target_wx": float,
            "target_wy": float,
            "target_wz": float,
            "gripper_vel": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    # ----------------------------------------------------------- connection

    @property
    def is_connected(self) -> bool:
        return self._listener is not None and self._listener.is_alive()

    def connect(self, calibrate: bool = True) -> None:
        self._listener = make_listener(
            self.config.key_source, self._on_press, self._on_release, self.config.browser_port
        )
        self._listener.start()

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------- keyboard

    def _on_press(self, key: str):
        with self._lock:
            if key == SPACE and key not in self._held:
                self.gripper_open = not self.gripper_open
            self._held.add(key)

    def _on_release(self, key: str):
        with self._lock:
            self._held.discard(key)

    # --------------------------------------------------------------- action

    def reset_state(self) -> None:
        """Zero the accumulated pose offset (call together with the pipeline reset)."""
        self._offset = np.zeros(3)
        self._pitch = 0.0
        self._roll = 0.0
        self.gripper_open = True
        self._last_t = None

    def get_action(self) -> dict[str, Any]:
        now = time.perf_counter()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.1)
        self._last_t = now

        with self._lock:
            held = set(self._held)
            gripper_open = self.gripper_open

        move = np.zeros(3)
        for key, direction in MOVE_KEYS.items():
            if key in held:
                move += direction
        if np.any(move):
            self._offset += self.config.move_speed * dt * move / np.linalg.norm(move)

        pitch_dir = sum(v for k, v in PITCH_KEYS.items() if k in held)
        self._pitch = float(
            np.clip(self._pitch + pitch_dir * self.config.pitch_speed * dt, *PITCH_RANGE)
        )
        roll_dir = sum(v for k, v in ROLL_KEYS.items() if k in held)
        self._roll = float(
            np.clip(self._roll + roll_dir * self.config.pitch_speed * dt, *ROLL_RANGE)
        )

        # pitch about local y, then roll about the (pitched) tool axis; encoded
        # as a single rotation vector for EEReferenceAndDelta
        wx, wy, wz = Rotation.from_matrix(
            Rotation.from_euler("y", self._pitch).as_matrix()
            @ Rotation.from_euler("z", self._roll).as_matrix()
        ).as_rotvec()

        return {
            "enabled": 1.0,
            "target_x": float(self._offset[0]),
            "target_y": float(self._offset[1]),
            "target_z": float(self._offset[2]),
            "target_wx": float(wx),
            "target_wy": float(wy),
            "target_wz": float(wz),
            "gripper_vel": self.config.gripper_vel if gripper_open else -self.config.gripper_vel,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass
