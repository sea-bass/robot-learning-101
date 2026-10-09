"""Key names and listener factory shared by the two key sources.

Keys are canonical strings: single lowercase characters ("w", "x") and
pynput's names for the rest ("up", "down", "left", "right", "space", "esc").

``make_listener`` returns an object with ``start`` / ``stop`` / ``is_alive``
(the pynput.Listener surface) whose callbacks receive canonical names, from
either the local display (pynput) or the viser viewer's browser tab.
"""

from collections.abc import Callable

UP, DOWN, LEFT, RIGHT, SPACE, ESC = "up", "down", "left", "right", "space", "esc"


def canonical_pynput(key) -> str | None:
    """pynput Key / KeyCode -> canonical name."""
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    return getattr(key, "name", None)


def make_listener(
    source: str,
    on_press: Callable[[str], None] | None = None,
    on_release: Callable[[str], None] | None = None,
    browser_port: int = 8081,
):
    if source == "browser":
        from .browser_keys import BrowserKeys

        return BrowserKeys.get(browser_port).listener(on_press=on_press, on_release=on_release)
    if source == "pynput":
        from pynput import keyboard  # grabs an X display on import, hence lazy

        def wrap(cb):
            if cb is None:
                return None

            def _(key):
                name = canonical_pynput(key)
                if name is not None:
                    cb(name)

            return _

        return keyboard.Listener(on_press=wrap(on_press), on_release=wrap(on_release))
    raise ValueError(f"unknown key source {source!r}, expected 'pynput' or 'browser'")
