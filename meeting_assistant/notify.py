from __future__ import annotations

import subprocess


def notify(title: str, message: str, sound: str = "Glass") -> None:
    """macOS banner via osascript; never raises (a failed notification must not break the app)."""
    script = (
        f'display notification "{_esc(message)}" with title "{_esc(title)}" sound name "{sound}"'
    )
    try:
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')
