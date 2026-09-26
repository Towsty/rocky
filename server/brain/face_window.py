"""Open the companion /face window on the desktop."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

from . import config


def face_url() -> str:
    host = config.LIVE_VIEW_BIND if config.LIVE_VIEW_BIND not in ("0.0.0.0", "::") else "127.0.0.1"
    return f"http://{host}:{config.LIVE_VIEW_PORT}/face"


def _edge_or_chrome() -> str | None:
    """Prefer an installed Edge/Chrome binary for an --app= window (no browser chrome)."""
    if sys.platform != "win32":
        return shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    candidates = []
    for env in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        root = os.environ.get(env)
        if not root:
            continue
        candidates.extend(
            [
                os.path.join(root, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(root, "Google", "Chrome", "Application", "chrome.exe"),
            ]
        )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return shutil.which("msedge") or shutil.which("chrome")


def open_face_window() -> None:
    """Log the face URL; if OPEN_FACE is on, open a desktop window shortly after serve starts."""
    url = face_url()
    print(f"face: {url}")
    if not config.OPEN_FACE:
        print("face: auto-open off (OPEN_FACE=0)")
        return

    def _open() -> None:
        time.sleep(0.4)  # let the HTTP thread bind
        browser = _edge_or_chrome()
        try:
            if browser:
                # App mode = dedicated window, no tabs/URL bar. Always-on-top is
                # not a stable Chromium flag; a normal app window is acceptable.
                subprocess.Popen(
                    [browser, f"--app={url}", f"--window-size=480,620"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            webbrowser.open(url)
        except Exception as e:
            print(f"face: could not open window ({e}) — visit {url}")

    threading.Thread(target=_open, daemon=True).start()
