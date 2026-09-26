"""Desk-pet process entry: must own the main thread for pywebview on Windows.

Started by brain.face_window as a child of python -m brain.main:
  python -m brain.face_pet http://127.0.0.1:8766/face
"""

from __future__ import annotations

import sys
import threading
import time
import webbrowser

from .face_window import (
    PET_H,
    PET_W,
    _PetApi,
    _dock_and_chrome_loop,
    _pet_xy,
    _resolve_hwnd,
    _style_tool_topmost,
    console_url,
)
from . import face_window as fw


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    url = argv[0] if argv else "http://127.0.0.1:8766/face"
    try:
        import webview
    except ImportError:
        print("face: pywebview missing in this venv", flush=True)
        return 1

    x, y = _pet_xy()
    api = _PetApi()
    window = webview.create_window(
        title="Rocky",
        url=url,
        width=PET_W,
        height=PET_H,
        x=x,
        y=y,
        resizable=False,
        frameless=True,
        easy_drag=False,
        on_top=True,
        focus=False,
        transparent=True,
        background_color="#000000",
        shadow=False,
        js_api=api,
    )
    fw._window = window

    def _shown() -> None:
        time.sleep(0.25)
        fw._hwnd = _resolve_hwnd(window)
        _style_tool_topmost(fw._hwnd)
        try:
            window.move(*_pet_xy())
        except Exception:
            pass

    try:
        window.events.shown += lambda: threading.Thread(target=_shown, daemon=True).start()
    except Exception:
        threading.Thread(target=_shown, daemon=True).start()

    threading.Thread(target=_dock_and_chrome_loop, daemon=True).start()
    print(f"face: desk pet {PET_W}x{PET_H} via pywebview (bottom-right above taskbar)", flush=True)
    try:
        webview.start(gui="edgechromium")
    except Exception:
        webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
