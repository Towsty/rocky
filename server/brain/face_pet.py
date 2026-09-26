"""Desk-pet process entry: must own the main thread for pywebview on Windows.

Started by brain.face_window as a child of python -m brain.main:
  python -m brain.face_pet http://127.0.0.1:8766/face

Exit codes:
  0 — normal quit
  2 — load timeout / hung WebView (parent may respawn opaque)
"""

from __future__ import annotations

import os
import sys
import threading
import time

from .face_window import (
    CHROMA_KEY,
    OPAQUE_BG,
    PET_H,
    PET_W,
    READY_TIMEOUT_S,
    _PetApi,
    _dock_and_chrome_loop,
    _force_destroy_hwnd,
    _pet_xy,
    _resolve_hwnd,
    _start_tray,
    _style_tool_topmost,
    apply_saved_dock,
    is_opaque_mode,
    mark_opaque_fallback,
    set_opaque_mode,
)
from . import config
from . import face_window as fw


def _log(msg: str) -> None:
    print(f"pet: {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    base_url = argv[0] if argv else "http://127.0.0.1:8766/face"
    apply_saved_dock()

    # Alpha/chroma/layered experiments wedge WebView2 on Ian's box — opaque first.
    opaque = is_opaque_mode()
    if len(argv) > 1 and argv[1] in ("opaque", "chroma"):
        opaque = argv[1] == "opaque"
    set_opaque_mode(opaque)

    if opaque:
        os.environ.pop("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", None)
        bg = OPAQUE_BG
        mode_q = "mode=opaque"
        from .face_window import _MODE_FILE

        if _MODE_FILE.is_file():
            _log("alpha disabled, opaque fallback")
        else:
            _log("opaque mode")
    else:
        # Explicit PET_ALPHA=1 only — still risky.
        os.environ.setdefault(
            "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
            "--disable-gpu --disable-gpu-compositing",
        )
        bg = CHROMA_KEY
        mode_q = "mode=chroma"
        _log("chroma/alpha mode (experimental)")

    sep = "&" if "?" in base_url else "?"
    url = f"{base_url}{sep}{mode_q}"

    try:
        import webview
    except ImportError:
        _log("pywebview missing in this venv")
        return 1

    x, y = _pet_xy()
    api = _PetApi()
    ready = threading.Event()
    api.on_ready = ready.set

    _log(f"creating window {PET_W}x{PET_H} bg={bg}")
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
        transparent=False,
        background_color=bg,
        shadow=False,
        js_api=api,
    )
    fw._window = window
    fw._opaque_mode = opaque

    def _destroy(reason: str) -> None:
        _log(f"destroy ({reason})")
        hwnd = fw._hwnd or _resolve_hwnd(window)
        try:
            window.destroy()
        except Exception as e:
            _log(f"window.destroy failed: {e}")
        _force_destroy_hwnd(hwnd)
        fw._window = None

    def _shown() -> None:
        time.sleep(0.15)
        fw._hwnd = _resolve_hwnd(window)
        _log(f"shown hwnd={fw._hwnd}")
        _style_tool_topmost(fw._hwnd, opaque=opaque)
        try:
            window.resize(PET_W, PET_H)
            window.move(*_pet_xy())
        except Exception as e:
            _log(f"resize/move: {e}")

    def _watchdog() -> None:
        _log(f"waiting for rockyReady (≤{READY_TIMEOUT_S:.0f}s)")
        if ready.wait(READY_TIMEOUT_S):
            _log("ready")
            return
        _log("load timeout")
        mark_opaque_fallback()
        _destroy("load timeout")
        # Exit so parent can clean up; code 2 → opaque respawn once.
        os._exit(2)

    try:
        window.events.shown += lambda: threading.Thread(target=_shown, daemon=True).start()
    except Exception:
        threading.Thread(target=_shown, daemon=True).start()

    threading.Thread(target=_watchdog, daemon=True, name="pet-watchdog").start()
    threading.Thread(target=_dock_and_chrome_loop, daemon=True, name="pet-dock").start()
    threading.Thread(
        target=lambda: _start_tray(lambda: _destroy("tray Quit pet")),
        daemon=True,
        name="pet-tray",
    ).start()

    _log(
        f"desk pet {PET_W}x{PET_H} "
        f"monitor={config.PET_MONITOR} corner={config.PET_CORNER}"
    )
    try:
        webview.start(gui="edgechromium")
    except Exception as e:
        _log(f"webview.start edgechromium failed: {e}")
        try:
            webview.start()
        except Exception as e2:
            _log(f"webview.start failed: {e2}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
