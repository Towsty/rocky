"""Desk pet launcher: frameless transparent always-on-top window above the taskbar.

pywebview must own the process main thread on Windows, so the pet runs as a
short-lived child: ``python -m brain.face_pet <url>``.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from ctypes import wintypes

from . import config

PET_W = 220
PET_H = 260
PET_MARGIN = 8

_window = None
_hwnd = 0
_hit_screen: tuple[float, float, float, float] | None = None
_click_through = False
_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def face_url() -> str:
    host = config.LIVE_VIEW_BIND if config.LIVE_VIEW_BIND not in ("0.0.0.0", "::") else "127.0.0.1"
    return f"http://{host}:{config.LIVE_VIEW_PORT}/face"


def console_url() -> str:
    host = config.LIVE_VIEW_BIND if config.LIVE_VIEW_BIND not in ("0.0.0.0", "::") else "127.0.0.1"
    return f"http://{host}:{config.LIVE_VIEW_PORT}/"


def _work_area() -> tuple[int, int, int, int]:
    """Primary monitor working area (excludes taskbar): left, top, right, bottom."""
    if sys.platform != "win32":
        return 0, 0, 1920, 1080
    SPI_GETWORKAREA = 0x0030

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    rect = RECT()
    ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
    return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)


def _pet_xy() -> tuple[int, int]:
    left, top, right, bottom = _work_area()
    x = right - PET_W - PET_MARGIN
    y = bottom - PET_H - PET_MARGIN
    return max(left, x), max(top, y)


def _resolve_hwnd(window) -> int:
    native = getattr(window, "native", None)
    if native is None:
        return 0
    handle = getattr(native, "Handle", None)
    if handle is not None:
        try:
            return int(handle.ToInt32())
        except Exception:
            try:
                return int(handle)
            except Exception:
                pass
    for attr in ("Hwnd", "hwnd", "handle"):
        h = getattr(native, attr, None)
        if h:
            try:
                return int(h)
            except Exception:
                continue
    return 0


def _style_tool_topmost(hwnd: int) -> None:
    if not hwnd or sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    GWL_EXSTYLE = -20
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_APPWINDOW = 0x00040000
    WS_EX_LAYERED = 0x00080000
    WS_EX_TOPMOST = 0x00000008
    HWND_TOPMOST = -1
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_NOACTIVATE = 0x0010
    SWP_FRAMECHANGED = 0x0020
    SWP_SHOWWINDOW = 0x0040

    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style = (style | WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_TOPMOST) & ~WS_EX_APPWINDOW
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
    user32.SetWindowPos(
        hwnd,
        HWND_TOPMOST,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_SHOWWINDOW,
    )


def _set_click_through(hwnd: int, enabled: bool) -> None:
    if not hwnd or sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    GWL_EXSTYLE = -20
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_LAYERED = 0x00080000
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    if enabled:
        style |= WS_EX_TRANSPARENT | WS_EX_LAYERED
    else:
        style = (style | WS_EX_LAYERED) & ~WS_EX_TRANSPARENT
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)


class _PetApi:
    """JS bridge from /face."""

    def update_hit_rect(self, left: float, top: float, right: float, bottom: float) -> None:
        global _hit_screen
        with _lock:
            win = _window
            if win is None:
                _hit_screen = None
                return
            try:
                wx, wy = int(win.x), int(win.y)
            except Exception:
                wx, wy = _pet_xy()
            _hit_screen = (wx + left, wy + top, wx + right, wy + bottom)

    def open_console(self) -> None:
        webbrowser.open(console_url())

    def quit_face(self) -> None:
        win = _window
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass


def _dock_and_chrome_loop() -> None:
    """Keep pet on the bottom-right of the work area; click-through off the body."""
    global _hwnd, _click_through
    while True:
        time.sleep(0.25)
        win = _window
        if win is None:
            continue
        try:
            if not _hwnd:
                _hwnd = _resolve_hwnd(win)
                if _hwnd:
                    _style_tool_topmost(_hwnd)
            x, y = _pet_xy()
            try:
                if abs(int(win.x) - x) > 2 or abs(int(win.y) - y) > 2:
                    win.move(x, y)
            except Exception:
                pass
            if _hwnd and sys.platform == "win32":
                pt = wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                with _lock:
                    hit = _hit_screen
                over = bool(
                    hit
                    and hit[0] <= pt.x <= hit[2]
                    and hit[1] <= pt.y <= hit[3]
                )
                want_through = not over
                if want_through != _click_through:
                    _set_click_through(_hwnd, want_through)
                    _click_through = want_through
                    if not want_through:
                        _style_tool_topmost(_hwnd)
        except Exception:
            pass


def open_face_window() -> None:
    """Log face URL; spawn the desk-pet child process (no browser popup)."""
    global _proc
    url = face_url()
    print(f"face: {url}")
    if not config.OPEN_FACE:
        print("face: auto-open off (OPEN_FACE=0)")
        return

    def _spawn() -> None:
        global _proc
        time.sleep(0.5)
        if _proc is not None and _proc.poll() is None:
            return
        py = sys.executable
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            _proc = subprocess.Popen(
                [py, "-m", "brain.face_pet", url],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                env=env,
                stdout=None,
                stderr=None,
            )
            print(
                f"face: desk pet {PET_W}x{PET_H} via pywebview sidecar "
                f"(bottom-right above taskbar, pid={_proc.pid})",
                flush=True,
            )
        except Exception as e:
            print(f"face: could not start pet ({e}) — visit {url}")

    threading.Thread(target=_spawn, daemon=True, name="rocky-face-pet").start()
