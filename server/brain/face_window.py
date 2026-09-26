"""Desk pet launcher: frameless transparent always-on-top window above the taskbar.

pywebview must own the process main thread on Windows, so the pet runs as a
short-lived child: ``python -m brain.face_pet <url>``.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from ctypes import wintypes
from pathlib import Path

from . import config

PET_W = 220
PET_H = 260
PET_MARGIN = 8
_DOCK_FILE = Path(__file__).resolve().parent.parent / "pet_dock.json"
_CORNERS = ("bottom-right", "bottom-left", "top-right", "top-left")

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


def _load_dock_file() -> dict:
    try:
        if _DOCK_FILE.is_file():
            data = json.loads(_DOCK_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_dock(monitor: int | None = None, corner: str | None = None) -> dict:
    """Update in-memory config + pet_dock.json (+ best-effort .env keys)."""
    if monitor is not None:
        config.PET_MONITOR = max(0, int(monitor))
    if corner is not None:
        c = corner.strip().lower().replace("_", "-")
        if c in _CORNERS:
            config.PET_CORNER = c
    data = {"monitor": int(config.PET_MONITOR), "corner": config.PET_CORNER}
    try:
        _DOCK_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        print(f"face: could not write pet_dock.json ({e})")
    _persist_env_dock(data["monitor"], data["corner"])
    return data


def _persist_env_dock(monitor: int, corner: str) -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    out: list[str] = []
    seen_m = seen_c = False
    for line in lines:
        if line.startswith("PET_MONITOR="):
            out.append(f"PET_MONITOR={monitor}")
            seen_m = True
        elif line.startswith("PET_CORNER="):
            out.append(f"PET_CORNER={corner}")
            seen_c = True
        else:
            out.append(line)
    if not seen_m:
        out.append(f"PET_MONITOR={monitor}")
    if not seen_c:
        out.append(f"PET_CORNER={corner}")
    try:
        env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    except OSError:
        pass


def apply_saved_dock() -> None:
    """Load pet_dock.json over env defaults (file wins for live moves)."""
    data = _load_dock_file()
    if "monitor" in data:
        try:
            config.PET_MONITOR = max(0, int(data["monitor"]))
        except (TypeError, ValueError):
            pass
    if "corner" in data:
        c = str(data["corner"]).strip().lower().replace("_", "-")
        if c in _CORNERS:
            config.PET_CORNER = c


def list_monitor_work_areas() -> list[tuple[int, int, int, int]]:
    """Per-monitor work areas (excludes taskbar): (left, top, right, bottom)."""
    if sys.platform != "win32":
        return [_primary_work_area()]

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    monitors: list[tuple[int, int, int, int]] = []
    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_int, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM
    )

    def _callback(hmon, hdc, lprect, lparam):  # noqa: ARG001
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if ctypes.windll.user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            w = info.rcWork
            monitors.append((int(w.left), int(w.top), int(w.right), int(w.bottom)))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(_callback), 0)
    return monitors or [_primary_work_area()]


def _primary_work_area() -> tuple[int, int, int, int]:
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
    apply_saved_dock()
    areas = list_monitor_work_areas()
    idx = min(max(0, int(config.PET_MONITOR)), len(areas) - 1)
    left, top, right, bottom = areas[idx]
    corner = config.PET_CORNER
    if corner == "bottom-left":
        x, y = left + PET_MARGIN, bottom - PET_H - PET_MARGIN
    elif corner == "top-right":
        x, y = right - PET_W - PET_MARGIN, top + PET_MARGIN
    elif corner == "top-left":
        x, y = left + PET_MARGIN, top + PET_MARGIN
    else:  # bottom-right
        x, y = right - PET_W - PET_MARGIN, bottom - PET_H - PET_MARGIN
    return x, y


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
    """Keep pet on the chosen monitor/corner work area; click-through off the body."""
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
    apply_saved_dock()
    url = face_url()
    print(f"face: {url}")
    print(f"face: dock monitor={config.PET_MONITOR} corner={config.PET_CORNER}")
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
                f"(monitor {config.PET_MONITOR} {config.PET_CORNER}, pid={_proc.pid})",
                flush=True,
            )
        except Exception as e:
            print(f"face: could not start pet ({e}) — visit {url}")

    threading.Thread(target=_spawn, daemon=True, name="rocky-face-pet").start()
