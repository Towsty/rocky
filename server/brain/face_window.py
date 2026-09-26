"""Desk pet launcher: frameless always-on-top window above the taskbar.

pywebview must own the process main thread on Windows, so the pet runs as a
short-lived child: ``python -m brain.face_pet <url>``.

Default is **opaque** (dark card). Chroma/layered alpha is opt-in via PET_ALPHA=1
because it wedges WebView2 into a white “not responding” ghost on Ian’s box.
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

PET_W = 300
PET_H = 400
PET_MARGIN = 8
READY_TIMEOUT_S = 3.0
OPAQUE_BG = "#0a1018"
CHROMA_KEY = "#00FE01"
_DOCK_FILE = Path(__file__).resolve().parent.parent / "pet_dock.json"
_MODE_FILE = Path(__file__).resolve().parent.parent / "pet_mode.json"
_CORNERS = ("bottom-right", "bottom-left", "top-right", "top-left")

_window = None
_hwnd = 0
_hit_screen: tuple[float, float, float, float] | None = None
_click_through = False
_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_opaque_mode = True
_opaque_forced = False


def face_url() -> str:
    host = config.LIVE_VIEW_BIND if config.LIVE_VIEW_BIND not in ("0.0.0.0", "::") else "127.0.0.1"
    return f"http://{host}:{config.LIVE_VIEW_PORT}/face"


def console_url() -> str:
    host = config.LIVE_VIEW_BIND if config.LIVE_VIEW_BIND not in ("0.0.0.0", "::") else "127.0.0.1"
    return f"http://{host}:{config.LIVE_VIEW_PORT}/"


def is_opaque_mode() -> bool:
    """Opaque unless PET_ALPHA=1 and we have not forced opaque after a hang."""
    if _MODE_FILE.is_file():
        try:
            data = json.loads(_MODE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("opaque") is True:
                return True
        except Exception:
            pass
    # PET_ALPHA=1 opts into chroma; anything else → opaque (safe).
    alpha = (os.environ.get("PET_ALPHA") or "").strip().lower() in ("1", "true", "yes", "on")
    return not alpha


def set_opaque_mode(opaque: bool) -> None:
    global _opaque_mode
    _opaque_mode = bool(opaque)


def mark_opaque_fallback() -> None:
    """Persist opaque so the next launch skips the hanging alpha path."""
    global _opaque_forced
    _opaque_forced = True
    try:
        _MODE_FILE.write_text(json.dumps({"opaque": True}, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    # Attribute used by face_pet log line
    mark_opaque_fallback.was_forced = True  # type: ignore[attr-defined]


mark_opaque_fallback.was_forced = False  # type: ignore[attr-defined]


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
    else:
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


def _force_destroy_hwnd(hwnd: int) -> None:
    """Last-resort HWND teardown so a wedged WebView cannot leave a white ghost."""
    if not hwnd or sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    try:
        user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass
    try:
        user32.DestroyWindow(hwnd)
    except Exception:
        pass
    try:
        # If DestroyWindow fails (wrong thread), post close.
        user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
    except Exception:
        pass


def _chroma_colorref() -> int:
    h = CHROMA_KEY.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return r | (g << 8) | (b << 16)


def _apply_colorkey_hwnd(hwnd: int) -> None:
    if not hwnd:
        return
    user32 = ctypes.windll.user32
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    LWA_COLORKEY = 0x00000001
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED)
    user32.SetLayeredWindowAttributes(hwnd, _chroma_colorref(), 255, LWA_COLORKEY)


def _enum_child_hwnds(parent: int) -> list[int]:
    kids: list[int] = []
    if not parent:
        return kids
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lp):  # noqa: ANN001
        kids.append(int(hwnd))
        return 1

    ctypes.windll.user32.EnumChildWindows(parent, WNDENUMPROC(_cb), 0)
    return kids


def _apply_cutout(hwnd: int) -> None:
    """Chroma color-key — only for experimental PET_ALPHA=1."""
    if not hwnd or sys.platform != "win32":
        return
    _apply_colorkey_hwnd(hwnd)
    for child in _enum_child_hwnds(hwnd):
        _apply_colorkey_hwnd(child)


def _style_tool_topmost(hwnd: int, opaque: bool | None = None) -> None:
    """Tool window + topmost. Opaque path never touches LWA_COLORKEY / child layering."""
    if not hwnd or sys.platform != "win32":
        return
    if opaque is None:
        opaque = _opaque_mode
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
    style = (style | WS_EX_TOOLWINDOW | WS_EX_TOPMOST) & ~WS_EX_APPWINDOW
    if opaque:
        # Layered + color-key is what left the white ghost — stay non-layered.
        style &= ~WS_EX_LAYERED
    else:
        style |= WS_EX_LAYERED
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
    if not opaque:
        _apply_cutout(hwnd)
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
        # Need LAYERED for TRANSPARENT to behave; do not color-key in opaque mode.
        style |= WS_EX_TRANSPARENT | WS_EX_LAYERED
    else:
        style &= ~WS_EX_TRANSPARENT
        if _opaque_mode:
            style &= ~WS_EX_LAYERED
        else:
            style |= WS_EX_LAYERED
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
    if not _opaque_mode:
        _apply_cutout(hwnd)


def _start_tray(quit_cb) -> None:
    """System tray “Quit pet” — works even when the WebView UI is wedged."""
    if sys.platform != "win32":
        return
    try:
        from System.Drawing import SystemIcons  # type: ignore
        from System.Windows.Forms import ContextMenu, MenuItem, NotifyIcon  # type: ignore

        ni = NotifyIcon()
        ni.Icon = SystemIcons.Application
        ni.Text = "Rocky pet"
        ni.Visible = True
        item = MenuItem("Quit pet")

        def _on_quit(sender, args):  # noqa: ARG001
            try:
                ni.Visible = False
            except Exception:
                pass
            quit_cb()

        item.Click += _on_quit
        menu = ContextMenu()
        menu.MenuItems.Add(item)
        ni.ContextMenu = menu
        fw_tray_ref[0] = ni
        print("pet: tray Quit pet ready", flush=True)
    except Exception as e:
        print(f"pet: tray unavailable ({e})", flush=True)


fw_tray_ref: list = [None]


class _PetApi:
    """JS bridge from /face."""

    def __init__(self) -> None:
        self.on_ready = None  # type: ignore

    def pet_ready(self) -> None:
        print("pet: rockyReady", flush=True)
        cb = self.on_ready
        if callable(cb):
            try:
                cb()
            except Exception:
                pass

    def pet_log(self, msg: str) -> None:
        print(f"pet: {msg}", flush=True)

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
        print("pet: quit requested", flush=True)
        win = _window
        hwnd = _hwnd or (_resolve_hwnd(win) if win is not None else 0)
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
        _force_destroy_hwnd(hwnd)


def _dock_and_chrome_loop() -> None:
    """Keep pet on the chosen monitor/corner; click-through off the body."""
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
                    _style_tool_topmost(_hwnd, opaque=_opaque_mode)
            x, y = _pet_xy()
            try:
                if abs(int(win.x) - x) > 2 or abs(int(win.y) - y) > 2:
                    win.move(x, y)
            except Exception:
                pass
            try:
                if abs(int(win.width) - PET_W) > 4 or abs(int(win.height) - PET_H) > 4:
                    win.resize(PET_W, PET_H)
                    win.move(*_pet_xy())
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
                        _style_tool_topmost(_hwnd, opaque=_opaque_mode)
        except Exception:
            pass


def open_face_window() -> None:
    """Spawn the desk-pet child (never blocks brain.main / STT)."""
    global _proc
    apply_saved_dock()
    url = face_url()
    print(f"face: {url}", flush=True)
    print(f"face: dock monitor={config.PET_MONITOR} corner={config.PET_CORNER}", flush=True)
    if not config.OPEN_FACE:
        print("face: auto-open off (OPEN_FACE=0)", flush=True)
        return

    def _spawn(opaque_force: bool = False) -> None:
        global _proc
        time.sleep(0.4)
        if _proc is not None and _proc.poll() is None:
            return
        py = sys.executable
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if opaque_force:
            env["PET_ALPHA"] = "0"
            mark_opaque_fallback()
        mode = "opaque" if (opaque_force or is_opaque_mode()) else "chroma"
        try:
            _proc = subprocess.Popen(
                [py, "-m", "brain.face_pet", url, mode],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                env=env,
                stdout=None,
                stderr=None,
            )
            print(
                f"face: desk pet {PET_W}x{PET_H} sidecar mode={mode} "
                f"(monitor {config.PET_MONITOR} {config.PET_CORNER}, pid={_proc.pid})",
                flush=True,
            )
        except Exception as e:
            print(f"face: could not start pet ({e}) — visit {url}", flush=True)
            return

        def _reap() -> None:
            global _proc
            proc = _proc
            if proc is None:
                return
            code = proc.wait()
            print(f"face: pet exited code={code}", flush=True)
            if code == 2 and mode == "chroma":
                print("face: respawning opaque after load timeout", flush=True)
                _proc = None
                _spawn(opaque_force=True)
            elif code == 2:
                print("pet: load timeout — ghost destroyed; not respawning", flush=True)

        threading.Thread(target=_reap, daemon=True, name="rocky-face-reap").start()

    threading.Thread(target=_spawn, daemon=True, name="rocky-face-pet").start()
