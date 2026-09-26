"""Desk-pet dock helpers + process launcher (no WebView).

Pet renderer: ``python -m brain.face_pet`` (moderngl + glfw, per-pixel alpha).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

from . import config

PET_W = 300
PET_H = 400
PET_MARGIN = 8
_UI_FILE = Path(__file__).resolve().parent.parent / "pet-ui.json"
_LEGACY_DOCK = Path(__file__).resolve().parent.parent / "pet_dock.json"
_CORNERS = ("bottom-right", "bottom-left", "top-right", "top-left")

_proc: subprocess.Popen | None = None


def face_url() -> str:
    host = config.LIVE_VIEW_BIND if config.LIVE_VIEW_BIND not in ("0.0.0.0", "::") else "127.0.0.1"
    return f"http://{host}:{config.LIVE_VIEW_PORT}/face"


def console_url() -> str:
    host = config.LIVE_VIEW_BIND if config.LIVE_VIEW_BIND not in ("0.0.0.0", "::") else "127.0.0.1"
    return f"http://{host}:{config.LIVE_VIEW_PORT}/"


def _load_ui() -> dict:
    for path in (_UI_FILE, _LEGACY_DOCK):
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def save_dock(monitor: int | None = None, corner: str | None = None) -> dict:
    """Update config + pet-ui.json (+ best-effort .env)."""
    if monitor is not None:
        config.PET_MONITOR = max(0, int(monitor))
    if corner is not None:
        c = corner.strip().lower().replace("_", "-")
        if c in _CORNERS:
            config.PET_CORNER = c
    data = {"monitor": int(config.PET_MONITOR), "corner": config.PET_CORNER}
    try:
        _UI_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        print(f"face: could not write pet-ui.json ({e})", flush=True)
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
    data = _load_ui()
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
    if sys.platform != "win32":
        return [_primary_work_area()]
    import ctypes

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
    import ctypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    rect = RECT()
    ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
    return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)


def pet_xy() -> tuple[int, int]:
    apply_saved_dock()
    areas = list_monitor_work_areas()
    idx = min(max(0, int(config.PET_MONITOR)), len(areas) - 1)
    left, top, right, bottom = areas[idx]
    corner = config.PET_CORNER
    if corner == "bottom-left":
        return left + PET_MARGIN, bottom - PET_H - PET_MARGIN
    if corner == "top-right":
        return right - PET_W - PET_MARGIN, top + PET_MARGIN
    if corner == "top-left":
        return left + PET_MARGIN, top + PET_MARGIN
    return right - PET_W - PET_MARGIN, bottom - PET_H - PET_MARGIN


def open_face_window() -> None:
    """Spawn the GL pet child — never blocks brain.main / STT."""
    global _proc
    apply_saved_dock()
    print(f"face: console {console_url()}", flush=True)
    print(
        f"face: dock monitor={config.PET_MONITOR} corner={config.PET_CORNER}",
        flush=True,
    )
    if not config.OPEN_FACE:
        print("face: auto-open off (OPEN_FACE=0)", flush=True)
        return

    def _spawn() -> None:
        global _proc
        time.sleep(0.35)
        if _proc is not None and _proc.poll() is None:
            return
        py = sys.executable
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            _proc = subprocess.Popen(
                [py, "-m", "brain.face_pet"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                env=env,
                stdout=None,
                stderr=None,
            )
            print(
                f"face: GL pet {PET_W}x{PET_H} "
                f"(monitor {config.PET_MONITOR} {config.PET_CORNER}, pid={_proc.pid})",
                flush=True,
            )
        except Exception as e:
            print(f"face: could not start pet ({e})", flush=True)
            return

        def _reap() -> None:
            proc = _proc
            if proc is None:
                return
            code = proc.wait()
            print(f"face: pet exited code={code}", flush=True)

        threading.Thread(target=_reap, daemon=True, name="rocky-face-reap").start()

    threading.Thread(target=_spawn, daemon=True, name="rocky-face-pet").start()
