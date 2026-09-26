"""Native OpenGL desk pet — moderngl + glfw (no WebView).

  python -m brain.face_pet
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from dataclasses import dataclass, field

import numpy as np

from . import config
from .face_window import (
    PET_H,
    PET_W,
    _CORNERS,
    apply_saved_dock,
    console_url,
    list_monitor_work_areas,
    pet_xy,
    save_dock,
)

STATUS_URL = f"http://127.0.0.1:{config.LIVE_VIEW_PORT}/status"

POSES = {
    "neutral": dict(eye_sx=1.15, eye_sy=1.0, glow=1.0, pitch=0.0, yaw=0.0, bounce=0.02, ant=0.06),
    "happy": dict(eye_sx=1.35, eye_sy=1.25, glow=1.35, pitch=-0.05, yaw=0.0, bounce=0.1, ant=0.12),
    "sad": dict(eye_sx=1.05, eye_sy=0.5, glow=0.45, pitch=0.2, yaw=0.0, bounce=0.0, ant=0.03),
    "angry": dict(eye_sx=1.45, eye_sy=0.26, glow=0.9, pitch=-0.1, yaw=0.0, bounce=0.0, ant=0.05),
    "surprised": dict(eye_sx=1.6, eye_sy=1.55, glow=1.5, pitch=-0.12, yaw=0.0, bounce=0.04, ant=0.18),
    "sleepy": dict(eye_sx=1.3, eye_sy=0.16, glow=0.3, pitch=0.3, yaw=0.1, bounce=0.0, ant=0.02),
    "thinking": dict(eye_sx=1.1, eye_sy=0.95, glow=1.0, pitch=0.04, yaw=-0.28, bounce=0.01, ant=0.38),
}


def _sphere(radius: float, sectors: int = 28, stacks: int = 18, scale=(1.0, 1.0, 1.0)):
    sx, sy, sz = scale
    verts: list[float] = []
    idx: list[int] = []
    for i in range(stacks + 1):
        v = i / stacks
        phi = math.pi * v
        for j in range(sectors + 1):
            u = j / sectors
            theta = 2 * math.pi * u
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            verts.extend([x * radius * sx, y * radius * sy, z * radius * sz, x, y, z])
    for i in range(stacks):
        for j in range(sectors):
            a = i * (sectors + 1) + j
            b = a + sectors + 1
            idx.extend([a, b, a + 1, a + 1, b, b + 1])
    return np.array(verts, dtype="f4"), np.array(idx, dtype="i4")


def _disc(radius: float, segments: int = 32):
    verts = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    idx: list[int] = []
    for i in range(segments + 1):
        a = 2 * math.pi * i / segments
        verts.extend([math.cos(a) * radius, 0.0, math.sin(a) * radius, 0.0, 1.0, 0.0])
    for i in range(1, segments + 1):
        idx.extend([0, i, i + 1 if i < segments else 1])
    return np.array(verts, dtype="f4"), np.array(idx, dtype="i4")


def _mat4_identity():
    return np.eye(4, dtype="f4")


def _mat4_translate(x, y, z):
    m = _mat4_identity()
    m[3, 0], m[3, 1], m[3, 2] = x, y, z
    return m


def _mat4_scale(x, y, z):
    m = _mat4_identity()
    m[0, 0], m[1, 1], m[2, 2] = x, y, z
    return m


def _mat4_rotate_x(a):
    c, s = math.cos(a), math.sin(a)
    m = _mat4_identity()
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, s, -s, c
    return m


def _mat4_rotate_y(a):
    c, s = math.cos(a), math.sin(a)
    m = _mat4_identity()
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, -s, s, c
    return m


def _mat4_rotate_z(a):
    c, s = math.cos(a), math.sin(a)
    m = _mat4_identity()
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, s, -s, c
    return m


def _mat4_mul(a, b):
    return a @ b


def _perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = -1.0
    m[3, 2] = (2 * far * near) / (near - far)
    return m


def _look_at(eye, target, up=(0, 1, 0)):
    eye = np.array(eye, dtype="f4")
    target = np.array(target, dtype="f4")
    up = np.array(up, dtype="f4")
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    m = _mat4_identity()
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[3, 0:3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
    return m


VERT = """
#version 330
in vec3 in_pos;
in vec3 in_nrm;
uniform mat4 mvp;
uniform mat4 model;
out vec3 v_nrm;
out vec3 v_pos;
void main() {
    vec4 wp = model * vec4(in_pos, 1.0);
    v_pos = wp.xyz;
    v_nrm = mat3(model) * in_nrm;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAG = """
#version 330
in vec3 v_nrm;
in vec3 v_pos;
uniform vec3 u_color;
uniform vec3 u_emissive;
uniform float u_alpha;
uniform float u_gloss;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_nrm);
    vec3 L = normalize(vec3(0.45, 0.85, 0.55));
    vec3 V = normalize(vec3(0.0, 0.1, 1.0));
    float ndl = max(dot(n, L), 0.0);
    float rim = pow(1.0 - max(dot(n, V), 0.0), 2.0) * 0.35;
    float spec = pow(max(dot(reflect(-L, n), V), 0.0), 48.0) * u_gloss;
    vec3 col = u_color * (0.28 + 0.72 * ndl) + vec3(spec) + u_emissive + vec3(rim * 0.15);
    f_color = vec4(col, u_alpha);
}
"""


@dataclass
class Mesh:
    vbo: object
    ibo: object
    vao: object
    count: int
    color: tuple[float, float, float]
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha: float = 1.0
    gloss: float = 0.45
    model: np.ndarray = field(default_factory=_mat4_identity)


@dataclass
class PetState:
    emotion: str = "neutral"
    awake: bool = True
    listening: bool = False
    speaking: bool = False
    last_said: str = ""
    monitors: int = 1


def _poll_status(state: PetState, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            with urllib.request.urlopen(STATUS_URL, timeout=0.8) as r:
                data = json.loads(r.read().decode("utf-8"))
            state.emotion = data.get("emotion") or "neutral"
            state.awake = bool(data.get("awake", True))
            state.listening = bool(data.get("listening", False))
            state.speaking = bool(data.get("speaking", False))
            state.last_said = data.get("last_said") or ""
            state.monitors = int(data.get("pet_monitors") or data.get("monitors") or 1)
            mon = data.get("pet_monitor")
            corner = data.get("pet_corner")
            if mon is not None:
                config.PET_MONITOR = int(mon)
            if corner:
                c = str(corner).strip().lower().replace("_", "-")
                if c in _CORNERS:
                    config.PET_CORNER = c
        except Exception:
            pass
        stop.wait(0.25)


def _style_tool_topmost(hwnd: int) -> None:
    """Tool window + always-on-top. No DWM blur-behind (it can erase the mesh)."""
    if sys.platform != "win32" or not hwnd:
        return
    import ctypes

    user32 = ctypes.windll.user32
    GWL_EXSTYLE = -20
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style = (style | 0x00000080 | 0x00000008) & ~0x00040000  # TOOLWINDOW|TOPMOST, ~APPWINDOW
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
    user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0010 | 0x0020 | 0x0040)


def _set_mouse_passthrough(hwnd: int, enabled: bool) -> None:
    if sys.platform != "win32" or not hwnd:
        return
    import ctypes

    user32 = ctypes.windll.user32
    GWL_EXSTYLE = -20
    WS_EX_TRANSPARENT = 0x00000020
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    if enabled:
        style |= WS_EX_TRANSPARENT
    else:
        style &= ~WS_EX_TRANSPARENT
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)


def _popup_menu(hwnd: int, sx: int, sy: int, state: PetState) -> str | None:
    if sys.platform != "win32":
        return None
    import ctypes

    user32 = ctypes.windll.user32
    hmenu = user32.CreatePopupMenu()
    cmds: dict[int, str] = {}
    nid = 1

    def add(label: str, cmd: str, checked: bool = False) -> None:
        nonlocal nid
        flags = 0x0000 | (0x0008 if checked else 0)
        user32.AppendMenuW(hmenu, flags, nid, label)
        cmds[nid] = cmd
        nid += 1

    add("Open console", "console")
    add("Hide pet", "hide")
    add("Quit pet", "quit")
    user32.AppendMenuW(hmenu, 0x0800, 0, None)
    for i in range(len(list_monitor_work_areas())):
        add(f"Monitor {i}", f"mon:{i}", checked=(i == config.PET_MONITOR))
    user32.AppendMenuW(hmenu, 0x0800, 0, None)
    for c in _CORNERS:
        add(c, f"corner:{c}", checked=(c == config.PET_CORNER))
    user32.SetForegroundWindow(hwnd)
    chosen = user32.TrackPopupMenu(hmenu, 0x0100 | 0x0002, sx, sy, 0, hwnd, None)
    user32.DestroyMenu(hmenu)
    user32.PostMessageW(hwnd, 0, 0, 0)
    return cmds.get(int(chosen))


def _create_glfw_window(glfw, alpha: bool):
    glfw.default_window_hints()
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE if alpha else glfw.FALSE)
    window = glfw.create_window(PET_W, PET_H, "Rocky", None, None)
    if not window:
        return None, False
    alpha_ok = bool(alpha and glfw.get_window_attrib(window, glfw.TRANSPARENT_FRAMEBUFFER))
    return window, alpha_ok


def main() -> int:
    print("pet: boot", flush=True)
    try:
        import glfw
        import moderngl
    except ImportError as e:
        print(f"pet: FAIL import ({e}) — pip install moderngl glfw", flush=True)
        return 1

    apply_saved_dock()
    areas = list_monitor_work_areas()
    if not areas:
        print("pet: FAIL no monitor work areas", flush=True)
        return 1
    if config.PET_MONITOR < 0 or config.PET_MONITOR >= len(areas):
        print(
            f"pet: PET_MONITOR={config.PET_MONITOR} out of range ({len(areas)} screens) → 0",
            flush=True,
        )
        save_dock(monitor=0, corner=config.PET_CORNER)
    x, y = pet_xy()
    left, top, right, bottom = areas[min(config.PET_MONITOR, len(areas) - 1)]
    print(
        f"pet: dock monitor={config.PET_MONITOR}/{len(areas)} "
        f"corner={config.PET_CORNER} pos=({x},{y}) "
        f"work=({left},{top})-({right},{bottom})",
        flush=True,
    )

    if not glfw.init():
        print("pet: FAIL glfw.init", flush=True)
        return 1
    print("pet: glfw.init ok", flush=True)

    # Default opaque (always visible). PET_ALPHA=1 tries cutout first.
    want_alpha = (os.environ.get("PET_ALPHA") or "").strip().lower() in ("1", "true", "yes", "on")
    window = None
    alpha_ok = False
    try:
        if want_alpha:
            print("pet: trying transparent framebuffer...", flush=True)
            window, alpha_ok = _create_glfw_window(glfw, alpha=True)
            if window is None or not alpha_ok:
                if window is not None:
                    glfw.destroy_window(window)
                print("pet: glfw alpha failed, opaque window", flush=True)
                window, _ = _create_glfw_window(glfw, alpha=False)
                alpha_ok = False
        else:
            print("pet: opaque window (PET_ALPHA=1 for cutout)", flush=True)
            window, alpha_ok = _create_glfw_window(glfw, alpha=False)
            alpha_ok = False
    except Exception as e:
        print(f"pet: FAIL create_window ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        glfw.terminate()
        return 1

    if not window:
        print("pet: FAIL create_window (null)", flush=True)
        glfw.terminate()
        return 1

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.show_window(window)
    glfw.focus_window(window)
    glfw.set_window_pos(window, x, y)

    hwnd = 0
    if sys.platform == "win32":
        try:
            hwnd = int(glfw.get_win32_window(window))
        except Exception as e:
            print(f"pet: hwnd resolve failed ({e})", flush=True)
        _style_tool_topmost(hwnd)

    print(
        f"pet: glfw window ok hwnd={hwnd} alpha={'yes' if alpha_ok else 'no'}",
        flush=True,
    )

    try:
        ctx = moderngl.create_context()
    except Exception as e:
        print(f"pet: FAIL moderngl ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        glfw.destroy_window(window)
        glfw.terminate()
        return 1

    ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

    def make_mesh(verts, indices, color, emissive=(0, 0, 0), alpha=1.0, gloss=0.45):
        vbo = ctx.buffer(verts.tobytes())
        ibo = ctx.buffer(indices.tobytes())
        vao = ctx.vertex_array(
            prog, [(vbo, "3f 3f", "in_pos", "in_nrm")], index_buffer=ibo, index_element_size=4
        )
        return Mesh(vbo, ibo, vao, int(indices.size), color, emissive, alpha, gloss)

    white = (0.96, 0.97, 0.99)
    accent = (0.24, 0.71, 1.0)
    visor_c = (0.03, 0.06, 0.09)
    eye_c = (0.48, 0.93, 1.0)

    body = make_mesh(*_sphere(0.55, scale=(0.85, 1.15, 0.8)), white, gloss=0.7)
    head = make_mesh(*_sphere(0.58, scale=(1.08, 1.0, 1.0)), white, gloss=0.7)
    visor = make_mesh(*_sphere(0.48, scale=(0.9, 0.55, 0.22)), visor_c, gloss=0.95)
    eye_l = make_mesh(*_sphere(0.09, scale=(1.2, 1.0, 0.35)), eye_c, emissive=(0.15, 0.55, 0.7), gloss=0.2)
    eye_r = make_mesh(*_sphere(0.09, scale=(1.2, 1.0, 0.35)), eye_c, emissive=(0.15, 0.55, 0.7), gloss=0.2)
    tip = make_mesh(*_sphere(0.075), accent, emissive=(0.05, 0.2, 0.35), gloss=0.5)
    stem = make_mesh(*_sphere(0.03, scale=(1.0, 6.0, 1.0)), white, gloss=0.5)
    chest = make_mesh(*_disc(0.09), accent, emissive=(0.08, 0.25, 0.4), gloss=0.3)
    shadow = make_mesh(*_disc(0.55), (0.0, 0.0, 0.0), alpha=0.22, gloss=0.0)

    state = PetState()
    stop = threading.Event()
    threading.Thread(target=_poll_status, args=(state, stop), daemon=True).start()

    passthrough = False
    t0 = time.perf_counter()
    speak_phase = 0.0
    last_dock = 0.0
    frame_n = 0

    def draw_mesh(mesh: Mesh, model: np.ndarray, proj_view: np.ndarray) -> None:
        mvp = proj_view @ model
        prog["mvp"].write(np.ascontiguousarray(mvp.T).astype("f4").tobytes())
        prog["model"].write(np.ascontiguousarray(model.T).astype("f4").tobytes())
        prog["u_color"].value = mesh.color
        prog["u_emissive"].value = mesh.emissive
        prog["u_alpha"].value = mesh.alpha
        prog["u_gloss"].value = mesh.gloss
        mesh.vao.render()

    def on_mouse(win, button, action, mods):  # noqa: ARG001
        if action != glfw.PRESS:
            return
        if button == glfw.MOUSE_BUTTON_RIGHT:
            cmd = None
            if sys.platform == "win32" and hwnd:
                import ctypes

                pt = ctypes.wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                cmd = _popup_menu(hwnd, int(pt.x), int(pt.y), state)
            if not cmd:
                return
            if cmd in ("quit", "hide"):
                print(f"pet: {cmd}", flush=True)
                glfw.set_window_should_close(window, True)
            elif cmd == "console":
                webbrowser.open(console_url())
            elif cmd.startswith("mon:"):
                save_dock(monitor=int(cmd.split(":")[1]), corner=config.PET_CORNER)
                glfw.set_window_pos(window, *pet_xy())
                print(f"pet: dock monitor={config.PET_MONITOR}", flush=True)
            elif cmd.startswith("corner:"):
                save_dock(monitor=config.PET_MONITOR, corner=cmd.split(":", 1)[1])
                glfw.set_window_pos(window, *pet_xy())
                print(f"pet: dock corner={config.PET_CORNER}", flush=True)
        elif button == glfw.MOUSE_BUTTON_LEFT:
            text = (state.last_said or "").strip() or ("…" if state.awake else "(sleeping)")
            print(f"pet: {text}", flush=True)

    glfw.set_mouse_button_callback(window, on_mouse)
    print(f"pet: ready path={'cutout' if alpha_ok else 'opaque-dark'}", flush=True)

    while not glfw.window_should_close(window):
        now = time.perf_counter()
        dt = min(0.05, now - t0)
        t0 = now
        sway = now

        if now - last_dock > 0.5:
            last_dock = now
            tx, ty = pet_xy()
            cx, cy = glfw.get_window_pos(window)
            if abs(cx - tx) > 2 or abs(cy - ty) > 2:
                glfw.set_window_pos(window, tx, ty)
            if hwnd:
                _style_tool_topmost(hwnd)

        pose = POSES.get(state.emotion, POSES["neutral"])
        slumped = not state.awake
        pitch = max(pose["pitch"], 0.38) if slumped else pose["pitch"]
        yaw = pose["yaw"] * 0.4 if slumped else (
            pose["yaw"] - 0.08 if state.listening and not state.speaking else pose["yaw"]
        )
        bounce = 0.0 if slumped else 0.032 + pose["bounce"] * 0.45
        root_y = math.sin(sway * 1.55) * bounce
        if state.speaking:
            speak_phase += dt * 11
            root_y += abs(math.sin(speak_phase)) * 0.018
        if slumped:
            root_y = -0.15

        ant_z = math.sin(sway * (2.4 + pose["ant"] * 6)) * pose["ant"]
        glow = pose["glow"] * (0.25 if slumped else 1.0)
        if state.speaking and not slumped:
            glow *= 1.12 + 0.25 * abs(math.sin(speak_phase))

        if alpha_ok:
            ctx.clear(0.0, 0.0, 0.0, 0.0)
        else:
            ctx.clear(0.04, 0.06, 0.09, 1.0)

        aspect = PET_W / max(PET_H, 1)
        pv = _perspective(26, aspect, 0.1, 40) @ _look_at((0.0, 0.02, 5.8), (0.0, -0.08, 0.0))
        root = _mat4_mul(
            _mat4_translate(0.0, root_y, 0.0),
            _mat4_rotate_z(math.sin(sway * 1.05) * (0.03 if not slumped else 0.04)),
        )

        shadow.alpha = 0.12 if slumped else 0.22
        draw_mesh(shadow, _mat4_mul(root, _mat4_translate(0.0, -1.05, 0.0)), pv)
        draw_mesh(body, _mat4_mul(root, _mat4_translate(0.0, -0.42, 0.0)), pv)
        chest.emissive = (0.08 * glow, 0.25 * glow, 0.4 * glow)
        draw_mesh(
            chest,
            _mat4_mul(root, _mat4_mul(_mat4_translate(0.0, -0.28, 0.42), _mat4_rotate_x(math.pi / 2))),
            pv,
        )

        hm = _mat4_mul(root, _mat4_translate(0.0, 0.42, 0.0))
        hm = _mat4_mul(hm, _mat4_rotate_x(pitch))
        hm = _mat4_mul(hm, _mat4_rotate_y(yaw))
        draw_mesh(head, hm, pv)
        vm = _mat4_mul(hm, _mat4_translate(0.0, 0.04, 0.5))
        draw_mesh(visor, vm, pv)

        esx, esy = pose["eye_sx"], pose["eye_sy"] * (0.12 if slumped else 1.0)
        look_x = 0.12 if state.emotion == "thinking" else 0.0
        eye_l.emissive = (0.12 * glow, 0.5 * glow, 0.65 * glow)
        eye_r.emissive = eye_l.emissive
        draw_mesh(
            eye_l,
            _mat4_mul(vm, _mat4_mul(_mat4_translate(-0.16 + look_x * 0.1, 0.06, 0.12), _mat4_scale(esx, esy, 1.0))),
            pv,
        )
        draw_mesh(
            eye_r,
            _mat4_mul(vm, _mat4_mul(_mat4_translate(0.16 + look_x * 0.1, 0.06, 0.12), _mat4_scale(esx, esy, 1.0))),
            pv,
        )

        am = _mat4_mul(hm, _mat4_mul(_mat4_translate(0.1, 0.7, 0.0), _mat4_rotate_z(ant_z)))
        draw_mesh(stem, _mat4_mul(am, _mat4_translate(0.0, 0.12, 0.0)), pv)
        if state.emotion == "angry":
            tip.color = (1.0, 0.48, 0.42)
            tip.emissive = (0.3, 0.08, 0.05)
        elif state.emotion == "thinking":
            tip.color = (1.0, 0.78, 0.34)
            tip.emissive = (0.3, 0.2, 0.05)
        else:
            tip.color = accent
            tip.emissive = (0.05, 0.2, 0.35)
        draw_mesh(tip, _mat4_mul(am, _mat4_translate(0.0, 0.28, 0.0)), pv)

        glfw.swap_buffers(window)
        glfw.poll_events()
        frame_n += 1
        if frame_n == 1:
            print("pet: first frame swapped", flush=True)

        if alpha_ok and sys.platform == "win32" and hwnd:
            mx, my = glfw.get_cursor_pos(window)
            if 0 <= mx < PET_W and 0 <= my < PET_H:
                try:
                    pix = ctx.screen.read(viewport=(int(mx), PET_H - 1 - int(my), 1, 1), components=4)
                    want = (pix[3] if len(pix) >= 4 else 255) < 12
                except Exception:
                    want = False
            else:
                want = True
            if want != passthrough:
                passthrough = want
                _set_mouse_passthrough(hwnd, want)
                if not want:
                    _style_tool_topmost(hwnd)

    stop.set()
    glfw.destroy_window(window)
    glfw.terminate()
    print("pet: quit", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"pet: FAIL ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
