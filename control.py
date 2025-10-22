# control.py
import os
import time
import math
import json
import subprocess
import pyautogui
import pandas as pd
from collections import defaultdict

# ------------------ Runtime tuning for slower PCs ------------------
# You can override these via environment variables if needed.
# Example (Windows CMD):  set PAINT_SLOW=1.5   (makes waits ~50% longer)
SLOW_FACTOR = float(os.environ.get("PAINT_SLOW", "1.0"))
BASE_SHORT = 0.12
BASE_MED   = 0.30
BASE_LONG  = 0.60

def _sleep_short(): time.sleep(BASE_SHORT * SLOW_FACTOR)
def _sleep_med():   time.sleep(BASE_MED   * SLOW_FACTOR)
def _sleep_long():  time.sleep(BASE_LONG  * SLOW_FACTOR)

# PyAutoGUI safe/pause tuning
# Keep FAILSAFE on by default (move mouse to a corner to abort).
# If you *must* disable (not recommended), set PAINT_FAILSAFE=0 in env.
pyautogui.FAILSAFE = (os.environ.get("PAINT_FAILSAFE", "1") != "0")
# Small pause after each PyAutoGUI call
pyautogui.PAUSE = float(os.environ.get("PAINT_PAUSE", "0.05")) * SLOW_FACTOR

# ------------------ Config: one file per chat ------------------
SAVE_ROOT = os.path.abspath("./data/saved_drawings")
os.makedirs(SAVE_ROOT, exist_ok=True)

STATE_CSV = "data/history/shapes_state.csv"  # persistent shapes state per session

# ------------------ Scaling helpers ------------------
SIZE_TO_SCALE = {
    "small": 0.3,
    "medium": 0.6,   # default
    "big": 1,
}

TOP_INSET_PCT = 0.40
BOTTOM_INSET_PCT = 0.27
SAFE_PAD_PX = 16

def _normalize_position(position: str) -> str:
    if not position:
        return "center"
    p = position.strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "down": "bottom",
        "up": "top",
        "centre": "center",
        "middle": "center",
        "left_bottom": "bottom_left",
        "right_bottom": "bottom_right",
        "left_top": "top_left",
        "right_top": "top_right",
    }
    return alias.get(p, p)

def _normalize_size(size: str) -> str:
    s = (size or "medium").strip().lower()
    if s == "large":
        s = "big"
    return s if s in SIZE_TO_SCALE else "medium"

def _make_scaler(size: str):
    scale = SIZE_TO_SCALE[_normalize_size(size)]
    def S(x):
        return int(round(x * scale))
    return S

def _file_for_session(session_name: str) -> str:
    os.makedirs(SAVE_ROOT, exist_ok=True)
    return os.path.join(SAVE_ROOT, f"{session_name}.png")

# ------------------ Persistence helpers (CSV) ------------------
def _ensure_state_csv():
    if not os.path.exists(STATE_CSV):
        df = pd.DataFrame(columns=["session", "shapes_json", "last_deleted_json", "last_action", "counter"])
        df.to_csv(STATE_CSV, index=False)

def _load_state_row(session_name: str):
    _ensure_state_csv()
    df = pd.read_csv(STATE_CSV)
    if df.empty or not (df["session"] == session_name).any():
        return {"session": session_name, "shapes": [], "last_deleted": None, "last_action": "", "counter": 0}
    row = df[df["session"] == session_name].iloc[0]
    shapes_json = row.get("shapes_json", "[]")
    last_deleted_json = row.get("last_deleted_json", "")
    try:
        shapes = json.loads(shapes_json or "[]")
    except Exception:
        shapes = []
    try:
        last_deleted = json.loads(last_deleted_json) if isinstance(last_deleted_json, str) and last_deleted_json else None
    except Exception:
        last_deleted = None
    last_action = "" if pd.isna(row.get("last_action", "")) else str(row.get("last_action", ""))
    counter = int(row.get("counter", 0) or 0)
    return {
        "session": session_name,
        "shapes": shapes,
        "last_deleted": last_deleted,
        "last_action": last_action,
        "counter": counter
    }

def _save_state_row(session_name: str, shapes: list, last_deleted: dict | None, last_action: str, counter: int):
    _ensure_state_csv()
    try:
        df = pd.read_csv(STATE_CSV)
    except Exception:
        df = pd.DataFrame(columns=["session", "shapes_json", "last_deleted_json", "last_action", "counter"])

    shapes_json = json.dumps(shapes, ensure_ascii=False)
    last_deleted_json = json.dumps(last_deleted, ensure_ascii=False) if last_deleted is not None else ""

    if "session" not in df.columns:
        df = pd.DataFrame(columns=["session", "shapes_json", "last_deleted_json", "last_action", "counter"])

    if (df["session"] == session_name).any():
        df.loc[df["session"] == session_name, ["shapes_json", "last_deleted_json", "last_action", "counter"]] = [
            shapes_json, last_deleted_json, last_action, counter
        ]
    else:
        df = pd.concat([df, pd.DataFrame([{
            "session": session_name,
            "shapes_json": shapes_json,
            "last_deleted_json": last_deleted_json,
            "last_action": last_action,
            "counter": counter
        }])], ignore_index=True)

    df.to_csv(STATE_CSV, index=False)

class Control:
    session_processes = {}
    session_files = {}

    session_shapes = defaultdict(list)
    session_shape_counter = defaultdict(int)
    session_last_deleted = defaultdict(lambda: None)
    session_last_action = defaultdict(str)

    @staticmethod
    def set_save_root(path: str):
        global SAVE_ROOT
        SAVE_ROOT = os.path.abspath(path or SAVE_ROOT)
        os.makedirs(SAVE_ROOT, exist_ok=True)

    @staticmethod
    def _load_session_state(session_name: str):
        if session_name in Control.session_shape_counter and Control.session_shapes[session_name]:
            return
        row = _load_state_row(session_name)
        Control.session_shapes[session_name] = row["shapes"]
        Control.session_shape_counter[session_name] = row["counter"]
        Control.session_last_deleted[session_name] = row["last_deleted"]
        Control.session_last_action[session_name] = row["last_action"]

    @staticmethod
    def _persist_session_state(session_name: str):
        _save_state_row(
            session_name,
            Control.session_shapes[session_name],
            Control.session_last_deleted[session_name],
            Control.session_last_action[session_name],
            Control.session_shape_counter[session_name]
        )

    @staticmethod
    def _register_shape(session_name: str, shape_type: str, bbox: tuple, position: str, size: str):
        Control._load_session_state(session_name)
        Control.session_shape_counter[session_name] += 1
        shape_id = Control.session_shape_counter[session_name]
        shape = {
            "id": shape_id,
            "type": shape_type,
            "position": _normalize_position(position),
            "size": _normalize_size(size),
            "bbox": tuple(int(v) for v in bbox),
            "ts": time.time(),
        }
        Control.session_shapes[session_name].append(shape)
        Control.session_last_action[session_name] = "draw"
        Control.session_last_deleted[session_name] = None
        Control._persist_session_state(session_name)
        return shape_id

    # ---------- Window helpers ----------
    @staticmethod
    def _has_paint_window(session_name: str) -> bool:
        try:
            for t in (f"{session_name}.png - Paint", f"{session_name} - Paint"):
                if pyautogui.getWindowsWithTitle(t):
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _activate_paint_window(session_name: str) -> bool:
        try:
            for t in (f"{session_name}.png - Paint", f"{session_name} - Paint"):
                wins = pyautogui.getWindowsWithTitle(t)
                if not wins:
                    continue
                w = wins[0]
                try:
                    if w.isMinimized:
                        w.restore(); _sleep_med()
                    w.activate(); _sleep_med()
                    return True
                except Exception:
                    pass
        except Exception:
            pass
        return False

    @staticmethod
    def _maximize_window():
        try:
            pyautogui.hotkey('alt', 'space'); _sleep_med()
            pyautogui.press('x'); _sleep_med()
        except Exception:
            pass

    @staticmethod
    def _normalize_zoom():
        try:
            pyautogui.hotkey('ctrl', '1'); _sleep_med()
        except Exception:
            pass

    @staticmethod
    def _set_canvas_size(width_px=2000, height_px=800):
        try:
            pyautogui.hotkey('ctrl', 'e'); _sleep_long()
            try:
                pyautogui.hotkey('alt', 'p'); _sleep_short()
            except Exception:
                pass
            pyautogui.typewrite(str(width_px)); _sleep_short()
            pyautogui.press('tab'); _sleep_short()
            pyautogui.typewrite(str(height_px)); _sleep_short()
            pyautogui.press('enter'); _sleep_long()
        except Exception as e:
            print("Canvas resize failed:", e)

    @staticmethod
    def _save_as(filepath: str):
        try:
            pyautogui.press('f12'); _sleep_long()
            pyautogui.typewrite(filepath); _sleep_short()
            pyautogui.press('enter'); _sleep_long()
        except Exception:
            pass

    @staticmethod
    def _save():
        try:
            pyautogui.hotkey('ctrl', 's'); _sleep_med()
        except Exception:
            pass

    @staticmethod
    def _minimize_paint_window():
        try:
            pyautogui.hotkey('alt', 'space'); _sleep_med()
            pyautogui.press('n'); _sleep_med()
        except Exception:
            pass

    @staticmethod
    def _close_paint():
        try:
            pyautogui.hotkey('alt', 'f4'); _sleep_long()
        except Exception:
            pass

    @staticmethod
    def setup_mspaint(session_name: str, position: str = "center"):
        filepath = _file_for_session(session_name)

        # drop dead process if user closed Paint
        if session_name in Control.session_processes:
            try:
                if Control.session_processes[session_name].poll() is not None:
                    Control.session_processes.pop(session_name, None)
            except Exception:
                Control.session_processes.pop(session_name, None)

        if session_name not in Control.session_processes:
            try:
                if os.path.exists(filepath):
                    proc = subprocess.Popen(["mspaint", filepath])
                else:
                    proc = subprocess.Popen("mspaint")
                Control.session_processes[session_name] = proc
            except FileNotFoundError:
                print("MS Paint not found! Make sure you are on Windows.")
                return

            # Give slower PCs more time to bring up Paint
            _sleep_long(); _sleep_long()

            Control._activate_paint_window(session_name)
            if not os.path.exists(filepath):
                Control._maximize_window()
                Control._normalize_zoom()
                Control._set_canvas_size(width_px=2000, height_px=800)
                Control._save_as(filepath)
            else:
                Control._maximize_window()
                Control._normalize_zoom()
        else:
            if not Control._activate_paint_window(session_name):
                try:
                    proc = subprocess.Popen(["mspaint", filepath])
                    Control.session_processes[session_name] = proc
                    _sleep_long(); _sleep_long()
                    Control._activate_paint_window(session_name)
                except Exception:
                    pass

        # final safety relaunch if no window
        if not Control._has_paint_window(session_name):
            try:
                proc = subprocess.Popen(["mspaint", _file_for_session(session_name)])
                Control.session_processes[session_name] = proc
                _sleep_long(); _sleep_long()
                Control._activate_paint_window(session_name)
            except Exception:
                pass

        Control._load_session_state(session_name)

    @staticmethod
    def _ensure_paint_ready(session_name: str):
        """
        Ensure MS Paint is running, the session file is loaded, and the window is active.
        Used for actions (like delete) that must work even if Paint was closed.
        """
        Control.setup_mspaint(session_name)
        Control._activate_paint_window(session_name)
        _sleep_med()

    @staticmethod
    def _get_anchor_center(position: str, margin_ratio=0.15):
        sw, sh = pyautogui.size()
        pos = (position or "center").strip().lower()
        if pos == "down":
            pos = "bottom"

        mx = int(sw * margin_ratio) + 200
        my_top = max(int(sh * TOP_INSET_PCT), SAFE_PAD_PX)
        my_bottom = max(int(sh * BOTTOM_INSET_PCT), SAFE_PAD_PX)
        toolbar_offset = int(sh * 0.05)

        centers = {
            "center":        (sw // 2, sh // 2 + toolbar_offset),
            "left":          (mx,       sh // 2 + toolbar_offset),
            "right":         (sw - mx,  sh // 2 + toolbar_offset),
            "top":           (sw // 2,  my_top),
            "bottom":        (sw // 2,  sh - my_bottom),
            "top_left":      (mx,       my_top),
            "top_right":     (sw - mx,  my_top),
            "bottom_left":   (mx,       sh - my_bottom),
            "bottom_right":  (sw - mx,  sh - my_bottom),
        }
        return centers.get(pos, centers["center"])

    # -------------- drawing primitives (each returns bbox) --------------
    @staticmethod
    def draw_house_at(cx, cy, S):
        house_width  = S(250)
        house_height = S(150)
        roof_height  = S(120)
        overhang     = S(20)
        chimney_w    = S(30)
        chimney_h    = S(70)
        chimney_offset_from_house_top = S(60)

        total_width  = house_width + 2 * overhang
        extra_above_apex = max(0, (chimney_offset_from_house_top + chimney_h) - roof_height)
        total_height = house_height + roof_height + extra_above_apex

        bottom_y = cy + total_height // 2
        left_footprint_x = cx - total_width // 2
        start_x = left_footprint_x + overhang
        start_y = bottom_y

        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragRel(0, -house_height, duration=0.35)
        pyautogui.dragRel(house_width, 0, duration=0.35)
        pyautogui.dragRel(0, house_height, duration=0.35)
        pyautogui.dragRel(-house_width, 0, duration=0.35)

        pyautogui.moveTo(start_x - overhang, start_y - house_height)
        pyautogui.dragRel(house_width // 2 + overhang, -roof_height, duration=0.28)
        pyautogui.dragRel(house_width // 2 + overhang,  roof_height, duration=0.28)
        pyautogui.dragRel(-(house_width + 2 * overhang), 0, duration=0.28)

        door_width  = S(60)
        door_height = S(90)
        door_x = cx - door_width // 2
        door_y = start_y
        pyautogui.moveTo(door_x, door_y)
        pyautogui.dragRel(0, -door_height, duration=0.22)
        pyautogui.dragRel(door_width, 0, duration=0.22)
        pyautogui.dragRel(0,  door_height, duration=0.22)
        pyautogui.dragRel(-door_width, 0, duration=0.22)

        knob_x = door_x + door_width - S(15)
        knob_y = door_y - door_height // 2
        pyautogui.moveTo(knob_x, knob_y); pyautogui.click(); _sleep_short()

        window_w = S(50); window_h = S(50)
        win1_x = start_x + S(20); win1_y = start_y - S(40)
        pyautogui.moveTo(win1_x, win1_y)
        pyautogui.dragRel(0, -window_h, duration=0.14)
        pyautogui.dragRel(window_w, 0, duration=0.14)
        pyautogui.dragRel(0,  window_h, duration=0.14)
        pyautogui.dragRel(-window_w, 0, duration=0.14)
        pyautogui.moveTo(win1_x + window_w // 2, win1_y)
        pyautogui.dragRel(0, -window_h, duration=0.10)
        pyautogui.moveTo(win1_x, win1_y - window_h // 2)
        pyautogui.dragRel(window_w, 0, duration=0.10)

        win2_x = start_x + house_width - S(20) - window_w; win2_y = win1_y
        pyautogui.moveTo(win2_x, win2_y)
        pyautogui.dragRel(0, -window_h, duration=0.14)
        pyautogui.dragRel(window_w, 0, duration=0.14)
        pyautogui.dragRel(0,  window_h, duration=0.14)
        pyautogui.dragRel(-window_w, 0, duration=0.14)
        pyautogui.moveTo(win2_x + window_w // 2, win2_y)
        pyautogui.dragRel(0, -window_h, duration=0.10)
        pyautogui.moveTo(win2_x, win2_y - window_h // 2)
        pyautogui.dragRel(window_w, 0, duration=0.10)

        chimney_x = start_x + house_width - S(70)
        chimney_base_y = start_y - house_height - chimney_offset_from_house_top
        pyautogui.moveTo(chimney_x, chimney_base_y)
        pyautogui.dragRel(0, -chimney_h, duration=0.14)
        pyautogui.dragRel(chimney_w, 0, duration=0.14)
        pyautogui.dragRel(0,  chimney_h, duration=0.14)
        pyautogui.dragRel(-chimney_w, 0, duration=0.14)

        x1 = left_footprint_x
        y1 = start_y - house_height - roof_height - extra_above_apex
        x2 = left_footprint_x + total_width
        y2 = bottom_y
        return (x1, y1, x2, y2)

    @staticmethod
    def draw_tree_at(cx, cy, S):
        trunk_w = S(30); trunk_h = S(100)
        leaf_base = S(170); leaf_h = S(80)
        layers = 3; layer_gap = S(60)

        top_extra = leaf_h + (layers - 1) * layer_gap
        total_h = trunk_h + top_extra

        bottom_y = cy + total_h // 2
        trunk_x = cx - trunk_w // 2; trunk_y = bottom_y

        pyautogui.moveTo(trunk_x, trunk_y)
        pyautogui.dragRel(0, -trunk_h, duration=0.22)
        pyautogui.dragRel(trunk_w, 0, duration=0.22)
        pyautogui.dragRel(0,  trunk_h, duration=0.22)
        pyautogui.dragRel(-trunk_w, 0, duration=0.22)

        leaf_start_y = trunk_y - trunk_h
        for i in range(layers):
            layer_base = leaf_base - S(20) * i
            layer_height = S(80)
            layer_x_start = cx - layer_base // 2
            layer_y_start = leaf_start_y - layer_gap * i
            pyautogui.moveTo(layer_x_start, layer_y_start)
            pyautogui.dragRel(layer_base // 2, -layer_height, duration=0.22)
            pyautogui.dragRel(layer_base // 2,  layer_height, duration=0.22)
            pyautogui.dragRel(-layer_base,      0,           duration=0.22)

        total_w = max(trunk_w, leaf_base)
        x1 = cx - total_w // 2
        y1 = bottom_y - total_h
        x2 = cx + total_w // 2
        y2 = bottom_y
        return (x1, y1, x2, y2)

    @staticmethod
    def draw_flower_at(cx, cy, S):
        center_radius = S(20)
        steps = 36
        for i in range(steps):
            angle = 2 * math.pi * i / steps
            x = int(cx + center_radius * math.cos(angle))
            y = int(cy + center_radius * math.sin(angle))
            if i == 0: pyautogui.moveTo(x, y)
            else:      pyautogui.dragTo(x, y, duration=0.012, button='left')

        petal_radius = S(30)
        num_petals = 8
        for i in range(num_petals):
            angle_offset = 2 * math.pi * i / num_petals
            pcx = int(cx + (center_radius + petal_radius) * math.cos(angle_offset))
            pcy = int(cy + (center_radius + petal_radius) * math.sin(angle_offset))
            for j in range(steps):
                angle = 2 * math.pi * j / steps
                x = int(pcx + petal_radius * math.cos(angle))
                y = int(pcy + petal_radius * math.sin(angle))
                if j == 0: pyautogui.moveTo(x, y)
                else:      pyautogui.dragTo(x, y, duration=0.012, button='left')

        stem_height = S(100)
        stem_x = cx
        stem_y_start = cy + center_radius + petal_radius
        pyautogui.moveTo(stem_x, stem_y_start)
        pyautogui.dragRel(0, stem_height, duration=0.26)

        leaf_size = S(40)
        pyautogui.moveTo(stem_x, stem_y_start + S(20))
        pyautogui.dragRel(-leaf_size, leaf_size // 2, duration=0.12)
        pyautogui.dragRel(leaf_size, 0, duration=0.12)
        pyautogui.dragRel(-leaf_size, -leaf_size // 2, duration=0.12)

        pyautogui.moveTo(stem_x, stem_y_start + S(60))
        pyautogui.dragRel(leaf_size, leaf_size // 2, duration=0.12)
        pyautogui.dragRel(-leaf_size, 0, duration=0.12)
        pyautogui.dragRel(leaf_size, -leaf_size // 2, duration=0.12)

        r = center_radius + petal_radius
        stem_bottom = stem_y_start + stem_height
        leaf_span = S(40)
        x1 = cx - (r + leaf_span)
        y1 = cy - r
        x2 = cx + (r + leaf_span)
        y2 = stem_bottom + leaf_span // 2
        return (x1, y1, x2, y2)

    @staticmethod
    def draw_boat_at(cx, cy, S):
        hull_width = S(200); hull_height = S(50)
        hull_top_x = cx - hull_width // 2
        hull_top_y = cy + S(50)
        hull_bottom_y = hull_top_y + hull_height

        pyautogui.moveTo(hull_top_x, hull_top_y)
        pyautogui.dragRel(hull_width, 0, duration=0.22)
        pyautogui.dragRel(-hull_width, hull_height, duration=0.22)
        pyautogui.dragRel(0, -hull_height, duration=0.22)

        mast_height = S(120); mast_x = cx
        mast_y_top = hull_top_y - mast_height
        mast_y_bottom = hull_top_y
        pyautogui.moveTo(mast_x, mast_y_bottom)
        pyautogui.dragTo(mast_x, mast_y_top, duration=0.26)

        sail_width = S(80); sail_height = S(100)
        pyautogui.moveTo(mast_x, mast_y_top)
        pyautogui.dragRel(-sail_width, sail_height, duration=0.16)
        pyautogui.dragRel(sail_width, 0, duration=0.16)
        pyautogui.dragRel(0, -sail_height, duration=0.16)

        wave_length = S(60); wave_count = 3
        for i in range(wave_count):
            start_x = cx - hull_width // 2 + S(50) * i
            start_y = hull_bottom_y + S(10) + S(10) * i
            pyautogui.moveTo(start_x, start_y)
            pyautogui.dragRel(wave_length // 2, -S(10), duration=0.12)
            pyautogui.dragRel(wave_length // 2,  S(10), duration=0.12)

        x1 = cx - hull_width // 2
        y1 = mast_y_top
        x2 = cx + hull_width // 2
        y2 = hull_bottom_y + S(20)
        return (x1, y1, x2, y2)

    @staticmethod
    def draw_sun_at(cx, cy, S):
        sun_radius = S(50)
        steps = 36
        for i in range(steps):
            angle = 2 * math.pi * i / steps
            x = int(cx + sun_radius * math.cos(angle))
            y = int(cy + sun_radius * math.sin(angle))
            if i == 0:
                pyautogui.moveTo(x, y)
            else:
                pyautogui.dragTo(x, y, duration=0.04, button='left')
        ray_length = S(40)
        for i in range(12):
            angle = 2 * math.pi * i / 12
            x_start = int(cx + sun_radius * math.cos(angle))
            y_start = int(cy + sun_radius * math.sin(angle))
            x_end = int(cx + (sun_radius + ray_length) * math.cos(angle))
            y_end = int(cy + (sun_radius + ray_length) * math.sin(angle))
            pyautogui.moveTo(x_start, y_start)
            pyautogui.dragTo(x_end, y_end, duration=0.04, button='left')
        min_x = cx - (sun_radius + ray_length)
        min_y = cy - (sun_radius + ray_length)
        max_x = cx + (sun_radius + ray_length)
        max_y = cy + (sun_radius + ray_length)
        return (min_x, min_y, max_x, max_y)

    @staticmethod
    def _select_rect_and_delete(x1, y1, x2, y2, pad=6):
        try:
            # Home -> Select -> Rectangular selection
            pyautogui.keyDown('alt'); pyautogui.press('h'); pyautogui.keyUp('alt'); _sleep_med()
            pyautogui.press('s'); _sleep_med()
            pyautogui.press('r'); _sleep_med()
            _sleep_long()
        except Exception:
            pass
        tx1 = max(0, int(min(x1, x2) - pad))
        ty1 = max(0, int(min(y1, y2) - pad))
        tx2 = int(max(x1, x2) + pad)
        ty2 = int(max(y1, y2) + pad)
        pyautogui.moveTo(tx1, ty1); _sleep_short()
        pyautogui.dragTo(tx2, ty2, duration=0.30); _sleep_short()
        pyautogui.press('delete'); _sleep_med()
        # Return to brush
        try:
            pyautogui.hotkey('alt', 'b'); _sleep_med()
        except Exception:
            pass

    @staticmethod
    def _is_session_process_alive(session_name: str) -> bool:
        proc = Control.session_processes.get(session_name)
        try:
            return bool(proc) and (proc.poll() is None)
        except Exception:
            return False

    @staticmethod
    def ms_paint(
        session_name: str,
        draw: str | None = None,
        position: str = "center",
        size: str = "medium",
        action: str | None = None,
        shape: str | None = None,
    ):
        if action is None:
            action = "draw"
            shape = draw

        action = (action or "").strip().lower()
        pos = _normalize_position(position)
        size_norm = _normalize_size(size)

        if action == "draw":
            shape_key = (shape or "").strip().lower()
            if not shape_key:
                return {"ok": False, "reason": "no_shape"}

            # Check occupancy BEFORE opening Paint
            Control._load_session_state(session_name)
            if any(s.get("position") == pos for s in Control.session_shapes[session_name]):
                return {"ok": False, "reason": "occupied", "position": pos}

            # Only now open/activate Paint
            Control.setup_mspaint(session_name, position=pos)
            Control._activate_paint_window(session_name)

            # ensure brush
            def _force_brush():
                try:
                    pyautogui.hotkey('alt', 'b'); _sleep_med()
                except Exception:
                    pass
            _force_brush()

            S = _make_scaler(size_norm)
            cx, cy = Control._get_anchor_center(pos)
            bbox = None
            if shape_key == 'house':
                bbox = Control.draw_house_at(cx, cy, S)
            elif shape_key == 'tree':
                bbox = Control.draw_tree_at(cx, cy, S)
            elif shape_key == 'sun':
                bbox = Control.draw_sun_at(cx, cy, S)
            elif shape_key == 'flower':
                bbox = Control.draw_flower_at(cx, cy, S)
            elif shape_key == 'boat':
                bbox = Control.draw_boat_at(cx, cy, S)
            else:
                return {"ok": False, "reason": "unknown_shape", "shape": shape_key}

            if bbox:
                Control._register_shape(session_name, shape_key, bbox, pos, size_norm)
            Control._save()
            _sleep_med()
            Control._minimize_paint_window()
            return {"ok": True, "shape": shape_key, "position": pos, "size": size_norm}

        elif action == "delete":
            # Delete should work even if closed: reopen and act
            Control._ensure_paint_ready(session_name)

            shape_key = (shape or "").strip().lower()
            if shape_key or pos:
                return Control.delete_by(session_name, shape_key, position=pos)
            else:
                return Control.delete_previous(session_name)

        elif action == "redo":
            # Redo MUST NOT reopen Paint. Only proceed if Paint is already open.
            return Control.redo_previous_delete(session_name)

        else:
            return {"ok": False, "reason": "unknown_action", "action": action}

    @staticmethod
    def delete_previous(session_name: str):
        # Always ensure Paint + file are ready, even if it was closed.
        Control._ensure_paint_ready(session_name)

        Control._load_session_state(session_name)
        shapes = Control.session_shapes[session_name]
        if not shapes:
            Control._minimize_paint_window()
            return {"ok": False, "reason": "no_shapes"}

        shape = shapes.pop()
        x1, y1, x2, y2 = shape["bbox"]
        Control._select_rect_and_delete(x1, y1, x2, y2)
        Control._save()
        Control.session_last_deleted[session_name] = shape
        Control.session_last_action[session_name] = "delete"
        Control._persist_session_state(session_name)
        Control._minimize_paint_window()
        return {"ok": True, "deleted": shape}

    @staticmethod
    def delete_by(session_name: str, shape_type: str | None, position: str | None = None, size: str | None = None):
        """
        Delete logic (match from latest backward):
          - Any combination of filters is allowed.
          - Filters: type (shape_type), position, size.
          - If none provided, fallback to delete_previous (but the caller already guards this).
        """
        # Always ensure Paint + file are ready
        Control._ensure_paint_ready(session_name)

        Control._load_session_state(session_name)
        shape_type = (shape_type or "").strip().lower()
        pos_norm = _normalize_position(position) if position else None
        size_norm = _normalize_size(size) if size else None
        shapes = Control.session_shapes[session_name]

        def match(s):
            type_ok = (not shape_type) or (s["type"] == shape_type)
            pos_ok  = (not pos_norm)  or (s.get("position") == pos_norm)
            size_ok = (not size_norm) or (s.get("size") == size_norm)
            return type_ok and pos_ok and size_ok

        target_idx = None
        for i in range(len(shapes) - 1, -1, -1):
            if match(shapes[i]):
                target_idx = i
                break

        if target_idx is None:
            Control._minimize_paint_window()
            return {"ok": False, "reason": "no_match",
                    "type": shape_type or None, "position": pos_norm, "size": size_norm}

        shape = shapes.pop(target_idx)
        x1, y1, x2, y2 = shape["bbox"]
        Control._select_rect_and_delete(x1, y1, x2, y2)
        Control._save()
        Control.session_last_deleted[session_name] = shape
        Control.session_last_action[session_name] = "delete"
        Control._persist_session_state(session_name)
        Control._minimize_paint_window()
        return {"ok": True, "deleted": shape}

    @staticmethod
    def redo_previous_delete(session_name: str):
        """
        Redo should NOT reopen Paint. It only works if the SAME Paint process
        (tracked for this session) is still running, and the session's window is open.
        If the user closed Paint and reopened manually, redo is unavailable.
        """
        Control._load_session_state(session_name)

        if not Control._is_session_process_alive(session_name):
            return {"ok": False, "reason": "process_closed"}

        if not Control._has_paint_window(session_name):
            return {"ok": False, "reason": "window_closed"}

        if Control.session_last_action[session_name] != "delete":
            return {"ok": False, "reason": "nothing_to_redo"}

        shape = Control.session_last_deleted[session_name]
        if not shape:
            return {"ok": False, "reason": "no_last_deleted"}

        if not Control._activate_paint_window(session_name):
            return {"ok": False, "reason": "cannot_activate"}

        try:
            pyautogui.hotkey('ctrl', 'z'); _sleep_med()
            Control._save()
            Control.session_shapes[session_name].append(shape)
            Control.session_last_deleted[session_name] = None
            Control.session_last_action[session_name] = "redo"
            Control._persist_session_state(session_name)
        finally:
            Control._minimize_paint_window()

        return {"ok": True, "redid": shape}

    @staticmethod
    def delete_session(session_name: str):
        """
        Close the MS Paint window for this session ONLY if the tracked process is still alive
        and the window is present. If Paint was closed manually, skip touching Paint and
        just clear in-memory state.
        """
        try:
            alive = Control._is_session_process_alive(session_name)
            has_win = Control._has_paint_window(session_name) if alive else False
            if alive and has_win:
                Control._activate_paint_window(session_name)
                Control._save()
                Control._close_paint()
                _sleep_med()
            if session_name in Control.session_processes:
                Control.session_processes.pop(session_name, None)
        except Exception:
            pass

        # Always clear our in-memory bookkeeping for this session
        try:
            Control.session_shapes.pop(session_name, None)
            Control.session_shape_counter.pop(session_name, None)
            Control.session_last_deleted.pop(session_name, None)
            Control.session_last_action.pop(session_name, None)
        except Exception:
            pass

    @staticmethod
    def save_canvas(session_name: str):
        if session_name in Control.session_processes:
            Control._save()

    @staticmethod
    def save_canvas_and_close(session_name: str, outfile: str = None):
        if session_name not in Control.session_processes:
            return
        if outfile:
            Control._save_as(outfile)
        else:
            Control._save()
        Control._close_paint()
        try:
            Control.session_processes.pop(session_name, None)
            _sleep_med()
        except Exception:
            pass

if __name__ == "__main__":
    Control.ms_paint("chat-1", action="draw", shape="house", position="center", size="big")
    Control.ms_paint("chat-1", action="delete")
    Control.ms_paint("chat-1", action="redo")
