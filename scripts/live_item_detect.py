from __future__ import annotations

import argparse
import ctypes
import re
import sys
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageGrab, ImageTk


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itemdetect.config import load_config  # noqa: E402
from itemdetect.infer import (  # noqa: E402
    aggregate_topk_by_class,
    decode_vector,
)


TARGET_TITLES = ("\u539f\u795e", "Genshin Impact")
PREVIEW_WINDOW = "ItemDetect Live"
MASK_WINDOW = "HSV Mask"
CONTROL_WINDOW = "ItemDetect Control"

HSV_DEFAULTS = {
    "H min": 20,
    "H max": 24,
    "S min": 10,
    "S max": 17,
    "V min": 223,
    "V max": 233,
    "Area min": 2100,
    "Area max": 6700,
    "Crop W": 124,
    "Crop H": 124,
    "Crop X Offset": 0,
    "Crop Y Offset": 0,
}

DISPLAY_DEFAULTS = {
    "Show HSV rect": True,
    "Show locate rect": True,
    "Show crop rect": True,
}

CARD_BOTTOM_MARGIN = 25
CACHE_GRID_PX = 4
CACHE_TTL_SECONDS = 0.5

user32 = ctypes.windll.user32


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def set_dpi_aware() -> None:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


def get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def find_target_window() -> tuple[int | None, str]:
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_proc(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        title = get_window_title(hwnd)
        if any(token in title for token in TARGET_TITLES):
            found.append((hwnd, title))
            return False
        return True

    user32.EnumWindows(enum_proc, 0)
    return found[0] if found else (None, "")


def get_client_bbox(hwnd: int) -> tuple[int, int, int, int] | None:
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    top_left = POINT(0, 0)
    bottom_right = POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        return None
    if bottom_right.x <= top_left.x or bottom_right.y <= top_left.y:
        return None
    return top_left.x, top_left.y, bottom_right.x, bottom_right.y


def capture_client(hwnd: int) -> np.ndarray | None:
    bbox = get_client_bbox(hwnd)
    if bbox is None:
        return None
    image = ImageGrab.grab(bbox=bbox)
    return np.asarray(image.convert("RGB"))


def preprocess_rgb(image_rgb: np.ndarray, cfg: dict) -> np.ndarray:
    image_size = cfg["data"].get("image_size", [125, 125])
    width, height = int(image_size[0]), int(image_size[1])
    resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)
    norm = cfg.get("normalization", {})
    mean = np.asarray(norm.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(norm.get("std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
    return ((tensor - mean) / std)[None, ...].astype(np.float32)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_text(
    frame_bgr: np.ndarray,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    x, y = xy
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_text_lines_top_right(
    frame_bgr: np.ndarray,
    lines: list[str],
    top_right: tuple[int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int] = (80, 255, 80),
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    x, y = top_right
    x = min(max(0, x), image.width - 1)
    y = min(max(0, y), image.height - 1)
    line_gap = 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = max(0, x - text_w)
        text_y = min(max(0, y), max(0, image.height - text_h))
        draw.text((text_x + 1, text_y + 1), line, font=font, fill=(0, 0, 0))
        draw.text((text_x, text_y), line, font=font, fill=fill)
        y += text_h + line_gap

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class LiveUi:
    def __init__(self) -> None:
        self.running = True
        self.root = tk.Tk()
        self.root.title(CONTROL_WINDOW)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.variables: dict[str, tk.IntVar] = {}
        self.display_variables: dict[str, tk.BooleanVar] = {}
        self.last_frame_bgr: np.ndarray | None = None
        self.last_mask_bgr: np.ndarray | None = None
        self.test_requested = False
        self.save_requested = False

        for row, (name, value) in enumerate(HSV_DEFAULTS.items()):
            if name.startswith("Area"):
                maximum = 20000
            elif name.startswith("H"):
                maximum = 179
            elif name.endswith("Offset"):
                maximum = 30
            elif name.startswith("Crop"):
                maximum = 150
            else:
                maximum = 255
            if name.endswith("Offset"):
                minimum = -30
            elif name.startswith("Crop"):
                minimum = 100
            else:
                minimum = 0
            variable = tk.IntVar(value=value)
            self.variables[name] = variable
            label = tk.Label(self.root, text=name, width=10, anchor="w")
            label.grid(row=row, column=0, padx=6, pady=3, sticky="w")
            scale = tk.Scale(
                self.root,
                from_=minimum,
                to=maximum,
                orient=tk.HORIZONTAL,
                length=340,
                variable=variable,
                showvalue=True,
                resolution=1,
            )
            scale.grid(row=row, column=1, padx=6, pady=3, sticky="ew")

        display_start_row = len(HSV_DEFAULTS)
        for offset, (name, value) in enumerate(DISPLAY_DEFAULTS.items()):
            variable = tk.BooleanVar(value=value)
            self.display_variables[name] = variable
            checkbox = tk.Checkbutton(self.root, text=name, variable=variable, anchor="w")
            checkbox.grid(row=display_start_row + offset, column=0, columnspan=2, padx=6, pady=3, sticky="w")

        self.preview = tk.Toplevel(self.root)
        self.preview.title(PREVIEW_WINDOW)
        self.preview.protocol("WM_DELETE_WINDOW", self.stop)
        self.preview.geometry("960x540")
        self.preview_label = tk.Label(self.preview, bg="black")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        self.photo: ImageTk.PhotoImage | None = None
        self.preview.bind("<Configure>", lambda _event: self.refresh_preview())

        self.mask_preview = tk.Toplevel(self.root)
        self.mask_preview.title(MASK_WINDOW)
        self.mask_preview.protocol("WM_DELETE_WINDOW", self.stop)
        self.mask_preview.geometry("640x360")
        self.mask_label = tk.Label(self.mask_preview, bg="black")
        self.mask_label.pack(fill=tk.BOTH, expand=True)
        self.mask_photo: ImageTk.PhotoImage | None = None
        self.mask_preview.bind("<Configure>", lambda _event: self.refresh_mask_preview())

        for widget in (self.root, self.preview, self.mask_preview):
            widget.bind("<Escape>", lambda _event: self.stop())
            widget.bind("<Key-q>", lambda _event: self.stop())
            widget.bind("<Key-Q>", lambda _event: self.stop())
            widget.bind("<Key-t>", lambda _event: self.request_test())
            widget.bind("<Key-T>", lambda _event: self.request_test())
            widget.bind("<Key-s>", lambda _event: self.request_save())
            widget.bind("<Key-S>", lambda _event: self.request_save())

    def stop(self) -> None:
        self.running = False

    def request_test(self) -> None:
        self.test_requested = True

    def request_save(self) -> None:
        self.save_requested = True

    def consume_test_request(self) -> bool:
        requested = self.test_requested
        self.test_requested = False
        return requested

    def consume_save_request(self) -> bool:
        requested = self.save_requested
        self.save_requested = False
        return requested

    def read_controls(self) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int, int]:
        values = {name: variable.get() for name, variable in self.variables.items()}
        h_min, h_max = sorted((values["H min"], values["H max"]))
        s_min, s_max = sorted((values["S min"], values["S max"]))
        v_min, v_max = sorted((values["V min"], values["V max"]))
        area_min, area_max = sorted((values["Area min"], values["Area max"]))
        crop_w = max(1, int(values["Crop W"]))
        crop_h = max(1, int(values["Crop H"]))
        crop_offset_x = int(values["Crop X Offset"])
        crop_offset_y = int(values["Crop Y Offset"])
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        return lower, upper, area_min, area_max, crop_w, crop_h, crop_offset_x, crop_offset_y

    def display_options(self) -> dict[str, bool]:
        return {name: variable.get() for name, variable in self.display_variables.items()}

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        self.last_frame_bgr = frame_bgr
        self.refresh_preview()

    def resize_to_label(self, frame_bgr: np.ndarray, label: tk.Label) -> Image.Image:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        label_w = max(1, label.winfo_width())
        label_h = max(1, label.winfo_height())
        scale = min(label_w / image.width, label_h / image.height)
        target_w = max(1, int(round(image.width * scale)))
        target_h = max(1, int(round(image.height * scale)))
        if (target_w, target_h) != image.size:
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return image

    def refresh_preview(self) -> None:
        if self.last_frame_bgr is None:
            return
        image = self.resize_to_label(self.last_frame_bgr, self.preview_label)
        self.photo = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self.photo)

    def show_mask(self, mask: np.ndarray) -> None:
        if mask.ndim == 2:
            self.last_mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            self.last_mask_bgr = mask
        self.refresh_mask_preview()

    def refresh_mask_preview(self) -> None:
        if self.last_mask_bgr is None:
            return
        image = self.resize_to_label(self.last_mask_bgr, self.mask_label)
        self.mask_photo = ImageTk.PhotoImage(image=image)
        self.mask_label.configure(image=self.mask_photo)

    def update(self) -> bool:
        if not self.running:
            return False
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.running = False
        return self.running

    def close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def find_all_rects(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rects: list[tuple[int, int, int, int, int]] = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        rects.append((int(x), int(y), int(w), int(h), int(area)))
    rects.sort(key=lambda item: item[4], reverse=True)
    return rects


def filter_rects_by_area(
    rects: list[tuple[int, int, int, int, int]],
    area_min: int,
    area_max: int,
) -> list[tuple[int, int, int, int, int]]:
    return [rect for rect in rects if area_min <= rect[4] <= area_max]


def draw_mask_debug(
    mask: np.ndarray,
    rects: list[tuple[int, int, int, int, int]],
    area_min: int,
    area_max: int,
) -> np.ndarray:
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for x, y, w, h, area in rects[:80]:
        in_range = area_min <= area <= area_max
        color = (0, 255, 0) if in_range else (0, 180, 255)
        thickness = 2 if in_range else 1
        cv2.rectangle(debug, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            debug,
            str(area),
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return debug


def load_prototypes(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    prototypes = pd.read_csv(path, dtype=str).fillna("")
    if "quality_level" not in prototypes.columns and "rank_level" in prototypes.columns:
        prototypes["quality_level"] = prototypes["rank_level"]
    if "allowed_quality_levels" not in prototypes.columns and "allowed_rank_levels" in prototypes.columns:
        prototypes["allowed_quality_levels"] = prototypes["allowed_rank_levels"]
    for column in ["variant_id", "item_class_id", "item_name", "food_base_name", "quality_level", "weapon_state"]:
        if column not in prototypes.columns:
            prototypes[column] = ""
    required = {"embedding", "item_class_id", "item_name", "quality_level"}
    missing = required - set(prototypes.columns)
    if missing:
        raise ValueError(f"prototypes.csv missing columns: {sorted(missing)}")
    matrix = np.stack([decode_vector(value) for value in prototypes["embedding"]]).astype(np.float32)
    return prototypes, matrix


def classify_crop(
    crop_rgb: np.ndarray,
    cfg: dict,
    session: ort.InferenceSession,
    input_name: str,
    prototypes: pd.DataFrame,
    matrix: np.ndarray,
    top_k: int,
) -> list[tuple[pd.Series, float]]:
    outputs = session.run(None, {input_name: preprocess_rgb(crop_rgb, cfg)})
    embedding = outputs[0][0].astype(np.float32)
    embedding = embedding / max(np.linalg.norm(embedding), 1e-12)
    scores = matrix @ embedding

    top = aggregate_topk_by_class(prototypes, scores, max(2, int(top_k)))
    return [(prototypes.iloc[idx], score) for idx, score in top]


def item_label_lines(
    rows: list[tuple[pd.Series, float]],
) -> list[str]:
    if not rows:
        return ["No match"]
    row, score = rows[0]
    top2_score = rows[1][1] if len(rows) > 1 else score
    gap = score - top2_score
    return [
        f"{row.get('item_name', '')}",
        f"Score: {score:.3f}",
        f"T2 Gap: {gap:.3f}",
    ]


def safe_filename_part(text: object, max_length: int = 80) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", str(text).strip())
    cleaned = cleaned.strip("._")
    return cleaned[:max_length] or "unknown"


def crop_box_from_rect(
    rect: tuple[int, int, int, int, int],
    crop_w: int,
    crop_h: int,
    crop_offset_x: int,
    crop_offset_y: int,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    x, y, w, h, area = rect
    anchor_x = x + w
    anchor_y = y + h
    locate_x1 = anchor_x - crop_w + crop_offset_x
    locate_y1 = anchor_y - crop_h - CARD_BOTTOM_MARGIN + crop_offset_y
    locate_x2 = anchor_x + crop_offset_x
    locate_y2 = anchor_y + crop_offset_y
    crop_x1 = locate_x1
    crop_y1 = locate_y1
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    return locate_x1, locate_y1, locate_x2, locate_y2, crop_x1, crop_y1, crop_x2, crop_y2, area


def print_top5_snapshot(
    frame_rgb: np.ndarray,
    rects: list[tuple[int, int, int, int, int]],
    crop_w: int,
    crop_h: int,
    crop_offset_x: int,
    crop_offset_y: int,
    cfg: dict,
    session: ort.InferenceSession,
    input_name: str,
    prototypes: pd.DataFrame,
    matrix: np.ndarray,
) -> None:
    height, width = frame_rgb.shape[:2]
    print("")
    print(f"===== Top5 snapshot rects={len(rects)} crop={crop_w}x{crop_h} offset=({crop_offset_x},{crop_offset_y}) =====")
    valid_count = 0
    for rect_index, rect in enumerate(rects, start=1):
        x, y, w, h, area = rect
        _lx1, _ly1, _lx2, _ly2, crop_x1, crop_y1, crop_x2, crop_y2, _area = crop_box_from_rect(
            rect,
            crop_w,
            crop_h,
            crop_offset_x,
            crop_offset_y,
        )
        if crop_x1 < 0 or crop_y1 < 0 or crop_x2 > width or crop_y2 > height:
            print(f"Rect{rect_index}: skip out_of_bounds hsv=({x},{y},{w},{h}) area={area}")
            continue
        crop_rgb = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_rgb.shape[:2] != (crop_h, crop_w):
            print(f"Rect{rect_index}: skip invalid_crop_shape hsv=({x},{y},{w},{h}) area={area}")
            continue

        top_rows = classify_crop(
            crop_rgb,
            cfg,
            session,
            input_name,
            prototypes,
            matrix,
            5,
        )
        valid_count += 1
        gap = top_rows[0][1] - top_rows[1][1] if len(top_rows) > 1 else 0.0
        print(f"Rect{rect_index}: hsv=({x},{y},{w},{h}) area={area} crop=({crop_x1},{crop_y1},{crop_w},{crop_h}) gap={gap:.4f}")
        for position, (row, score) in enumerate(top_rows[:5], start=1):
            print(
                f"  Top{position}: score={score:.4f} "
                f"name={row.get('item_name', '')} "
                f"food_base={row.get('food_base_name', '')} "
                f"quality={row.get('quality_level', '')} "
                f"weapon={row.get('weapon_state', '')} "
                f"class={row.get('item_class_id', '')}"
            )
    print(f"===== Top5 snapshot done valid={valid_count}/{len(rects)} =====")


def save_crops_snapshot(
    frame_rgb: np.ndarray,
    rects: list[tuple[int, int, int, int, int]],
    crop_w: int,
    crop_h: int,
    crop_offset_x: int,
    crop_offset_y: int,
    cfg: dict,
    session: ort.InferenceSession,
    input_name: str,
    prototypes: pd.DataFrame,
    matrix: np.ndarray,
    save_dir: Path,
    top_k: int,
) -> None:
    height, width = frame_rgb.shape[:2]
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    valid_count = 0
    print("")
    print(f"===== Save crops rects={len(rects)} dir={save_dir} =====")
    for rect_index, rect in enumerate(rects, start=1):
        x, y, w, h, area = rect
        _lx1, _ly1, _lx2, _ly2, crop_x1, crop_y1, crop_x2, crop_y2, _area = crop_box_from_rect(
            rect,
            crop_w,
            crop_h,
            crop_offset_x,
            crop_offset_y,
        )
        if crop_x1 < 0 or crop_y1 < 0 or crop_x2 > width or crop_y2 > height:
            continue
        crop_rgb = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_rgb.shape[:2] != (crop_h, crop_w):
            continue

        top_rows = classify_crop(
            crop_rgb,
            cfg,
            session,
            input_name,
            prototypes,
            matrix,
            top_k,
        )
        if not top_rows:
            continue
        row, score = top_rows[0]
        top2_score = top_rows[1][1] if len(top_rows) > 1 else score
        gap = score - top2_score
        name = safe_filename_part(row.get("item_name", "unknown"))
        state = safe_filename_part(row.get("weapon_state", "") or row.get("quality_level", "") or "item")
        filename = (
            f"{timestamp}_rect{rect_index:03d}_{name}_{state}_"
            f"score{score:.4f}_gap{gap:.4f}_xy{crop_x1}_{crop_y1}_{crop_w}x{crop_h}.png"
        )
        out_path = save_dir / filename
        Image.fromarray(crop_rgb).save(out_path)
        valid_count += 1
        print(f"saved: {out_path}")
    print(f"===== Save crops done valid={valid_count}/{len(rects)} =====")


def draw_status(frame_bgr: np.ndarray, message: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> np.ndarray:
    return draw_text(frame_bgr, message, (20, 24), font, fill=(255, 80, 80))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ItemDetect 实机窗口识别测试")
    parser.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--model", default=str(ROOT / "outputs" / "item.onnx"))
    parser.add_argument("--prototypes", default=str(ROOT / "outputs" / "prototypes.csv"))
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-dir", default=str(ROOT / "temp" / "live_crops"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    prototypes_path = Path(args.prototypes)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not prototypes_path.exists():
        raise FileNotFoundError(f"prototypes.csv not found: {prototypes_path}")
    save_dir = Path(args.save_dir)

    set_dpi_aware()
    cfg = load_config(args.config)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.provider == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    prototypes, matrix = load_prototypes(prototypes_path)
    font = load_font(18)

    ui = LiveUi()
    target_delay = 1.0 / max(args.fps, 1.0)
    hwnd: int | None = None
    title = ""
    last_search = 0.0
    result_cache: dict[
        tuple[int, int, int, int],
        tuple[float, list[tuple[pd.Series, float]]],
    ] = {}

    print("Press T for Top5 snapshot, S to save current crops, q/Esc to exit.")
    while ui.running:
        frame_start = time.perf_counter()
        now = time.perf_counter()
        if hwnd is None or now - last_search > 2.0 or user32.IsIconic(hwnd):
            hwnd, title = find_target_window()
            last_search = now

        if hwnd is None:
            preview = np.zeros((360, 640, 3), dtype=np.uint8)
            preview = draw_status(preview, "Target window not found. Retrying...", font)
            ui.show_frame(preview)
            ui.show_mask(np.zeros((360, 640), dtype=np.uint8))
        else:
            frame_rgb = capture_client(hwnd)
            if frame_rgb is None:
                preview = np.zeros((360, 640, 3), dtype=np.uint8)
                preview = draw_status(preview, f"Unable to capture window: {title}", font)
                ui.show_frame(preview)
                ui.show_mask(np.zeros((360, 640), dtype=np.uint8))
            else:
                lower, upper, area_min, area_max, crop_w, crop_h, crop_offset_x, crop_offset_y = ui.read_controls()
                hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                all_rects = find_all_rects(mask)
                rects = filter_rects_by_area(all_rects, area_min, area_max)
                ui.show_mask(draw_mask_debug(mask, all_rects, area_min, area_max))
                display_options = ui.display_options()
                preview = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                height, width = frame_rgb.shape[:2]
                if ui.consume_test_request():
                    print_top5_snapshot(
                        frame_rgb,
                        rects,
                        crop_w,
                        crop_h,
                        crop_offset_x,
                        crop_offset_y,
                        cfg,
                        session,
                        input_name,
                        prototypes,
                        matrix,
                    )
                if ui.consume_save_request():
                    save_crops_snapshot(
                        frame_rgb,
                        rects,
                        crop_w,
                        crop_h,
                        crop_offset_x,
                        crop_offset_y,
                        cfg,
                        session,
                        input_name,
                        prototypes,
                        matrix,
                        save_dir,
                        args.top_k,
                    )
                cache_cutoff = now - CACHE_TTL_SECONDS * 3
                result_cache = {
                    key: value for key, value in result_cache.items() if value[0] >= cache_cutoff
                }

                for rect in rects:
                    x, y, w, h, area = rect
                    locate_x1, locate_y1, locate_x2, locate_y2, crop_x1, crop_y1, crop_x2, crop_y2, area = crop_box_from_rect(
                        rect,
                        crop_w,
                        crop_h,
                        crop_offset_x,
                        crop_offset_y,
                    )

                    if display_options["Show HSV rect"]:
                        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 1)
                    if locate_x1 < 0 or locate_y1 < 0 or locate_x2 > width or locate_y2 > height:
                        continue
                    if display_options["Show locate rect"]:
                        cv2.rectangle(preview, (locate_x1, locate_y1), (locate_x2, locate_y2), (255, 160, 0), 1)
                    if crop_x1 < 0 or crop_y1 < 0 or crop_x2 > width or crop_y2 > height:
                        continue

                    crop_rgb = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop_rgb.shape[:2] != (crop_h, crop_w):
                        continue

                    cache_key = (
                        round(crop_x1 / CACHE_GRID_PX),
                        round(crop_y1 / CACHE_GRID_PX),
                        crop_w,
                        crop_h,
                    )
                    cached = result_cache.get(cache_key)
                    if cached is not None and now - cached[0] <= CACHE_TTL_SECONDS:
                        top_rows = cached[1]
                    else:
                        top_rows = classify_crop(
                            crop_rgb,
                            cfg,
                            session,
                            input_name,
                            prototypes,
                            matrix,
                            args.top_k,
                        )
                        result_cache[cache_key] = (now, top_rows)
                    if display_options["Show crop rect"]:
                        cv2.rectangle(preview, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
                    preview = draw_text_lines_top_right(
                        preview,
                        item_label_lines(top_rows),
                        (crop_x2 - 4, crop_y1 + 4),
                        font,
                        fill=(80, 255, 80),
                    )
                    cv2.putText(
                        preview,
                        f"area={area}",
                        (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                status = (
                    f"{title}  all={len(all_rects)}  candidates={len(rects)}  "
                    f"HSV={tuple(int(v) for v in lower)}-{tuple(int(v) for v in upper)}  "
                    f"area={area_min}-{area_max}  crop={crop_w}x{crop_h}  offset={crop_offset_x},{crop_offset_y}"
                )
                preview = draw_text(preview, status, (12, 12), font, fill=(255, 255, 255))
                ui.show_frame(preview)

        elapsed = time.perf_counter() - frame_start
        if not ui.update():
            break
        remaining = target_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    ui.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None
