#!/usr/bin/env python3
"""Make a video of how the foam cells change through a compression series.

For every image in the folder (in file-name order, i.e. by timestamp) it does
the same analysis as foam_anisotropy.py (cells from cell_data.csv, the foam
divided into 8 squares, anisotropy per square) and turns each step into one
video frame:

  left   the foam, cells coloured by their lean from vertical, the grid of
         squares with a direction bar each (longer = more aligned). The view is
         the same for every frame, so you see the foam compress and recover.
  right  the current step's numbers (DA, AR, alignment, direction, cells) and
         graphs of DA and alignment over the whole series with the current
         step marked.

Each step is held for --hold seconds, with a --fade second crossfade to the
next. Writes <out>/anisotropy_video.mp4 and a smaller anisotropy_video.gif
(plays anywhere, e.g. in PowerPoint or a browser).

Example:
  python anisotropy_video.py videocompressed\\images videocompressed\\Segmentation
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import foam_anisotropy as fa  # noqa: E402
from cell_analysis import load_images, pair_files  # noqa: E402

FRAME_W, FRAME_H = 1920, 1080
MAP_W = 1320  # width of the foam panel
MAP_AREA = (MAP_W - 40, FRAME_H - 140)  # (w, h) of the foam picture inside it
BG = (255, 255, 255)
INK = "#222222"
MUTED = "#666666"
DA_COLOUR = "#1c5cab"
ALIGN_COLOUR = "#c4302f"


def step_label(path: Path, index: int) -> str:
    m = re.search(r"step(\d+)", path.stem, re.I)
    return f"Step {m.group(1)}" if m else f"Frame {index + 1}"


def analyse_step(image_path: Path, seg_path: Path, args, csv_rows) -> dict:
    image, seg = load_images(image_path, seg_path)
    row = fa.pick_row(csv_rows, image_path, args.row) if csv_rows is not None else None
    cell_labels, cells, info, footprint = fa.foam_cells(image, seg, args, row)
    if cells.empty:
        raise ValueError("no interior foam cells")
    ys, xs = np.nonzero(footprint)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    rows, cols = args.grid if args.grid else fa.grid_shape(box[2] - box[0], box[3] - box[1], args.squares)
    squares = fa.by_square(cells, footprint, box, rows, cols, args.min_foam)
    return {
        "image": cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "cell_labels": cell_labels, "cells": cells,
        "footprint": footprint, "box": box, "rows": rows, "cols": cols, "squares": squares,
        "whole": fa.summarise(cells), "mm_per_px": info.get("mm_per_px"),
    }


def common_view(steps: list[dict], pad_frac: float = 0.04) -> tuple[int, int, int, int]:
    """One crop for all frames: the union of the foam boxes, padded, at the map panel's aspect ratio."""
    x0 = min(s["box"][0] for s in steps); y0 = min(s["box"][1] for s in steps)
    x1 = max(s["box"][2] for s in steps); y1 = max(s["box"][3] for s in steps)
    h_img, w_img = steps[0]["image"].shape[:2]
    pad = int(pad_frac * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    # widen or heighten to the panel's aspect ratio, centred
    aspect = MAP_AREA[0] / MAP_AREA[1]
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        extra = int(h * aspect) - w
        x0 -= extra // 2; x1 += extra - extra // 2
    else:
        extra = int(w / aspect) - h
        y0 -= extra // 2; y1 += extra - extra // 2
    # shift back inside the image where possible
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    return max(x0, 0), max(y0, 0), min(x1, w_img), min(y1, h_img)


def render_panel(steps: list[dict], labels: list[str], k: int) -> np.ndarray:
    """Right-hand side: current numbers plus DA and alignment over the series."""
    w_px, h_px = FRAME_W - MAP_W, FRAME_H
    dpi = 100
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    s = steps[k]["whole"]

    fig.text(0.06, 0.94, labels[k], fontsize=30, weight="bold", color=INK, va="top")
    fig.text(0.06, 0.875, f"{k + 1} of {len(steps)}   ·   {len(steps[k]['cells'])} cells",
             fontsize=13, color=MUTED, va="top")
    items = [
        ("DA", f"{s['degree_of_anisotropy']:.2f}", "cells together (1 = isotropic)"),
        ("Alignment", f"{100 * s['alignment']:.0f}%", "how parallel the cells are"),
        ("AR", f"{s['aspect_median']:.2f}", "median cell length / width"),
        ("Direction", f"{s['direction_deg']:.0f}°", fa.lean_text(s["direction_deg"])),
    ]
    for i, (name, value, note) in enumerate(items):
        x0, y = (0.06, 0.80) if i < 2 else (0.53, 0.80)
        y -= 0.13 * (i % 2)
        fig.text(x0, y, name, fontsize=13, color=MUTED, va="top")
        fig.text(x0, y - 0.03, value, fontsize=26, weight="bold", color=INK, va="top")
        fig.text(x0, y - 0.085, note, fontsize=10, color=MUTED, va="top")

    x = np.arange(1, len(steps) + 1)
    da = [st["whole"]["degree_of_anisotropy"] for st in steps]
    al = [st["whole"]["alignment"] for st in steps]
    for i, (vals, name, col, top) in enumerate([(da, "DA (degree of anisotropy)", DA_COLOUR, 0.47),
                                                (al, "Alignment", ALIGN_COLOUR, 0.22)]):
        ax = fig.add_axes([0.15, top - 0.17, 0.79, 0.19])
        ax.plot(x, vals, "-", color=col, alpha=0.35, lw=2)
        ax.plot(x[: k + 1], vals[: k + 1], "o-", color=col, lw=2.5, ms=6)
        ax.plot([x[k]], [vals[k]], "o", color=col, ms=14, mec="white", mew=2.5)
        ax.axvline(x[k], color="#999999", lw=1, ls=":")
        ax.set_xlim(0.5, len(steps) + 0.5)
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 0.05)
        ax.set_ylim(lo - 0.15 * span, hi + 0.2 * span)
        ax.set_title(name, fontsize=12, color=INK, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x] if i == 1 else [], fontsize=10)
        ax.tick_params(labelsize=10, colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        if i == 1:
            ax.set_xlabel("step", fontsize=11, color=MUTED)

    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return cv2.resize(panel, (w_px, h_px), interpolation=cv2.INTER_AREA)


def lean_key(width: int) -> np.ndarray:
    """Small horizontal colour key for the lean colours."""
    h = 54
    key = np.full((h, width, 3), 255, np.uint8)
    bar_w = width - 400
    lean = np.linspace(-fa.LEAN_MAX_DEG, fa.LEAN_MAX_DEG, bar_w)  # leaning left .. right
    cols = fa.lean_rgb(90 - lean)  # angle from horizontal = 90 - lean
    key[8:26, 130:130 + bar_w] = cols[None, :, :]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(key, "cell lean:", (10, 24), font, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(key, f"{fa.LEAN_MAX_DEG}+ left", (130, 46), font, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(key, "vertical", (130 + bar_w // 2 - 30, 46), font, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(key, f"{fa.LEAN_MAX_DEG}+ right", (130 + bar_w - 75, 46), font, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(key, "black bar = square's direction", (130 + bar_w + 25, 24), font, 0.5, (90, 90, 90), 1,
                cv2.LINE_AA)
    cv2.putText(key, "(longer = more aligned)", (130 + bar_w + 25, 46), font, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return key


def render_frame(steps, labels, k, view, key) -> np.ndarray:
    st = steps[k]
    fmap = fa.orientation_map(st["image"], st["cell_labels"], st["cells"], st["squares"], st["box"],
                              st["rows"], st["cols"], st["footprint"], crop=view)
    aw, ah = MAP_AREA
    scale = min(aw / fmap.shape[1], ah / fmap.shape[0])  # keep proportions
    fmap = cv2.resize(fmap, (int(fmap.shape[1] * scale), int(fmap.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    frame = np.full((FRAME_H, FRAME_W, 3), 255, np.uint8)
    ox, oy = 20 + (aw - fmap.shape[1]) // 2, 45 + (ah - fmap.shape[0]) // 2
    frame[oy:oy + fmap.shape[0], ox:ox + fmap.shape[1]] = fmap
    frame[FRAME_H - 80:FRAME_H - 80 + key.shape[0], 20:20 + key.shape[1]] = key
    cv2.putText(frame, "Where the cells point", (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (34, 34, 34), 2,
                cv2.LINE_AA)
    frame[:, MAP_W:] = render_panel(steps, labels, k)
    return frame


class Writer:
    """MP4 writer taking RGB frames. Uses H.264 through imageio-ffmpeg when it is
    installed (plays in Windows, PowerPoint, browsers); otherwise OpenCV's
    MPEG-4 (plays in VLC, may not in the Windows player)."""

    def __init__(self, path: Path, fps: float):
        try:
            import imageio_ffmpeg
            self._gen = imageio_ffmpeg.write_frames(str(path), (FRAME_W, FRAME_H), fps=fps, codec="libx264",
                                                    pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
            self._gen.send(None)
            self._cv = None
            self.codec = "H.264"
        except ImportError:
            self._gen = None
            self._cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_W, FRAME_H))
            if not self._cv.isOpened():
                raise SystemExit("Could not open a video writer for MP4")
            self.codec = "MPEG-4 (install imageio-ffmpeg for H.264)"

    def write(self, rgb: np.ndarray) -> None:
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(rgb))
        else:
            self._cv.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
        else:
            self._cv.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path, help="folder of images")
    ap.add_argument("segmentation", type=Path, help="folder of segmentations")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: next to the images)")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds each step is shown (default 1.5)")
    ap.add_argument("--fade", type=float, default=0.5, help="crossfade between steps, seconds (default 0.5)")
    ap.add_argument("--fps", type=float, default=25)
    ap.add_argument("--no-gif", action="store_true", help="only write the MP4")
    ap.add_argument("--squares", type=int, default=fa.DEFAULT_SQUARES)
    ap.add_argument("--grid", type=int, nargs=2, metavar=("ROWS", "COLS"))
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--row", default=None)
    ap.add_argument("--min-mm", type=float, default=None)
    ap.add_argument("--max-mm", type=float, default=None)
    ap.add_argument("--hue", type=int, nargs=2, default=fa.DEFAULT_HUE_RANGE, metavar=("MIN", "MAX"))
    ap.add_argument("--min-saturation", type=int, default=fa.DEFAULT_MIN_SATURATION)
    ap.add_argument("--no-colour-check", action="store_true")
    ap.add_argument("--min-foam", type=float, default=0.25)
    args = ap.parse_args()

    pairs, unmatched = pair_files(args.images, args.segmentation)
    for img in unmatched:
        print(f"no segmentation for {img.name} (skipped)")
    # file-name order, with numbers compared as numbers (step2 before step10)
    pairs.sort(key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p[0].stem)])
    if not pairs:
        raise SystemExit("no image/segmentation pairs found")

    csv_rows = None
    if not args.no_csv:
        csv_path = args.csv or fa.find_csv(args.images)
        if csv_path:
            print(f"reading {csv_path}")
            csv_rows = fa.read_csv_rows(csv_path)
        else:
            print("no cell_data.csv found: picking cells by colour")

    steps, labels = [], []
    for i, (img, seg) in enumerate(pairs):
        try:
            st = analyse_step(img, seg, args, csv_rows)
        except Exception as exc:
            print(f"[{i + 1}/{len(pairs)}] {img.name}: FAILED ({exc}), left out of the video")
            continue
        labels.append(step_label(img, i))
        steps.append(st)
        w = st["whole"]
        print(f"[{i + 1}/{len(pairs)}] {img.name}: DA {w['degree_of_anisotropy']:.2f}  "
              f"align {w['alignment']:.2f}  AR {w['aspect_median']:.2f}")
    if not steps:
        raise SystemExit("nothing to put in the video")

    out = args.out or args.images.parent
    out.mkdir(parents=True, exist_ok=True)
    view = common_view(steps)
    key = lean_key(MAP_W - 40)
    print("rendering frames...")
    frames = [render_frame(steps, labels, k, view, key) for k in range(len(steps))]

    mp4 = out / "anisotropy_video.mp4"
    writer = Writer(mp4, args.fps)
    hold_n, fade_n = max(1, round(args.hold * args.fps)), round(args.fade * args.fps)
    for k, frame in enumerate(frames):
        for _ in range(hold_n):
            writer.write(frame)
        if k + 1 < len(frames):
            for j in range(1, fade_n + 1):
                t = j / (fade_n + 1)
                writer.write(cv2.addWeighted(frame, 1 - t, frames[k + 1], t, 0))
    for _ in range(hold_n):  # linger on the last step
        writer.write(frames[-1])
    writer.close()
    print(f"wrote {mp4} ({writer.codec})")

    if not args.no_gif:
        gif = out / "anisotropy_video.gif"
        small = [Image.fromarray(cv2.resize(f, (FRAME_W // 2, FRAME_H // 2), interpolation=cv2.INTER_AREA))
                 for f in frames]
        durations = [int(args.hold * 1000)] * (len(small) - 1) + [int(args.hold * 2000)]
        small[0].save(gif, save_all=True, append_images=small[1:], duration=durations, loop=0, optimize=True)
        print(f"wrote {gif}")


if __name__ == "__main__":
    main()
