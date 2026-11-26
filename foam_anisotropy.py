#!/usr/bin/env python3
"""Foam cell anisotropy in a grid of squares.

Takes a micrograph and its cell segmentation (dark wall lines on white), finds
the foam cells (same as cell_analysis.py: regions that are blue in the
micrograph; cells cut by the foam's edge are left out), divides the foam into
a grid of squares (8 by default, laid out to be as square as possible) and
reports the anisotropy of the cells in each square and in the whole foam.

The anisotropy measures are the same as pinhole_overlay.py on the kingpin
branch (used there for the insulation foams):

  Shape        each cell gets an ellipse fitted to its edge (cv2.fitEllipse):
               length, width, aspect ratio AR = length / width (1 = round),
               angle = long axis from horizontal, counter-clockwise on screen,
               0-180 deg (90 = vertical)
  alignment    0-1: how parallel the cells are (0 = random directions,
               1 = all parallel); length of the mean doubled-angle vector,
               weighted by area x (1 - width/length) so big elongated cells
               count most and near-round ones (direction is noise) least
  direction    that weighted mean direction, deg
  DA           degree of anisotropy: length / width of the combined ellipse
               of all the cells' ellipses (each weighted by its area);
               1 = isotropic

Output (default: anisotropy_results/ next to the images): one picture per
image, <name>_anisotropy.png, with the cells coloured by lean from vertical, a
direction bar per square (length = alignment) and a table of the squares laid
out like the grid. Pass --tables to also write a CSV per image and an
anisotropy.xlsx with all images.

Examples:
  python foam_anisotropy.py foam.png foam_seg.png
  python foam_anisotropy.py videocompressed\\images videocompressed\\Segmentation
  python foam_anisotropy.py videocompressed\\images videocompressed\\Segmentation --grid 2 4
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

from cell_analysis import (  # noqa: E402
    DEFAULT_HUE_RANGE,
    DEFAULT_MIN_AREA_PX,
    DEFAULT_MIN_SATURATION,
    DEFAULT_WALL_THRESHOLD,
    classify_foam_regions,
    label_regions,
    load_images,
    neighbour_info,
    pair_files,
)

# Same look and thresholds as pinhole_overlay.py (kingpin).
ROUND_ASPECT_MAX = 1.2
LEAN_MAX_DEG = 30
LEAN_COLORS = ["#1c5cab", "#6da7ec", "#a9a7a0", "#ec8a89", "#c4302f"]
DIRECTION_BAR_COLOR = (30, 30, 30)
MAP_GRID_COLOR = (255, 255, 255)
INK = "#222222"
MUTED_INK = "#666666"
DEFAULT_SQUARES = 8


# ---------------------------------------------------------------- cells
def fit_edge_ellipse(mask: np.ndarray) -> tuple[float, ...] | None:
    """Ellipse fitted to the edge of mask: (cx, cy, length, width, angle deg), as in pinhole_overlay.py."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    edge = np.vstack([c[:, 0] for c in contours]).astype(np.float32)
    if len(edge) < 5:
        return None
    (cx, cy), (axis_1, axis_2), rotation = cv2.fitEllipse(edge)
    long_axis_rotation = rotation if axis_1 >= axis_2 else rotation + 90
    return cx, cy, max(axis_1, axis_2), min(axis_1, axis_2), (-long_axis_rotation) % 180


def foam_cells(image: np.ndarray, seg: np.ndarray, args) -> tuple[np.ndarray, pd.DataFrame]:
    """Interior foam cells with their edge-fitted ellipses. Returns (cell label image, table)."""
    labels = label_regions(seg, DEFAULT_WALL_THRESHOLD)
    is_foam = classify_foam_regions(labels, image, tuple(args.hue), args.min_saturation,
                                    DEFAULT_MIN_AREA_PX, None)
    if not is_foam.any():
        raise ValueError("no foam cells found (check --hue)")
    _, is_edge = neighbour_info(labels, is_foam)
    keep = is_foam & ~is_edge
    cell_labels = np.where(keep, np.arange(keep.size), 0)[labels]

    rows = []
    for lab, sl in enumerate(ndimage.find_objects(cell_labels), start=1):
        if sl is None:
            continue
        mask = np.pad(cell_labels[sl] == lab, 1)
        fit = fit_edge_ellipse(mask)
        if fit is None:
            continue
        cx, cy, length, width, angle = fit
        rows.append({
            "cell_id": lab,
            "x": sl[1].start - 1 + cx, "y": sl[0].start - 1 + cy,
            "length_px": length, "width_px": width,
            "aspect_ratio": length / max(width, 1e-9),
            "angle_deg": angle,
            "area_px": int(mask.sum()),
        })
    return cell_labels, pd.DataFrame(rows)


# ---------------------------------------------------------------- anisotropy
def anisotropy_summary(major, minor, angle, area) -> dict:
    """Aspect ratio stats, alignment (0-1), mean direction, DA. Identical to pinhole_overlay.py."""
    aspect = major / np.maximum(minor, 1e-9)
    weight = area * (1 - minor / np.maximum(major, 1e-9))
    mean_vector = np.sum(weight * np.exp(2j * np.radians(angle))) / max(weight.sum(), 1e-9)

    theta = np.radians(angle)
    var_major, var_minor = (major / 4) ** 2, (minor / 4) ** 2
    cxx = np.sum(area * (var_major * np.cos(theta) ** 2 + var_minor * np.sin(theta) ** 2))
    cyy = np.sum(area * (var_major * np.sin(theta) ** 2 + var_minor * np.cos(theta) ** 2))
    cxy = np.sum(area * (var_major - var_minor) * np.cos(theta) * np.sin(theta))
    eig = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
    return {
        "aspect_mean": float(aspect.mean()),
        "aspect_median": float(np.median(aspect)),
        "aspect_p90": float(np.percentile(aspect, 90)),
        "round_count": int((aspect < ROUND_ASPECT_MAX).sum()),
        "alignment": float(abs(mean_vector)),
        "direction_deg": float(np.degrees(np.angle(mean_vector)) / 2 % 180),
        "degree_of_anisotropy": float(np.sqrt(eig[1] / max(eig[0], 1e-9))),
    }


def summarise(cells: pd.DataFrame) -> dict | None:
    if cells.empty:
        return None
    return anisotropy_summary(cells["length_px"].to_numpy(), cells["width_px"].to_numpy(),
                              cells["angle_deg"].to_numpy(), cells["area_px"].to_numpy())


def grid_shape(w: float, h: float, n: int) -> tuple[int, int]:
    """(rows, cols) with rows * cols == n whose squares are closest to square."""
    options = [(r, n // r) for r in range(1, n + 1) if n % r == 0]
    return min(options, key=lambda rc: abs(math.log((w / rc[1]) / (h / rc[0]))))


def by_square(cells: pd.DataFrame, box: tuple[int, int, int, int], rows: int, cols: int) -> list[dict]:
    x0, y0, x1, y1 = box
    r = np.clip(((cells["y"] - y0) / ((y1 - y0) / rows)).astype(int), 0, rows - 1)
    c = np.clip(((cells["x"] - x0) / ((x1 - x0) / cols)).astype(int), 0, cols - 1)
    squares = []
    for i in range(rows):
        for j in range(cols):
            inside = cells[(r == i) & (c == j)]
            squares.append({"grid_row": i, "grid_col": j, "count": len(inside), "anisotropy": summarise(inside)})
    return squares


# ---------------------------------------------------------------- drawing
def lean_deg(angle):
    """Lean from vertical, deg: > 0 = the top of the cell leans right, < 0 = left."""
    return 90.0 - np.asarray(angle)


def lean_text(angle: float) -> str:
    lean = float(lean_deg(angle))
    return "vertical" if abs(lean) < 2 else f"{abs(lean):.0f}° {'right' if lean > 0 else 'left'} of vertical"


def lean_colormap():
    return LinearSegmentedColormap.from_list("lean", LEAN_COLORS), Normalize(-LEAN_MAX_DEG, LEAN_MAX_DEG)


def lean_rgb(angle) -> np.ndarray:
    cmap, norm = lean_colormap()
    lean = np.clip(np.nan_to_num(lean_deg(angle)), -LEAN_MAX_DEG, LEAN_MAX_DEG)
    return (cmap(norm(lean))[..., :3] * 255).astype(np.uint8)


def orientation_map(image_rgb, cell_labels, cells, squares, box, rows, cols) -> np.ndarray:
    """Cells coloured by lean, grid lines, and a direction bar per square (length = alignment)."""
    out = (image_rgb * 0.45).astype(np.uint8)
    colours = np.zeros((cell_labels.max() + 1, 3), np.uint8)
    has = np.zeros(cell_labels.max() + 1, bool)
    colours[cells["cell_id"]] = lean_rgb(cells["angle_deg"].to_numpy())
    has[cells["cell_id"]] = True
    m = has[cell_labels]
    out[m] = colours[cell_labels][m]

    x0, y0, x1, y1 = box
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
    scale = max(out.shape[:2]) / 1500
    lt = max(1, int(round(2 * scale)))
    for i in range(rows + 1):
        y = int(round(y0 + i * ch))
        cv2.line(out, (x0, y), (x1, y), MAP_GRID_COLOR, lt, cv2.LINE_AA)
    for j in range(cols + 1):
        x = int(round(x0 + j * cw))
        cv2.line(out, (x, y0), (x, y1), MAP_GRID_COLOR, lt, cv2.LINE_AA)

    bt = max(2, int(round(5 * scale)))
    for sq in squares:
        a = sq["anisotropy"]
        if a is None:
            continue
        cx, cy = x0 + (sq["grid_col"] + 0.5) * cw, y0 + (sq["grid_row"] + 0.5) * ch
        half = 0.4 * min(cw, ch) * a["alignment"]
        t = math.radians(a["direction_deg"])
        dx, dy = half * math.cos(t), -half * math.sin(t)
        p0, p1 = (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy))
        cv2.line(out, p0, p1, (255, 255, 255), bt * 3, cv2.LINE_AA)
        cv2.line(out, p0, p1, DIRECTION_BAR_COLOR, bt, cv2.LINE_AA)
        label = f"{sq['grid_row'] * cols + sq['grid_col'] + 1}"
        org = (int(x0 + sq["grid_col"] * cw + 10 * scale), int(y0 + sq["grid_row"] * ch + 30 * scale))
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (0, 0, 0), lt * 4, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (255, 255, 255), lt, cv2.LINE_AA)

    pad = int(30 * scale)
    return out[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]


def save_plot(map_rgb, squares, whole, rows, cols, title, dst: Path) -> None:
    """Orientation map beside a plain-language summary and a per-square table laid out like the grid."""
    cmap, norm = lean_colormap()
    fig = plt.figure(figsize=(18, 9.6), dpi=130)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.05)
    right = outer[1].subgridspec(3, 1, height_ratios=[0.3, 0.62, 0.08], hspace=0.16)

    ax_map = fig.add_subplot(outer[0])
    ax_map.imshow(map_rgb)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_title("Where the cells point", fontsize=13, color=INK, loc="left")

    ax_text = fig.add_subplot(right[0])
    ax_text.set_axis_off()
    a = whole
    lines = [
        ("Shape", f"Cells are typically {a['aspect_median']:.2f}x longer than wide (median aspect ratio, AR)."),
        ("Direction", f"On average they point {a['direction_deg']:.0f}° -- {lean_text(a['direction_deg'])}."),
        ("Alignment", f"{100 * a['alignment']:.0f}% aligned (0% = random directions, 100% = all parallel)."),
        ("Cell space", f"As a whole {a['degree_of_anisotropy']:.2f}x longer along that direction than across it "
                       f"(DA; 1 = isotropic)."),
    ]
    ax_text.text(0, 1.0, "Whole foam", fontsize=13, color=INK, weight="bold", va="top", transform=ax_text.transAxes)
    for k, (lab, txt) in enumerate(lines):
        y = 0.78 - 0.2 * k
        ax_text.text(0, y, lab, fontsize=10.5, color=MUTED_INK, va="top", transform=ax_text.transAxes)
        ax_text.text(0.17, y, txt, fontsize=10.5, color=INK, va="top", transform=ax_text.transAxes, wrap=True)

    ax_tab = fig.add_subplot(right[1])
    ax_tab.set_xlim(0, cols)
    ax_tab.set_ylim(rows, 0)
    ax_tab.set_aspect("equal")
    ax_tab.set_axis_off()
    ax_tab.set_title("Each square (same layout as the map)", fontsize=12, color=INK, loc="left")
    big = 13 if cols <= 4 else 10
    small = big * 0.72
    for sq in squares:
        r, c = sq["grid_row"], sq["grid_col"]
        ax_tab.add_patch(Rectangle((c + 0.03, r + 0.03), 0.94, 0.94, facecolor="#fcfcfb", edgecolor="#cccccc"))
        ax_tab.text(c + 0.9, r + 0.12, f"#{r * cols + c + 1}", fontsize=small, color=MUTED_INK, ha="right", va="center")
        s = sq["anisotropy"]
        if s is None:
            ax_tab.text(c + 0.5, r + 0.5, "no cells", ha="center", va="center", fontsize=small, color=MUTED_INK)
            continue
        d = s["direction_deg"]
        t = math.radians(d)
        half = 0.13 * s["alignment"]
        bx, by = c + 0.2, r + 0.3
        ax_tab.plot([bx - half * math.cos(t), bx + half * math.cos(t)],
                    [by + half * math.sin(t), by - half * math.sin(t)],
                    color=cmap(norm(np.clip(lean_deg(d), -LEAN_MAX_DEG, LEAN_MAX_DEG))), lw=4,
                    solid_capstyle="round")
        ax_tab.text(c + 0.36, r + 0.3, f"{d:.0f}°", fontsize=big, color=INK, weight="bold", va="center")
        ax_tab.text(c + 0.1, r + 0.52, lean_text(d).replace(" of vertical", ""), fontsize=small, color=MUTED_INK,
                    va="center")
        ax_tab.text(c + 0.1, r + 0.68, f"align {s['alignment']:.2f}   n = {sq['count']}", fontsize=small,
                    color=INK, va="center")
        ax_tab.text(c + 0.1, r + 0.84, f"AR {s['aspect_median']:.2f}   DA {s['degree_of_anisotropy']:.2f}",
                    fontsize=small, color=INK, va="center", weight="bold")

    ax_key = fig.add_subplot(right[2])
    key = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_key, orientation="horizontal",
                       ticks=[-LEAN_MAX_DEG, -LEAN_MAX_DEG / 2, 0, LEAN_MAX_DEG / 2, LEAN_MAX_DEG])
    key.ax.set_xticklabels([f"{LEAN_MAX_DEG}°+ left", f"{LEAN_MAX_DEG // 2}° left", "vertical",
                            f"{LEAN_MAX_DEG // 2}° right", f"{LEAN_MAX_DEG}°+ right"], fontsize=9, color=INK)
    key.set_label("Cell colour = lean from vertical.  Black bar on the map = a square's average direction "
                  "(longer = more aligned).", fontsize=9, color=MUTED_INK)
    key.outline.set_edgecolor("#cccccc")

    fig.suptitle(title, fontsize=10, color=MUTED_INK, x=0.01, ha="left")
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- run
def table_row(name, sq_id, row, col, count, s) -> dict:
    out = {"square": sq_id, "row": row, "col": col, "n_cells": count}
    if s:
        out.update(AR_median=s["aspect_median"], AR_mean=s["aspect_mean"], AR_p90=s["aspect_p90"],
                   alignment=s["alignment"], direction_deg=s["direction_deg"],
                   lean_from_vertical_deg=float(lean_deg(s["direction_deg"])),
                   DA=s["degree_of_anisotropy"], near_round_cells=s["round_count"])
    return out


def analyse(image_path: Path, seg_path: Path, out: Path, args) -> pd.DataFrame:
    image, seg = load_images(image_path, seg_path)
    cell_labels, cells = foam_cells(image, seg, args)
    if cells.empty:
        raise ValueError("no interior foam cells")

    ys, xs = np.nonzero(cell_labels)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    rows, cols = args.grid if args.grid else grid_shape(box[2] - box[0], box[3] - box[1], args.squares)
    squares = by_square(cells, box, rows, cols)
    whole = summarise(cells)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fmap = orientation_map(rgb, cell_labels, cells, squares, box, rows, cols)
    save_plot(fmap, squares, whole, rows, cols, f"{image_path.name}  ({len(cells)} cells, {rows} x {cols} squares)",
              out / f"{image_path.stem}_anisotropy.png")

    table = [table_row(image_path.name, sq["grid_row"] * cols + sq["grid_col"] + 1, sq["grid_row"] + 1,
                       sq["grid_col"] + 1, sq["count"], sq["anisotropy"]) for sq in squares]
    table.append(table_row(image_path.name, "whole foam", "", "", len(cells), whole))
    df = pd.DataFrame(table)
    if args.tables:
        df.round(3).to_csv(out / f"{image_path.stem}_anisotropy.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path, help="image file, or folder of images")
    ap.add_argument("segmentation", type=Path, help="segmentation file, or folder of segmentations")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output folder")
    ap.add_argument("--squares", type=int, default=DEFAULT_SQUARES, help="number of squares (default 8)")
    ap.add_argument("--grid", type=int, nargs=2, metavar=("ROWS", "COLS"),
                    help="explicit grid instead of --squares, e.g. --grid 2 4")
    ap.add_argument("--hue", type=int, nargs=2, default=DEFAULT_HUE_RANGE, metavar=("MIN", "MAX"),
                    help="OpenCV hue range (0-179) of the foam colour; default blue")
    ap.add_argument("--min-saturation", type=int, default=DEFAULT_MIN_SATURATION)
    ap.add_argument("--tables", action="store_true",
                    help="also write a CSV per image and anisotropy.xlsx (default: pictures only)")
    args = ap.parse_args()

    if args.image.is_dir():
        if not args.segmentation.is_dir():
            raise SystemExit("When IMAGE is a folder, SEGMENTATION must be a folder too.")
        pairs, unmatched = pair_files(args.image, args.segmentation)
        for img in unmatched:
            print(f"no segmentation for {img.name} (skipped)")
    else:
        pairs = [(args.image, args.segmentation)]
    out = args.out or args.image.parent / "anisotropy_results"
    out.mkdir(parents=True, exist_ok=True)

    tables = []
    for i, (img, seg) in enumerate(pairs, 1):
        try:
            t = analyse(img, seg, out, args)
        except Exception as exc:
            print(f"[{i}/{len(pairs)}] {img.name}: FAILED ({exc})")
            continue
        w = t.iloc[-1]
        sq = t.iloc[:-1]
        print(f"[{i}/{len(pairs)}] {img.name}: DA {w.DA:.2f}  AR {w.AR_median:.2f}  align {w.alignment:.2f}  "
              f"dir {w.direction_deg:.0f} deg   (squares: DA {sq.DA.min():.2f}-{sq.DA.max():.2f})")
        t.insert(0, "image", img.name)
        tables.append(t)

    if tables and args.tables:
        pd.concat(tables, ignore_index=True).round(3).to_excel(out / "anisotropy.xlsx", index=False)
    print(f"\nPictures in {out}")


if __name__ == "__main__":
    main()
