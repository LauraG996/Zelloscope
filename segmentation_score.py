#!/usr/bin/env python3
"""Score Zelloscope cell segmentations of a foam image, to find the best settings.

Give it the original image and one or more segmentations of it (e.g. made with
different Zelloscope presets or settings). Each segmentation is judged against
the image itself, inside the foam only (the jig/holder/background is found
and left out automatically, or set it with --roi):

  on-wall lines     share of the segmentation's line pixels lying on a bright
                    cell wall in the image (within 2 px)
  boundaries        every boundary between two neighbouring cells is scored by
                    how bright the image is along it, 0 = as dark as inside a
                    cell, 1 = as bright as a wall:
                      real wall   > 0.5
                      weak        0.2 - 0.5
                      false split < 0.2   (the line cuts through a cell)
  merged            cells > 2.5x the median area (usually two or more cells
                    joined because a wall was missed)
  fragments         cells < 0.25x the median area (bits of a cell cut off)
  straight runs     perfectly horizontal/vertical line pieces >= 15 px per 100
                    cells; real walls are rarely ruler-straight, so many of
                    these mean boundaries are drawn geometrically between seeds
                    instead of following the wall
  neighbours        mean neighbours per cell (a 2D foam averages about 6)

  error rate = false splits % + merged % + fragments %   (lower is better)

The segmentations are ranked by error rate. Output (default: next to the image):
  segmentation_scores.csv      one row per segmentation
  <segmentation>_score.png     where the errors are: the foam with every
                               boundary coloured (green = real wall, orange =
                               weak, red = false split), merged cells tinted
                               magenta, plus close-ups

Examples:
  python segmentation_score.py segmentation_test\\image.png segmentation_test\\image_segmentation.png
  python segmentation_score.py segmentation_test\\image.png segmentation_test\\settings_tests\\
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import label
from skimage.segmentation import expand_labels

from cell_analysis import _read_on_white

WALL_THRESHOLD = 230          # segmentation pixels darker than this are lines
MIN_CELL_PX = 30
MERGED_FACTOR, FRAGMENT_FACTOR = 2.5, 0.25
GOOD, WEAK = 0.5, 0.2
STRAIGHT_RUN_PX = 15
COLOURS = {"wall": (60, 200, 80), "weak": (255, 170, 0), "split": (235, 40, 40), "merged": (230, 60, 230)}


# ---------------------------------------------------------------- inputs
def zelloscope_metadata(path: Path) -> dict:
    """The ZelloscopeData JSON Zelloscope writes into its PNGs ({} if none)."""
    try:
        data = path.read_bytes()
    except OSError:
        return {}
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return {}
    i = 8
    while i + 8 <= len(data):
        n = struct.unpack(">I", data[i:i + 4])[0]
        kind, body = data[i + 4:i + 8], data[i + 8:i + 8 + n]
        if kind in (b"iTXt", b"tEXt") and body.startswith(b"ZelloscopeData\0"):
            text = body.split(b"\0")[-1]
            for decode in (lambda t: json.loads(base64.b64decode(t)), lambda t: json.loads(t)):
                try:
                    return decode(text)
                except Exception:
                    pass
        if kind == b"IEND":
            break
        i += 12 + n
    return {}


def read_gray(path: Path) -> np.ndarray:
    img = _read_on_white(path)
    if img is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def cell_labels(seg: np.ndarray) -> np.ndarray:
    if np.median(seg) < 128:  # light lines on dark
        seg = 255 - seg
    return label(seg >= WALL_THRESHOLD, connectivity=1)


# ---------------------------------------------------------------- foam region
def foam_region(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Where the foam is: many normal-sized cells close together AND strong image texture.

    The jig and background are either not segmented into small cells, or have
    much smoother texture. The largest such piece is kept, holes filled, and
    a band along its edge dropped (edge cells are cut off)."""
    areas = np.bincount(labels.ravel())
    ok = areas >= MIN_CELL_PX
    ok[0] = False
    med = np.median(areas[ok])
    normal = ok & (areas <= 4 * med)
    win = int(max(51, 5 * np.sqrt(med))) | 1
    density = cv2.blur(normal[labels].astype(np.float32), (win, win))
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (win, win))
    std = np.sqrt(np.maximum(cv2.blur(g * g, (win, win)) - mean ** 2, 0))
    dense = density > 0.75
    if not dense.any():
        raise SystemExit("could not find the foam automatically; give it with --roi X0 Y0 X1 Y1")
    # the foam dominates the densely segmented area; the jig there has much smoother texture
    cand = (dense & (std > 0.7 * np.median(std[dense]))).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win)))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n <= 1:
        raise SystemExit("could not find the foam automatically; give it with --roi X0 Y0 X1 Y1")
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = ndimage.binary_fill_holes(comp == main).astype(np.uint8)
    margin = int(2 * np.sqrt(med))
    return cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)).astype(bool)


# ---------------------------------------------------------------- scoring
def score(gray: np.ndarray, labels: np.ndarray, region: np.ndarray, p2mm: float | None) -> tuple[dict, dict]:
    g = gray.astype(np.float32)
    wall = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 25)
    lines = labels == 0

    areas = np.bincount(labels.ravel())
    in_region = np.bincount(labels[region], minlength=areas.size)
    cells = np.flatnonzero((areas >= MIN_CELL_PX) & (in_region > 0.5 * areas))
    cells = cells[cells > 0]
    if len(cells) < 20:
        raise ValueError("fewer than 20 cells in the foam region")
    is_cell = np.zeros(areas.size, bool)
    is_cell[cells] = True
    med = float(np.median(areas[cells]))
    interior = is_cell[labels]

    wall_px = wall > np.percentile(wall[interior], 80)
    near_wall = cv2.dilate(wall_px.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    line_in_foam = lines & region
    on_wall = float(near_wall[line_in_foam].mean())

    # boundaries between neighbouring cells
    grown = expand_labels(labels, 3)
    keys, vals, pos = [], [], []
    nmax = labels.max() + 1
    for (a, b, w, sl) in ((grown[:, :-1], grown[:, 1:], wall[:, :-1], (slice(None), slice(0, -1))),
                          (grown[:-1, :], grown[1:, :], wall[:-1, :], (slice(0, -1), slice(None)))):
        m = (a != b) & is_cell[a] & is_cell[b]
        lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
        keys.append(lo.astype(np.int64) * nmax + hi)
        vals.append(w[m])
        yy, xx = np.nonzero(m)
        pos.append(np.c_[yy, xx])
    keys, vals, pos = np.concatenate(keys), np.concatenate(vals), np.concatenate(pos)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    means = np.bincount(inv, weights=vals) / counts
    inside_level, wall_level = np.median(wall[interior]), np.median(wall[wall_px & region])
    bscore = (means - inside_level) / max(wall_level - inside_level, 1e-6)
    keep = counts >= 6
    b = bscore[keep]
    n_neigh = np.bincount(np.concatenate([uniq[keep] // nmax, uniq[keep] % nmax]), minlength=nmax)[cells]

    merged = cells[areas[cells] > MERGED_FACTOR * med]
    fragments = cells[areas[cells] < FRAGMENT_FACTOR * med]

    def straight_runs(mask):
        d = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
        starts, ends = np.nonzero(d == 1), np.nonzero(d == -1)
        return int(((ends[1] - starts[1]) >= STRAIGHT_RUN_PX).sum())
    straight = straight_runs(line_in_foam) + straight_runs(line_in_foam.T)

    diam = 2 * np.sqrt(areas[cells] / np.pi)
    res = {
        "cells": len(cells),
        "median_diameter_px": float(np.median(diam)),
        "median_diameter_mm": float(np.median(diam) * p2mm) if p2mm else None,
        "lines_on_walls_pct": 100 * on_wall,
        "boundaries": int(len(b)),
        "real_wall_pct": 100 * float((b > GOOD).mean()),
        "weak_pct": 100 * float(((b > WEAK) & (b <= GOOD)).mean()),
        "false_split_pct": 100 * float((b <= WEAK).mean()),
        "merged_pct": 100 * len(merged) / len(cells),
        "fragments_pct": 100 * len(fragments) / len(cells),
        "straight_runs_per_100_cells": 100 * straight / len(cells),
        "mean_neighbours": float(n_neigh.mean()),
    }
    res["error_rate_pct"] = res["false_split_pct"] + res["merged_pct"] + res["fragments_pct"]
    # per-pixel class for drawing
    cls = np.digitize(bscore, [WEAK, GOOD])  # 0 split, 1 weak, 2 wall
    draw = {"pos": pos[keep[inv]], "cls": cls[inv][keep[inv]], "merged": merged, "cells": cells}
    return res, draw


# ---------------------------------------------------------------- picture
def picture(image_rgb, labels, region, draw, res, name, dst: Path) -> None:
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    over = (image_rgb * 0.55).astype(np.uint8)
    merged_mask = np.isin(labels, draw["merged"])
    over[merged_mask] = (over[merged_mask] * 0.4 + np.array(COLOURS["merged"]) * 0.6).astype(np.uint8)
    cols = np.array([COLOURS["split"], COLOURS["weak"], COLOURS["wall"]], np.uint8)
    p = draw["pos"]
    over[p[:, 0], p[:, 1]] = cols[draw["cls"]]
    edge = region ^ cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    over[edge] = 255

    # whole foam (thicker lines so they show when shrunk) + two close-ups
    thick = over.copy()
    for c in (0, 2, 1):  # draw splits last so they stay visible
        m = np.zeros(labels.shape, np.uint8)
        sel = draw["cls"] == c
        m[p[sel, 0], p[sel, 1]] = 1
        m = cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)
        thick[m] = cols[c]
    full = thick[y0:y1, x0:x1]
    H = 900
    full = cv2.resize(full, (int(full.shape[1] * H / full.shape[0]), H), interpolation=cv2.INTER_AREA)

    size = int(min(500, (x1 - x0) // 3, (y1 - y0) // 3))
    crops = []
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    # second close-up where false splits are densest
    splits = np.zeros(labels.shape, np.float32)
    sel = draw["cls"] == 0
    splits[p[sel, 0], p[sel, 1]] = 1
    dens = cv2.blur(splits, (size, size)) * region
    sy, sx = np.unravel_index(np.argmax(dens), dens.shape)
    for (yy, xx, title) in [(cy, cx, "centre"), (sy, sx, "most false splits")]:
        yy = int(np.clip(yy, y0 + size // 2, y1 - size // 2)); xx = int(np.clip(xx, x0 + size // 2, x1 - size // 2))
        c = over[yy - size // 2:yy + size // 2, xx - size // 2:xx + size // 2]
        c = cv2.resize(c, (H // 2 - 10, H // 2 - 10), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(c, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        crops.append(c)
    right = np.vstack([crops[0], np.full((20, crops[0].shape[1], 3), 255, np.uint8), crops[1]])
    right = cv2.copyMakeBorder(right, 0, H - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    body = np.hstack([full, np.full((H, 20, 3), 255, np.uint8), right])

    head = np.full((150, body.shape[1], 3), 255, np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(head, name, (10, 35), f, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    txt1 = (f"error rate {res['error_rate_pct']:.1f}%  =  false splits {res['false_split_pct']:.1f}%  + merged "
            f"{res['merged_pct']:.1f}%  + fragments {res['fragments_pct']:.1f}%")
    txt2 = (f"{res['cells']} cells, median {res['median_diameter_px']:.0f} px   lines on walls "
            f"{res['lines_on_walls_pct']:.0f}%   boundaries on real walls {res['real_wall_pct']:.0f}%   "
            f"straight runs {res['straight_runs_per_100_cells']:.0f}/100 cells   neighbours {res['mean_neighbours']:.2f}")
    cv2.putText(head, txt1, (10, 75), f, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(head, txt2, (10, 105), f, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
    x = 10
    for lab, key in [("real wall", "wall"), ("weak", "weak"), ("false split", "split"), ("merged cell", "merged")]:
        cv2.rectangle(head, (x, 122), (x + 22, 140), COLOURS[key], -1)
        cv2.putText(head, lab, (x + 30, 138), f, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
        x += 170
    out = np.vstack([head, body])
    cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))[1].tofile(str(dst))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path, help="the original image")
    ap.add_argument("segmentations", type=Path, nargs="+",
                    help="segmentation image(s) of that image, or a folder of them")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: next to the image)")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="score only inside this rectangle instead of finding the foam automatically")
    args = ap.parse_args()

    segs = []
    for s in args.segmentations:
        if s.is_dir():
            segs += sorted(p for p in s.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp", ".jpg"}
                           and p.resolve() != args.image.resolve())
        else:
            segs.append(s)
    if not segs:
        raise SystemExit("no segmentation images given")

    meta = zelloscope_metadata(args.image)
    p2mm = meta.get("p2mm")
    if meta:
        print(f"{args.image.name}: Zelloscope {meta.get('sw_ver', '?')}, preset {meta.get('preset', '?')!r}, "
              f"{p2mm} mm/px")
    image = _read_on_white(args.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out = args.out or args.image.parent
    out.mkdir(parents=True, exist_ok=True)

    region = None
    rows = []
    for seg_path in segs:
        seg = read_gray(seg_path)
        if seg.shape != gray.shape:
            print(f"{seg_path.name}: size {seg.shape[::-1]} differs from the image {gray.shape[::-1]}, skipped")
            continue
        labels = cell_labels(seg)
        if region is None:  # same region for every segmentation, so they are compared fairly
            if args.roi:
                region = np.zeros(gray.shape, bool)
                x0, y0, x1, y1 = args.roi
                region[y0:y1, x0:x1] = True
            else:
                region = foam_region(gray, labels)
            print(f"foam region: {region.mean() * 100:.0f}% of the image")
        try:
            res, draw = score(gray, labels, region, p2mm)
        except ValueError as exc:
            print(f"{seg_path.name}: {exc}")
            continue
        seg_meta = zelloscope_metadata(seg_path)
        # the preset is stored in the image Zelloscope saved; a segmentation file may carry its own
        preset = seg_meta.get("preset") or (meta.get("preset", "") if len(segs) == 1 else "")
        res = {"segmentation": seg_path.name, "preset": preset} | res
        rows.append(res)
        picture(rgb, labels, region, draw, res, seg_path.name, out / f"{seg_path.stem}_score.png")
        print(f"{seg_path.name}: error rate {res['error_rate_pct']:.1f}% (false splits {res['false_split_pct']:.1f}%, "
              f"merged {res['merged_pct']:.1f}%, fragments {res['fragments_pct']:.1f}%), "
              f"{res['cells']} cells, lines on walls {res['lines_on_walls_pct']:.0f}%")

    if rows:
        df = pd.DataFrame(rows).sort_values("error_rate_pct")
        df.round(3).to_csv(out / "segmentation_scores.csv", index=False)
        if len(df) > 1:
            print("\nranking (best first):")
            print(df[["segmentation", "error_rate_pct", "false_split_pct", "merged_pct", "fragments_pct",
                      "lines_on_walls_pct", "cells"]].round(1).to_string(index=False))
        print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
