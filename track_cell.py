#!/usr/bin/env python3
"""Follow single foam cells through a compression series.

Pick one or more cells in the first image; the script follows each of them
through every step and shows how its size and shape change.

How the tracking works: between two consecutive steps the motion of the whole
image is estimated (dense optical flow). The cell's pixels are moved along
with it, and the cell in the next step that overlaps the moved outline most is
taken as the same cell. The overlap (IoU, 0-1) is the match confidence:
  >= 0.5   good match (and about the same size): the matched cell is shown,
           measured, and carried on to the next step
  0.25-0.5 uncertain (orange), or a size jump of more than 40% (likely the
           segmentation merged it with a neighbour or split it)
  < 0.25   lost (red)
On uncertain and lost steps the carried outline itself is shown, measured and
carried on, so one bad step can't send the track onto a neighbouring cell.

Which cells are foam cells comes from cell_data.csv, as in foam_anisotropy.py.

Picking cells (coordinates are in the first image's pixels):
  --pick           click the cells in a window showing the first image
                   (default; Enter or Esc when done, right-click undoes)
  --cell X Y       a cell at X, Y (repeat for more cells)
  --auto N         N cells spread over the foam, chosen automatically

Output (default: cell_tracks/ next to the images), per tracked cell:
  cell_<n>_track.png   close-ups of the cell at every step with its outline,
                       and graphs of its area, aspect ratio and angle
  cell_<n>_track.mp4   the close-ups as a short video (--video)
and for all cells:
  cells_overview.png   the first image with the tracked cells marked
  cell_tracks.csv      every cell, every step: position, area, length, width,
                       aspect ratio, angle, match confidence

Examples:
  python track_cell.py videocompressed\\images videocompressed\\Segmentation
  python track_cell.py videocompressed\\images videocompressed\\Segmentation --cell 2500 2100
  python track_cell.py videocompressed\\images videocompressed\\Segmentation --auto 6 --video
  python track_cell.py --files before.png before_seg.png after.png after_seg.png --auto 6
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import foam_anisotropy as fa  # noqa: E402
from cell_analysis import load_images, pair_files  # noqa: E402
from segmentation_score import zelloscope_metadata  # noqa: E402

FLOW_SCALE = 0.25  # optical flow is computed on images shrunk by this factor
GOOD_IOU, LOST_IOU = 0.5, 0.25
MERGE_RATIO = 1.4  # matched area / carried area above this (or below 1/this) = merge (split)
TILE = 300  # close-up size in the summary picture, px
OUTLINE = (255, 220, 0)
CONF_COLOURS = {"good": (40, 170, 60), "uncertain": (240, 150, 0), "lost": (220, 40, 40)}


# ---------------------------------------------------------------- loading
def natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.stem)]


def step_label(path: Path, index: int) -> str:
    m = re.search(r"step(\d+)", path.stem, re.I)
    return f"step {m.group(1)}" if m else f"frame {index + 1}"


def load_steps(pairs, args, csv_rows) -> list[dict]:
    steps = []
    for i, (img_path, seg_path) in enumerate(pairs):
        image, seg = load_images(img_path, seg_path)
        row = fa.pick_row(csv_rows, img_path, None) if csv_rows is not None else None
        labels, _, info, footprint = fa.foam_cells(image, seg, args, row, keep_edge=True)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        steps.append({
            "path": img_path, "label": step_label(img_path, i), "rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "gray": gray,
            "small": cv2.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA),
            "labels": labels, "areas": np.bincount(labels.ravel()), "footprint": footprint,
            "mm_per_px": info.get("mm_per_px") or zelloscope_metadata(img_path).get("p2mm"),
        })
        print(f"  loaded {img_path.name}: {len(np.unique(labels)) - 1} foam cells (from {info['source']})")
    return steps


def feature_matches(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Matched SIFT feature positions (full-res px) between a (on its foam) and b."""
    s = 0.5
    ga = cv2.resize(a["gray"], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gb = cv2.resize(b["gray"], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(a["footprint"].astype(np.uint8) * 255, ga.shape[::-1], interpolation=cv2.INTER_NEAREST)
    sift = cv2.SIFT_create(nfeatures=30000)
    ka, da = sift.detectAndCompute(ga, mask)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return None
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good]) / s
    pb = np.float32([kb[m.trainIdx].pt for m in good]) / s
    return pa, pb


def displacement_guide(a: dict, b: dict, shape_small) -> tuple[np.ndarray, np.ndarray]:
    """Smooth displacement field a -> b on the low-res grid (full-res px) from feature matches:
    an overall affine fit (big shifts, overall squeezing) plus the matches' local deviations
    from it, interpolated and smoothed. Returns (field h x w x 2, affine 2x3)."""
    h, w = shape_small
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) / FLOW_SCALE
    identity = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    m = feature_matches(a, b)
    if m is None:
        return np.zeros((h, w, 2), np.float32), identity
    pa, pb = m
    A, inl = cv2.estimateAffine2D(pa, pb, ransacReprojThreshold=6, maxIters=5000)
    if A is None or inl.sum() < 20:
        return np.zeros((h, w, 2), np.float32), identity
    field = np.dstack([A[0, 0] * xs + A[0, 1] * ys + A[0, 2] - xs,
                       A[1, 0] * xs + A[1, 1] * ys + A[1, 2] - ys]).astype(np.float32)
    # local deviations of consistent matches from the affine fit
    res = pb - cv2.transform(pa[None], A)[0]
    ok = np.hypot(res[:, 0], res[:, 1]) < 60
    if ok.sum() >= 30:
        from scipy.interpolate import griddata
        pts = pa[ok] * FLOW_SCALE
        grid = (xs * FLOW_SCALE, ys * FLOW_SCALE)
        for c in range(2):
            lin = griddata(pts, res[ok, c], grid, method="linear")
            near = griddata(pts, res[ok, c], grid, method="nearest")
            dev = np.where(np.isnan(lin), near, lin).astype(np.float32)
            field[..., c] += cv2.GaussianBlur(dev, (0, 0), 60 * FLOW_SCALE)
    return field, A


def flow_between(a: dict, b: dict) -> np.ndarray:
    """Low-resolution flow a -> b, in full-resolution pixels.

    A smooth guide from matched features (handles large shifts and overall
    squeezing), refined by dense optical flow that may only move a short way
    from it, so it can't drift off where the texture changes a lot."""
    h, w = a["small"].shape
    guide, A = displacement_guide(a, b, (h, w))
    a["affine_to_next"] = A
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xs + guide[..., 0] * FLOW_SCALE
    map_y = ys + guide[..., 1] * FLOW_SCALE
    b_in_a = cv2.remap(b["small"], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    r = cv2.calcOpticalFlowFarneback(a["small"], b_in_a, None, 0.5, 2, 21, 5, 7, 1.5, 0)
    r = np.clip(r, -8, 8)  # at most ~32 full-res px away from the guide
    return guide + r / FLOW_SCALE


# ---------------------------------------------------------------- tracking
def move_mask(ys: np.ndarray, xs: np.ndarray, flow: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """Move pixel coordinates along the flow; returns the filled moved region's pixel coords."""
    fy = np.clip((ys * FLOW_SCALE).astype(int), 0, flow.shape[0] - 1)
    fx = np.clip((xs * FLOW_SCALE).astype(int), 0, flow.shape[1] - 1)
    nx = np.round(xs + flow[fy, fx, 0]).astype(int)
    ny = np.round(ys + flow[fy, fx, 1]).astype(int)
    ok = (nx >= 0) & (ny >= 0) & (nx < shape[1]) & (ny < shape[0])
    nx, ny = nx[ok], ny[ok]
    if len(nx) == 0:
        return ny, nx
    x0, y0 = nx.min() - 3, ny.min() - 3
    m = np.zeros((ny.max() - y0 + 4, nx.max() - x0 + 4), np.uint8)
    m[ny - y0, nx - x0] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    yy, xx = np.nonzero(m)
    yy, xx = yy + y0, xx + x0
    ok = (xx >= 0) & (yy >= 0) & (xx < shape[1]) & (yy < shape[0])
    return yy[ok], xx[ok]


def best_match(labels: np.ndarray, areas: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> tuple[int, float, int]:
    """Cell in labels overlapping the region (ys, xs) most: (label, IoU, its area); (0, 0, 0) if none."""
    under = labels[ys, xs]
    under = under[under > 0]
    if len(under) == 0:
        return 0, 0.0, 0
    ids, counts = np.unique(under, return_counts=True)
    best_iou, best_id, best_area = 0.0, 0, 0
    for lab, inter in zip(ids, counts):
        area = int(areas[lab])
        iou = inter / (area + len(ys) - inter)
        if iou > best_iou:
            best_iou, best_id, best_area = iou, int(lab), area
    return best_id, float(best_iou), best_area


def measure(ys: np.ndarray, xs: np.ndarray) -> dict:
    x0, y0 = xs.min() - 1, ys.min() - 1
    m = np.zeros((ys.max() - y0 + 2, xs.max() - x0 + 2), np.uint8)
    m[ys - y0, xs - x0] = 1
    fit = fa.fit_edge_ellipse(m)
    out = {"x": float(xs.mean()), "y": float(ys.mean()), "area_px": int(len(xs))}
    if fit:
        _, _, length, width, angle = fit
        out.update(length_px=length, width_px=width, aspect_ratio=length / max(width, 1e-9), angle_deg=angle)
    return out


def track(steps: list[dict], flows: list[np.ndarray], start_xy: tuple[float, float]) -> list[dict] | None:
    """Follow the cell under start_xy in the first step through all steps."""
    first = steps[0]["labels"]
    x, y = int(round(start_xy[0])), int(round(start_xy[1]))
    lab = int(first[y, x]) if 0 <= y < first.shape[0] and 0 <= x < first.shape[1] else 0
    if lab == 0:
        # clicked on a wall or just outside: take the nearest cell within 15 px
        y0, x0 = max(y - 15, 0), max(x - 15, 0)
        win = first[y0:y + 16, x0:x + 16]
        cand = np.argwhere(win > 0)
        if len(cand) == 0:
            return None
        d = np.hypot(cand[:, 0] + y0 - y, cand[:, 1] + x0 - x)
        cy, cx = cand[d.argmin()]
        lab = int(win[cy, cx])
    ys, xs = np.nonzero(first == lab)
    rec = measure(ys, xs) | {"step": steps[0]["label"], "iou": 1.0, "status": "good", "note": "picked"}
    rows = [rec | {"_ys": ys, "_xs": xs}]

    # The anchor is what gets carried to the next step. It is replaced by the
    # matched cell only when the match is good (high overlap, similar size);
    # otherwise the carried outline itself goes on, so a step where the
    # segmentation splits or merges cells can't send the track onto a neighbour.
    anchor_ys, anchor_xs = ys, xs
    for k in range(1, len(steps)):
        lab_img = steps[k]["labels"]
        pys, pxs = move_mask(anchor_ys, anchor_xs, flows[k - 1], lab_img.shape)
        if len(pys) == 0:
            break
        match, iou, area = best_match(lab_img, steps[k]["areas"], pys, pxs)
        note = ""
        if match and iou >= LOST_IOU:
            ratio = area / max(len(pys), 1)
            if ratio > MERGE_RATIO:
                note = "merged with a neighbour?"
            elif ratio < 1 / MERGE_RATIO:
                note = "split / partly lost?"
            if iou >= GOOD_IOU and not note:
                status = "good"
                ys, xs = np.nonzero(lab_img == match)
                anchor_ys, anchor_xs = ys, xs
            else:
                status = "uncertain"
                ys, xs = pys, pxs  # show and measure the carried outline, keep carrying it
                anchor_ys, anchor_xs = pys, pxs
                note = (note + "; " if note else "") + "carried outline shown"
        else:
            ys, xs = pys, pxs
            anchor_ys, anchor_xs = pys, pxs
            status, note = "lost", "no matching cell; carried outline shown"
        rec = measure(ys, xs) | {"step": steps[k]["label"], "iou": iou, "status": status, "note": note}
        # independent check: match the texture around the cell in the previous step
        prev = rows[-1]
        off = texture_check(steps[k - 1], steps[k], prev, rec)
        rec["texture_check_px"] = off
        diam = 2 * math.sqrt(max(prev["area_px"], 1) / math.pi)
        if off is not None and off > 0.5 * diam and rec["status"] == "good":
            rec["status"] = "uncertain"
            rec["note"] = (rec["note"] + "; " if rec["note"] else "") + f"texture check off by {off:.0f} px"
        rows.append(rec | {"_ys": ys, "_xs": xs})
    return rows


def texture_check(a: dict, b: dict, rec_a: dict, rec_b: dict) -> float | None:
    """Distance (px) between the tracked position in b and where the image patch around the cell
    in a best matches in b (the patch squeezed by the overall deformation). None if not possible."""
    diam = 2 * math.sqrt(max(rec_a["area_px"], 1) / math.pi)
    half = int(max(2.5 * diam, 30))
    x, y = int(round(rec_a["x"])), int(round(rec_a["y"]))
    h, w = a["gray"].shape
    if not (half <= x < w - half and half <= y < h - half):
        return None
    tpl = a["gray"][y - half:y + half, x - half:x + half]
    A = a.get("affine_to_next")
    if A is not None:  # squeeze the patch like the foam as a whole
        tpl = cv2.resize(tpl, (max(int(2 * half * abs(A[0, 0])), 8), max(int(2 * half * abs(A[1, 1])), 8)))
    search = int(3 * diam)
    tx, ty = int(round(rec_b["x"])), int(round(rec_b["y"]))
    y0, x0 = max(ty - search - tpl.shape[0] // 2, 0), max(tx - search - tpl.shape[1] // 2, 0)
    win = b["gray"][y0:ty + search + tpl.shape[0] // 2, x0:tx + search + tpl.shape[1] // 2]
    if win.shape[0] <= tpl.shape[0] or win.shape[1] <= tpl.shape[1]:
        return None
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(r)
    mx, my = x0 + loc[0] + tpl.shape[1] / 2, y0 + loc[1] + tpl.shape[0] / 2
    return float(math.hypot(mx - rec_b["x"], my - rec_b["y"]))


def auto_pick(steps: list[dict], n: int) -> list[tuple[float, float]]:
    """n cells spread over the foam: in each part of a grid, a large cell near its middle."""
    labels, foot = steps[0]["labels"], steps[0]["footprint"]
    fy, fx = np.nonzero(foot)
    x0, x1, y0, y1 = fx.min(), fx.max(), fy.min(), fy.max()
    rows, cols = fa.grid_shape(x1 - x0, y1 - y0, n)
    areas = np.bincount(labels.ravel())
    picks = []
    for r in range(rows):
        for c in range(cols):
            cx = x0 + (c + 0.5) * (x1 - x0) / cols
            cy = y0 + (r + 0.5) * (y1 - y0) / rows
            half = 0.3 * min((x1 - x0) / cols, (y1 - y0) / rows)
            win = labels[int(cy - half):int(cy + half), int(cx - half):int(cx + half)]
            ids = np.unique(win[win > 0])
            if len(ids) == 0:
                continue
            big = ids[areas[ids] >= np.percentile(areas[ids], 75)]
            best = None
            for lab in big:
                ys, xs = np.nonzero(labels == lab)
                d = math.hypot(xs.mean() - cx, ys.mean() - cy)
                if best is None or d < best[0]:
                    best = (d, xs.mean(), ys.mean())
            picks.append((best[1], best[2]))
    return picks


def click_pick(step: dict) -> list[tuple[float, float]]:
    """Window with the first image; click cells, right-click undoes, Enter/Esc finishes."""
    img = cv2.cvtColor(step["rgb"], cv2.COLOR_RGB2BGR)
    scale = min(1600 / img.shape[1], 950 / img.shape[0], 1.0)
    base = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    picks: list[tuple[float, float]] = []
    title = "Click cells to track (right-click = undo, Enter = done)"

    def redraw():
        view = base.copy()
        for i, (x, y) in enumerate(picks, 1):
            p = (int(x * scale), int(y * scale))
            cv2.circle(view, p, 9, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(view, str(i), (p[0] + 11, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2,
                        cv2.LINE_AA)
        cv2.imshow(title, view)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            picks.append((x / scale, y / scale))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and picks:
            picks.pop()
            redraw()

    try:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        raise SystemExit("Can't open a window here; use --cell X Y or --auto N instead.")
    cv2.setMouseCallback(title, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10, 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return picks


# ---------------------------------------------------------------- output
def crop_box(rec: dict, size: int, shape) -> tuple[int, int, int, int]:
    x0 = int(round(rec["x"] - size / 2)); y0 = int(round(rec["y"] - size / 2))
    x0 = min(max(x0, 0), shape[1] - size); y0 = min(max(y0, 0), shape[0] - size)
    return x0, y0, x0 + size, y0 + size


def path_box(rec_list: list[dict], size: int, shape) -> tuple[int, int, int, int]:
    """One square view that holds the cell's whole path (plus half a close-up around it)."""
    xs = [r["x"] for r in rec_list]; ys = [r["y"] for r in rec_list]
    side = int(max(max(xs) - min(xs), max(ys) - min(ys)) + size)
    side = min(side, min(shape))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return crop_box({"x": cx, "y": cy}, side, shape)


def tile(step: dict, rec: dict, size: int, box=None, out: int = TILE, trail=None) -> np.ndarray:
    """Close-up centred on the cell (or the fixed view box): neighbours' walls faint, the cell outlined.

    trail: the cell's earlier positions [(x, y), ...], drawn as a path of dots.
    """
    x0, y0, x1, y1 = box if box is not None else crop_box(rec, size, step["labels"].shape)
    size = x1 - x0
    rgb = step["rgb"][y0:y1, x0:x1].copy()
    lab = step["labels"][y0:y1, x0:x1]
    walls = (lab != np.roll(lab, 1, 0)) | (lab != np.roll(lab, 1, 1))
    rgb[walls & (lab > 0)] = (rgb[walls & (lab > 0)] * 0.5 + 120).astype(np.uint8)
    m = np.zeros(lab.shape, np.uint8)
    ys, xs = rec["_ys"] - y0, rec["_xs"] - x0
    ok = (ys >= 0) & (xs >= 0) & (ys < size) & (xs < size)
    m[ys[ok], xs[ok]] = 1
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    t = cv2.resize(rgb, (out, out), interpolation=cv2.INTER_AREA)
    s = out / size
    if trail:
        pts = [(int((x - x0) * s), int((y - y0) * s)) for x, y in trail] + [(int((rec["x"] - x0) * s),
                                                                                int((rec["y"] - y0) * s))]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(t, a, b, (255, 255, 255), 2, cv2.LINE_AA)
        for p in pts[:-1]:
            cv2.circle(t, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(t, p, 5, (0, 0, 0), 1, cv2.LINE_AA)
    scaled = [(c * s).astype(np.int32) for c in contours]
    style = cv2.LINE_AA
    if rec["status"] == "lost":
        for c in scaled:  # dashed-looking: thin red outline of the predicted position
            cv2.polylines(t, [c], True, CONF_COLOURS["lost"], 2, style)
    else:
        cv2.drawContours(t, scaled, -1, (0, 0, 0), 5, style)
        cv2.drawContours(t, scaled, -1, OUTLINE, 2, style)
    return t


def short_note(note: str) -> str:
    return ("merged?" if "merged" in note else "split?" if "split" in note else "lost" if "no matching" in note
            else "") + (" (carried outline)" if "carried" in note else "")


def summary_figure(steps, rec_list, n: int, size: int, mm_per_px, dst: Path) -> None:
    k = len(rec_list)
    cols = min(k, 6)
    trows = math.ceil(k / cols)
    fig = plt.figure(figsize=(3.0 * cols, 3.25 * trows + 3.4), dpi=110)
    gs = fig.add_gridspec(trows + 1, cols, height_ratios=[3.25] * trows + [3.4], hspace=0.35)
    unit = "mm" if mm_per_px else "px"
    f = mm_per_px if mm_per_px else 1.0

    for i, rec in enumerate(rec_list):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(tile(steps[i], rec, size))
        ax.set_xticks([]); ax.set_yticks([])
        col = np.array(CONF_COLOURS[rec["status"]]) / 255
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3)
        ar = rec.get("aspect_ratio", float("nan"))
        ang = rec.get("angle_deg", float("nan"))
        ax.set_title(f"{rec['step']}   match {rec['iou']:.2f}", fontsize=10, color=col if rec["status"] != "good"
                     else "#222222")
        dx, dy = rec["x"] - rec_list[0]["x"], rec["y"] - rec_list[0]["y"]
        moved = f"moved {dx * f:+.2f}, {dy * f:+.2f} {unit} from step 1" if i else "start position"
        ax.set_xlabel(f"AR {ar:.2f}   {ang:.0f}°   {rec['area_px'] * f * f:.3g} {unit}²\n{moved}"
                      + (f"\n{short_note(rec['note'])}" if rec["note"] and rec["note"] != "picked" else ""),
                      fontsize=8.5, color="#444444")

    x = np.arange(1, k + 1)
    good = np.array([r["status"] != "lost" for r in rec_list])
    series = [
        ([r["area_px"] * f * f for r in rec_list], f"Area [{unit}²]"),
        ([r.get("aspect_ratio", np.nan) for r in rec_list], "Aspect ratio (length / width)"),
        ([r.get("angle_deg", np.nan) for r in rec_list], "Angle [deg, 90 = vertical]"),
    ]
    sub = gs[trows, :].subgridspec(1, 3, wspace=0.3)
    for j, (vals, title) in enumerate(series):
        ax = fig.add_subplot(sub[0, j])
        vals = np.array(vals, float)
        ax.plot(x, vals, "-", color="#9bb5d8", lw=1.5)
        ax.plot(x[good], vals[good], "o", color="#1c5cab", ms=6)
        if (~good).any():
            ax.plot(x[~good], vals[~good], "o", mfc="white", mec="#dc2828", ms=7, label="lost (predicted)")
            ax.legend(fontsize=8, frameon=False)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([r["step"].split()[-1] for r in rec_list], fontsize=8)
        ax.set_xlabel("step", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Cell {n}: picked at ({rec_list[0]['x']:.0f}, {rec_list[0]['y']:.0f}) in {steps[0]['path'].name}"
                 "\nClose-ups follow the cell (the video and overview show its path). Frame colour = match quality: "
                 "green = matched to a segmented cell; orange/red = no clean match (the segmentation split or merged "
                 "it), so the outline carried along by the measured motion is shown and measured instead.",
                 fontsize=10, x=0.01, ha="left", color="#555555")
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def write_video(steps, rec_list, size: int, dst: Path, fps: float = 25, hold: float = 1.0) -> None:
    side = 720
    try:
        import imageio_ffmpeg
        gen = imageio_ffmpeg.write_frames(str(dst), (side, side + 60), fps=fps, codec="libx264",
                                          pix_fmt_out="yuv420p", quality=8, macro_block_size=4)
        gen.send(None)
        write, close = gen.send, gen.close
    except ImportError:
        vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (side, side + 60))
        write, close = (lambda f: vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))), vw.release
    frames = []
    box = path_box(rec_list, size, steps[0]["labels"].shape)  # same view for every step: you see it move
    for i, rec in enumerate(rec_list):
        trail = [(r["x"], r["y"]) for r in rec_list[:i]]
        t = tile(steps[i], rec, size, box=box, out=side, trail=trail)
        f = np.full((side + 60, side, 3), 255, np.uint8)
        f[60:] = t
        txt = f"{rec['step']}   AR {rec.get('aspect_ratio', float('nan')):.2f}   " \
              f"angle {rec.get('angle_deg', float('nan')):.0f}   match {rec['iou']:.2f}"
        cv2.putText(f, txt, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CONF_COLOURS[rec["status"]]
                    if rec["status"] != "good" else (30, 30, 30), 2, cv2.LINE_AA)
        frames.append(f)
    n_hold, n_fade = int(hold * fps), int(0.4 * fps)
    for i, f in enumerate(frames):
        for _ in range(n_hold):
            write(np.ascontiguousarray(f))
        if i + 1 < len(frames):
            for j in range(1, n_fade + 1):
                write(np.ascontiguousarray(cv2.addWeighted(f, 1 - j / (n_fade + 1), frames[i + 1], j / (n_fade + 1), 0)))
    close()


def pair_summary(steps, tracks: dict, sizes: dict, mm_per_px, dst: Path) -> None:
    """Two-image series: every tracked cell before | after, with how it changed."""
    n = len(tracks)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(5.2 * cols, 3.3 * rows + 0.8), dpi=110)
    unit = "mm" if mm_per_px else "px"
    f = mm_per_px if mm_per_px else 1.0
    for i, (cid, recs) in enumerate(tracks.items()):
        a, b = recs[0], recs[-1]
        ta = tile(steps[0], a, sizes[cid])
        tb = tile(steps[-1], b, sizes[cid])
        pair = np.hstack([ta, np.full((TILE, 8, 3), 255, np.uint8), tb])
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(pair)
        ax.set_xticks([]); ax.set_yticks([])
        col = np.array(CONF_COLOURS[b["status"]]) / 255
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3)
        da = 100 * (b["area_px"] / a["area_px"] - 1)
        ar_a, ar_b = a.get("aspect_ratio", np.nan), b.get("aspect_ratio", np.nan)
        an_a, an_b = a.get("angle_deg", np.nan), b.get("angle_deg", np.nan)
        ax.set_title(f"cell {cid}      before   |   after", fontsize=10, loc="left")
        note = short_note(b["note"]) if b["note"] else ""
        ax.set_xlabel(f"area {a['area_px'] * f * f:.3g} -> {b['area_px'] * f * f:.3g} {unit}² ({da:+.0f}%)   "
                      f"AR {ar_a:.2f} -> {ar_b:.2f}   angle {an_a:.0f} -> {an_b:.0f}°"
                      + (f"\n{note}" if note else ""), fontsize=8.5,
                      color="#222222" if b["status"] == "good" else col)
    fig.suptitle(f"{steps[0]['path'].name}  ->  {steps[-1]['path'].name}      green frame = after-image cell "
                 "matched cleanly; orange/red = the segmentation outlines it differently after compression, so the "
                 "outline carried along by the measured motion is shown", fontsize=9.5, x=0.01, ha="left",
                 color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(dst)
    plt.close(fig)


def overview(step: dict, tracks: dict, dst: Path, k: int = 0) -> None:
    """The k-th step's image with each tracked cell outlined and numbered, and its path drawn."""
    img = step["rgb"].copy()
    th = max(2, img.shape[0] // 600)
    for n, recs in tracks.items():
        if k >= len(recs):
            continue
        r = recs[k]
        pts = np.array([(int(q["x"]), int(q["y"])) for q in recs[:k + 1]], np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), th * 2, cv2.LINE_AA)
        for q in pts[1:]:
            cv2.circle(img, tuple(int(v) for v in q), th * 3, (255, 255, 255), -1, cv2.LINE_AA)
        m = np.zeros(img.shape[:2], np.uint8)
        m[r["_ys"], r["_xs"]] = 1
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, OUTLINE, th * 2, cv2.LINE_AA)
        p = (int(r["x"]) + 25 * th, int(r["y"]) - 10 * th)
        cv2.putText(img, str(n), p, cv2.FONT_HERSHEY_SIMPLEX, 1.4 * th / 2, (0, 0, 0), th * 4, cv2.LINE_AA)
        cv2.putText(img, str(n), p, cv2.FONT_HERSHEY_SIMPLEX, 1.4 * th / 2, OUTLINE, th, cv2.LINE_AA)
    s = 2000 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    cv2.imencode(".png", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))[1].tofile(str(dst))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path, nargs="?", help="folder of images (the series, in file-name order)")
    ap.add_argument("segmentation", type=Path, nargs="?", help="folder of segmentations")
    ap.add_argument("--files", type=Path, nargs="+", metavar="FILE",
                    help="instead of folders: IMAGE1 SEG1 IMAGE2 SEG2 ... in series order")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: cell_tracks/ next to images)")
    ap.add_argument("--cell", type=float, nargs=2, action="append", metavar=("X", "Y"),
                    help="a cell to track, by its pixel position in the first image (repeatable)")
    ap.add_argument("--auto", type=int, default=None, metavar="N", help="track N cells spread over the foam")
    ap.add_argument("--pick", action="store_true", help="click the cells in a window (the default)")
    ap.add_argument("--video", action="store_true", help="also write a close-up video per cell")
    ap.add_argument("--zoom", type=float, default=5.0,
                    help="close-up width in cell diameters (default 5; bigger = more surroundings)")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--hue", type=int, nargs=2, default=fa.DEFAULT_HUE_RANGE, metavar=("MIN", "MAX"))
    ap.add_argument("--min-saturation", type=int, default=fa.DEFAULT_MIN_SATURATION)
    ap.add_argument("--no-colour-check", action="store_true")
    args = ap.parse_args()
    args.min_mm = args.max_mm = None
    args.row = None

    if args.files:
        if len(args.files) % 2:
            raise SystemExit("--files takes pairs: IMAGE SEGMENTATION IMAGE SEGMENTATION ...")
        pairs = list(zip(args.files[::2], args.files[1::2]))
        args.images = args.files[0].parent
    else:
        if not (args.images and args.segmentation):
            raise SystemExit("give an images folder and a segmentation folder, or --files")
        pairs, unmatched = pair_files(args.images, args.segmentation)
        for img in unmatched:
            print(f"no segmentation for {img.name} (skipped)")
        pairs.sort(key=lambda p: natural_key(p[0]))
    if len(pairs) < 2:
        raise SystemExit("need at least two image/segmentation pairs")

    csv_rows = None
    if not args.no_csv:
        csv_path = args.csv or fa.find_csv(args.images)
        if csv_path:
            print(f"reading {csv_path}")
            csv_rows = fa.read_csv_rows(csv_path)
    print(f"loading {len(pairs)} steps...")
    steps = load_steps(pairs, args, csv_rows)

    if args.cell:
        starts = [tuple(c) for c in args.cell]
    elif args.auto:
        starts = auto_pick(steps, args.auto)
    else:
        starts = click_pick(steps[0])
    if not starts:
        raise SystemExit("no cells picked")

    print("estimating motion between steps...")
    flows = [flow_between(steps[k], steps[k + 1]) for k in range(len(steps) - 1)]
    for k in range(len(steps) - 1):
        A = steps[k]["affine_to_next"]
        print(f"  {steps[k]['label']} -> {steps[k + 1]['label']}: overall shift ({A[0, 2]:+.0f}, {A[1, 2]:+.0f}) px, "
              f"width x{A[0, 0]:.3f}, height x{A[1, 1]:.3f}")

    out = args.out or args.images.parent / "cell_tracks"
    out.mkdir(parents=True, exist_ok=True)
    mm_per_px = steps[0]["mm_per_px"]
    tracks, table, sizes = {}, [], {}
    for n, xy in enumerate(starts, 1):
        recs = track(steps, flows, xy)
        if recs is None:
            print(f"cell {n}: nothing at ({xy[0]:.0f}, {xy[1]:.0f}) - not on a foam cell, skipped")
            continue
        tracks[n] = recs
        diam = 2 * math.sqrt(recs[0]["area_px"] / math.pi)
        size = int(min(max(args.zoom * diam, 160), min(steps[0]["labels"].shape)))
        sizes[n] = size
        summary_figure(steps, recs, n, size, mm_per_px, out / f"cell_{n}_track.png")
        if args.video:
            write_video(steps, recs, size, out / f"cell_{n}_track.mp4")
        statuses = [r["status"] for r in recs]
        print(f"cell {n} at ({xy[0]:.0f}, {xy[1]:.0f}): {statuses.count('good')} good, "
              f"{statuses.count('uncertain')} uncertain, {statuses.count('lost')} lost of {len(recs)} steps; "
              f"AR {recs[0].get('aspect_ratio', float('nan')):.2f} -> {recs[-1].get('aspect_ratio', float('nan')):.2f}")
        for r in recs:
            row = {"cell": n} | {k: v for k, v in r.items() if not k.startswith("_")}
            if mm_per_px:
                for key in ("length_px", "width_px"):
                    if key in row:
                        row[key.replace("_px", "_um")] = row[key] * mm_per_px * 1000
                row["area_um2"] = row["area_px"] * (mm_per_px * 1000) ** 2
            table.append(row)

    if tracks:
        overview(steps[0], tracks, out / "cells_overview.png")
        overview(steps[-1], tracks, out / "cells_overview_last.png", k=len(steps) - 1)
        if len(steps) == 2:
            pair_summary(steps, tracks, sizes, mm_per_px, out / "before_after.png")
        pd.DataFrame(table).round(3).to_csv(out / "cell_tracks.csv", index=False)
        print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
