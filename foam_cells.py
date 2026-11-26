#!/usr/bin/env python3
"""Measure foam cell size and cell shape change across a compression series.

Each image shows a blue foam strip pressed against a yellow plate on a dark
background. For every image the script:

1. Segments the foam strip (blue-dominant pixels, largest component) and
   measures its height/thickness (the strip's vertical extent per column).
2. Measures the cells two independent ways inside the strip:
   a. Cell tessellation: pores are dark, cell walls (struts) bright. Seeds
      are placed at pore centres (local intensity minima) and a watershed
      grows them out to the walls. Each cell gets an equivalent diameter,
      fitted-ellipse axes, aspect ratio, angle and its extent along x and y.
   b. Texture autocorrelation (ACF): the 2-D autocorrelation of the strip's
      band-passed texture has a central peak whose width along x and y is
      the characteristic cell period in that direction. It is
      segmentation-free, so it is a robust check on the trend in (a). It
      also picks up how aligned/continuous the struts are, so its y/x ratio
      is a texture anisotropy and is larger than the per-cell aspect ratio;
      compare it between images rather than reading it as a cell shape.
3. Compares every image with the first one (the reference, uncompressed
   state, or whichever image you pass via --reference) and reports the
   relative change in cell size, the change in aspect ratio, and an
   apparent strain in x and y derived from the cell dimensions.

Outputs (in the output directory):
  <name>_cells.png      annotated image: strip outline, cell ellipses
                        colour-coded by aspect ratio, summary text
  <name>_cells.csv      one row per detected cell
  foam_summary.csv      one row per image, including change vs reference
  foam_profile.csv      cell size/shape binned by distance from the plate
                        (compression is usually not uniform along the strip)
  foam_summary.png      plots of the metrics across the series (if
                        matplotlib is installed)

Images are processed in filename order, which is assumed to be the order of
increasing compression. Use --pix2mm to report sizes in mm.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from straighten import IMAGE_EXTS

# --- Strip segmentation -----------------------------------------------------
# A pixel belongs to the foam if its blue channel beats green by this much
# (after blurring). Background is greenish-dark, the plate yellow, the metal
# fixture grey -- none of them are blue-dominant like the foam.
BLUE_DOMINANCE = 12
# Pixels this far (px) from the strip outline are ignored for cell
# measurements: cut edges and out-of-focus borders distort the cells.
EDGE_MARGIN = 25

# --- Cell segmentation ------------------------------------------------------
# Band-pass (difference of Gaussians, px) applied before the
# autocorrelation: removes pixel noise and slow illumination changes so only
# the cell-scale texture is correlated.
BANDPASS_LOW = 1.5
BANDPASS_HIGH = 8
# Light smoothing (px) applied before seed detection, to suppress sparkle
# noise on the struts.
DENOISE_SIGMA = 1.5
# Minimum distance (px) between watershed seeds (pore centres): roughly a
# third to half of a cell diameter.
MIN_SEED_DISTANCE = 3
# A seed must be this much darker (grey levels) than the Gaussian average of
# its surroundings (sigma SEED_CONTEXT_SIGMA px) to count as a pore centre.
SEED_DEPTH = 2.0
SEED_CONTEXT_SIGMA = 10
# Cells outside this equivalent-diameter range (px) are discarded as noise or
# merged regions.
MIN_CELL_DIAMETER = 5
MAX_CELL_DIAMETER = 80
# Regions more elongated than this are watershed leaks along a strut gap,
# not real cells.
MAX_CELL_ASPECT = 5.0

# Bin width (px) for the cell profile along the strip (distance from plate).
PROFILE_BIN = 100

# --- Autocorrelation --------------------------------------------------------
# Half-size (px) of the autocorrelation window analysed around the centre.
ACF_HALF = 60


# ----------------------------------------------------------------------------
# Strip
# ----------------------------------------------------------------------------
def foam_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of the blue foam strip."""
    blur = cv2.GaussianBlur(image.astype(np.int16), (0, 0), 5)
    b, g, r = blur[..., 0], blur[..., 1], blur[..., 2]
    raw = ((b - g > BLUE_DOMINANCE) & (b - r > BLUE_DOMINANCE)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if n <= 1:
        raise ValueError("No foam strip found")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Fill interior holes (dark folds, large pores).
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def strip_geometry(mask: np.ndarray) -> dict:
    """Thickness (vertical extent) statistics and bounding box of the strip."""
    cols = np.where(mask.any(axis=0))[0]
    heights = (mask[:, cols] > 0).sum(axis=0).astype(float)
    # Ignore the ragged ends of the strip.
    trim = max(1, len(cols) // 20)
    core = heights[trim:-trim] if len(heights) > 2 * trim else heights
    x, y, w, h = cv2.boundingRect(mask)
    return {
        "strip_thickness_px": float(np.median(core)),
        "strip_thickness_min_px": float(np.min(core)),
        "strip_thickness_max_px": float(np.max(core)),
        "strip_area_px": float((mask > 0).sum()),
        "strip_x0": int(x),
        "strip_bbox": (int(x), int(y), int(w), int(h)),
    }


# ----------------------------------------------------------------------------
# Cells: segmentation
# ----------------------------------------------------------------------------
def bandpass(image: np.ndarray) -> np.ndarray:
    """Cell-scale texture of the blue channel (difference of Gaussians)."""
    blue = image[..., 0].astype(np.float32)
    return cv2.GaussianBlur(blue, (0, 0), BANDPASS_LOW) - cv2.GaussianBlur(blue, (0, 0), BANDPASS_HIGH)


def segment_cells(image: np.ndarray, analysis_mask: np.ndarray) -> np.ndarray:
    """Label image tessellating the strip into individual cells (0 = outside).

    Every pore is a local intensity minimum (dark) surrounded by bright
    struts. Seeds are placed at local minima of the lightly smoothed blue
    channel that are clearly darker than their neighbourhood; a watershed on
    the intensity then grows each seed until it meets the strut ridges. The
    resulting regions are whole cells (pore plus half the surrounding wall),
    i.e. the cell size you would measure by hand from wall to wall.
    """
    blue = image[..., 0].astype(np.float32)
    smooth = cv2.GaussianBlur(blue, (0, 0), DENOISE_SIGMA)
    k = 2 * MIN_SEED_DISTANCE + 1
    local_min = cv2.erode(smooth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    neighbourhood = cv2.GaussianBlur(blue, (0, 0), SEED_CONTEXT_SIGMA)
    seeds = (
        (smooth == local_min) & (analysis_mask > 0) & (smooth < neighbourhood - SEED_DEPTH)
    ).astype(np.uint8)
    _, markers = cv2.connectedComponents(seeds, connectivity=8)

    relief = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    labels = cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
    labels[(labels < 0) | (analysis_mask == 0)] = 0
    return labels


def measure_cells(labels: np.ndarray) -> list[dict]:
    """Per-pore geometry from second moments (equivalent ellipse)."""
    cells = []
    n = labels.max()
    if n == 0:
        return cells
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    order = np.argsort(lab)
    lab, xs, ys = lab[order], xs[order].astype(float), ys[order].astype(float)
    bounds = np.searchsorted(lab, np.arange(1, n + 2))
    for i in range(n):
        a, b = bounds[i], bounds[i + 1]
        area = b - a
        if area < 5:
            continue
        cx, cy = xs[a:b].mean(), ys[a:b].mean()
        dx, dy = xs[a:b] - cx, ys[a:b] - cy
        cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]])
        cov += np.eye(2) / 12.0  # pixel discretisation
        evals, evecs = np.linalg.eigh(cov)
        minor, major = 4 * np.sqrt(evals)  # full axis lengths of equivalent ellipse
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])) % 180
        eq_d = 2 * np.sqrt(area / np.pi)
        if not (MIN_CELL_DIAMETER <= eq_d <= MAX_CELL_DIAMETER) or major / max(minor, 1e-6) > MAX_CELL_ASPECT:
            continue
        cells.append({
            "cx": cx, "cy": cy, "area_px": float(area), "eq_diameter_px": eq_d,
            "major_px": major, "minor_px": minor, "aspect_ratio": major / max(minor, 1e-6),
            "angle_deg": angle,
            # Extent along the image axes (Feret-like, from moments):
            "width_x_px": 4 * np.sqrt(cov[0, 0]), "height_y_px": 4 * np.sqrt(cov[1, 1]),
        })
    return cells


# ----------------------------------------------------------------------------
# Cells: autocorrelation
# ----------------------------------------------------------------------------
def autocorrelation_size(flat: np.ndarray, analysis_mask: np.ndarray) -> dict:
    """Mean cell dimensions from the central peak of the texture autocorrelation.

    The masked autocorrelation is normalised by the mask's own
    autocorrelation so the strip's outline doesn't bias it. The cell size in
    a direction is taken as twice the lag at which the correlation first
    crosses zero (one pore + one wall = one cell period). The second moments
    of the positive central lobe give the overall anisotropy and its angle.
    """
    m = (analysis_mask > 0).astype(np.float32)
    f = flat.astype(np.float32) * m
    f[m > 0] -= f[m > 0].mean()
    f *= m
    h, w = f.shape
    H, W = cv2.getOptimalDFTSize(2 * h), cv2.getOptimalDFTSize(2 * w)

    def acf(x):
        F = np.fft.rfft2(x, s=(H, W))
        return np.fft.fftshift(np.fft.irfft2(F * np.conj(F), s=(H, W)))

    num, den = acf(f), acf(m)
    cy, cx = H // 2, W // 2
    win = (slice(cy - ACF_HALF, cy + ACF_HALF + 1), slice(cx - ACF_HALF, cx + ACF_HALF + 1))
    num, den = num[win], den[win]
    c = num / np.maximum(den, 1.0)
    c /= c[ACF_HALF, ACF_HALF]

    def zero_cross(profile: np.ndarray) -> float:
        idx = np.where(profile <= 0)[0]
        if len(idx) == 0:
            return float("nan")
        i = idx[0]
        # linear interpolation between i-1 and i
        p0, p1 = profile[i - 1], profile[i]
        return (i - 1) + p0 / (p0 - p1)

    row = c[ACF_HALF, ACF_HALF:]
    col = c[ACF_HALF:, ACF_HALF]
    row = (row + c[ACF_HALF, ACF_HALF::-1]) / 2
    col = (col + c[ACF_HALF::-1, ACF_HALF]) / 2
    size_x = 2 * zero_cross(row)
    size_y = 2 * zero_cross(col)

    # Second moments of the connected positive central lobe.
    lobe = (c > 0).astype(np.uint8)
    _, lab = cv2.connectedComponents(lobe)
    lobe = (lab == lab[ACF_HALF, ACF_HALF])
    yy, xx = np.mgrid[-ACF_HALF:ACF_HALF + 1, -ACF_HALF:ACF_HALF + 1]
    wgt = np.where(lobe, c, 0)
    s = wgt.sum()
    cov = np.array([
        [(wgt * xx * xx).sum() / s, (wgt * xx * yy).sum() / s],
        [(wgt * xx * yy).sum() / s, (wgt * yy * yy).sum() / s],
    ])
    evals, evecs = np.linalg.eigh(cov)
    anis = float(np.sqrt(evals[1] / max(evals[0], 1e-9)))
    angle = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])) % 180)
    return {
        "acf_size_x_px": float(size_x),
        "acf_size_y_px": float(size_y),
        "acf_ratio_y_over_x": float(size_y / size_x) if size_x else float("nan"),
        "acf_anisotropy": anis,
        "acf_major_angle_deg": angle,
        "_acf": c,
    }


def distance_profile(cells: list[dict], mask: np.ndarray, bin_px: int) -> list[dict]:
    """Median cell size/shape in bins of distance from the strip's left end (the plate)."""
    if not cells:
        return []
    has = mask > 0
    left_edge = np.where(has.any(axis=1), has.argmax(axis=1), -1)
    dist = np.array([c["cx"] - left_edge[int(round(c["cy"]))] for c in cells])
    out = []
    for b in range(int(dist.max() // bin_px) + 1):
        sel = [c for c, d in zip(cells, dist) if b * bin_px <= d < (b + 1) * bin_px]
        if len(sel) < 20:
            continue
        wx = np.median([c["width_x_px"] for c in sel])
        hy = np.median([c["height_y_px"] for c in sel])
        out.append({
            "dist_from_plate_px": (b + 0.5) * bin_px,
            "n_cells": len(sel),
            "cell_diameter_median_px": float(np.median([c["eq_diameter_px"] for c in sel])),
            "cell_width_x_median_px": float(wx),
            "cell_height_y_median_px": float(hy),
            "cell_ratio_y_over_x": float(hy / wx),
            "cell_aspect_median": float(np.median([c["aspect_ratio"] for c in sel])),
        })
    return out


# ----------------------------------------------------------------------------
# Per image
# ----------------------------------------------------------------------------
def analyse(image: np.ndarray) -> tuple[dict, list[dict], np.ndarray, np.ndarray, list[dict]]:
    mask = foam_mask(image)
    geom = strip_geometry(mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_MARGIN + 1, 2 * EDGE_MARGIN + 1))
    inner = cv2.erode(mask, k)
    if not inner.any():
        raise ValueError("Foam strip too small after removing the edge margin")

    labels = segment_cells(image, inner)
    cells = measure_cells(labels)
    acf = autocorrelation_size(bandpass(image), inner)

    summary = dict(geom)
    summary["n_cells"] = len(cells)
    if cells:
        d = np.array([c["eq_diameter_px"] for c in cells])
        ar = np.array([c["aspect_ratio"] for c in cells])
        wx = np.array([c["width_x_px"] for c in cells])
        hy = np.array([c["height_y_px"] for c in cells])
        ang = np.radians([c["angle_deg"] for c in cells])
        # Mean orientation of elongated cells: doubled-angle average (axial data).
        wts = ar - 1
        mean_angle = np.degrees(0.5 * np.arctan2((wts * np.sin(2 * ang)).sum(), (wts * np.cos(2 * ang)).sum())) % 180
        summary.update({
            "cell_diameter_mean_px": float(d.mean()),
            "cell_diameter_median_px": float(np.median(d)),
            "cell_diameter_std_px": float(d.std()),
            "cell_aspect_mean": float(ar.mean()),
            "cell_aspect_median": float(np.median(ar)),
            "cell_width_x_median_px": float(np.median(wx)),
            "cell_height_y_median_px": float(np.median(hy)),
            "cell_ratio_y_over_x": float(np.median(hy) / np.median(wx)),
            "cell_mean_angle_deg": float(mean_angle),
        })
    summary.update({k: v for k, v in acf.items() if not k.startswith("_")})
    profile = distance_profile(cells, mask, PROFILE_BIN)
    return summary, cells, mask, inner, profile


def aspect_colour(ar: float) -> tuple[int, int, int]:
    """Green (round) -> yellow -> red (aspect >= 2) in BGR."""
    t = float(np.clip((ar - 1.0) / 1.0, 0, 1))
    if t < 0.5:
        return (0, 255, int(510 * t))
    return (0, int(255 * (2 - 2 * t)), 255)


def fmt(value_px: float, pix2mm: float | None) -> str:
    return f"{value_px * pix2mm:.3f} mm" if pix2mm else f"{value_px:.1f} px"


def draw(image: np.ndarray, mask: np.ndarray, inner: np.ndarray, cells: list[dict],
         summary: dict, pix2mm: float | None) -> np.ndarray:
    out = image.copy()
    for m, colour in ((mask, (255, 255, 255)), (inner, (200, 200, 0))):
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, colour, 2, cv2.LINE_AA)
    for c in cells:
        cv2.ellipse(
            out, (int(round(c["cx"])), int(round(c["cy"]))),
            (max(1, int(round(c["major_px"] / 2))), max(1, int(round(c["minor_px"] / 2)))),
            c["angle_deg"], 0, 360, aspect_colour(c["aspect_ratio"]), 1, cv2.LINE_AA,
        )
    lines = [
        f"cells: {summary['n_cells']}",
        f"cell diameter (median): {fmt(summary.get('cell_diameter_median_px', float('nan')), pix2mm)}",
        f"cell aspect (median): {summary.get('cell_aspect_median', float('nan')):.2f}",
        f"ACF cell size x: {fmt(summary['acf_size_x_px'], pix2mm)}  y: {fmt(summary['acf_size_y_px'], pix2mm)}",
        f"ACF y/x: {summary['acf_ratio_y_over_x']:.2f}",
        f"strip thickness: {fmt(summary['strip_thickness_px'], pix2mm)}",
    ]
    scale = image.shape[1] / 2000
    y = int(50 * scale)
    for line in lines:
        cv2.putText(out, line, (int(20 * scale) + image.shape[1] // 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1 * scale, (0, 0, 0), int(6 * scale), cv2.LINE_AA)
        cv2.putText(out, line, (int(20 * scale) + image.shape[1] // 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
        y += int(45 * scale)
    return out


# ----------------------------------------------------------------------------
# Series comparison
# ----------------------------------------------------------------------------
def add_relative_changes(rows: list[dict], ref_index: int) -> None:
    ref = rows[ref_index]
    for r in rows:
        def rel(key):
            if key not in r or key not in ref or not ref[key]:
                return float("nan")
            return r[key] / ref[key] - 1.0
        # Apparent strain along each axis from the cell dimensions:
        # negative = cells shorter than in the reference (compressed).
        r["strain_x_acf"] = rel("acf_size_x_px")
        r["strain_y_acf"] = rel("acf_size_y_px")
        r["strain_x_cells"] = rel("cell_width_x_median_px")
        r["strain_y_cells"] = rel("cell_height_y_median_px")
        r["change_cell_diameter"] = rel("cell_diameter_median_px")
        r["change_cell_aspect"] = rel("cell_aspect_median")
        r["change_acf_ratio_y_over_x"] = rel("acf_ratio_y_over_x")
        r["change_strip_thickness"] = rel("strip_thickness_px")
        r["plate_displacement_px"] = r["strip_x0"] - ref["strip_x0"]


SUMMARY_COLUMNS = [
    "image", "n_cells",
    "cell_diameter_median_px", "cell_diameter_mean_px", "cell_diameter_std_px",
    "cell_aspect_median", "cell_aspect_mean", "cell_mean_angle_deg",
    "cell_width_x_median_px", "cell_height_y_median_px", "cell_ratio_y_over_x",
    "acf_size_x_px", "acf_size_y_px", "acf_ratio_y_over_x", "acf_anisotropy", "acf_major_angle_deg",
    "strip_thickness_px", "strip_thickness_min_px", "strip_thickness_max_px", "strip_area_px",
    "plate_displacement_px",
    "strain_x_acf", "strain_y_acf", "strain_x_cells", "strain_y_cells",
    "change_cell_diameter", "change_cell_aspect", "change_acf_ratio_y_over_x", "change_strip_thickness",
]
LENGTH_COLUMNS = {c for c in SUMMARY_COLUMNS if c.endswith("_px") and c != "strip_area_px"}


def write_summary(rows: list[dict], path: Path, pix2mm: float | None) -> None:
    cols = list(SUMMARY_COLUMNS)
    if pix2mm:
        cols += [c[:-3] + "_mm" for c in SUMMARY_COLUMNS if c in LENGTH_COLUMNS]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            vals = []
            for c in cols:
                if c.endswith("_mm"):
                    v = r.get(c[:-3] + "_px", float("nan")) * pix2mm
                else:
                    v = r.get(c, "")
                vals.append(f"{v:.4f}" if isinstance(v, float) else v)
            w.writerow(vals)


PROFILE_COLUMNS = ["dist_from_plate_px", "n_cells", "cell_diameter_median_px", "cell_width_x_median_px",
                   "cell_height_y_median_px", "cell_ratio_y_over_x", "cell_aspect_median"]


def write_profile(rows: list[dict], path: Path, pix2mm: float | None) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        mm_cols = [c[:-3] + "_mm" for c in PROFILE_COLUMNS if c.endswith("_px")] if pix2mm else []
        w.writerow(["image"] + PROFILE_COLUMNS + mm_cols)
        for r in rows:
            for p in r["_profile"]:
                vals = [r["image"]] + [f"{p[c]:.4f}" if isinstance(p[c], float) else p[c] for c in PROFILE_COLUMNS]
                vals += [f"{p[c[:-3] + '_px'] * pix2mm:.4f}" for c in mm_cols]
                w.writerow(vals)


def write_cells(cells: list[dict], path: Path, pix2mm: float | None) -> None:
    keys = ["cx", "cy", "area_px", "eq_diameter_px", "major_px", "minor_px",
            "aspect_ratio", "angle_deg", "width_x_px", "height_y_px"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys + (["eq_diameter_mm", "major_mm", "minor_mm"] if pix2mm else []))
        for c in cells:
            row = [f"{c[k]:.3f}" for k in keys]
            if pix2mm:
                row += [f"{c[k] * pix2mm:.4f}" for k in ("eq_diameter_px", "major_px", "minor_px")]
            w.writerow(row)


def plot_summary(rows: list[dict], path: Path, pix2mm: float | None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping foam_summary.png")
        return
    unit, s = ("mm", pix2mm) if pix2mm else ("px", 1.0)
    x = np.arange(len(rows))
    names = [r["image"] for r in rows]
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)

    fig, ax = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    a = ax[0, 0]
    a.plot(x, g("acf_size_x_px") * s, "o-", label="cell size x (ACF)")
    a.plot(x, g("acf_size_y_px") * s, "s-", label="cell size y (ACF)")
    a.plot(x, g("cell_diameter_median_px") * s, "^--", label="cell eq. diameter (median)")
    a.set_ylabel(f"size [{unit}]"); a.set_title("Cell size"); a.legend()
    a = ax[0, 1]
    a.plot(x, g("acf_ratio_y_over_x"), "o-", label="y/x (ACF)")
    a.plot(x, g("cell_ratio_y_over_x"), "s--", label="y/x (segmented cells)")
    a.plot(x, g("cell_aspect_median"), "^--", label="per-cell aspect (median)")
    a.axhline(1, color="grey", lw=0.8)
    a.set_title("Cell shape"); a.legend()
    a = ax[1, 0]
    a.plot(x, g("strain_x_acf") * 100, "o-", label="x (ACF)")
    a.plot(x, g("strain_y_acf") * 100, "s-", label="y (ACF)")
    a.plot(x, g("strain_x_cells") * 100, "o--", label="x (cells)")
    a.plot(x, g("strain_y_cells") * 100, "s--", label="y (cells)")
    a.axhline(0, color="grey", lw=0.8)
    a.set_ylabel("% vs reference"); a.set_title("Apparent cell strain"); a.legend()
    a = ax[1, 1]
    a.plot(x, g("strip_thickness_px") * s, "o-")
    a.set_ylabel(f"thickness [{unit}]"); a.set_title("Strip thickness (median)")
    for a in ax[:, :2].flat:
        a.set_xticks(x); a.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        a.grid(alpha=0.3)
    for r in rows:
        p = r["_profile"]
        if not p:
            continue
        d = np.array([q["dist_from_plate_px"] for q in p]) * s
        ax[0, 2].plot(d, np.array([q["cell_diameter_median_px"] for q in p]) * s, "o-", ms=3, label=r["image"])
        ax[1, 2].plot(d, [q["cell_ratio_y_over_x"] for q in p], "o-", ms=3, label=r["image"])
    ax[0, 2].set_title("Cell diameter vs distance from plate"); ax[0, 2].set_ylabel(f"size [{unit}]")
    ax[1, 2].set_title("Cell y/x vs distance from plate"); ax[1, 2].set_xlabel(f"distance from plate [{unit}]")
    for a in ax[:, 2]:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    global EDGE_MARGIN
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Image file or directory of images (compression series)")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--pix2mm", type=float, default=None, help="mm per pixel, to report sizes in mm")
    parser.add_argument("--reference", default=None,
                        help="Filename of the reference (uncompressed) image; default: first in order")
    parser.add_argument("--edge-margin", type=int, default=EDGE_MARGIN,
                        help=f"Ignore cells within this many px of the strip outline (default {EDGE_MARGIN})")
    args = parser.parse_args()
    EDGE_MARGIN = args.edge_margin

    if args.input.is_dir():
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        files = [args.input]
    if not files:
        sys.exit(f"No images found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in files:
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            print(f"{src.name}: could not read, skipped")
            continue
        try:
            summary, cells, mask, inner, profile = analyse(image)
        except ValueError as e:
            print(f"{src.name}: {e}, skipped")
            continue
        summary["image"] = src.name
        summary["_profile"] = profile
        rows.append(summary)
        cv2.imwrite(str(args.output / f"{src.stem}_cells.png"), draw(image, mask, inner, cells, summary, args.pix2mm))
        write_cells(cells, args.output / f"{src.stem}_cells.csv", args.pix2mm)
        print(
            f"{src.name}: {summary['n_cells']} cells, "
            f"median diameter {fmt(summary.get('cell_diameter_median_px', float('nan')), args.pix2mm)}, "
            f"ACF size x {fmt(summary['acf_size_x_px'], args.pix2mm)} / y {fmt(summary['acf_size_y_px'], args.pix2mm)} "
            f"(y/x {summary['acf_ratio_y_over_x']:.2f}), "
            f"strip thickness {fmt(summary['strip_thickness_px'], args.pix2mm)}"
        )

    if not rows:
        sys.exit("No images could be analysed")
    ref_index = 0
    if args.reference:
        matches = [i for i, r in enumerate(rows) if r["image"] == args.reference]
        if not matches:
            sys.exit(f"Reference image {args.reference} not found among analysed images")
        ref_index = matches[0]
    add_relative_changes(rows, ref_index)

    write_summary(rows, args.output / "foam_summary.csv", args.pix2mm)
    write_profile(rows, args.output / "foam_profile.csv", args.pix2mm)
    if len(rows) > 1:
        plot_summary(rows, args.output / "foam_summary.png", args.pix2mm)
        print(f"\nChange vs reference ({rows[ref_index]['image']}):")
        for r in rows:
            print(
                f"  {r['image']}: cell strain x {r['strain_x_acf'] * 100:+.1f}% "
                f"y {r['strain_y_acf'] * 100:+.1f}% (ACF), "
                f"aspect {r['change_cell_aspect'] * 100:+.1f}%, "
                f"strip thickness {r['change_strip_thickness'] * 100:+.1f}%, "
                f"plate moved {r['plate_displacement_px']:+d} px"
            )
    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
