#!/usr/bin/env python3
"""Plot the pinholes (or cells) from one measurement row on its segmented
image, and report their size and spatial distribution.

pinhole_data.csv and cell_data.csv share one format: one row per measurement
run (timestamp, label, diameters, x_positions, y_positions, each a
Python-literal list of equal length). cell_data.csv holds every segmented
region; pinhole_data.csv only the ones counted as pinholes. Positions are
pixel coordinates in the segmented image; diameters are in mm. The segmented
image is a black-and-white boundary map: white regions (cells) separated by
thin black lines, where each pinhole is one of those regions. This matches
each (x, y) point to the white region it falls in and fills that region's
real outline, colored by diameter, over a faded copy of
the segmentation -- so only the regions from the chosen row stand out.
Colors run up to the 99th-percentile diameter so a few huge voids don't
flatten the scale for everything else; larger ones get the top color.

The mm-per-pixel scale is inferred from the data (reported diameter / the
matched region's equivalent circular diameter) unless --pix2mm is given; a
wide spread in that ratio means the row wasn't measured on this image.

Cutter noise: the machined surface carries circular tracks from the cutter,
and dark patches of foam along a track get segmented and measured like
pinholes even where there's no hole. Pass --original (the micrograph the
segmentation was made from) to find the tracks (see cutter_arcs.py) and drop
detections that sit on a track but have no solid dark core. Dropped ones are
drawn in grey with a red outline, the tracks are tinted, and the stats,
grid counts and distribution use only the rest (--keep-cutter-noise marks
them without dropping them).

Anisotropy: each region's shape is summarized by its equivalent ellipse (same
second moments): major and minor axis lengths, aspect ratio = major / minor
(1 = round), and orientation = the major axis's angle from horizontal,
counter-clockwise as seen on screen, 0-180 deg (90 = vertical). Over the
whole row it reports:
  alignment    0-1: how parallel the regions are (0 = random directions,
               1 = all parallel), the length of the mean of each region's
               doubled-angle direction vector, weighted by area x (1 - minor/
               major) so big elongated regions count most and near-round
               ones, whose direction is noise, count least
  direction    that weighted mean direction, deg
  DA           degree of anisotropy: major / minor of the ellipse of all
               the regions' second moments summed (area-weighted), i.e. the
               pore space as a whole; 1 = isotropic
Near-round regions (aspect ratio below ROUND_ASPECT_MAX) are reported but
their orientation isn't meaningful.

Outputs (in --output-dir, named after the segmented image and the CSV, e.g.
<image>_pinholes* for pinhole_data.csv, <image>_cells* for cell_data.csv):
  *.png               full-resolution overlay, with the --grid grid and each
                      grid square's count drawn on it
  *_plot.png          the same overlay with title and diameter colorbar
  *_distribution.png  size histogram + cumulative size curve (log diameter axis),
                      aspect ratio histogram, orientation rose diagram
  *.csv               one line per region (id, x, y, diameter, area, grid square,
                      major/minor axis mm, aspect ratio, orientation deg;
                      with --original also on_track_frac, dark_core_frac, cutter_noise)
"""
import argparse
import ast
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from cutter_arcs import TRACK_FRAC_MIN, find_cutter_tracks, is_cutter_noise, region_features
from void_analysis import DEFAULT_GRID_SIZE, Void, size_summary, spatial_distribution

# Some rows hold thousands of measurements per column; csv's default field
# size limit is too small for that (same cap as plot_cells.py).
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

# How much the non-pinhole segmentation is faded toward white (0 = hidden, 1 = unchanged).
BACKGROUND_OPACITY = 0.35

# Grid lines and per-square counts drawn on the overlay (RGB).
GRID_COLOR = (0, 90, 200)

# Above this many regions, per-region outlines would bury the regions
# themselves; the segmentation's own boundary lines already separate them.
OUTLINE_MAX_REGIONS = 2000

# Diameter percentile at the top of the color scale (see module docstring).
COLOR_MAX_PERCENTILE = 99

# Cutter-noise detections: fill and outline (RGB), and the cutter track tint.
NOISE_FILL = (175, 175, 175)
NOISE_OUTLINE = (220, 0, 0)
TRACK_TINT = (255, 150, 150)

# Below this aspect ratio a region is effectively round and its orientation is noise.
ROUND_ASPECT_MAX = 1.2

# Major-axis line drawn through each colored region on the overlay (RGB).
AXIS_COLOR = (255, 255, 255)


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"No rows found in {csv_path}")
    return rows


def list_rows(csv_path: Path) -> None:
    """Print each row's index, timestamp, label, count and diameter range, for picking --row."""
    print(f"{csv_path}: index  timestamp            count  diameter range (mm)  label")
    for i, row in enumerate(read_rows(csv_path)):
        diameters, _, _ = parse_pinholes(row)
        size = f"{diameters.min():.3f} - {diameters.max():.3f}" if len(diameters) else "-"
        print(f"  {i:5d}  {row['timestamp']:<19}  {len(diameters):5d}  {size:<19}  {row.get('label', '')}")


def load_row(csv_path: Path, selector: str | None) -> dict:
    """The row picked by selector, or the latest timestamp if None.

    selector is either a row index, Python-style (0 = first row, -1 = last), or
    text matched against the timestamps -- a full timestamp, or any part of one
    that picks out a single row, e.g. "14:19:36" or "2026-09-23 15:19".
    """
    rows = read_rows(csv_path)
    if selector is None:
        return max(rows, key=lambda r: r["timestamp"])
    try:
        index = int(selector)
    except ValueError:
        matches = [(i, r) for i, r in enumerate(rows) if selector.strip() in r["timestamp"]]
        if len(matches) == 1:
            return matches[0][1]
        if not matches:
            sys.exit(f"--row {selector!r}: no timestamp in {csv_path} contains that (see --list-rows)")
        sys.exit(f"--row {selector!r} matches {len(matches)} rows, be more specific: "
                 + ", ".join(f"{i} ({r['timestamp']})" for i, r in matches))
    try:
        return rows[index]
    except IndexError:
        sys.exit(f"--row {index} out of range: {csv_path} has {len(rows)} rows (0 to {len(rows) - 1})")


def parse_pinholes(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def parse(key: str) -> np.ndarray:
        return np.array([float(v) for v in ast.literal_eval(row[key])]) if row[key] else np.array([])

    diameters, x, y = parse("diameters"), parse("x_positions"), parse("y_positions")
    if not (len(diameters) == len(x) == len(y)):
        sys.exit(f"Row {row['timestamp']}: diameters/x/y lengths differ ({len(diameters)}/{len(x)}/{len(y)})")
    return diameters, x, y


def match_regions(segmented: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label the white regions and return (labels, stats, region id under each point; 0 = on a boundary line)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats((segmented > 127).astype(np.uint8), connectivity=4)
    h, w = segmented.shape
    cols = np.clip(np.round(x).astype(int), 0, w - 1)
    rows = np.clip(np.round(y).astype(int), 0, h - 1)
    return labels, stats, labels[rows, cols]


def region_shapes(labels: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per-label equivalent ellipse: (centroid x, centroid y, major px, minor px, orientation deg).

    Indexed by label value. Axis lengths are full lengths of the ellipse with
    the region's second moments; orientation is from horizontal, counter-
    clockwise on screen (image y points down, hence the sign flip), in [0, 180).
    """
    flat = labels.ravel()
    n = int(flat.max()) + 1
    h, w = labels.shape
    xs = np.tile(np.arange(w, dtype=np.float64), h)
    ys = np.repeat(np.arange(h, dtype=np.float64), w)
    area = np.maximum(np.bincount(flat, minlength=n), 1).astype(float)
    mx, my = np.bincount(flat, xs, n) / area, np.bincount(flat, ys, n) / area
    mu20 = np.bincount(flat, xs * xs, n) / area - mx ** 2
    mu02 = np.bincount(flat, ys * ys, n) / area - my ** 2
    mu11 = np.bincount(flat, xs * ys, n) / area - mx * my
    spread = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    major = 4 * np.sqrt(np.maximum((mu20 + mu02) / 2 + spread, 0))
    minor = 4 * np.sqrt(np.maximum((mu20 + mu02) / 2 - spread, 0))
    angle = (-np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))) % 180
    return mx, my, major, minor, angle


def anisotropy_summary(major: np.ndarray, minor: np.ndarray, angle: np.ndarray, area: np.ndarray) -> dict:
    """Row-level anisotropy: aspect ratio stats, alignment (0-1), mean direction, DA (see module docstring)."""
    aspect = major / np.maximum(minor, 1e-9)
    weight = area * (1 - minor / np.maximum(major, 1e-9))
    mean_vector = np.sum(weight * np.exp(2j * np.radians(angle))) / max(weight.sum(), 1e-9)

    # Sum each region's second-moment tensor (area x covariance) and take its ellipse's axis ratio.
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


def draw_major_axes(image: np.ndarray, cx: np.ndarray, cy: np.ndarray, major: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Draw each region's major axis (a line through its centroid along its orientation)."""
    output = image.copy()
    thickness = max(1, int(round(max(image.shape[:2]) / 1500)))
    theta = np.radians(angle)
    dx, dy = major / 2 * np.cos(theta), -major / 2 * np.sin(theta)  # screen angle -> image y down
    for x0, y0, ux, uy in zip(cx, cy, dx, dy):
        cv2.line(output, (int(round(x0 - ux)), int(round(y0 - uy))), (int(round(x0 + ux)), int(round(y0 + uy))),
                 AXIS_COLOR, thickness, cv2.LINE_AA)
    return output


def diameter_colors(diameters: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    norm = Normalize(diameters.min(), np.percentile(diameters, COLOR_MAX_PERCENTILE))
    return norm, plt.get_cmap("plasma")


def draw_overlay(
    segmented: np.ndarray, labels: np.ndarray, region_ids: np.ndarray, diameters: np.ndarray,
    noise: np.ndarray | None = None, tracks: np.ndarray | None = None,
) -> np.ndarray:
    """RGB image: faded segmentation with each matched region filled by its diameter color and outlined.

    Regions flagged in noise (per point, parallel to region_ids) are drawn in
    NOISE_FILL with a NOISE_OUTLINE outline instead; tracks, if given, is tinted.
    """
    if noise is None:
        noise = np.zeros(len(region_ids), bool)
    norm, cmap = diameter_colors(diameters[~noise] if (~noise).any() else diameters)
    n_labels = labels.max() + 1
    lut = np.zeros((n_labels, 3), np.uint8)
    is_pinhole = np.zeros(n_labels, bool)
    is_noise = np.zeros(n_labels, bool)
    for region_id, diameter, flagged in zip(region_ids, diameters, noise):
        if region_id > 0:
            is_noise[region_id] = flagged
            is_pinhole[region_id] = not flagged
            lut[region_id] = NOISE_FILL if flagged else (np.array(cmap(norm(diameter))[:3]) * 255).astype(np.uint8)

    background = 255 - (255 - segmented.astype(float)) * BACKGROUND_OPACITY
    output = np.repeat(background.astype(np.uint8)[:, :, None], 3, axis=2)
    if tracks is not None:
        output[tracks] = np.minimum(output[tracks], TRACK_TINT)
    mask = (is_pinhole | is_noise)[labels]
    output[mask] = lut[labels][mask]

    thickness = max(1, int(round(max(segmented.shape) / 1250)))
    for flags, color in ((is_pinhole, (0, 0, 0)), (is_noise, NOISE_OUTLINE)):
        if 0 < flags.sum() <= OUTLINE_MAX_REGIONS:
            contours, _ = cv2.findContours(flags[labels].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(output, contours, -1, color, thickness * (2 if color == NOISE_OUTLINE else 1))
    return output


def save_overlay_plot(overlay: np.ndarray, diameters: np.ndarray, title: str, dst: Path) -> None:
    import matplotlib.pyplot as plt

    norm, cmap = diameter_colors(diameters)
    fig, ax = plt.subplots(figsize=(10, 10.6), dpi=150)
    ax.imshow(overlay)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.02,
                            extend="max" if norm.vmax < diameters.max() else "neither")
    colorbar.set_label("Diameter (mm)")
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def draw_grid_counts(image: np.ndarray, spatial_rows: list[dict], grid_size: int) -> np.ndarray:
    """Draw the grid_size x grid_size grid on image, each square labeled with its count."""
    output = image.copy()
    h, w = output.shape[:2]
    cell_h, cell_w = h / grid_size, w / grid_size
    scale = max(h, w) / 1500

    line_thickness = max(1, int(round(3 * scale)))
    for i in range(1, grid_size):
        y = int(round(i * cell_h))
        x = int(round(i * cell_w))
        cv2.line(output, (0, y), (w, y), GRID_COLOR, line_thickness, cv2.LINE_AA)
        cv2.line(output, (x, 0), (x, h), GRID_COLOR, line_thickness, cv2.LINE_AA)

    # Dark digits with a white halo stay readable over both the faded
    # segmentation and a colored pinhole.
    font_scale = 2.5 * scale
    font_thickness = max(1, int(round(4 * scale)))
    for row in spatial_rows:
        text = str(row["void_count"])
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cx = int(round((row["grid_col"] + 0.5) * cell_w)) - text_w // 2
        cy = int(round((row["grid_row"] + 0.5) * cell_h)) + text_h // 2
        cv2.putText(output, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    font_thickness * 4, cv2.LINE_AA)
        cv2.putText(output, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, GRID_COLOR,
                    font_thickness, cv2.LINE_AA)
    return output


def save_distribution_plot(
    diameters: np.ndarray, aspect: np.ndarray, angle: np.ndarray, anisotropy: dict, noun: str, title: str, dst: Path,
) -> None:
    """2x2: size histogram, cumulative size curve (log diameter axis), aspect ratio histogram, orientation rose."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 9.6))
    ax1, ax2, ax3 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection="polar")

    # Log-spaced bins on a log axis: sizes span ~2 decades (0.05 mm cells to
    # multi-mm voids), which a linear axis squeezes into a couple of bins.
    bins = np.geomspace(diameters.min(), diameters.max(), 40)
    ax1.hist(diameters, bins=bins, color="#4477AA", edgecolor="white")
    ax1.set_xscale("log")
    for value, style, name in ((np.mean(diameters), "-", "mean"), (np.median(diameters), "--", "median")):
        ax1.axvline(value, color="#CC3311", linestyle=style, label=f"{name} {value:.3f} mm")
    ax1.set_xlabel("Diameter (mm)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Size distribution (n={len(diameters)})")
    ax1.legend()

    sorted_d = np.sort(diameters)
    ax2.plot(sorted_d, 100.0 * np.arange(1, len(sorted_d) + 1) / len(sorted_d), color="#4477AA")
    for pct in (10, 50, 90):
        ax2.axhline(pct, color="grey", linewidth=0.6, linestyle=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("Diameter (mm)")
    ax2.set_ylabel(f"Cumulative % of {noun}")
    ax2.set_title("Cumulative size distribution")
    ax2.set_ylim(0, 100)

    ax3.hist(aspect, bins=np.linspace(1, max(2.0, np.percentile(aspect, 99)), 30), color="#4477AA", edgecolor="white")
    ax3.axvline(anisotropy["aspect_median"], color="#CC3311", linestyle="--",
                label=f"median {anisotropy['aspect_median']:.2f}")
    ax3.set_xlabel("Aspect ratio (major / minor axis)")
    ax3.set_ylabel("Count")
    ax3.set_title("Elongation")
    ax3.legend()

    # Rose diagram: orientation is axial (0 and 180 deg are the same direction),
    # so each region is counted (once, unweighted) at angle and angle + 180; round
    # regions are left out since their orientation is noise.
    elongated = aspect >= ROUND_ASPECT_MAX
    edges = np.radians(np.arange(0, 361, 10))
    theta = np.radians(np.concatenate([angle[elongated], angle[elongated] + 180]))
    counts, _ = np.histogram(theta, bins=edges)
    ax4.bar(edges[:-1], counts, width=np.radians(10), align="edge", color="#4477AA", edgecolor="white")
    direction = np.radians(anisotropy["direction_deg"])
    ax4.plot([direction, direction + np.pi], [counts.max()] * 2, color="#CC3311", linewidth=2)
    ax4.set_yticklabels([])
    ax4.set_title(f"Orientation (0 deg = horizontal, 90 = vertical)\n"
                  f"alignment {anisotropy['alignment']:.2f}, direction {anisotropy['direction_deg']:.0f} deg, "
                  f"DA {anisotropy['degree_of_anisotropy']:.2f}", fontsize=10)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("segmented", type=Path, help="Segmented (black boundary / white region) image")
    parser.add_argument("--csv", type=Path, default=Path("pinhole_data.csv"), help="Measurement CSV, e.g. pinhole_data.csv or cell_data.csv (default: pinhole_data.csv)")
    parser.add_argument(
        "--row", default=None,
        help="Row to plot: an index (0 = first row, -1 = last) or a timestamp or part of one, "
             "e.g. \"14:19:36\"; default: latest timestamp. See --list-rows",
    )
    parser.add_argument("--list-rows", action="store_true", help="List the CSV's rows (index, timestamp, count) and exit")
    parser.add_argument(
        "--pix2mm", type=float, default=None,
        help="mm per pixel; default: inferred from the row's diameters vs the matched regions",
    )
    parser.add_argument(
        "--grid", type=int, default=DEFAULT_GRID_SIZE,
        help=f"Spatial distribution grid size (default: {DEFAULT_GRID_SIZE})",
    )
    parser.add_argument(
        "--original", type=Path, default=None,
        help="Micrograph the segmentation was made from; enables cutter-noise removal (see module docstring)",
    )
    parser.add_argument(
        "--keep-cutter-noise", action="store_true",
        help="With --original: mark cutter-noise detections but keep them in the counts and stats",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write outputs (default: next to the segmented image)")
    args = parser.parse_args()

    if args.list_rows:
        list_rows(args.csv)
        return

    segmented = cv2.imread(str(args.segmented), cv2.IMREAD_GRAYSCALE)
    if segmented is None:
        sys.exit(f"Could not read {args.segmented}")

    # "pinhole_data" -> "pinholes", "cell_data" -> "cells"; names outputs and labels.
    noun = args.csv.stem.removesuffix("_data") + "s"
    row = load_row(args.csv, args.row)
    diameters, x, y = parse_pinholes(row)
    if len(diameters) == 0:
        sys.exit(f"Row {row['timestamp']} has no {noun}")

    labels, stats, region_ids = match_regions(segmented, x, y)
    matched = region_ids > 0
    area_px = stats[region_ids, cv2.CC_STAT_AREA].astype(float)
    area_px[~matched] = 0.0
    print(f"{args.csv} row {row['timestamp']}: {len(diameters)} {noun}, "
          f"{matched.sum()} matched to a segmented region, {len(set(region_ids[matched]))} distinct regions")
    if not matched.all():
        print(f"  warning: {(~matched).sum()} points fall on a boundary line and aren't drawn")

    pix2mm = args.pix2mm
    if pix2mm is None and matched.any():
        ratio = diameters[matched] / (2.0 * np.sqrt(area_px[matched] / np.pi))
        pix2mm = float(np.median(ratio))
        spread = (np.percentile(ratio, 75) - np.percentile(ratio, 25)) / pix2mm
        print(f"  inferred scale: {pix2mm:.5f} mm/px (IQR {100 * spread:.1f}% of median)")
        if spread > 0.05:
            print("  warning: wide scale spread -- this row may not have been measured on this image")

    noise = np.zeros(len(diameters), bool)
    track_frac = core_frac = tracks = None
    if args.original is not None:
        original = cv2.imread(str(args.original), cv2.IMREAD_GRAYSCALE)
        if original is None:
            sys.exit(f"Could not read {args.original}")
        if original.shape != segmented.shape:
            sys.exit(f"{args.original} is {original.shape[::-1]}, segmented image is {segmented.shape[::-1]}")
        tracks, centers = find_cutter_tracks(original)
        label_track, label_core = region_features(original, labels, tracks)
        track_frac, core_frac = label_track[region_ids], label_core[region_ids]
        noise = matched & is_cutter_noise(track_frac, core_frac)
        print(f"  cutter tracks: {len(centers)} pass(es), {(track_frac >= TRACK_FRAC_MIN).sum()} {noun} on a track, "
              f"{noise.sum()} of them cutter noise (no solid dark core)"
              + (" -- kept, --keep-cutter-noise" if args.keep_cutter_noise else " -- removed"))
    counted = ~noise if not args.keep_cutter_noise else np.ones(len(diameters), bool)
    if not counted.any():
        sys.exit(f"Every {noun[:-1]} in row {row['timestamp']} was flagged as cutter noise")

    # Void objects let the repo's size/spatial helpers work on these regions.
    voids = [Void(i + 1, a, (cx, cy), (0, 0, 0, 0)) for i, (a, cx, cy) in enumerate(zip(area_px, x, y))]
    counted_voids = [v for v, keep in zip(voids, counted) if keep]
    summary = size_summary([v for v in counted_voids if v.area_px > 0], segmented.size, pix2mm)
    spatial_rows = spatial_distribution(counted_voids, segmented.shape, args.grid)
    all_diameters, diameters = diameters, diameters[counted]

    label_cx, label_cy, label_major, label_minor, label_angle = region_shapes(labels)
    major_px, minor_px, angle = label_major[region_ids], label_minor[region_ids], label_angle[region_ids]
    aspect = major_px / np.maximum(minor_px, 1e-9)
    shaped = counted & matched
    anisotropy = anisotropy_summary(major_px[shaped], minor_px[shaped], angle[shaped], area_px[shaped])

    print(f"  diameter (mm): mean {diameters.mean():.3f}  median {np.median(diameters):.3f}  "
          f"std {diameters.std():.3f}  min {diameters.min():.3f}  max {diameters.max():.3f}")
    p10, p25, p75, p90 = np.percentile(diameters, [10, 25, 75, 90])
    print(f"  percentiles (mm): p10 {p10:.3f}  p25 {p25:.3f}  p75 {p75:.3f}  p90 {p90:.3f}")
    print(f"  {noun} cover {summary['porosity_pct']:.2f}% of the image")
    print(f"  spatial distribution ({args.grid}x{args.grid} grid, {noun} per grid square, top row first):")
    for r in range(args.grid):
        cells = [c["void_count"] for c in spatial_rows if c["grid_row"] == r]
        print("    " + "  ".join(f"{n:6d}" for n in cells))
    print(f"  anisotropy: aspect ratio mean {anisotropy['aspect_mean']:.2f}  median {anisotropy['aspect_median']:.2f}  "
          f"p90 {anisotropy['aspect_p90']:.2f}  ({anisotropy['round_count']} near-round, < {ROUND_ASPECT_MAX})")
    print(f"              alignment {anisotropy['alignment']:.2f} (0 random - 1 parallel), direction "
          f"{anisotropy['direction_deg']:.1f} deg (90 = vertical), DA {anisotropy['degree_of_anisotropy']:.2f}")

    out_dir = args.output_dir or args.segmented.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.segmented.stem.removesuffix("_segmented")
    title = f"{args.csv.name} row {row['timestamp']}, n={len(diameters)}, on {args.segmented.name}"
    if noise.any():
        title += f"\n{noise.sum()} cutter-noise detections " + (
            "marked (grey, red outline), still counted" if args.keep_cutter_noise
            else "removed (grey, red outline); cutter tracks tinted")

    overlay = draw_overlay(segmented, labels, region_ids, all_diameters, noise if noise.any() else None, tracks)
    if shaped.sum() <= OUTLINE_MAX_REGIONS:
        ids = region_ids[shaped]
        overlay = draw_major_axes(overlay, label_cx[ids], label_cy[ids], label_major[ids], label_angle[ids])
    overlay = draw_grid_counts(overlay, spatial_rows, args.grid)
    paths = {
        "overlay": out_dir / f"{stem}_{noun}.png",
        "plot": out_dir / f"{stem}_{noun}_plot.png",
        "distribution": out_dir / f"{stem}_{noun}_distribution.png",
        "csv": out_dir / f"{stem}_{noun}.csv",
    }
    cv2.imwrite(str(paths["overlay"]), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    # Colors (and so the colorbar) cover only the regions drawn in color, never the grey noise.
    save_overlay_plot(overlay, all_diameters[~noise], title, paths["plot"])
    save_distribution_plot(diameters, aspect[shaped], angle[shaped], anisotropy, noun, title, paths["distribution"])

    cell_h, cell_w = segmented.shape[0] / args.grid, segmented.shape[1] / args.grid
    with open(paths["csv"], "w", newline="") as f:
        writer = csv.writer(f)
        header = ["id", "x_px", "y_px", "diameter_mm", "area_px", "grid_row", "grid_col",
                  "major_mm", "minor_mm", "aspect_ratio", "orientation_deg"]
        writer.writerow(header + (["on_track_frac", "dark_core_frac", "cutter_noise"] if track_frac is not None else []))
        for i, (v, d) in enumerate(zip(voids, all_diameters)):
            cx, cy = v.centroid
            line = [v.void_id, f"{cx:.3f}", f"{cy:.3f}", f"{d:.3f}", int(v.area_px),
                    min(args.grid - 1, int(cy // cell_h)), min(args.grid - 1, int(cx // cell_w)),
                    f"{major_px[i] * pix2mm:.3f}", f"{minor_px[i] * pix2mm:.3f}", f"{aspect[i]:.3f}", f"{angle[i]:.1f}"]
            if track_frac is not None:
                line += [f"{track_frac[i]:.3f}", f"{core_frac[i]:.3f}", int(noise[i])]
            writer.writerow(line)

    for path in paths.values():
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
