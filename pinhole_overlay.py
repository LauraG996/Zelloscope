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

Outputs (in --output-dir, named after the segmented image and the CSV, e.g.
<image>_pinholes* for pinhole_data.csv, <image>_cells* for cell_data.csv):
  *.png               full-resolution overlay, with the --grid grid and each
                      grid square's count drawn on it
  *_plot.png          the same overlay with title and diameter colorbar
  *_distribution.png  size histogram + cumulative size curve (log diameter axis)
  *.csv               one line per region (id, x, y, diameter, area, grid square)
"""
import argparse
import ast
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

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


def load_row(csv_path: Path, row_index: int | None) -> dict:
    """The row at row_index (Python-style, so -1 is the last row), or the latest timestamp if None."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"No rows found in {csv_path}")
    if row_index is None:
        return max(rows, key=lambda r: r["timestamp"])
    try:
        return rows[row_index]
    except IndexError:
        sys.exit(f"--row {row_index} out of range: {csv_path} has {len(rows)} rows")


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


def diameter_colors(diameters: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    norm = Normalize(diameters.min(), np.percentile(diameters, COLOR_MAX_PERCENTILE))
    return norm, plt.get_cmap("plasma")


def draw_overlay(segmented: np.ndarray, labels: np.ndarray, region_ids: np.ndarray, diameters: np.ndarray) -> np.ndarray:
    """RGB image: faded segmentation with each matched region filled by its diameter color and outlined."""
    norm, cmap = diameter_colors(diameters)
    n_labels = labels.max() + 1
    lut = np.zeros((n_labels, 3), np.uint8)
    is_pinhole = np.zeros(n_labels, bool)
    for region_id, diameter in zip(region_ids, diameters):
        if region_id > 0:
            lut[region_id] = (np.array(cmap(norm(diameter))[:3]) * 255).astype(np.uint8)
            is_pinhole[region_id] = True

    background = 255 - (255 - segmented.astype(float)) * BACKGROUND_OPACITY
    output = np.repeat(background.astype(np.uint8)[:, :, None], 3, axis=2)
    mask = is_pinhole[labels]
    output[mask] = lut[labels][mask]
    if is_pinhole.sum() > OUTLINE_MAX_REGIONS:
        return output

    thickness = max(1, int(round(max(segmented.shape) / 1250)))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, (0, 0, 0), thickness)
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


def save_distribution_plot(diameters: np.ndarray, noun: str, title: str, dst: Path) -> None:
    """Size histogram and cumulative size curve (log diameter axis), side by side."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

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

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("segmented", type=Path, help="Segmented (black boundary / white region) image")
    parser.add_argument("--csv", type=Path, default=Path("pinhole_data.csv"), help="Measurement CSV, e.g. pinhole_data.csv or cell_data.csv (default: pinhole_data.csv)")
    parser.add_argument(
        "--row", type=int, default=None,
        help="Row index to plot, Python-style (-1 = last row in the file); default: latest timestamp",
    )
    parser.add_argument(
        "--pix2mm", type=float, default=None,
        help="mm per pixel; default: inferred from the row's diameters vs the matched regions",
    )
    parser.add_argument(
        "--grid", type=int, default=DEFAULT_GRID_SIZE,
        help=f"Spatial distribution grid size (default: {DEFAULT_GRID_SIZE})",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write outputs (default: next to the segmented image)")
    args = parser.parse_args()

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

    # Void objects let the repo's size/spatial helpers work on these regions.
    voids = [Void(i + 1, a, (cx, cy), (0, 0, 0, 0)) for i, (a, cx, cy) in enumerate(zip(area_px, x, y))]
    summary = size_summary([v for v in voids if v.area_px > 0], segmented.size, pix2mm)
    spatial_rows = spatial_distribution(voids, segmented.shape, args.grid)

    print(f"  diameter (mm): mean {diameters.mean():.3f}  median {np.median(diameters):.3f}  "
          f"std {diameters.std():.3f}  min {diameters.min():.3f}  max {diameters.max():.3f}")
    p10, p25, p75, p90 = np.percentile(diameters, [10, 25, 75, 90])
    print(f"  percentiles (mm): p10 {p10:.3f}  p25 {p25:.3f}  p75 {p75:.3f}  p90 {p90:.3f}")
    print(f"  {noun} cover {summary['porosity_pct']:.2f}% of the image")
    print(f"  spatial distribution ({args.grid}x{args.grid} grid, {noun} per grid square, top row first):")
    for r in range(args.grid):
        cells = [c["void_count"] for c in spatial_rows if c["grid_row"] == r]
        print("    " + "  ".join(f"{n:6d}" for n in cells))

    out_dir = args.output_dir or args.segmented.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.segmented.stem.removesuffix("_segmented")
    title = f"{args.csv.name} row {row['timestamp']}, n={len(diameters)}, on {args.segmented.name}"

    overlay = draw_grid_counts(draw_overlay(segmented, labels, region_ids, diameters), spatial_rows, args.grid)
    paths = {
        "overlay": out_dir / f"{stem}_{noun}.png",
        "plot": out_dir / f"{stem}_{noun}_plot.png",
        "distribution": out_dir / f"{stem}_{noun}_distribution.png",
        "csv": out_dir / f"{stem}_{noun}.csv",
    }
    cv2.imwrite(str(paths["overlay"]), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    save_overlay_plot(overlay, diameters, title, paths["plot"])
    save_distribution_plot(diameters, noun, title, paths["distribution"])

    cell_h, cell_w = segmented.shape[0] / args.grid, segmented.shape[1] / args.grid
    with open(paths["csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x_px", "y_px", "diameter_mm", "area_px", "grid_row", "grid_col"])
        for v, d in zip(voids, diameters):
            cx, cy = v.centroid
            writer.writerow([v.void_id, f"{cx:.3f}", f"{cy:.3f}", f"{d:.3f}", int(v.area_px),
                             min(args.grid - 1, int(cy // cell_h)), min(args.grid - 1, int(cx // cell_w))])

    for path in paths.values():
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
