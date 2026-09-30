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
detections that sit on a track but have no solid dark core. The stats, grid
counts and distribution use only the rest (--keep-cutter-noise marks them
without dropping them); DRAW_CUTTER_TRACKS / DRAW_IGNORED_CELLS choose
whether the tracks and the dropped cells are drawn.

Shape and angle: each pinhole is measured with an ellipse fitted to its edge
(least-squares fit of an ellipse curve to the edge pixels, OpenCV
fitEllipse) -- the oval that best traces the outline, which follows a slit's
or a rounded void's straight sides the way a hand measurement does. It gives
the pinhole's length and width (the ellipse's major and minor axes), aspect
ratio = length / width (1 = round), and angle = the major axis's angle from
horizontal, counter-clockwise as seen on screen, 0-180 deg (90 = vertical).

Which edge: the segmented outline runs around the outside of each hole's
bright rim, so it's wider than the hole itself -- on the calibration image the
dark hole filled only ~55% of its outline -- which flattens narrow slits'
aspect ratios and can merge a hole with its neighbor. With --original the
ellipse is fitted to the edge of the hole itself: the pixels inside the
outline darker than HOLE_DARK_REL x the local foam background, opened to drop
texture specks, keeping every piece at least HOLE_MIN_PIECE_FRAC of the
largest (debris can split a void's dark floor), with enclosed bright spots
(debris lying in a void) filled in. Without --original it's fitted to the
segmented outline. Diameters always stay as the CSV reports them; the hole's
own equivalent diameter goes in hole_diameter_mm.

Over the whole row it reports:
  alignment    0-1: how parallel the pinholes are (0 = random directions,
               1 = all parallel), the length of the mean of each pinhole's
               doubled-angle direction vector, weighted by area x (1 - width/
               length) so big elongated pinholes count most and near-round
               ones, whose direction is noise, count least
  direction    that weighted mean direction, deg
  DA           degree of anisotropy: length / width of the combined ellipse
               of all the pinholes' ellipses (each weighted by its area),
               i.e. the pore space as a whole; 1 = isotropic
Near-round regions (aspect ratio below ROUND_ASPECT_MAX) are reported but
their angle isn't meaningful. The same measures (alignment, direction,
median aspect ratio, DA) are also worked out per grid square (--grid;
--grid 2 gives the four quadrants), to show whether the orientation changes
across the specimen. The orientation map writes each pinhole's angle next to
it (when there are at most ANGLE_LABEL_MAX of them).

Void area: how much of the specimen is void and how much is background
(solid foam). The specimen is the whole image minus any dark mount around it
(void_analysis.specimen_mask, on the original image; the whole image without
one). Void area = the total area of the counted cells (size filter, cutter
noise dropped), i.e. of their segmented outlines -- no brightness involved.
REPORT_DARK_HOLE_AREA also reports the dark holes inside those cells, which
leave out each hole's bright rim. Background = specimen - void.
Per grid square, each cell's area goes to the square its (x, y) is in.

Outputs (in --output-dir, named after the segmented image and the CSV, e.g.
<image>_pinholes* for pinhole_data.csv, <image>_cells* for cell_data.csv):
  *.png               full resolution: the original with the segmentation's
                      borders drawn clearly on it, nothing else
  *_plot.png          the same overlay with title and diameter colorbar
  *_distribution.png  size histogram + cumulative size curve (log diameter axis),
                      aspect ratio histogram, orientation rose diagram
  *_orientation.png   map of the holes colored by lean from vertical (blue = left,
                      red = right, grey = vertical) with each grid square's average
                      direction as a bar, beside a plain-language summary and a
                      table of direction / lean / alignment / AR / DA / count per
                      square laid out like the grid
  *_area.csv          void vs background area: one line per grid square plus a
                      total line (see Void area below)
  *.csv               one line per region (id, x, y, diameter, area, grid square,
                      length/width mm (edge-fitted ellipse axes), aspect ratio, angle deg,
                      shape_source (hole or outline), hole_diameter_mm;
                      with --original also on_track_frac, dark_core_frac, cutter_noise)
"""
import argparse
import ast
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from cutter_arcs import TRACK_FRAC_MIN, dark_mask, find_cutter_tracks, is_cutter_noise, region_features
from void_analysis import Void, size_summary, spatial_distribution, specimen_mask

# =============================================================================
# SETTINGS -- edit these to change what's counted and how. For a single run,
# the command-line option shown in [brackets] overrides the value here.
# =============================================================================

# Measurement file [--csv]. cell_data.csv lists every segmented cell;
# pinhole_data.csv only the ones already picked out as pinholes.
CSV_FILE = "cell_data.csv"

# Size filter [--min-mm / --max-mm]: only cells whose diameter (as the CSV
# reports it) is within these limits are counted and drawn. None = no limit.
# 0.9 mm is where cells stop fitting the normal foam size distribution
# (about 3.5x the median cell) on the 2026-09-23 sample; 1.3 mm keeps only
# the clearly abnormal voids.
MIN_DIAMETER_MM = 0.9
MAX_DIAMETER_MM = None

# Folder holding the original micrographs [--original to give one directly].
# The one named like the segmented image (e.g. 2026-09-23_13-29-36.png for
# 2026-09-23_13-29-36_segmented.png) is used automatically, for cutter-noise
# removal and for measuring shapes on the dark hole. None = don't look.
ORIGINAL_DIR = "phenolic"

# Remove detections along the cutter tracks that have no real hole
# [--keep-cutter-noise keeps them, only marked]. Needs the original image.
REMOVE_CUTTER_NOISE = True

# What to draw of that: the cutter tracks themselves (pink tint), and the
# cells ignored as cutter noise (grey, red outline). False = leave them out
# of the pictures entirely; they're still ignored in the counts either way.
DRAW_CUTTER_TRACKS = False
DRAW_IGNORED_CELLS = False

# Finding the cutter tracks is the slow part of a run (~50 s). The result is
# saved in this folder the first time and reused while the original image is
# unchanged. None = always recompute.
CACHE_DIR = ".zelloscope_cache"

# Empty area around the specimen: in the segmentation it's one or a few huge
# white regions touching the image edge. Any region touching the edge that's
# bigger than this (mm2) is treated as empty -- never counted as a void and
# left out of the specimen area. (The biggest real void seen so far is ~20 mm2.)
EMPTY_REGION_MIN_MM2 = 50.0

# A void at the specimen's edge can open straight into that empty area (the
# segmentation draws no line across its mouth), making it part of the empty
# region. The specimen's outline is found by closing gaps up to twice this
# size (mm) in its edge; the part of an empty region inside the outline is a
# void and is counted (if it passes the size filter), the rest is empty.
# Bigger = bridges wider void mouths but also turns wide notches in a ragged
# specimen edge into "voids" -- on a ragged edge that happens at 2 mm already.
# 0 = don't look for such voids (the default).
EDGE_VOID_GAP_MM = 0.0

# The CSV can leave out a cell that's much bigger than the rest -- on
# 2026-09-29_13-50-13 it skipped the biggest void (5.4 mm) completely. True =
# any closed segmented region inside the specimen (not touching the image edge)
# that has no CSV point and passes the size filter is added as a void, sized
# from its own area.
INCLUDE_UNLISTED_REGIONS = True

# Stop if the CSV has no row with the segmented image's timestamp, instead of
# falling back to the latest row (which then belongs to a different image and
# gives nonsense). --row still picks a row explicitly.
REQUIRE_MATCHING_ROW = True

# Grid for the per-square counts and orientation [--grid]; 2 = quadrants.
GRID_SIZE = 4

# The dark hole inside a cell: pixels darker than this fraction of the local
# foam brightness. Lower = stricter, a smaller hole (0.35 cuts into the edge,
# 0.6 follows the visible edge).
HOLE_DARK_REL = 0.6

# Holes with an aspect ratio below this count as round: their angle is
# reported but not used for the direction/alignment.
ROUND_ASPECT_MAX = 1.2

# Draw the pictures on the original micrograph with the segmentation's cell
# borders laid over it (needs the original image; False = on the faded
# segmentation alone). Border color (RGB) and how strongly the borders and
# the void fills cover the original (0 = invisible, 1 = opaque).
SUPERIMPOSE_ORIGINAL = True
SEGMENTATION_LINE_COLOR = (255, 200, 0)
SEGMENTATION_LINE_OPACITY = 0.25

# <image>_cells.png: the original with every segmentation border drawn on it
# and nothing else (no void colors, axis lines, grid or labels), for checking
# the segmentation against the real foam. Border color (RGB), opacity
# (1 = solid) and width in px. Without an original it's the segmentation alone.
CHECK_LINE_COLOR = (0, 230, 255)  # cyan: clearest over both black pores and white foam; magenta (255, 0, 200) is next
CHECK_LINE_OPACITY = 0.5
CHECK_LINE_WIDTH_PX = 2
VOID_FILL_OPACITY = 0.7

# Axis names for the specimen directions: image x (left -> right) and image y
# (bottom -> top). Shown on every picture.
X_AXIS_LABEL = "MD"
Y_AXIS_LABEL = "Z"

# How much the original is faded toward white behind the voids on the plot
# (0 = original as is, 1 = white), so the colored voids stand out.
PLOT_BACKGROUND_FADE = 0.55

# Write each hole's angle (deg) next to it on the orientation map.
SHOW_HOLE_ANGLES = True

# Void area. False = the total area of the counted cells (their segmented
# outlines, as the CSV sizes them) -- no brightness involved. True = also
# report the dark-hole area inside those cells (pixels darker than
# HOLE_DARK_REL x the foam around them; needs the original image), which
# leaves out each hole's bright rim.
REPORT_DARK_HOLE_AREA = False

# (Cutter-noise thresholds -- how much of a cell must lie on a track, and how
# much solid dark core a real hole needs -- are TRACK_FRAC_MIN and
# CORE_FRAC_MIN in cutter_arcs.py.)
# =============================================================================

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

# The dark hole (HOLE_DARK_REL above) is opened with this radius to drop texture specks.
HOLE_OPEN_RADIUS_PX = 3
# Dark pieces at least this fraction of the largest one count as part of the
# hole: grey debris lying in a big void can split its dark floor in two, and
# keeping only the largest piece would cut the void short.
HOLE_MIN_PIECE_FRAC = 0.1

# Per-pinhole angle labels on the orientation map, up to this many pinholes (RGB).
ANGLE_LABEL_MAX = 400
ANGLE_LABEL_COLOR = (70, 70, 70)


# Major-axis line drawn through each colored region on the overlay (RGB).
AXIS_COLOR = (255, 255, 255)

# Cyclic colormap for orientation (0 and 180 deg are the same direction) and
# the per-square mean-direction bars on the orientation map (RGB).
# Orientation map colors: how far each hole leans from vertical, blue = to the
# left, red = to the right, grey = vertical; LEAN_MAX_DEG or more is full color.
# (Diverging blue <-> red around a neutral grey; the grey is darker than a
# page-grey so vertical holes still stand out from the faded background.)
LEAN_MAX_DEG = 30
LEAN_COLORS = ["#1c5cab", "#6da7ec", "#a9a7a0", "#ec8a89", "#c4302f"]
DIRECTION_BAR_COLOR = (30, 30, 30)
MAP_GRID_COLOR = (150, 150, 150)
INK = "#222222"
MUTED_INK = "#666666"


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


def load_row(csv_path: Path, selector: str | None, image_timestamp: str | None = None) -> dict:
    """The row picked by selector; if None, the row timestamped image_timestamp, else the latest.

    selector is either a row index, Python-style (0 = first row, -1 = last), or
    text matched against the timestamps -- a full timestamp, or any part of one
    that picks out a single row, e.g. "14:19:36" or "2026-09-23 15:19".
    """
    rows = read_rows(csv_path)
    if selector is None:
        same_capture = [r for r in rows if image_timestamp and r["timestamp"].strip() == image_timestamp]
        if same_capture:
            return same_capture[-1]
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


def fit_edge_ellipse(mask: np.ndarray) -> tuple[float, ...] | None:
    """Ellipse fitted to the edge of mask (all its outer contours): (cx, cy, length, width, angle deg).

    cx, cy are in mask coords; angle is the long axis from horizontal,
    counter-clockwise on screen, in [0, 180). None if the edge has fewer than
    the 5 points an ellipse fit needs.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    edge = np.vstack([c[:, 0] for c in contours]).astype(np.float32)
    if len(edge) < 5:
        return None
    (cx, cy), (axis_1, axis_2), rotation = cv2.fitEllipse(edge)
    # OpenCV's rotation is axis_1's direction, clockwise in image coords (y
    # down); turn it into the long axis's counter-clockwise screen angle.
    long_axis_rotation = rotation if axis_1 >= axis_2 else rotation + 90
    return cx, cy, max(axis_1, axis_2), min(axis_1, axis_2), (-long_axis_rotation) % 180


def pinhole_ellipses(
    labels: np.ndarray, stats: np.ndarray, region_ids: np.ndarray, original: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Edge-fitted ellipse for each point's pinhole (see module docstring for which edge).

    Returns per-point arrays (cx, cy, length px, width px, angle deg, area px,
    from_hole bool); NaN where there's no region or no fittable edge. area is
    the measured shape's pixel area (the hole's with original, else the outline's).
    """
    dark = dark_mask(original, HOLE_DARK_REL) if original is not None else None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HOLE_OPEN_RADIUS_PX + 1,) * 2)
    out = np.full((len(region_ids), 7), np.nan)
    cache = {}
    for i, label in enumerate(region_ids):
        if label <= 0:
            continue
        if label not in cache:
            x0, y0, w, h, _ = stats[label]
            region = (labels[y0:y0 + h, x0:x0 + w] == label).astype(np.uint8)
            shape, from_hole = region, False
            if dark is not None:
                hole = cv2.morphologyEx(region & dark[y0:y0 + h, x0:x0 + w], cv2.MORPH_OPEN, kernel)
                n, pieces, piece_stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
                if n > 1:
                    areas = piece_stats[1:, cv2.CC_STAT_AREA]
                    hole = np.isin(pieces, np.flatnonzero(areas >= HOLE_MIN_PIECE_FRAC * areas.max()) + 1).astype(np.uint8)
                    # Fill enclosed bright spots: flood the outside from a padded border, keep what it can't reach.
                    padded = np.pad(hole, 1)
                    outside = padded.copy()
                    cv2.floodFill(outside, None, (0, 0), 1)
                    shape, from_hole = (padded | (1 - outside))[1:-1, 1:-1], True
            fit = fit_edge_ellipse(shape)
            cache[label] = None if fit is None else (x0 + fit[0], y0 + fit[1], *fit[2:], shape.sum(), from_hole)
        if cache[label] is not None:
            out[i] = cache[label]
    cx, cy, length, width, angle, area, from_hole = out.T
    return cx, cy, length, width, angle, area, from_hole == 1


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


def orientation_by_square(
    x: np.ndarray, y: np.ndarray, major: np.ndarray, minor: np.ndarray, angle: np.ndarray, area: np.ndarray,
    image_shape: tuple[int, int], grid_size: int,
) -> list[dict]:
    """anisotropy_summary for the regions in each grid square (by (x, y)), row-major; None where empty."""
    h, w = image_shape
    rows = np.minimum(grid_size - 1, (y // (h / grid_size)).astype(int))
    cols = np.minimum(grid_size - 1, (x // (w / grid_size)).astype(int))
    squares = []
    for r in range(grid_size):
        for c in range(grid_size):
            inside = (rows == r) & (cols == c)
            summary = anisotropy_summary(major[inside], minor[inside], angle[inside], area[inside]) if inside.any() else None
            squares.append({"grid_row": r, "grid_col": c, "count": int(inside.sum()), "anisotropy": summary})
    return squares


def lean_deg(angle: np.ndarray | float) -> np.ndarray | float:
    """Lean from vertical, deg: > 0 = the top of the hole leans right, < 0 = left."""
    return 90.0 - np.asarray(angle)


def lean_text(angle: float) -> str:
    lean = float(lean_deg(angle))
    return "vertical" if abs(lean) < 2 else f"{abs(lean):.0f}\u00b0 {'right' if lean > 0 else 'left'} of vertical"


def lean_colormap():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    return LinearSegmentedColormap.from_list("lean", LEAN_COLORS), Normalize(-LEAN_MAX_DEG, LEAN_MAX_DEG)


def orientation_colors(angle: np.ndarray) -> np.ndarray:
    """RGB uint8 color for each orientation (deg, 0-180): its lean from vertical (see LEAN_COLORS)."""
    cmap, norm = lean_colormap()
    lean = np.nan_to_num(lean_deg(angle))
    return (cmap(norm(np.clip(lean, -LEAN_MAX_DEG, LEAN_MAX_DEG)))[:, :3] * 255).astype(np.uint8)


def draw_square_directions(image: np.ndarray, squares: list[dict], grid_size: int) -> np.ndarray:
    """Light grid lines, plus in each square a bar along its mean direction (length = alignment).

    The numbers for each square go in the table beside the map, not on it.
    """
    output = image.copy()
    h, w = output.shape[:2]
    cell_h, cell_w = h / grid_size, w / grid_size
    scale = max(h, w) / 1500

    line_thickness = max(1, int(round(2 * scale)))
    for i in range(1, grid_size):
        cv2.line(output, (0, int(round(i * cell_h))), (w, int(round(i * cell_h))), MAP_GRID_COLOR, line_thickness, cv2.LINE_AA)
        cv2.line(output, (int(round(i * cell_w)), 0), (int(round(i * cell_w)), h), MAP_GRID_COLOR, line_thickness, cv2.LINE_AA)

    bar_thickness = max(2, int(round(5 * scale)))
    for square in squares:
        summary = square["anisotropy"]
        if summary is None:
            continue
        cx, cy = (square["grid_col"] + 0.5) * cell_w, (square["grid_row"] + 0.5) * cell_h
        half = 0.4 * min(cell_w, cell_h) * summary["alignment"]
        theta = np.radians(summary["direction_deg"])
        dx, dy = half * np.cos(theta), -half * np.sin(theta)
        p0, p1 = (int(round(cx - dx)), int(round(cy - dy))), (int(round(cx + dx)), int(round(cy + dy)))
        cv2.line(output, p0, p1, (255, 255, 255), bar_thickness * 3, cv2.LINE_AA)
        cv2.line(output, p0, p1, DIRECTION_BAR_COLOR, bar_thickness, cv2.LINE_AA)
    return output


def save_orientation_plot(
    image: np.ndarray, squares: list[dict], anisotropy: dict, grid_size: int, noun: str, title: str, dst: Path,
) -> None:
    """Orientation map beside a plain-language summary and a per-square table laid out like the grid."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cmap, norm = lean_colormap()
    fig = plt.figure(figsize=(17, 9.6), dpi=150)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.05)
    right = outer[1].subgridspec(3, 1, height_ratios=[0.27, 0.66, 0.07], hspace=0.14)

    ax_map = fig.add_subplot(outer[0])
    ax_map.imshow(image)
    label_axes(ax_map)
    ax_map.set_title("Where the holes point", fontsize=13, color=INK, loc="left")

    # Plain-language summary for the whole specimen.
    ax_text = fig.add_subplot(right[0])
    ax_text.set_axis_off()
    a = anisotropy
    lines = [
        ("Shape", f"Holes are typically {a['aspect_median']:.1f}x longer than wide (median aspect ratio, AR)."),
        ("Direction", f"On average they point {a['direction_deg']:.0f}\u00b0 -- {lean_text(a['direction_deg'])}."),
        ("Alignment", f"{100 * a['alignment']:.0f}% aligned (0% = random directions, 100% = all parallel)."),
        ("Pore space", f"As a whole {a['degree_of_anisotropy']:.1f}x longer along that direction than across it "
                       f"(DA; 1 = no direction)."),
    ]
    ax_text.text(0, 1.0, "Whole specimen", fontsize=13, color=INK, weight="bold", va="top", transform=ax_text.transAxes)
    for k, (label, text) in enumerate(lines):
        y = 0.78 - 0.2 * k
        ax_text.text(0, y, label, fontsize=10.5, color=MUTED_INK, va="top", transform=ax_text.transAxes)
        ax_text.text(0.17, y, text, fontsize=10.5, color=INK, va="top", transform=ax_text.transAxes, wrap=True)

    # Per-square table, same layout as the map's grid.
    ax_tab = fig.add_subplot(right[1])
    ax_tab.set_xlim(0, grid_size)
    ax_tab.set_ylim(grid_size, 0)
    ax_tab.set_aspect("equal")
    ax_tab.set_axis_off()
    ax_tab.set_title("Each grid square (same layout as the map)", fontsize=12, color=INK, loc="left")
    big = 15 if grid_size <= 2 else 11 if grid_size <= 4 else 8
    small = big * 0.72
    for square in squares:
        r, c = square["grid_row"], square["grid_col"]
        ax_tab.add_patch(Rectangle((c + 0.03, r + 0.03), 0.94, 0.94, facecolor="#fcfcfb", edgecolor="#cccccc", lw=1))
        summary = square["anisotropy"]
        if summary is None:
            ax_tab.text(c + 0.5, r + 0.5, "no holes", ha="center", va="center", fontsize=small, color=MUTED_INK)
            continue
        direction = summary["direction_deg"]
        # A mini direction bar, colored by lean like the map, length = alignment.
        theta = np.radians(direction)
        half = 0.13 * summary["alignment"]
        x0, y0 = c + 0.2, r + 0.3
        ax_tab.plot([x0 - half * np.cos(theta), x0 + half * np.cos(theta)],
                    [y0 + half * np.sin(theta), y0 - half * np.sin(theta)],
                    color=cmap(norm(np.clip(lean_deg(direction), -LEAN_MAX_DEG, LEAN_MAX_DEG))),
                    lw=4, solid_capstyle="round")
        ax_tab.text(c + 0.36, r + 0.3, f"{direction:.0f}\u00b0", fontsize=big, color=INK, weight="bold", va="center")
        ax_tab.text(c + 0.1, r + 0.52, lean_text(direction).replace(" of vertical", ""), fontsize=small, color=MUTED_INK,
                    va="center")
        ax_tab.text(c + 0.1, r + 0.68, f"align {summary['alignment']:.2f}   n = {square['count']}", fontsize=small,
                    color=INK, va="center")
        ax_tab.text(c + 0.1, r + 0.84, f"AR {summary['aspect_median']:.1f}   DA {summary['degree_of_anisotropy']:.1f}",
                    fontsize=small, color=INK, va="center")

    # Lean color key.
    ax_key = fig.add_subplot(right[2])
    key = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_key, orientation="horizontal",
                       ticks=[-LEAN_MAX_DEG, -LEAN_MAX_DEG / 2, 0, LEAN_MAX_DEG / 2, LEAN_MAX_DEG])
    key.ax.set_xticklabels([f"{LEAN_MAX_DEG}\u00b0+ left", f"{LEAN_MAX_DEG // 2}\u00b0 left", "vertical",
                            f"{LEAN_MAX_DEG // 2}\u00b0 right", f"{LEAN_MAX_DEG}\u00b0+ right"], fontsize=9, color=INK)
    key.set_label(f"Hole color = lean from vertical.  Black bar on the map = a square's average direction "
                  f"(longer = more aligned).", fontsize=9, color=MUTED_INK)
    key.outline.set_edgecolor("#cccccc")

    fig.suptitle(title, fontsize=10, color=MUTED_INK, x=0.01, ha="left")
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def draw_angle_labels(
    image: np.ndarray, cx: np.ndarray, cy: np.ndarray, width: np.ndarray, angle: np.ndarray,
) -> np.ndarray:
    """Write each pinhole's angle (deg) just to the right of it."""
    output = image.copy()
    scale = max(image.shape[:2]) / 1500
    font_scale, font_thickness = 0.75 * scale, max(1, int(round(1.6 * scale)))
    for x0, y0, w, a in zip(cx, cy, width, angle):
        text = f"{a:.0f}"
        (_, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        org = (int(round(x0 + w / 2 + 8 * scale)), int(round(y0 + text_h / 2)))
        cv2.putText(output, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness * 4, cv2.LINE_AA)
        cv2.putText(output, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, ANGLE_LABEL_COLOR, font_thickness, cv2.LINE_AA)
    return output


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
    noise: np.ndarray | None = None, tracks: np.ndarray | None = None, colors: np.ndarray | None = None,
    original: np.ndarray | None = None,
) -> np.ndarray:
    """RGB image: each matched region filled by its diameter color and outlined, over a background.

    The background is the faded segmentation, or -- given original (the
    grayscale micrograph) -- the micrograph with the segmentation's cell
    borders laid over it in SEGMENTATION_LINE_COLOR, and the fills then only
    VOID_FILL_OPACITY opaque so the holes show through. Regions flagged in
    noise (per point, parallel to region_ids) are drawn in NOISE_FILL with a
    NOISE_OUTLINE outline instead; tracks, if given, is tinted. colors (per
    point RGB) replaces the diameter coloring.
    """
    if noise is None:
        noise = np.zeros(len(region_ids), bool)
    norm, cmap = diameter_colors(diameters[~noise] if (~noise).any() else diameters)
    n_labels = labels.max() + 1
    lut = np.zeros((n_labels, 3), np.uint8)
    is_pinhole = np.zeros(n_labels, bool)
    is_noise = np.zeros(n_labels, bool)
    for i, (region_id, diameter, flagged) in enumerate(zip(region_ids, diameters, noise)):
        if region_id > 0:
            is_noise[region_id] = flagged
            is_pinhole[region_id] = not flagged
            if flagged:
                lut[region_id] = NOISE_FILL
            elif colors is not None:
                lut[region_id] = colors[i]
            else:
                lut[region_id] = (np.array(cmap(norm(diameter))[:3]) * 255).astype(np.uint8)

    mask = (is_pinhole | is_noise)[labels]
    if original is None:
        background = 255 - (255 - segmented.astype(float)) * BACKGROUND_OPACITY
        output = np.repeat(background.astype(np.uint8)[:, :, None], 3, axis=2)
        fill_opacity = 1.0
    else:
        output = np.repeat(original[:, :, None], 3, axis=2).astype(np.float32)
        borders = segmented < 128
        output[borders] = ((1 - SEGMENTATION_LINE_OPACITY) * output[borders]
                           + SEGMENTATION_LINE_OPACITY * np.array(SEGMENTATION_LINE_COLOR, np.float32))
        output = output.astype(np.uint8)
        fill_opacity = VOID_FILL_OPACITY
    if tracks is not None:
        output[tracks] = np.minimum(output[tracks], TRACK_TINT)
    output[mask] = ((1 - fill_opacity) * output[mask] + fill_opacity * lut[labels][mask]).astype(np.uint8)

    thickness = max(1, int(round(max(segmented.shape) / 1250)))
    for flags, color in ((is_pinhole, (0, 0, 0)), (is_noise, NOISE_OUTLINE)):
        if 0 < flags.sum() <= OUTLINE_MAX_REGIONS:
            contours, _ = cv2.findContours(flags[labels].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(output, contours, -1, color, thickness * (2 if color == NOISE_OUTLINE else 1))
    return output


def label_axes(ax) -> None:
    """Frame an image axes with the specimen direction labels instead of hiding it."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#999999")
    ax.set_xlabel(f"\u2190  {X_AXIS_LABEL}  \u2192", fontsize=18, weight="bold", labelpad=8)
    ax.set_ylabel(f"\u2190  {Y_AXIS_LABEL}  \u2192", fontsize=18, weight="bold", labelpad=8)


def draw_axis_key(image: np.ndarray) -> np.ndarray:
    """Double-headed MD / Z arrows in the image's bottom-left corner (for the full-resolution outputs)."""
    output = image.copy()
    h, w = output.shape[:2]
    scale = max(h, w) / 1500
    length, thick = int(170 * scale), max(2, int(round(4 * scale)))
    gap = int(45 * scale)  # space between the two arrows and the corner
    cx, cy = gap + length // 2 + int(40 * scale), h - gap - int(40 * scale)  # centre of the MD arrow
    zx, zy = gap, h - gap - int(40 * scale) - length // 2                     # centre of the Z arrow
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 1.8 * scale, max(2, int(round(4 * scale)))
    md = ((cx - length // 2, cy), (cx + length // 2, cy))
    z = ((zx, zy + length // 2), (zx, zy - length // 2))
    for color, width in (((255, 255, 255), thick * 3), ((0, 0, 0), thick)):
        for p0, p1 in (md, z):
            cv2.arrowedLine(output, p0, p1, color, width, cv2.LINE_AA, tipLength=0.14)
            cv2.arrowedLine(output, p1, p0, color, width, cv2.LINE_AA, tipLength=0.14)
    (mw, mh), _ = cv2.getTextSize(X_AXIS_LABEL, font, fs, ft)
    (zw, zh), _ = cv2.getTextSize(Y_AXIS_LABEL, font, fs, ft)
    labels = ((X_AXIS_LABEL, (cx - mw // 2, cy - int(18 * scale))),
              (Y_AXIS_LABEL, (zx + int(18 * scale), zy + zh // 2)))
    for text, org in labels:
        cv2.putText(output, text, org, font, fs, (255, 255, 255), ft * 4, cv2.LINE_AA)
        cv2.putText(output, text, org, font, fs, (0, 0, 0), ft, cv2.LINE_AA)
    return output


def draw_grid_lines(image: np.ndarray, grid_size: int) -> np.ndarray:
    """Thin grey grid lines, no labels."""
    output = image.copy()
    h, w = output.shape[:2]
    thickness = max(1, int(round(2 * max(h, w) / 1500)))
    for i in range(1, grid_size):
        y, x = int(round(i * h / grid_size)), int(round(i * w / grid_size))
        cv2.line(output, (0, y), (w, y), MAP_GRID_COLOR, thickness, cv2.LINE_AA)
        cv2.line(output, (x, 0), (x, h), MAP_GRID_COLOR, thickness, cv2.LINE_AA)
    return output


# Per-square table shading: void % of the square, light -> dark blue.
VOID_PCT_COLORS = ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]


def save_overlay_plot(
    overlay: np.ndarray, diameters: np.ndarray, title: str, dst: Path, info: list[str] | None = None,
    area_rows: list[dict] | None = None, grid_size: int = 4, noun: str = "cells",
) -> None:
    """Void map (colored by diameter) beside a summary and a per-square table of count and void %.

    info: summary sentences, an empty string, then the detailed number lines.
    area_rows: void_area_table's rows (per square, then the total).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    info = info or []
    split = info.index("") if "" in info else len(info)
    sentences, details = info[:split], info[split + 1:]

    norm, cmap = diameter_colors(diameters)
    fig = plt.figure(figsize=(17, 9.8), dpi=150)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.06)
    left = outer[0].subgridspec(2, 1, height_ratios=[0.955, 0.045], hspace=0.12)
    right = outer[1].subgridspec(3, 1, height_ratios=[0.2, 0.58, 0.22], hspace=0.12)

    ax_map = fig.add_subplot(left[0])
    ax_map.imshow(overlay)
    label_axes(ax_map)
    ax_map.set_title("Where the voids are (color = size)", fontsize=13, color=INK, loc="left")
    ax_bar = fig.add_subplot(left[1])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_bar, orientation="horizontal",
                       extend="max" if norm.vmax < diameters.max() else "neither")
    bar.set_label("Void diameter (mm)", fontsize=10, color=INK)
    bar.outline.set_edgecolor("#cccccc")

    ax_text = fig.add_subplot(right[0])
    ax_text.set_axis_off()
    ax_text.text(0, 1.0, "Whole specimen", fontsize=13, color=INK, weight="bold", va="top", transform=ax_text.transAxes)
    for k, line in enumerate(sentences):
        ax_text.text(0, 0.78 - 0.2 * k, line, fontsize=10.5, color=INK, va="top", transform=ax_text.transAxes)

    ax_tab = fig.add_subplot(right[1])
    ax_tab.set_xlim(0, grid_size)
    ax_tab.set_ylim(grid_size, 0)
    ax_tab.set_aspect("equal")
    ax_tab.set_axis_off()
    ax_tab.set_title("Each grid square (same layout as the map)", fontsize=12, color=INK, loc="left")
    if area_rows:
        squares = area_rows[:-1]
        top = max(max(a["void_cell_pct"] for a in squares), 1e-9)
        shade = LinearSegmentedColormap.from_list("void_pct", VOID_PCT_COLORS)
        shade_norm = Normalize(0, top)
        big = 20 if grid_size <= 2 else 15 if grid_size <= 4 else 10
        for a in squares:
            r, c = a["grid_row"], a["grid_col"]
            level = shade_norm(a["void_cell_pct"])
            ax_tab.add_patch(Rectangle((c + 0.03, r + 0.03), 0.94, 0.94, facecolor=shade(level), edgecolor="#cccccc"))
            ink = "white" if level > 0.6 else INK
            ax_tab.text(c + 0.5, r + 0.42, f"{a['void_cell_pct']:.1f}%", ha="center", va="center",
                        fontsize=big, weight="bold", color=ink)
            ax_tab.text(c + 0.5, r + 0.7, f"{a['cells']} {noun}", ha="center", va="center",
                        fontsize=big * 0.6, color=ink)
    ax_tab.text(0, grid_size + 0.12, "Big number = void % of the square; darker = more void.",
                fontsize=9, color=MUTED_INK, va="top")

    ax_det = fig.add_subplot(right[2])
    ax_det.set_axis_off()
    ax_det.text(0, 0.9, "\n".join(line.replace("   (% under each grid count = void % of that square)", "")
                                   for line in details),
                fontsize=9.5, family="monospace", color=MUTED_INK, va="top", linespacing=1.5, transform=ax_det.transAxes)

    fig.suptitle(title, fontsize=10, color=MUTED_INK, x=0.01, ha="left")
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def draw_grid_counts(
    image: np.ndarray, spatial_rows: list[dict], grid_size: int, void_pct: list[float] | None = None,
) -> np.ndarray:
    """Draw the grid_size x grid_size grid on image, each square labeled with its count.

    void_pct (row-major, one per square), if given, is written under each count.
    """
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
        # A light box behind the label keeps it readable over the micrograph.
        box_w, box_bottom = text_w, cy
        if void_pct is not None:
            label = f"{void_pct[row['grid_row'] * grid_size + row['grid_col']]:.2f}% void"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * font_scale, font_thickness)
            box_w, box_bottom = max(text_w, label_w), cy + int(1.8 * label_h) + label_h // 2
        pad = int(10 * scale)
        mid = int(round((row["grid_col"] + 0.5) * cell_w))
        x0, x1 = max(0, mid - box_w // 2 - pad), min(w, mid + box_w // 2 + pad)
        y0, y1 = max(0, cy - text_h - pad), min(h, box_bottom + pad)
        output[y0:y1, x0:x1] = (0.45 * output[y0:y1, x0:x1] + 0.55 * 255).astype(np.uint8)
        cv2.putText(output, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    font_thickness * 4, cv2.LINE_AA)
        cv2.putText(output, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, GRID_COLOR,
                    font_thickness, cv2.LINE_AA)
        if void_pct is not None:
            small = 0.45 * font_scale
            label = f"{void_pct[row['grid_row'] * grid_size + row['grid_col']]:.2f}% void"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, small, font_thickness)
            org = (int(round((row["grid_col"] + 0.5) * cell_w)) - label_w // 2, cy + int(1.8 * label_h))
            cv2.putText(output, label, org, cv2.FONT_HERSHEY_SIMPLEX, small, (255, 255, 255),
                        font_thickness * 4, cv2.LINE_AA)
            cv2.putText(output, label, org, cv2.FONT_HERSHEY_SIMPLEX, small, GRID_COLOR, font_thickness, cv2.LINE_AA)
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
    ax4.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax4.set_xticklabels([X_AXIS_LABEL, "45\u00b0", Y_AXIS_LABEL, "135\u00b0", X_AXIS_LABEL, "225\u00b0",
                         Y_AXIS_LABEL, "315\u00b0"])
    ax4.set_title(f"Orientation (0\u00b0 = along {X_AXIS_LABEL}, 90\u00b0 = along {Y_AXIS_LABEL})\n"
                  f"alignment {anisotropy['alignment']:.2f}, direction {anisotropy['direction_deg']:.0f} deg, "
                  f"DA {anisotropy['degree_of_anisotropy']:.2f}", fontsize=10)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def void_area_table(
    specimen: np.ndarray, x: np.ndarray, y: np.ndarray, outline_px: np.ndarray, hole_px: np.ndarray | None,
    grid_size: int, pix2mm: float,
) -> list[dict]:
    """Specimen / void / background area per grid square plus a "total" row (see module docstring).

    x, y, outline_px (and hole_px, NaN-free, or None) describe the counted
    cells; specimen is the bool specimen mask.
    """
    h, w = specimen.shape
    cell_h, cell_w = h / grid_size, w / grid_size
    rows_of = np.minimum(grid_size - 1, (y // cell_h).astype(int))
    cols_of = np.minimum(grid_size - 1, (x // cell_w).astype(int))
    mm2 = pix2mm ** 2
    table = []
    squares = [(r, c) for r in range(grid_size) for c in range(grid_size)] + [("total", "total")]
    for r, c in squares:
        if r == "total":
            inside, spec_px = np.ones(len(x), bool), int(specimen.sum())
        else:
            inside = (rows_of == r) & (cols_of == c)
            y0, y1 = int(round(r * cell_h)), int(round((r + 1) * cell_h))
            x0, x1 = int(round(c * cell_w)), int(round((c + 1) * cell_w))
            spec_px = int(specimen[y0:y1, x0:x1].sum())
        entry = {"grid_row": r, "grid_col": c, "cells": int(inside.sum()),
                 "specimen_px": spec_px, "specimen_mm2": spec_px * mm2}
        for kind, px in (("cell", outline_px), ("hole", hole_px)):
            if px is None:
                continue
            void = float(px[inside].sum())
            entry.update({f"void_{kind}_px": int(void), f"void_{kind}_mm2": void * mm2,
                          f"void_{kind}_pct": 100 * void / max(spec_px, 1),
                          f"background_{kind}_px": int(spec_px - void),
                          f"background_{kind}_mm2": (spec_px - void) * mm2})
        table.append(entry)
    return table


def cached_cutter_tracks(original: np.ndarray, original_path: Path) -> tuple[np.ndarray, list]:
    """find_cutter_tracks, saved to CACHE_DIR and reused while original_path is unchanged."""
    if not CACHE_DIR:
        return find_cutter_tracks(original)
    stat = original_path.stat()
    # The file's size and modification time identify this version of the image;
    # the cutter_arcs settings are included so changing them recomputes.
    from cutter_arcs import DEFAULT_RIDGE_THRESHOLD, MAX_PASSES
    key = f"{original_path.stem}_{stat.st_size}_{int(stat.st_mtime)}_{DEFAULT_RIDGE_THRESHOLD}_{MAX_PASSES}"
    cache = Path(CACHE_DIR) / f"{key}_cutter_tracks.npz"
    if cache.exists():
        data = np.load(cache)
        print(f"  cutter tracks: reused from {cache}")
        return np.unpackbits(data["tracks"])[: original.size].reshape(original.shape).astype(bool), \
            [tuple(c) for c in data["centers"]]
    tracks, centers = find_cutter_tracks(original)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, tracks=np.packbits(tracks.ravel()), centers=np.array(centers).reshape(-1, 2))
    return tracks, centers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("segmented", type=Path, help="Segmented (black boundary / white region) image")
    parser.add_argument("--csv", type=Path, default=Path(CSV_FILE),
                        help=f"Measurement CSV, e.g. cell_data.csv or pinhole_data.csv (default: {CSV_FILE})")
    parser.add_argument("--min-mm", type=float, default=MIN_DIAMETER_MM,
                        help=f"Only count cells at least this wide, mm (default: {MIN_DIAMETER_MM}); 0 = no limit")
    parser.add_argument("--max-mm", type=float, default=MAX_DIAMETER_MM,
                        help=f"Only count cells at most this wide, mm (default: {MAX_DIAMETER_MM})")
    parser.add_argument(
        "--row", default=None,
        help="Row to plot: an index (0 = first row, -1 = last) or a timestamp or part of one, "
             "e.g. \"14:19:36\"; default: the row with the segmented image's timestamp "
             "(from its filename), else the latest. See --list-rows",
    )
    parser.add_argument("--list-rows", action="store_true", help="List the CSV's rows (index, timestamp, count) and exit")
    parser.add_argument(
        "--pix2mm", type=float, default=None,
        help="mm per pixel; default: inferred from the row's diameters vs the matched regions",
    )
    parser.add_argument(
        "--grid", type=int, default=GRID_SIZE,
        help=f"Spatial distribution grid size (default: {GRID_SIZE})",
    )
    parser.add_argument(
        "--original", type=Path, default=None,
        help="Micrograph the segmentation was made from; enables cutter-noise removal and hole-edge shapes "
             "(default: found in ORIGINAL_DIR by the segmented image's name)",
    )
    parser.add_argument(
        "--keep-cutter-noise", action="store_true",
        help="With --original: mark cutter-noise detections but keep them in the counts and stats",
    )
    parser.add_argument("--no-original", action="store_true", help="Don't use an original image even if one is found")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write outputs (default: next to the segmented image)")
    args = parser.parse_args()

    if args.list_rows:
        list_rows(args.csv)
        return

    segmented = cv2.imread(str(args.segmented), cv2.IMREAD_GRAYSCALE)
    if segmented is None:
        sys.exit(f"Could not read {args.segmented}")

    stem = args.segmented.stem.removesuffix("_segmented")
    if args.no_original:
        args.original = None
    elif args.original is None and ORIGINAL_DIR:
        found = sorted(p for p in Path(ORIGINAL_DIR).glob(f"{stem}.*") if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".bmp"))
        args.original = found[0] if found else None
    if args.original is not None:
        print(f"original image: {args.original}")
    if not REMOVE_CUTTER_NOISE:
        args.keep_cutter_noise = True

    # "pinhole_data" -> "pinholes", "cell_data" -> "cells"; names outputs and labels.
    noun = args.csv.stem.removesuffix("_data") + "s"
    # A filename like 2026-09-23_13-29-36_segmented.png names its capture time.
    parts = stem.split("_")
    image_timestamp = f"{parts[0]} {parts[1].replace('-', ':')}" if len(parts) >= 2 else None
    row = load_row(args.csv, args.row, image_timestamp)
    if args.row is None and image_timestamp and row["timestamp"].strip() != image_timestamp:
        message = (f"no {args.csv} row timestamped {image_timestamp} (the segmented image's time); the latest row "
                   f"is {row['timestamp']}, which is from a different image. Add this image's row to {args.csv}, "
                   f"or pick one with --row (see --list-rows).")
        if REQUIRE_MATCHING_ROW:
            sys.exit("error: " + message)
        print("warning: " + message + " Using the latest row anyway.")
    diameters, x, y = parse_pinholes(row)
    n_row = len(diameters)
    all_x, all_y = x.copy(), y.copy()
    lower = args.min_mm if args.min_mm else -np.inf
    upper = args.max_mm if args.max_mm is not None else np.inf
    in_range = (diameters >= lower) & (diameters <= upper)
    diameters, x, y = diameters[in_range], x[in_range], y[in_range]
    if args.min_mm and args.max_mm is not None:
        size_filter = f"{args.min_mm:g}-{args.max_mm:g} mm"
    elif args.min_mm:
        size_filter = f">= {args.min_mm:g} mm"
    elif args.max_mm is not None:
        size_filter = f"<= {args.max_mm:g} mm"
    else:
        size_filter = ""
    if size_filter:
        print(f"{args.csv} row {row['timestamp']}: {n_row} {noun}, {len(diameters)} of them {size_filter}")
    if len(diameters) == 0:
        sys.exit(f"Row {row['timestamp']} has no {noun}")

    labels, stats, region_ids = match_regions(segmented, x, y)

    # Empty regions: huge white regions touching the image edge (see EMPTY_REGION_MIN_MM2).
    # The scale isn't known yet, so size them with --pix2mm or the usual 0.014 mm/px.
    h_img, w_img = segmented.shape
    rx, ry, rw, rh, ra = (stats[:, k] for k in range(5))
    on_edge = (rx <= 0) | (ry <= 0) | (rx + rw >= w_img) | (ry + rh >= h_img)
    scale_guess = args.pix2mm or 0.01402
    empty_label = on_edge & (ra * scale_guess ** 2 >= EMPTY_REGION_MIN_MM2)
    empty_label[0] = False
    in_empty = empty_label[region_ids]
    # Each region counts once, however many CSV points fall in it.
    first = np.zeros(len(region_ids), bool)
    first[np.unique(region_ids, return_index=True)[1]] = True
    duplicate = (region_ids > 0) & ~first
    if in_empty.any() or duplicate.any():
        print(f"  dropped {in_empty.sum()} {noun} in the empty area around the specimen and "
              f"{(duplicate & ~in_empty).sum()} repeat points in an already-counted region")
    keep = ~in_empty & ~duplicate
    listed = np.zeros(stats.shape[0], bool)
    listed[region_ids] = True
    diameters, x, y, region_ids = diameters[keep], x[keep], y[keep], region_ids[keep]
    if INCLUDE_UNLISTED_REGIONS:
        # listed covers every CSV point, before the size filter, so a region the
        # CSV sized below --min-mm isn't re-added from its (bigger) outline area.
        csv_ids = match_regions(segmented, all_x, all_y)[2]
        listed[csv_ids] = True
        eq_mm = 2 * np.sqrt(ra / np.pi) * scale_guess
        unlisted = np.flatnonzero(~listed & ~on_edge & (eq_mm >= lower) & (eq_mm <= upper))
        unlisted = unlisted[unlisted > 0]
        if len(unlisted):
            pts = []
            for k in unlisted:
                # A point inside the region (its centroid may fall outside a curved shape).
                x0, y0, w0, h0, _ = stats[k]
                ys_k, xs_k = np.nonzero(labels[y0:y0 + h0, x0:x0 + w0] == k)
                cx0, cy0 = xs_k.mean(), ys_k.mean()
                j = np.argmin((xs_k - cx0) ** 2 + (ys_k - cy0) ** 2)
                pts.append((eq_mm[k], x0 + xs_k[j], y0 + ys_k[j], k))
            d_add, x_add, y_add, id_add = map(np.array, zip(*pts))
            diameters, x, y = np.concatenate([diameters, d_add]), np.concatenate([x, x_add]), np.concatenate([y, y_add])
            region_ids = np.concatenate([region_ids, id_add.astype(region_ids.dtype)])
            print(f"  added {len(pts)} region(s) the CSV left out (closed, inside the specimen): "
                  + ", ".join(f"{v:.2f} mm at ({xx:.0f}, {yy:.0f})" for v, xx, yy, _ in pts))

    # The specimen outline: everything that isn't empty, with gaps in its edge
    # up to 2 x EDGE_VOID_GAP_MM closed (done at 1/4 scale for speed).
    empty_px = empty_label[labels]
    outline = ~empty_px
    if EDGE_VOID_GAP_MM and empty_px.any():
        f = 4
        small = cv2.resize((~empty_px).astype(np.uint8), (w_img // f, h_img // f), interpolation=cv2.INTER_AREA)
        radius = max(1, int(round(EDGE_VOID_GAP_MM / scale_guess / f)))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        # Pad so closing isn't cut short at the frame, then fill enclosed holes.
        padded = cv2.copyMakeBorder((small > 0).astype(np.uint8), radius, radius, radius, radius, cv2.BORDER_CONSTANT, 0)
        closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, disk)
        # Flood from the padded border: empty area touching any image edge is outside.
        cv2.floodFill(closed, np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), np.uint8), (0, 0), 2)
        closed = (closed != 2)[radius:-radius, radius:-radius].astype(np.uint8)
        outline = cv2.resize(closed, (w_img, h_img), interpolation=cv2.INTER_NEAREST).astype(bool)
        # Pieces of the empty region inside the outline are voids opening onto the edge.
        edge_voids = (empty_px & outline).astype(np.uint8)
        edge_voids = cv2.morphologyEx(edge_voids, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n_ev, ev_labels, ev_stats, ev_cent = cv2.connectedComponentsWithStats(edge_voids, connectivity=4)
        added = []
        for k in range(1, n_ev):
            d_mm = 2 * np.sqrt(ev_stats[k, cv2.CC_STAT_AREA] / np.pi) * scale_guess
            if not (lower <= d_mm <= upper):
                continue
            new_label = stats.shape[0]
            labels[ev_labels == k] = new_label
            stats = np.vstack([stats, ev_stats[k]])
            empty_label = np.append(empty_label, False)
            # A point inside the piece (its centroid may fall outside a curved shape).
            ys_k, xs_k = np.nonzero(ev_labels == k)
            j = np.argmin((xs_k - ev_cent[k][0]) ** 2 + (ys_k - ev_cent[k][1]) ** 2)
            added.append((d_mm, float(xs_k[j]), float(ys_k[j]), new_label))
        if added:
            d_add, x_add, y_add, id_add = map(np.array, zip(*added))
            diameters, x, y = np.concatenate([diameters, d_add]), np.concatenate([x, x_add]), np.concatenate([y, y_add])
            region_ids = np.concatenate([region_ids, id_add.astype(region_ids.dtype)])
            print(f"  found {len(added)} void(s) opening onto the specimen edge (split off the empty area): "
                  + ", ".join(f"{v:.2f} mm" for v in d_add))
    if len(diameters) == 0:
        sys.exit(f"Row {row['timestamp']} has no {noun} inside the specimen")
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
        tracks, centers = cached_cutter_tracks(original, args.original)
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

    axis_cx, axis_cy, major_px, minor_px, angle, shape_area, from_hole = pinhole_ellipses(
        labels, stats, region_ids, original if args.original is not None else None)
    fitted = ~np.isnan(major_px) & (minor_px > 0)
    hole_area = np.where(from_hole, shape_area, np.nan)
    if args.original is not None:
        print(f"  edge-fitted ellipse on the dark hole for {(from_hole & counted).sum()} of {counted.sum()} {noun} "
              f"(the rest on the segmented outline); holes fill "
              f"{np.nanmedian(hole_area[counted] / np.maximum(area_px[counted], 1)):.0%} of their outline (median)")
    aspect = major_px / np.maximum(minor_px, 1e-9)
    shaped = counted & matched & fitted
    anisotropy = anisotropy_summary(major_px[shaped], minor_px[shaped], angle[shaped], shape_area[shaped])
    squares = orientation_by_square(x[shaped], y[shaped], major_px[shaped], minor_px[shaped], angle[shaped],
                                    shape_area[shaped], segmented.shape, args.grid)

    print(f"  diameter (mm): mean {diameters.mean():.3f}  median {np.median(diameters):.3f}  "
          f"std {diameters.std():.3f}  min {diameters.min():.3f}  max {diameters.max():.3f}")
    p10, p25, p75, p90 = np.percentile(diameters, [10, 25, 75, 90])
    print(f"  percentiles (mm): p10 {p10:.3f}  p25 {p25:.3f}  p75 {p75:.3f}  p90 {p90:.3f}")
    specimen = (specimen_mask(original) > 0) if args.original is not None else np.ones(segmented.shape, bool)
    specimen &= outline
    area_rows = void_area_table(
        specimen, x[counted & matched], y[counted & matched], area_px[counted & matched],
        np.nan_to_num(hole_area[counted & matched]) if REPORT_DARK_HOLE_AREA and args.original is not None else None,
        args.grid, pix2mm)
    total = area_rows[-1]
    print(f"  area: specimen {total['specimen_px']} px = {total['specimen_mm2']:.1f} mm2"
          + (f" (empty area around it left out: {100 * (1 - specimen.mean()):.1f}% of the image)"
             if not specimen.all() else " (the whole image)"))
    for kind, label in (("cell", "cell area"), ("hole", "dark holes in them")):
        if f"void_{kind}_px" in total:
            ratio = total[f"background_{kind}_px"] / max(total[f"void_{kind}_px"], 1)
            print(f"    void, {label:<24} {total[f'void_{kind}_px']:>9} px = {total[f'void_{kind}_mm2']:7.2f} mm2 "
                  f"= {total[f'void_{kind}_pct']:5.2f}% of specimen;  background "
                  f"{total[f'background_{kind}_mm2']:.1f} mm2;  void : background = 1 : {ratio:.0f}")
    main_kind = "cell"
    print(f"  void % of specimen per grid square (cell area, top row first):")
    for r in range(args.grid):
        print("    " + "  ".join(f"{a[f'void_{main_kind}_pct']:6.2f}" for a in area_rows[r * args.grid:(r + 1) * args.grid]))
    print(f"  spatial distribution ({args.grid}x{args.grid} grid, {noun} per grid square, top row first):")
    for r in range(args.grid):
        cells = [c["void_count"] for c in spatial_rows if c["grid_row"] == r]
        print("    " + "  ".join(f"{n:6d}" for n in cells))
    print(f"  anisotropy: aspect ratio mean {anisotropy['aspect_mean']:.2f}  median {anisotropy['aspect_median']:.2f}  "
          f"p90 {anisotropy['aspect_p90']:.2f}  ({anisotropy['round_count']} near-round, < {ROUND_ASPECT_MAX})")
    print(f"              alignment {anisotropy['alignment']:.2f} (0 random - 1 parallel), direction "
          f"{anisotropy['direction_deg']:.1f} deg (90 = vertical), DA {anisotropy['degree_of_anisotropy']:.2f}")
    print(f"  anisotropy per grid square (direction deg / alignment / median aspect ratio / DA / count, top row first):")
    for r in range(args.grid):
        cells = []
        for square in squares[r * args.grid:(r + 1) * args.grid]:
            a = square["anisotropy"]
            cells.append(f"{a['direction_deg']:5.1f} / {a['alignment']:.2f} / {a['aspect_median']:.2f} / "
                         f"{a['degree_of_anisotropy']:.2f} / {square['count']:<4d}" if a else f"{'-':^34}")
        print("    " + "   ".join(cells))

    out_dir = args.output_dir or args.segmented.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (f"{args.csv.name} row {row['timestamp']}, n={len(diameters)}"
             + (f" {noun} {size_filter}" if size_filter else "") + f", on {args.segmented.name}")
    title += "\nvoid area " + ", ".join(
        f"{total[f'void_{kind}_pct']:.2f}%" + (f" ({label})" if f"void_hole_pct" in total else "")
        for kind, label in (("cell", "cell area"), ("hole", "dark holes"))
        if f"void_{kind}_pct" in total) + " of specimen"
    if noise.any():
        if args.keep_cutter_noise:
            title += f"\n{noise.sum()} cutter-noise detections marked (grey, red outline), still counted"
        else:
            title += f"\n{noise.sum()} {noun} on cutter tracks ignored" + (
                " (grey, red outline)" if DRAW_IGNORED_CELLS else " (not drawn)")

    # Ignored cutter-noise cells are drawn grey only if asked (or when they're
    # kept in the counts); otherwise their region id is blanked so they're skipped.
    show_noise = noise.any() and (args.keep_cutter_noise or DRAW_IGNORED_CELLS)
    draw_ids = region_ids if show_noise or not noise.any() else np.where(noise, 0, region_ids)
    draw_noise = noise if show_noise else None
    draw_tracks = tracks if DRAW_CUTTER_TRACKS else None
    backdrop = original if (SUPERIMPOSE_ORIGINAL and args.original is not None) else None
    # Plot map: the original lightened so the colored voids stand out, no axis lines.
    light = None if backdrop is None else (PLOT_BACKGROUND_FADE * 255 + (1 - PLOT_BACKGROUND_FADE) * backdrop).astype(np.uint8)
    overlay = draw_overlay(segmented, labels, draw_ids, all_diameters, draw_noise, draw_tracks, original=light)
    orientation_map = draw_overlay(segmented, labels, draw_ids, all_diameters, draw_noise, draw_tracks, original=backdrop,
                                   colors=orientation_colors(angle))
    if shaped.sum() <= OUTLINE_MAX_REGIONS:
        orientation_map = draw_major_axes(orientation_map, axis_cx[shaped], axis_cy[shaped], major_px[shaped], angle[shaped])
    if SHOW_HOLE_ANGLES and shaped.sum() <= ANGLE_LABEL_MAX:
        orientation_map = draw_angle_labels(orientation_map, axis_cx[shaped], axis_cy[shaped], minor_px[shaped], angle[shaped])
    orientation_map = draw_square_directions(orientation_map, squares, args.grid)
    overlay = draw_grid_lines(overlay, args.grid)
    paths = {
        "overlay": out_dir / f"{stem}_{noun}.png",
        "plot": out_dir / f"{stem}_{noun}_plot.png",
        "distribution": out_dir / f"{stem}_{noun}_distribution.png",
        "orientation": out_dir / f"{stem}_{noun}_orientation.png",
        "csv": out_dir / f"{stem}_{noun}.csv",
        "area": out_dir / f"{stem}_{noun}_area.csv",
    }
    base = original if args.original is not None else np.full(segmented.shape, 255, np.uint8)
    check = np.repeat(base[:, :, None], 3, axis=2).astype(np.float32)
    borders = (segmented < 128).astype(np.uint8)
    if CHECK_LINE_WIDTH_PX > 1:
        borders = cv2.dilate(borders, np.ones((CHECK_LINE_WIDTH_PX, CHECK_LINE_WIDTH_PX), np.uint8))
    borders = borders.astype(bool)
    line_color = np.array(CHECK_LINE_COLOR if args.original is not None else (0, 0, 0), np.float32)
    check[borders] = (1 - CHECK_LINE_OPACITY) * check[borders] + CHECK_LINE_OPACITY * line_color
    cv2.imwrite(str(paths["overlay"]), cv2.cvtColor(draw_axis_key(check.astype(np.uint8)), cv2.COLOR_RGB2BGR))
    # Colors (and so the colorbar) cover only the regions drawn in color, never the grey noise.
    # Plain-language summary first, then the numbers.
    squares_only = area_rows[:-1]
    worst = max(squares_only, key=lambda a: a["void_cell_pct"])
    empty = sum(a["cells"] == 0 for a in squares_only)
    counted_area = area_px[counted & matched]
    biggest_share = 100 * counted_area.max() / max(counted_area.sum(), 1) if len(counted_area) else 0.0
    where = ("top" if worst["grid_row"] == 0 else "bottom" if worst["grid_row"] == args.grid - 1 else
             f"row {worst['grid_row'] + 1}") + ", " + (
             "left" if worst["grid_col"] == 0 else "right" if worst["grid_col"] == args.grid - 1 else
             f"column {worst['grid_col'] + 1}")
    info = [f"{total['void_cell_pct']:.2f}% of the specimen is void; {100 - total['void_cell_pct']:.2f}% is solid foam.",
            f"That is {total['void_cell_mm2']:.1f} mm2 of void in {total['specimen_mm2']:.0f} mm2 of specimen: "
            f"1 mm2 of void for every {total['background_cell_px'] / max(total['void_cell_px'], 1):.0f} mm2 of foam.",
            f"The largest single void ({diameters.max():.2f} mm) is {biggest_share:.0f}% of all the void area.",
            f"Most void-rich grid square: {where}, {worst['void_cell_pct']:.1f}% void"
            + (f"; {empty} square{'s' if empty != 1 else ''} with no voids." if empty else "."),
            "",
            f"Void vs background  (specimen {total['specimen_mm2']:.1f} mm2 = {total['specimen_px']:,} px)",
            f"{'Void (' + str(len(diameters)) + ' ' + noun + (' ' + size_filter if size_filter else '') + '):':<28}"
            f"{total['void_cell_mm2']:9.2f} mm2 = {total['void_cell_pct']:6.2f} %",
            f"{'Background (foam):':<28}{total['background_cell_mm2']:9.2f} mm2 = {100 - total['void_cell_pct']:6.2f} %",
            f"Void : background = 1 : {total['background_cell_px'] / max(total['void_cell_px'], 1):.0f}"
            f"   (% under each grid count = void % of that square)"]
    if "void_hole_pct" in total:
        info.append(f"Dark holes only: {total['void_hole_mm2']:.2f} mm2 = {total['void_hole_pct']:.2f} %, "
                    f"void : background = 1 : {total['background_hole_px'] / max(total['void_hole_px'], 1):.0f}")
    print("  summary:")
    for line in info[:info.index("")]:
        print(f"    {line}")
    save_overlay_plot(overlay, all_diameters[~noise], title, paths["plot"], info, area_rows, args.grid, noun)
    save_orientation_plot(orientation_map, squares, anisotropy, args.grid, noun, title, paths["orientation"])
    save_distribution_plot(diameters, aspect[shaped], angle[shaped], anisotropy, noun, title, paths["distribution"])

    cell_h, cell_w = segmented.shape[0] / args.grid, segmented.shape[1] / args.grid
    with open(paths["csv"], "w", newline="") as f:
        writer = csv.writer(f)
        header = ["id", "x_px", "y_px", "diameter_mm", "area_px", "grid_row", "grid_col",
                  "length_mm", "width_mm", "aspect_ratio", "angle_deg", "shape_source", "hole_diameter_mm"]
        writer.writerow(header + (["on_track_frac", "dark_core_frac", "cutter_noise"] if track_frac is not None else []))
        for i, (v, d) in enumerate(zip(voids, all_diameters)):
            cx, cy = v.centroid
            line = [v.void_id, f"{cx:.3f}", f"{cy:.3f}", f"{d:.3f}", int(v.area_px),
                    min(args.grid - 1, int(cy // cell_h)), min(args.grid - 1, int(cx // cell_w)),
                    f"{major_px[i] * pix2mm:.3f}", f"{minor_px[i] * pix2mm:.3f}", f"{aspect[i]:.3f}", f"{angle[i]:.1f}",
                    "hole" if from_hole[i] else "outline",
                    f"{2 * np.sqrt(hole_area[i] / np.pi) * pix2mm:.3f}" if from_hole[i] else ""]
            if track_frac is not None:
                line += [f"{track_frac[i]:.3f}", f"{core_frac[i]:.3f}", int(noise[i])]
            writer.writerow(line)

    with open(paths["area"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(area_rows[-1].keys()))
        writer.writeheader()
        for entry in area_rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in entry.items()})

    for path in paths.values():
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
