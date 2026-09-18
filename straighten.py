#!/usr/bin/env python3
"""Auto-straighten tilted microscopy images by rotating them level.

Segments the bright foreground object from a dark background, then finds
the rotation angle that makes its top boundary as left-right symmetric as
possible (see estimate_symmetry_tilt) and rotates the image by that angle.
Then, on that rotated image, draws both the outer border line (the top
cap/legs transition) and the inner border line (the V-notch's shoulders),
and measures the horizontal distance between the inner shoulders plus the
vertical wall thickness between the outer and inner borders at the
object's center. Works on single files or a whole directory of images;
for a directory, also writes a measurements.csv summary.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def largest_foreground_mask(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove dust/speckle noise and keep only the main connected object.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


# Width (px) of the moving-average smoothing applied to the top profile,
# to average out foam-cell texture noise before slope estimation.
SMOOTH_WINDOW = 31
# Column step (px) used to estimate local slope of the top profile.
SLOPE_STEP = 40
# dy/dx magnitude beyond which the profile is considered to have left the
# rounded top cap and entered a leg's side.
SLOPE_THRESHOLD = 0.6

# straighten(): images whose detected tilt is below this (degrees) are left
# untouched entirely, rather than applying a tiny "correction" that would
# just add padding/interpolation for no real benefit.
DEFAULT_MIN_ROTATION_DEG = 1.0

# estimate_symmetry_tilt(): the mask is downsampled by this factor before the
# angle search, since many candidate rotations are tested and only the
# object's coarse silhouette symmetry matters, not per-pixel precision.
SYMMETRY_SEARCH_SCALE = 0.2
SYMMETRY_COARSE_RANGE_DEG = 20.0
SYMMETRY_COARSE_STEP_DEG = 1.0
SYMMETRY_FINE_STEP_DEG = 0.1


def find_top_border_shoulder_points(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Locate the two points where the object's rounded top cap transitions into its legs.

    Traces the outer top boundary (topmost foreground pixel per column),
    smooths it to average out foam-cell texture noise, and walks outward
    from its peak until the local slope steepens past a threshold.
    """
    ys_all, xs_all = np.where(mask > 0)
    if len(xs_all) == 0:
        raise ValueError("No object found in image")
    x_min, x_max = xs_all.min(), xs_all.max()
    w = x_max - x_min + 1
    if w < 2 * SLOPE_STEP:
        raise ValueError("Object too narrow to reliably locate its top-border shoulders")

    top_row_per_col = np.full(w, -1, dtype=np.int64)
    for col in range(w):
        col_ys = np.where(mask[:, x_min + col] > 0)[0]
        if len(col_ys):
            top_row_per_col[col] = col_ys.min()
    has_fg = top_row_per_col >= 0
    xs = np.where(has_fg)[0] + x_min
    ys = top_row_per_col[has_fg]

    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    ys_smooth = np.convolve(ys.astype(float), kernel, mode="same")

    center = int(np.argmin(ys_smooth))  # peak of the top cap
    left = center
    while left - SLOPE_STEP > 0:
        slope = (ys_smooth[left] - ys_smooth[left - SLOPE_STEP]) / SLOPE_STEP
        if slope < -SLOPE_THRESHOLD:
            break
        left -= 1
    right = center
    while right + SLOPE_STEP < len(xs) - 1:
        slope = (ys_smooth[right + SLOPE_STEP] - ys_smooth[right]) / SLOPE_STEP
        if slope > SLOPE_THRESHOLD:
            break
        right += 1

    left_point = (int(xs[left]), int(round(ys_smooth[left])))
    right_point = (int(xs[right]), int(round(ys_smooth[right])))
    return left_point, right_point


def _top_profile_asymmetry(mask: np.ndarray, angle_deg: float) -> float:
    """Mean squared mismatch between the top boundary and its own mirror image, after rotating by angle_deg.

    Lower is more symmetric. Used to search for the rotation that makes the
    object's top boundary as left-right symmetric as possible -- a global
    shape measure using the whole boundary, rather than a single detected
    corner, so a local texture bump on one side can't dominate the result.
    """
    h, w = mask.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)

    ys, xs = np.where(rotated > 0)
    if len(xs) == 0:
        return float("inf")
    x_min, x_max = xs.min(), xs.max()
    region = rotated[:, x_min:x_max + 1]
    has_fg = region.any(axis=0)
    if not has_fg.any():
        return float("inf")
    top_row_per_col = np.where(has_fg, region.argmax(axis=0), -1)

    center = (x_max - x_min) / 2
    cols = np.arange(0, int(center) + 1, 4)
    mirror_cols = np.round(2 * center - cols).astype(int)
    valid = (mirror_cols >= 0) & (mirror_cols < top_row_per_col.shape[0])
    cols, mirror_cols = cols[valid], mirror_cols[valid]
    valid = (top_row_per_col[cols] >= 0) & (top_row_per_col[mirror_cols] >= 0)
    if not valid.any():
        return float("inf")
    diffs = top_row_per_col[cols[valid]].astype(float) - top_row_per_col[mirror_cols[valid]].astype(float)
    return float(np.mean(diffs ** 2))


def estimate_symmetry_tilt(mask: np.ndarray) -> float:
    """Find the rotation angle (degrees) that makes the object's top boundary most left-right symmetric.

    These objects are inherently bilaterally symmetric (a horseshoe/arch
    shape), so the true tilt is whatever angle best restores that symmetry.
    Searches a coarse grid, then refines around the best candidate.
    """
    small = cv2.resize(
        mask, None, fx=SYMMETRY_SEARCH_SCALE, fy=SYMMETRY_SEARCH_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    best_angle, best_score = 0.0, _top_profile_asymmetry(small, 0.0)
    coarse_angles = np.arange(
        -SYMMETRY_COARSE_RANGE_DEG, SYMMETRY_COARSE_RANGE_DEG + SYMMETRY_COARSE_STEP_DEG,
        SYMMETRY_COARSE_STEP_DEG,
    )
    for angle in coarse_angles:
        score = _top_profile_asymmetry(small, angle)
        if score < best_score:
            best_score, best_angle = score, angle

    fine_angles = np.arange(
        best_angle - SYMMETRY_COARSE_STEP_DEG,
        best_angle + SYMMETRY_COARSE_STEP_DEG + SYMMETRY_FINE_STEP_DEG,
        SYMMETRY_FINE_STEP_DEG,
    )
    for angle in fine_angles:
        score = _top_profile_asymmetry(small, angle)
        if score < best_score:
            best_score, best_angle = score, angle

    return best_angle


def _notch_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No object found in image")
    contour = max(contours, key=cv2.contourArea)

    hull_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(contour), 255)
    defect_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(mask))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    if num_labels <= 1:
        raise ValueError("No notch (concavity) found on the object's contour")
    notch_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    notch_mask = np.where(labels == notch_label, 255, 0).astype(np.uint8)
    x, y, w, h = stats[notch_label, :4]
    return notch_mask, (x, y, w, h)


def find_notch_shoulder_points(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Locate the two "inner" shoulder points, where the notch void meets the material.

    Same profile-walk technique as find_top_border_shoulder_points, applied to
    the notch's own top boundary instead of the object's outer top boundary.
    """
    notch_mask, (x, y, w, h) = _notch_mask(mask)

    region = notch_mask[y:y + h, x:x + w]
    top_row_per_col = region.argmax(axis=0)
    has_fg = region.any(axis=0)
    if has_fg.sum() < 2 * SLOPE_STEP:
        raise ValueError("Notch too small to reliably locate its shoulders")
    xs = np.where(has_fg)[0] + x
    ys = top_row_per_col[has_fg] + y

    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    ys_smooth = np.convolve(ys.astype(float), kernel, mode="same")

    center = len(xs) // 2
    left = center
    while left - SLOPE_STEP > 0:
        slope = (ys_smooth[left] - ys_smooth[left - SLOPE_STEP]) / SLOPE_STEP
        if slope < -SLOPE_THRESHOLD:
            break
        left -= 1
    right = center
    while right + SLOPE_STEP < len(xs) - 1:
        slope = (ys_smooth[right + SLOPE_STEP] - ys_smooth[right]) / SLOPE_STEP
        if slope > SLOPE_THRESHOLD:
            break
        right += 1

    left_point = (int(xs[left]), int(round(ys_smooth[left])))
    right_point = (int(xs[right]), int(round(ys_smooth[right])))
    return left_point, right_point


def inner_shoulder_line_angle(mask: np.ndarray) -> float:
    """Angle (degrees) of the line through the inner (notch) shoulder points, from horizontal."""
    (x1, y1), (x2, y2) = find_notch_shoulder_points(mask)
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def find_notch_wall_points_at_depth(
    mask: np.ndarray, offset_pct: float = 0.0
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Locate the two wall points offset_pct further down each wall than the shoulder corners.

    offset_pct=0 returns the same points as find_notch_shoulder_points.
    offset_pct is a percentage of the notch's total height (e.g. 10 = 10%
    of the way from the shoulder toward the notch's bottom). Each side's
    point is moved straight down by that amount from its own shoulder
    corner, then snapped onto the actual wall at that row (leftmost/
    rightmost notch-void pixel in that row) -- so the line tracks the real
    material edge rather than a straight offset.
    """
    notch_mask, (x, y, w, h) = _notch_mask(mask)
    (lx, ly), (rx, ry) = find_notch_shoulder_points(mask)
    if offset_pct <= 0:
        return (lx, ly), (rx, ry)

    offset_px = int(round(offset_pct / 100.0 * h))
    bottom = y + h - 1
    left_row = min(bottom, ly + offset_px)
    xs = np.where(notch_mask[left_row, :] > 0)[0]
    left_point = (int(xs.min()), left_row) if len(xs) else (lx, ly)

    right_row = min(bottom, ry + offset_px)
    xs = np.where(notch_mask[right_row, :] > 0)[0]
    right_point = (int(xs.max()), right_row) if len(xs) else (rx, ry)

    return left_point, right_point


def inner_shoulder_distance(mask: np.ndarray, offset_pct: float = 0.0) -> float:
    """Euclidean distance (px) between the two inner (notch) points, offset_pct below the shoulders."""
    (x1, y1), (x2, y2) = find_notch_wall_points_at_depth(mask, offset_pct)
    return float(np.hypot(x2 - x1, y2 - y1))


def vertical_wall_thickness(mask: np.ndarray) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """Vertical distance (px) from the outer top surface to the inner notch's top surface.

    Measured at the object's own horizontal center (midpoint of its overall
    bounding box) rather than at the corner points found by
    find_top_border_shoulder_points, since that corner detection can be
    thrown off by local texture -- a plain column scan at the center is far
    more robust and, after straighten()'s symmetry-based rotation, the
    center column is where the top surface and notch are expected to align.

    Returns (thickness_px, outer_point, inner_point).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("No object found in image")
    cx = int((xs.min() + xs.max()) // 2)

    outer_col = np.where(mask[:, cx] > 0)[0]
    if len(outer_col) == 0:
        raise ValueError("Object does not span its own center column")
    outer_y = int(outer_col.min())

    notch_mask, _ = _notch_mask(mask)
    inner_col = np.where(notch_mask[:, cx] > 0)[0]
    if len(inner_col) == 0:
        raise ValueError("Notch does not span the object's center column")
    inner_y = int(inner_col.min())

    return float(inner_y - outer_y), (cx, outer_y), (cx, inner_y)


def rectangular_measurements(mask: np.ndarray) -> dict:
    """For a rectangular specimen with a rectangular hole: wall thickness on all four
    sides, plus the inner hole's width and height.

    Each side's thickness is measured at the center of the hole's own
    bounding box on that axis (e.g. top/bottom thickness at the hole's
    horizontal center, left/right at its vertical center) via a plain
    column/row scan -- robust and shape-agnostic, no corner detection.

    Returns a dict with keys "top", "bottom", "left", "right" (each a
    (thickness_px, outer_point, inner_point) tuple) and "inner_width_px",
    "inner_height_px".
    """
    notch_mask, (nx, ny, nw, nh) = _notch_mask(mask)
    ccx, ccy = nx + nw // 2, ny + nh // 2

    def scan(line: np.ndarray, want_min: bool) -> int | None:
        idx = np.where(line > 0)[0]
        if len(idx) == 0:
            return None
        return int(idx.min()) if want_min else int(idx.max())

    results = {}
    for side, want_min in [("top", True), ("bottom", False)]:
        outer_y = scan(mask[:, ccx], want_min)
        inner_y = scan(notch_mask[:, ccx], want_min)
        if outer_y is None or inner_y is None:
            raise ValueError(f"Object/notch does not span the center column for '{side}'")
        results[side] = (float(abs(inner_y - outer_y)), (ccx, outer_y), (ccx, inner_y))
    for side, want_min in [("left", True), ("right", False)]:
        outer_x = scan(mask[ccy, :], want_min)
        inner_x = scan(notch_mask[ccy, :], want_min)
        if outer_x is None or inner_x is None:
            raise ValueError(f"Object/notch does not span the center row for '{side}'")
        results[side] = (float(abs(inner_x - outer_x)), (outer_x, ccy), (inner_x, ccy))

    results["inner_width_px"] = float(nw)
    results["inner_height_px"] = float(nh)
    return results


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found")
    return max(contours, key=cv2.contourArea)


def _format_measurement(value_px: float, pix2mm: float | None) -> str:
    return f"{value_px * pix2mm:.2f} mm" if pix2mm is not None else f"{value_px:.1f} px"


def _measurement_style(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    scale = max(h, w) / 1500  # so lines/text stay proportional at any resolution
    return {
        "scale": scale,
        "contour_thickness": max(2, int(round(2.5 * scale))),
        "measure_thickness": max(1, int(round(2 * scale))),
        "font_scale": max(0.5, 0.9 * scale),
        "font_thickness": max(1, int(round(1.5 * scale))),
    }


def _draw_border_contours(output: np.ndarray, mask: np.ndarray, notch_mask: np.ndarray, thickness: int) -> None:
    # Outer border contour (green) -- the object's own outline, as actually seen in frame.
    cv2.drawContours(output, [_largest_contour(mask)], -1, (0, 220, 0), thickness, cv2.LINE_AA)
    # Inner border contour (orange) -- the notch/hole's outline.
    cv2.drawContours(output, [_largest_contour(notch_mask)], -1, (0, 140, 255), thickness, cv2.LINE_AA)


def draw_measurements(
    image: np.ndarray,
    mask: np.ndarray,
    inner_points: tuple[tuple[int, int], tuple[int, int]],
    inner_distance_px: float,
    thickness_px: float,
    thickness_points: tuple[tuple[int, int], tuple[int, int]],
    pix2mm: float | None = None,
) -> np.ndarray:
    """Trace the outer and inner border contours, and draw the two measurements."""
    output = image.copy()
    style = _measurement_style(image)
    scale = style["scale"]

    def label(text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
        cv2.putText(
            output, text, pos, cv2.FONT_HERSHEY_SIMPLEX, style["font_scale"], color,
            style["font_thickness"], cv2.LINE_AA,
        )

    notch_mask, _ = _notch_mask(mask)
    _draw_border_contours(output, mask, notch_mask, style["contour_thickness"])

    # Horizontal inner-shoulder distance (cyan).
    p1, p2 = inner_points
    cv2.line(output, p1, p2, (255, 255, 0), style["measure_thickness"], cv2.LINE_AA)
    mx, my = (p1[0] + p2[0]) // 2, min(p1[1], p2[1])
    label(_format_measurement(inner_distance_px, pix2mm), (mx, max(0, my - int(15 * scale))), (255, 255, 0))

    # Vertical wall thickness (yellow), from outer surface down to inner surface.
    outer_pt, inner_pt = thickness_points
    cv2.line(output, outer_pt, inner_pt, (0, 255, 255), style["measure_thickness"], cv2.LINE_AA)
    label(
        _format_measurement(thickness_px, pix2mm),
        (min(image.shape[1] - 10, outer_pt[0] + int(10 * scale)), (outer_pt[1] + inner_pt[1]) // 2),
        (0, 255, 255),
    )

    return output


def draw_square_measurements(
    image: np.ndarray, mask: np.ndarray, measurements: dict, pix2mm: float | None = None,
) -> np.ndarray:
    """Trace the outer/inner contours and draw wall thickness on all four sides."""
    output = image.copy()
    style = _measurement_style(image)
    scale = style["scale"]

    def label(text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
        cv2.putText(
            output, text, pos, cv2.FONT_HERSHEY_SIMPLEX, style["font_scale"], color,
            style["font_thickness"], cv2.LINE_AA,
        )

    notch_mask, _ = _notch_mask(mask)
    _draw_border_contours(output, mask, notch_mask, style["contour_thickness"])

    color = (0, 255, 255)  # yellow for every thickness indicator
    offset = int(12 * scale)
    for side in ("top", "bottom", "left", "right"):
        thickness_px, outer_pt, inner_pt = measurements[side]
        cv2.line(output, outer_pt, inner_pt, color, style["measure_thickness"], cv2.LINE_AA)
        text = _format_measurement(thickness_px, pix2mm)
        mx, my = (outer_pt[0] + inner_pt[0]) // 2, (outer_pt[1] + inner_pt[1]) // 2
        if side in ("top", "bottom"):
            label(text, (mx + offset, my), color)
        else:
            label(text, (mx, my - offset if side == "left" else my + offset * 3), color)

    return output


def rotation_matrix_expand(h: int, w: int, angle_deg: float) -> tuple[np.ndarray, int, int]:
    """Rotation matrix for angle_deg (CCW positive) plus the canvas size needed to avoid cropping."""
    cx, cy = w / 2, h / 2
    matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - cx
    matrix[1, 2] += (new_h / 2) - cy
    return matrix, new_w, new_h


def rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by angle_deg (counter-clockwise positive) expanding the canvas."""
    h, w = image.shape[:2]
    matrix, new_w, new_h = rotation_matrix_expand(h, w, angle_deg)
    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )


def straighten(
    image: np.ndarray, min_rotation_deg: float = DEFAULT_MIN_ROTATION_DEG
) -> tuple[np.ndarray, float]:
    # Levels the object by its own bilateral symmetry (estimate_symmetry_tilt),
    # not by hunting for a single "shoulder corner" -- that approach turned
    # out to be fragile: a local texture bump right next to the top could
    # get mistaken for the true corner, producing a badly wrong angle that
    # even *looked* self-consistent on re-measurement (confirmed by cross-
    # checking against a simple model-free left/right height comparison).
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = largest_foreground_mask(gray)
    angle = estimate_symmetry_tilt(mask)
    if abs(angle) < min_rotation_deg:
        # Already straight enough -- leave the image untouched rather than
        # applying a tiny "correction" that just adds padding/interpolation.
        return image, 0.0
    return rotate_bound(image, angle), angle


class Measurement:
    """Results from process_file (V-notch mode): rotation applied plus both measurements,
    in px and (if pix2mm was given) mm. mm fields are None when no pix2mm was provided."""

    def __init__(
        self, rotation_deg: float,
        inner_distance_px: float, inner_distance_mm: float | None,
        thickness_px: float, thickness_mm: float | None,
    ):
        self.rotation_deg = rotation_deg
        self.inner_distance_px = inner_distance_px
        self.inner_distance_mm = inner_distance_mm
        self.thickness_px = thickness_px
        self.thickness_mm = thickness_mm


class SquareMeasurement:
    """Results from process_file (square mode): rotation applied plus wall thickness on
    all four sides and the inner hole's width/height, in px and (if pix2mm) mm."""

    def __init__(self, rotation_deg: float, measurements: dict, pix2mm: float | None):
        self.rotation_deg = rotation_deg
        self.top_px = measurements["top"][0]
        self.bottom_px = measurements["bottom"][0]
        self.left_px = measurements["left"][0]
        self.right_px = measurements["right"][0]
        self.inner_width_px = measurements["inner_width_px"]
        self.inner_height_px = measurements["inner_height_px"]
        self.pix2mm = pix2mm

    def as_mm(self, value_px: float) -> float | None:
        return value_px * self.pix2mm if self.pix2mm is not None else None


def process_file(
    src: Path,
    dst: Path,
    pix2mm: float | None = None,
    shoulder_offset_pct: float = 0.0,
    min_rotation_deg: float = DEFAULT_MIN_ROTATION_DEG,
    square: bool = False,
) -> Measurement | SquareMeasurement:
    """Straighten src, draw the outer/inner border lines and measurements, save to dst.

    The measurements are drawn and computed on the straightened (rotated)
    image. shoulder_offset_pct moves the inner-distance measurement points
    that percentage of the notch's height down each wall from the detected
    shoulder corner (0 = at the corner itself) -- the wall-thickness
    measurement is unaffected by this and is always taken at the top.
    min_rotation_deg: images tilted less than this are left unrotated
    (rotation_deg will read 0.0 for those). square: use rectangular_measurements
    (wall thickness on all four sides + inner width/height) instead of the
    V-notch-specific shoulder distance and top-only thickness.
    """
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {src}")
    straightened, angle = straighten(image, min_rotation_deg)

    gray = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY) if straightened.ndim == 3 else straightened
    mask = largest_foreground_mask(gray)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if square:
        measurements = rectangular_measurements(mask)
        annotated = draw_square_measurements(straightened, mask, measurements, pix2mm)
        cv2.imwrite(str(dst), annotated)
        return SquareMeasurement(angle, measurements, pix2mm)

    inner_points = find_notch_wall_points_at_depth(mask, shoulder_offset_pct)
    inner_distance_px = float(np.hypot(
        inner_points[1][0] - inner_points[0][0], inner_points[1][1] - inner_points[0][1],
    ))
    thickness_px, outer_thickness_pt, inner_thickness_pt = vertical_wall_thickness(mask)

    inner_distance_mm = inner_distance_px * pix2mm if pix2mm is not None else None
    thickness_mm = thickness_px * pix2mm if pix2mm is not None else None

    annotated = draw_measurements(
        straightened, mask, inner_points, inner_distance_px,
        thickness_px, (outer_thickness_pt, inner_thickness_pt), pix2mm,
    )
    cv2.imwrite(str(dst), annotated)
    return Measurement(angle, inner_distance_px, inner_distance_mm, thickness_px, thickness_mm)


def _format_distance(distance_px: float, distance_mm: float | None) -> str:
    if distance_mm is not None:
        return f"{distance_px:.1f} px ({distance_mm:.2f} mm)"
    return f"{distance_px:.1f} px"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image file or directory")
    parser.add_argument("output", type=Path, help="Output image file or directory")
    parser.add_argument(
        "--pix2mm", type=float, default=None,
        help="Millimeters per pixel, to also report the inner shoulder distance in mm",
    )
    parser.add_argument(
        "--shoulder-offset-pct", type=float, default=0.0,
        help="Move the measurement points down each wall by this percentage of the "
             "notch's height from the detected shoulder corner before measuring "
             "(0 = at the corner itself, e.g. 10 = 10%% of the way down)",
    )
    parser.add_argument(
        "--min-rotation-deg", type=float, default=DEFAULT_MIN_ROTATION_DEG,
        help=f"Images tilted less than this many degrees are left unrotated "
             f"entirely, rather than applying a tiny correction (default: "
             f"{DEFAULT_MIN_ROTATION_DEG})",
    )
    parser.add_argument(
        "--square", action="store_true",
        help="Use this for a rectangular specimen with a rectangular hole: measures wall "
             "thickness on all four sides plus the inner hole's width/height, instead of "
             "the V-notch-specific shoulder distance and top-only thickness",
    )
    args = parser.parse_args()

    def describe(m: Measurement | SquareMeasurement) -> str:
        rotated_note = f"rotated {m.rotation_deg:+.2f} deg" if m.rotation_deg != 0.0 else "left unrotated"
        if isinstance(m, SquareMeasurement):
            sides = ", ".join(
                f"{side} {_format_distance(getattr(m, f'{side}_px'), m.as_mm(getattr(m, f'{side}_px')))}"
                for side in ("top", "bottom", "left", "right")
            )
            return f"{rotated_note}, wall thickness [{sides}]"
        return (
            f"{rotated_note}, "
            f"inner shoulder distance {_format_distance(m.inner_distance_px, m.inner_distance_mm)}, "
            f"wall thickness {_format_distance(m.thickness_px, m.thickness_mm)}"
        )

    def csv_header() -> list[str]:
        if args.square:
            header = ["filename", "rotation_deg", "top_px", "bottom_px", "left_px", "right_px",
                      "inner_width_px", "inner_height_px"]
            if args.pix2mm is not None:
                header += ["top_mm", "bottom_mm", "left_mm", "right_mm", "inner_width_mm", "inner_height_mm"]
            return header
        header = ["filename", "rotation_deg", "inner_shoulder_distance_px", "wall_thickness_px"]
        if args.pix2mm is not None:
            header += ["inner_shoulder_distance_mm", "wall_thickness_mm"]
        return header

    def csv_row(name: str, m: Measurement | SquareMeasurement) -> list[str]:
        if isinstance(m, SquareMeasurement):
            row = [name, f"{m.rotation_deg:.2f}", f"{m.top_px:.1f}", f"{m.bottom_px:.1f}",
                   f"{m.left_px:.1f}", f"{m.right_px:.1f}", f"{m.inner_width_px:.1f}", f"{m.inner_height_px:.1f}"]
            if args.pix2mm is not None:
                row += [
                    f"{m.as_mm(m.top_px):.2f}", f"{m.as_mm(m.bottom_px):.2f}",
                    f"{m.as_mm(m.left_px):.2f}", f"{m.as_mm(m.right_px):.2f}",
                    f"{m.as_mm(m.inner_width_px):.2f}", f"{m.as_mm(m.inner_height_px):.2f}",
                ]
            return row
        row = [name, f"{m.rotation_deg:.2f}", f"{m.inner_distance_px:.1f}", f"{m.thickness_px:.1f}"]
        if args.pix2mm is not None:
            row += [f"{m.inner_distance_mm:.2f}", f"{m.thickness_mm:.2f}"]
        return row

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"No images found in {args.input}")
        csv_path = args.output / "measurements.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header())
            for src in files:
                m = process_file(
                    src, args.output / src.name, args.pix2mm,
                    args.shoulder_offset_pct, args.min_rotation_deg, args.square,
                )
                print(f"{src.name}: {describe(m)} -> {args.output / src.name}")
                writer.writerow(csv_row(src.name, m))
        print(f"Measurements written to {csv_path}")
    else:
        m = process_file(
            args.input, args.output, args.pix2mm,
            args.shoulder_offset_pct, args.min_rotation_deg, args.square,
        )
        print(f"{args.input.name}: {describe(m)} -> {args.output}")


if __name__ == "__main__":
    main()
