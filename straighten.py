#!/usr/bin/env python3
"""Auto-straighten tilted microscopy images by rotating them level.

Segments the bright foreground object from a dark background, finds the two
points where its outer top cap transitions into the legs (see
find_top_border_shoulder_points), and rotates the image so the line between
them is horizontal. Then, on that rotated image, measures the distance
between the two "inner" shoulder points where the V-notch meets the
material (see find_notch_shoulder_points) -- a different pair of points
from the ones used for rotation. Works on single files or a whole
directory of images; for a directory, also writes a measurements.csv
summary.
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

# straighten(): stop iterating once the residual angle is below this (degrees),
# or after this many rotation passes, whichever comes first.
STRAIGHTEN_TOLERANCE_DEG = 0.1
MAX_STRAIGHTEN_ITERATIONS = 5


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


def shoulder_line_angle(mask: np.ndarray) -> float:
    """Angle (degrees) of the line through the top-border shoulder points, from horizontal."""
    (x1, y1), (x2, y2) = find_top_border_shoulder_points(mask)
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


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


def draw_inner_shoulder_measurement(
    image: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    distance_px: float,
    pix2mm: float | None = None,
) -> np.ndarray:
    """Draw the two inner shoulder points, the segment between them, and a distance label."""
    output = image.copy()
    scale = max(image.shape[:2]) / 1500  # so markers stay visible at any resolution
    radius = max(6, int(round(8 * scale)))
    thickness = max(2, int(round(3 * scale)))

    cv2.line(output, p1, p2, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.circle(output, p1, radius, (0, 255, 0), -1)
    cv2.circle(output, p2, radius, (0, 255, 0), -1)

    label = f"{distance_px:.1f}px"
    if pix2mm is not None:
        label += f" / {distance_px * pix2mm:.2f}mm"
    mx, my = (p1[0] + p2[0]) // 2, min(p1[1], p2[1])
    font_scale = max(0.6, 1.2 * scale)
    cv2.putText(
        output, label, (mx, max(0, my - int(20 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), max(1, int(round(2 * scale))), cv2.LINE_AA,
    )
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


def straighten(image: np.ndarray) -> tuple[np.ndarray, float]:
    # Levels by the OUTER top-border shoulder line (not the inner notch line --
    # the inner shoulder distance is measured separately, after this rotation,
    # by process_file). The detector scans by image column, so its result is
    # itself somewhat orientation-dependent: a single rotation by the measured
    # angle doesn't always fully zero out the residual tilt. Iterate a few
    # times, re-measuring on each rotated result, until it converges.
    current = image
    total_angle = 0.0
    for _ in range(MAX_STRAIGHTEN_ITERATIONS):
        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim == 3 else current
        mask = largest_foreground_mask(gray)
        angle = shoulder_line_angle(mask)
        if abs(angle) < STRAIGHTEN_TOLERANCE_DEG:
            break
        # Rotating by the measured angle itself brings the shoulder line to horizontal
        # (cv2's rotation direction convention already matches atan2's here).
        current = rotate_bound(current, angle)
        total_angle += angle
    return current, total_angle


def process_file(
    src: Path, dst: Path, pix2mm: float | None = None, shoulder_offset_pct: float = 0.0
) -> tuple[float, float, float | None]:
    """Straighten src, draw the inner-shoulder measurement on it, save to dst.

    Returns (rotation_deg, inner_shoulder_distance_px, inner_shoulder_distance_mm).
    The measurement is drawn and computed on the straightened (rotated) image.
    shoulder_offset_pct moves the measurement points that percentage of the
    notch's height down each wall from the detected shoulder corner (0 = at
    the corner itself).
    """
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {src}")
    straightened, angle = straighten(image)

    gray = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY) if straightened.ndim == 3 else straightened
    mask = largest_foreground_mask(gray)
    p1, p2 = find_notch_wall_points_at_depth(mask, shoulder_offset_pct)
    distance_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    distance_mm = distance_px * pix2mm if pix2mm is not None else None

    annotated = draw_inner_shoulder_measurement(straightened, p1, p2, distance_px, pix2mm)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), annotated)
    return angle, distance_px, distance_mm


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
    args = parser.parse_args()

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"No images found in {args.input}")
        csv_path = args.output / "measurements.csv"
        header = ["filename", "rotation_deg", "inner_shoulder_distance_px"]
        if args.pix2mm is not None:
            header.append("inner_shoulder_distance_mm")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for src in files:
                angle, distance_px, distance_mm = process_file(
                    src, args.output / src.name, args.pix2mm, args.shoulder_offset_pct
                )
                print(
                    f"{src.name}: rotated {angle:+.2f} deg, "
                    f"inner shoulder distance {_format_distance(distance_px, distance_mm)} "
                    f"-> {args.output / src.name}"
                )
                row = [src.name, f"{angle:.2f}", f"{distance_px:.1f}"]
                if distance_mm is not None:
                    row.append(f"{distance_mm:.2f}")
                writer.writerow(row)
        print(f"Measurements written to {csv_path}")
    else:
        angle, distance_px, distance_mm = process_file(
            args.input, args.output, args.pix2mm, args.shoulder_offset_pct
        )
        print(
            f"{args.input.name}: rotated {angle:+.2f} deg, "
            f"inner shoulder distance {_format_distance(distance_px, distance_mm)} -> {args.output}"
        )


if __name__ == "__main__":
    main()
