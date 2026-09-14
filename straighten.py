#!/usr/bin/env python3
"""Auto-straighten tilted microscopy images by rotating them level.

Segments the bright foreground object from a dark background, finds the two
points where its outer top cap transitions into the legs (see
find_top_border_shoulder_points), and rotates the image so the line between
them is horizontal. Works on single files or a whole directory of images.
"""
import argparse
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = largest_foreground_mask(gray)
    angle = shoulder_line_angle(mask)
    # Rotating by the measured angle itself brings the shoulder line to horizontal
    # (cv2's rotation direction convention already matches atan2's here).
    return rotate_bound(image, angle), angle


def process_file(src: Path, dst: Path) -> float:
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {src}")
    straightened, angle = straighten(image)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), straightened)
    return angle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image file or directory")
    parser.add_argument("output", type=Path, help="Output image file or directory")
    args = parser.parse_args()

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"No images found in {args.input}")
        for src in files:
            angle = process_file(src, args.output / src.name)
            print(f"{src.name}: rotated {angle:+.2f} deg -> {args.output / src.name}")
    else:
        angle = process_file(args.input, args.output)
        print(f"{args.input.name}: rotated {angle:+.2f} deg -> {args.output}")


if __name__ == "__main__":
    main()
