#!/usr/bin/env python3
"""Draw a reference line connecting the two "shoulder" points of a V-notch.

The shoulder points are where the notch void meets the material at its top
(where the notch ceiling meets the left and right legs). Found by tracing
the notch's top boundary profile (topmost background pixel per column) and
walking outward from its center until the profile's local slope steepens
past a threshold -- i.e. until the boundary stops being the roughly flat
ceiling and starts being a leg wall.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from straighten import IMAGE_EXTS, largest_foreground_mask

# Width (px) of the moving-average smoothing applied to the top profile,
# to average out foam-cell texture noise before slope estimation.
SMOOTH_WINDOW = 31
# Column step (px) used to estimate local slope of the top profile.
SLOPE_STEP = 40
# dy/dx magnitude beyond which the profile is considered to have left the
# notch ceiling and entered a leg wall.
SLOPE_THRESHOLD = 0.6


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
    """Locate the two points where the notch void meets the material at its top."""
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


def extend_line_to_edges(p1: tuple[int, int], p2: tuple[int, int], width: int, height: int):
    (x1, y1), (x2, y2) = p1, p2
    if x1 == x2:
        return (x1, 0), (x1, height - 1)
    slope = (y2 - y1) / (x2 - x1)
    y_at = lambda x: y1 + slope * (x - x1)
    return (0, int(round(y_at(0)))), (width - 1, int(round(y_at(width - 1))))


def draw_shoulder_line(image: np.ndarray, extend: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = largest_foreground_mask(gray)
    p1, p2 = find_notch_shoulder_points(mask)

    output = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if extend:
        h, w = gray.shape
        p1, p2 = extend_line_to_edges(p1, p2, w, h)
    cv2.line(output, p1, p2, (0, 0, 255), 3, cv2.LINE_AA)
    return output


def process_file(src: Path, dst: Path, extend: bool) -> None:
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {src}")
    annotated = draw_shoulder_line(image, extend=extend)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), annotated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image file or directory")
    parser.add_argument("output", type=Path, help="Output image file or directory")
    parser.add_argument(
        "--no-extend", action="store_true",
        help="Draw only the segment between the two shoulder points, without extending to image edges",
    )
    args = parser.parse_args()
    extend = not args.no_extend

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"No images found in {args.input}")
        for src in files:
            process_file(src, args.output / src.name, extend)
            print(f"{src.name} -> {args.output / src.name}")
    else:
        process_file(args.input, args.output, extend)
        print(f"{args.input.name} -> {args.output}")


if __name__ == "__main__":
    main()
