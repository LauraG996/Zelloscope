#!/usr/bin/env python3
"""Draw a reference line across the object's outer top border.

Traces the outer top boundary (topmost foreground pixel per column across
the whole object), smooths it to average out foam-cell texture noise, and
walks outward from its peak until the local slope steepens past a
threshold -- i.e. until the boundary leaves the rounded top cap and enters
a leg's relatively straight side. The line connects those two points.
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
# rounded top cap and entered a leg's side.
SLOPE_THRESHOLD = 0.6


def find_top_border_shoulder_points(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Locate the two points where the rounded top cap transitions into the legs."""
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
    p1, p2 = find_top_border_shoulder_points(mask)

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
