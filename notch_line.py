#!/usr/bin/env python3
"""Draw a reference line across the object's outer top border.

See straighten.py's find_top_border_shoulder_points for how the two
shoulder points are found. This script just draws the line between them
(extended across the image) instead of using it to rotate the image.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from straighten import IMAGE_EXTS, find_top_border_shoulder_points, largest_foreground_mask


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
