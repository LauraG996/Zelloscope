#!/usr/bin/env python3
"""Count the big, dark holes (voids) in a material micrograph.

A hole is a patch clearly darker than its immediate surroundings (see
void_analysis.detect_void_mask for how); --min-diameter-px (or
--min-diameter-mm, given --pix2mm) sets what counts as "big" -- smaller dark
specks (material grain texture, noise) are ignored. Holes touching the image
frame are excluded by default, since they're cut off and don't reflect the
hole's true size -- pass --include-edge-holes to count them anyway. Prints
one line per image with the count, and optionally saves an annotated copy
with each counted hole outlined.

For material whose own texture is grainy enough to be mistaken for holes
(e.g. a CLAHE-processed image), pass --median-blur-k 25 --bg-kernel-frac 0.10
--min-solidity 0.65 -- see void_analysis.py's help for what each one does.
Solidity above ~0.7 starts rejecting real voids with a slightly irregular
(non-convex) outline, e.g. a curved or gently notched shape -- 0.65 is a
safer default that still rejects sprawling grain-texture clusters.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from straighten import IMAGE_EXTS
from void_analysis import detect_void_mask, draw_void_contours, find_voids, specimen_mask

# What counts as "big" by default: holes narrower than this (px) are ignored
# as texture/noise. Tune to your image's scale and what you consider a hole.
DEFAULT_MIN_DIAMETER_PX = 15.0


def _drop_edge_holes(holes: list, labels: np.ndarray, width: int, height: int) -> tuple[list, np.ndarray]:
    """Remove holes whose bounding box touches the image frame -- they're cut
    off there, so their true size/shape isn't fully visible in this image."""
    kept = []
    kept_labels = labels.copy()
    for hole in holes:
        x, y, w, h = hole.bbox
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            kept_labels[kept_labels == hole.void_id] = 0
        else:
            kept.append(hole)
    return kept, kept_labels


def count_holes(
    image: np.ndarray,
    min_diameter_px: float,
    median_blur_k: int = 0,
    bg_kernel_frac: float = 0.025,
    min_solidity: float = 0.0,
    ignore_edges: bool = True,
) -> tuple[list, np.ndarray]:
    """Detect holes at least min_diameter_px wide in image.

    ignore_edges drops any hole touching the image frame (see _drop_edge_holes).
    Returns (holes, labels) -- see void_analysis.find_voids.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    region = specimen_mask(gray)
    border_px = max(5, int(round(min(gray.shape) * 0.003)) | 1)
    border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_px * 2 + 1,) * 2)
    region = cv2.erode(region, border_kernel)

    mask = detect_void_mask(gray, region, bg_kernel_frac=bg_kernel_frac, median_blur_k=median_blur_k)
    min_area_px = np.pi * (min_diameter_px / 2) ** 2
    holes, labels = find_voids(mask, min_area_px=min_area_px, min_solidity=min_solidity)

    if ignore_edges:
        height, width = gray.shape
        holes, labels = _drop_edge_holes(holes, labels, width, height)
    return holes, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument(
        "--min-diameter-px", type=float, default=DEFAULT_MIN_DIAMETER_PX,
        help=f"Only count holes at least this wide, in pixels (default: {DEFAULT_MIN_DIAMETER_PX:g}); "
             "overridden by --min-diameter-mm if that's given",
    )
    parser.add_argument(
        "--pix2mm", type=float, default=None,
        help="Millimeters per pixel, to specify --min-diameter-mm instead of pixels and to print "
             "each hole's diameter range in mm alongside the count",
    )
    parser.add_argument(
        "--min-diameter-mm", type=float, default=None,
        help="Only count holes at least this wide, in mm; requires --pix2mm",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save an annotated copy here (a directory if input is a directory)",
    )
    parser.add_argument(
        "--median-blur-k", type=int, default=0,
        help="For grainy/noisy material (e.g. CLAHE-processed): try 25",
    )
    parser.add_argument(
        "--bg-kernel-frac", type=float, default=0.025,
        help="For images with very large holes: try 0.10",
    )
    parser.add_argument(
        "--min-solidity", type=float, default=0.0,
        help="For grainy material, reject non-blob-shaped noise: try 0.65 "
             "(above ~0.7 starts rejecting real but non-convex voids)",
    )
    parser.add_argument(
        "--include-edge-holes", action="store_true",
        help="Count holes touching the image frame too (excluded by default, since "
             "they're cut off and don't reflect the hole's true size)",
    )
    args = parser.parse_args()

    if args.min_diameter_mm is not None:
        if args.pix2mm is None:
            sys.exit("--min-diameter-mm requires --pix2mm")
        min_diameter_px = args.min_diameter_mm / args.pix2mm
    else:
        min_diameter_px = args.min_diameter_px

    files = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        sys.exit(f"No images found in {args.input}")

    if args.output is not None and len(files) > 1:
        args.output.mkdir(parents=True, exist_ok=True)

    total = 0
    for src in files:
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            print(f"{src.name}: could not read image", file=sys.stderr)
            continue

        holes, labels = count_holes(
            image, min_diameter_px, args.median_blur_k, args.bg_kernel_frac, args.min_solidity,
            ignore_edges=not args.include_edge_holes,
        )
        line = f"{src.name}: {len(holes)} holes"
        if args.pix2mm is not None and holes:
            diameters_mm = [h.diameter_px * args.pix2mm for h in holes]
            line += f" (diameter {min(diameters_mm):.2f}-{max(diameters_mm):.2f} mm)"
        print(line)
        total += len(holes)

        if args.output is not None:
            dst = args.output / src.name if len(files) > 1 else args.output
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), draw_void_contours(image, labels, len(holes)))

    if len(files) > 1:
        print(f"total: {total} holes across {len(files)} images")


if __name__ == "__main__":
    main()
