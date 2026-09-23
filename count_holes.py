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
from void_analysis import (
    detect_void_mask,
    draw_void_contours,
    find_voids,
    size_summary,
    spatial_distribution,
    specimen_mask,
)

# What counts as "big" by default: holes narrower than this (px) are ignored
# as texture/noise. Tune to your image's scale and what you consider a hole.
DEFAULT_MIN_DIAMETER_PX = 15.0


def draw_distribution_grid(image: np.ndarray, rows: list[dict], grid_size: int) -> np.ndarray:
    """Draw the grid_size x grid_size distribution grid on image, each cell labeled with its hole count."""
    output = image.copy()
    h, w = output.shape[:2]
    cell_h, cell_w = h / grid_size, w / grid_size
    scale = max(h, w) / 1500

    line_thickness = max(1, int(round(2 * scale)))
    for i in range(1, grid_size):
        y = int(round(i * cell_h))
        cv2.line(output, (0, y), (w, y), (0, 255, 255), line_thickness, cv2.LINE_AA)
    for j in range(1, grid_size):
        x = int(round(j * cell_w))
        cv2.line(output, (x, 0), (x, h), (0, 255, 255), line_thickness, cv2.LINE_AA)

    font_scale = max(0.7, 1.3 * scale)
    font_thickness = max(1, int(round(2 * scale)))
    for row in rows:
        text = str(row["void_count"])
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cx = int(round((row["grid_col"] + 0.5) * cell_w)) - text_w // 2
        cy = int(round((row["grid_row"] + 0.5) * cell_h)) + text_h // 2
        cv2.putText(output, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    font_thickness + 2, cv2.LINE_AA)
        cv2.putText(output, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255),
                    font_thickness, cv2.LINE_AA)
    return output


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
) -> tuple[list, np.ndarray, int]:
    """Detect holes at least min_diameter_px wide in image.

    ignore_edges drops any hole touching the image frame (see _drop_edge_holes).
    Returns (holes, labels, specimen_area_px) -- holes/labels as per
    void_analysis.find_voids; specimen_area_px for size_summary's porosity.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    region = specimen_mask(gray)
    mask = detect_void_mask(gray, region, bg_kernel_frac=bg_kernel_frac, median_blur_k=median_blur_k)
    min_area_px = np.pi * (min_diameter_px / 2) ** 2
    holes, labels = find_voids(mask, min_area_px=min_area_px, min_solidity=min_solidity)

    if ignore_edges:
        height, width = gray.shape
        holes, labels = _drop_edge_holes(holes, labels, width, height)
    return holes, labels, int(np.count_nonzero(region))


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
    parser.add_argument(
        "--stats", action="store_true",
        help="Also print size stats (mean/median/std/min/max diameter, porosity) for each image",
    )
    parser.add_argument(
        "--distribution", type=int, default=None, metavar="GRID_SIZE",
        help="Also print a GRID_SIZE x GRID_SIZE spatial breakdown (hole count and porosity "
             "per region) for each image, e.g. --distribution 4",
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

        holes, labels, specimen_area_px = count_holes(
            image, min_diameter_px, args.median_blur_k, args.bg_kernel_frac, args.min_solidity,
            ignore_edges=not args.include_edge_holes,
        )
        line = f"{src.name}: {len(holes)} holes"
        if args.pix2mm is not None and holes:
            diameters_mm = [h.diameter_px * args.pix2mm for h in holes]
            line += f" (diameter {min(diameters_mm):.2f}-{max(diameters_mm):.2f} mm)"
        print(line)
        total += len(holes)

        if args.stats:
            summary = size_summary(holes, specimen_area_px, args.pix2mm)
            print(
                f"  size: diameter mean {summary['diameter_px_mean']:.1f}px "
                f"(median {summary['diameter_px_median']:.1f}, std {summary['diameter_px_std']:.1f}, "
                f"range {summary['diameter_px_min']:.1f}-{summary['diameter_px_max']:.1f}), "
                f"porosity {summary['porosity_pct']:.2f}%"
            )
            if "diameter_mm_mean" in summary:
                print(
                    f"  size (mm): diameter mean {summary['diameter_mm_mean']:.3f}mm "
                    f"(median {summary['diameter_mm_median']:.3f}, std {summary['diameter_mm_std']:.3f}, "
                    f"range {summary['diameter_mm_min']:.3f}-{summary['diameter_mm_max']:.3f})"
                )

        distribution_rows = None
        if args.distribution:
            distribution_rows = spatial_distribution(holes, image.shape[:2], args.distribution)
            print(f"  distribution ({args.distribution}x{args.distribution} grid):")
            for row in distribution_rows:
                print(
                    f"    [{row['grid_row']},{row['grid_col']}] "
                    f"{row['void_count']} holes, {row['porosity_pct']:.2f}% porosity"
                )

        if args.output is not None:
            dst = args.output / src.name if len(files) > 1 else args.output
            dst.parent.mkdir(parents=True, exist_ok=True)
            annotated = draw_void_contours(image, labels, len(holes))
            if distribution_rows is not None:
                annotated = draw_distribution_grid(annotated, distribution_rows, args.distribution)
            cv2.imwrite(str(dst), annotated)

    if len(files) > 1:
        print(f"total: {total} holes across {len(files)} images")


if __name__ == "__main__":
    main()
