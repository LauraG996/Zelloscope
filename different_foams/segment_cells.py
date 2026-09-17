#!/usr/bin/env python3
"""Segment individual cells in open-cell foam microscopy images.

Cell walls (struts) show up bright and cell interiors darker. The approach:
flatten uneven illumination (a raw threshold fails on images with a
lighting gradient), threshold to get a binary mask of cell interiors, then
split touching cells apart at their walls with a distance-transform +
watershed (the same trick used to separate touching coins/cells in
standard OpenCV/scikit-image tutorials).

Outputs an annotated image with each detected cell's boundary drawn, plus
a CSV of per-cell measurements (area, equivalent diameter, centroid).
Works on a single file or a whole directory of images.
"""
import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import find_boundaries, watershed

# remove_small_objects/remove_small_holes's min_size/area_threshold params
# still work as used here; skimage just warns ahead of a future rename.
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*(min_size|area_threshold).*")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Sigma (px) of the background blur used to flatten uneven illumination
# before thresholding.
ILLUMINATION_SIGMA = 51
# Minimum separation (px) between watershed seed points, i.e. the smallest
# center-to-center distance at which two neighboring cells are still split
# apart rather than merged into one.
MIN_PEAK_DISTANCE = 8
# Cell regions (and mask specks) smaller than this many pixels are dropped
# as noise/texture rather than real cells.
MIN_CELL_AREA_PX = 60


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Correct uneven lighting by dividing by a heavily-blurred version of the image."""
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=ILLUMINATION_SIGMA)
    flat = cv2.divide(gray.astype(np.float32), background.astype(np.float32) + 1e-3, scale=128)
    return np.clip(flat, 0, 255).astype(np.uint8)


def cell_interior_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (255 = cell interior) after illumination flattening and Otsu thresholding."""
    flat = flatten_illumination(gray)
    blurred = cv2.GaussianBlur(flat, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bool_mask = remove_small_objects(mask > 0, min_size=MIN_CELL_AREA_PX)
    bool_mask = remove_small_holes(bool_mask, area_threshold=MIN_CELL_AREA_PX)
    return (bool_mask * 255).astype(np.uint8)


def segment_cells(mask: np.ndarray) -> np.ndarray:
    """Split the cell-interior mask into individually labeled cells via watershed.

    Returns an int32 label image (0 = background/walls, 1..N = cell ids).
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=MIN_PEAK_DISTANCE, labels=mask, exclude_border=False)
    markers = np.zeros(dist.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    return watershed(-dist, markers, mask=mask)


def cell_measurements(labels: np.ndarray, pix2mm: float | None = None) -> list[dict]:
    """Per-cell area, equivalent diameter, and centroid, dropping regions below MIN_CELL_AREA_PX."""
    cells = []
    for prop in regionprops(labels):
        if prop.area < MIN_CELL_AREA_PX:
            continue
        equiv_diameter_px = math.sqrt(4 * prop.area / math.pi)
        cy, cx = prop.centroid
        cell = {
            "cell_id": prop.label,
            "area_px": prop.area,
            "equiv_diameter_px": equiv_diameter_px,
            "centroid_x": cx,
            "centroid_y": cy,
        }
        if pix2mm is not None:
            cell["area_mm2"] = prop.area * pix2mm**2
            cell["equiv_diameter_mm"] = equiv_diameter_px * pix2mm
        cells.append(cell)
    return cells


def draw_cell_boundaries(image: np.ndarray, labels: np.ndarray, cells: list[dict]) -> np.ndarray:
    """Draw the boundary of each measured cell in red, vectorized over the whole label image."""
    kept_ids = {c["cell_id"] for c in cells}
    filtered = np.where(np.isin(labels, list(kept_ids)), labels, 0)
    boundaries = find_boundaries(filtered, mode="outer")

    output = image.copy()
    output[boundaries] = (0, 0, 255)
    return output


def process_file(src: Path, dst: Path, pix2mm: float | None = None) -> list[dict]:
    """Segment src's cells, draw their boundaries, save to dst, and return per-cell measurements."""
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {src}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cell_interior_mask(gray)
    labels = segment_cells(mask)
    cells = cell_measurements(labels, pix2mm)

    annotated = draw_cell_boundaries(image, labels, cells)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), annotated)

    csv_path = dst.with_name(dst.stem + "_cells.csv")
    header = ["cell_id", "area_px", "equiv_diameter_px", "centroid_x", "centroid_y"]
    if pix2mm is not None:
        header += ["area_mm2", "equiv_diameter_mm"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for cell in cells:
            writer.writerow(cell)

    return cells


def _summarize(cells: list[dict], pix2mm: float | None) -> str:
    if not cells:
        return "0 cells"
    diameters = [c["equiv_diameter_px"] for c in cells]
    mean_diam = sum(diameters) / len(diameters)
    if pix2mm is not None:
        mean_diam_mm = mean_diam * pix2mm
        return f"{len(cells)} cells, mean diameter {mean_diam:.1f}px ({mean_diam_mm:.3f}mm)"
    return f"{len(cells)} cells, mean diameter {mean_diam:.1f}px"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image file or directory")
    parser.add_argument("output", type=Path, help="Output image file or directory")
    parser.add_argument(
        "--pix2mm", type=float, default=None,
        help="Millimeters per pixel, to also report cell size in mm",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"No images found in {args.input}")
        summary_path = args.output / "cell_summary.csv"
        header = ["filename", "num_cells", "mean_diameter_px"]
        if args.pix2mm is not None:
            header.append("mean_diameter_mm")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for src in files:
                cells = process_file(src, args.output / src.name, args.pix2mm)
                print(f"{src.name}: {_summarize(cells, args.pix2mm)} -> {args.output / src.name}")
                diameters = [c["equiv_diameter_px"] for c in cells]
                mean_diam = sum(diameters) / len(diameters) if diameters else 0.0
                row = [src.name, len(cells), f"{mean_diam:.2f}"]
                if args.pix2mm is not None:
                    row.append(f"{mean_diam * args.pix2mm:.4f}")
                writer.writerow(row)
        print(f"Summary written to {summary_path}")
    else:
        cells = process_file(args.input, args.output, args.pix2mm)
        print(f"{args.input.name}: {_summarize(cells, args.pix2mm)} -> {args.output}")


if __name__ == "__main__":
    main()
