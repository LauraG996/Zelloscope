#!/usr/bin/env python3
"""Segment individual foam cells and report their size distribution.

Restricts analysis to the material (see straighten.py's
largest_foreground_mask) and runs a marker-controlled watershed:
one seed per cell is placed at the local minima of a heavily-smoothed
copy of the image (each cell forms a shallow "bowl" between its bright
cell-wall ridges, so its darkest point is its center), then the actual
wall boundaries are traced by flooding outward from those seeds on a
lightly-smoothed, contrast-enhanced copy of the image, which keeps the
thin wall ridges sharp enough to stop the flood accurately.

Works on single files or a whole directory of images; for a directory,
also writes a cell_measurements.csv summary (one row per image) plus a
per-image "<name>_cells.csv" with one row per detected cell.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from straighten import IMAGE_EXTS, largest_foreground_mask

# Gaussian sigma (px) used to smooth the image before seeding: coarse enough
# that a cell's internal texture/cracks disappear and only the broad
# dark-center/bright-wall "bowl" shape of each cell remains.
SEED_SMOOTH_SIGMA = 14.0
# Minimum pixel distance enforced between two seed points, so one cell
# doesn't get split into several by noise in its center.
SEED_MIN_SEPARATION = 28
# Light smoothing (px) applied to the contrast-enhanced image used for the
# watershed flood itself -- just enough to suppress pixel-level speckle
# without blurring away the actual wall ridges.
WALL_SMOOTH_SIGMA = 1.5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (16, 16)


def _cell_seed_markers(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """One connected-component label per candidate cell center, 0 elsewhere."""
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=SEED_SMOOTH_SIGMA)
    inverted = 255 - smoothed

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (SEED_MIN_SEPARATION, SEED_MIN_SEPARATION)
    )
    local_max = cv2.dilate(inverted, kernel)
    seeds = ((inverted == local_max) & (mask > 0)).astype(np.uint8) * 255
    # Merge near-duplicate peaks (flat/noisy minima) into a single seed.
    seeds = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    _, labels = cv2.connectedComponents(seeds)
    return labels


def segment_cells(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Segment the foam cells inside mask (or the largest foreground object).

    Returns (labels, num_cells): labels is an int32 array the same size as
    image, where -1 marks a cell-wall boundary pixel, 1 marks background
    (outside the material), and 2..num_cells+1 each mark one cell's interior.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if mask is None:
        mask = largest_foreground_mask(gray)

    seed_labels = _cell_seed_markers(gray, mask)
    num_cells = int(seed_labels.max())

    markers = np.zeros(gray.shape, dtype=np.int32)
    markers[mask == 0] = 1  # sure background
    markers[seed_labels > 0] = seed_labels[seed_labels > 0] + 1

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=WALL_SMOOTH_SIGMA)
    landscape = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    cv2.watershed(landscape, markers)
    return markers, num_cells


def cell_stats(
    labels: np.ndarray, pix2mm: float | None = None, min_area_px: int = 0
) -> list[dict]:
    """Per-cell area/diameter, one dict per cell label (2..labels.max()), sorted by id."""
    stats = []
    for label in range(2, int(labels.max()) + 1):
        area_px = int(np.count_nonzero(labels == label))
        if area_px < min_area_px:
            continue
        diameter_px = 2.0 * np.sqrt(area_px / np.pi)
        entry = {"cell_id": label - 1, "area_px": area_px, "diameter_px": diameter_px}
        if pix2mm is not None:
            entry["area_mm2"] = area_px * pix2mm**2
            entry["diameter_mm"] = diameter_px * pix2mm
        stats.append(entry)
    return stats


def draw_cell_segmentation(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Draw the detected cell-wall boundaries over the (material area of the) image."""
    output = image.copy()
    scale = max(image.shape[:2]) / 1500  # so lines stay visible at any resolution
    thickness = max(1, int(round(2 * scale)))

    boundaries = (labels == -1) & (labels != 1)
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        boundaries = cv2.dilate(boundaries.astype(np.uint8), kernel) > 0
    output[boundaries] = (0, 255, 255)
    return output


def process_file(
    src: Path, dst: Path, pix2mm: float | None = None, min_area_px: int = 0
) -> list[dict]:
    """Segment src's foam cells, draw the result, save to dst.

    Returns the per-cell stats list (see cell_stats).
    """
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {src}")

    labels, _ = segment_cells(image)
    stats = cell_stats(labels, pix2mm, min_area_px)

    annotated = draw_cell_segmentation(image, labels)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), annotated)
    return stats


def _write_cell_csv(path: Path, stats: list[dict], pix2mm: float | None) -> None:
    header = ["cell_id", "area_px", "diameter_px"]
    if pix2mm is not None:
        header += ["area_mm2", "diameter_mm"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for entry in stats:
            row = [entry["cell_id"], entry["area_px"], f"{entry['diameter_px']:.2f}"]
            if pix2mm is not None:
                row += [f"{entry['area_mm2']:.4f}", f"{entry['diameter_mm']:.4f}"]
            writer.writerow(row)


def _diameter_summary(stats: list[dict]) -> str:
    if not stats:
        return "0 cells"
    diameters = np.array([entry["diameter_px"] for entry in stats])
    return (
        f"{len(stats)} cells, diameter mean {diameters.mean():.1f}px "
        f"(median {np.median(diameters):.1f}px, std {diameters.std():.1f}px)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image file or directory")
    parser.add_argument("output", type=Path, help="Output image file or directory")
    parser.add_argument(
        "--pix2mm", type=float, default=None,
        help="Millimeters per pixel, to also report cell area/diameter in mm",
    )
    parser.add_argument(
        "--min-area", type=int, default=50,
        help="Discard detected cells smaller than this many pixels, "
             "treating them as segmentation noise rather than real cells (default: 50)",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"No images found in {args.input}")
        csv_path = args.output / "cell_measurements.csv"
        header = ["filename", "cell_count", "mean_diameter_px", "median_diameter_px", "std_diameter_px"]
        if args.pix2mm is not None:
            header += ["mean_diameter_mm"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for src in files:
                stats = process_file(src, args.output / src.name, args.pix2mm, args.min_area)
                _write_cell_csv(args.output / f"{src.stem}_cells.csv", stats, args.pix2mm)
                print(f"{src.name}: {_diameter_summary(stats)} -> {args.output / src.name}")

                diameters = np.array([e["diameter_px"] for e in stats]) if stats else np.array([0.0])
                row = [src.name, len(stats), f"{diameters.mean():.2f}", f"{np.median(diameters):.2f}", f"{diameters.std():.2f}"]
                if args.pix2mm is not None:
                    mm = np.array([e["diameter_mm"] for e in stats]) if stats else np.array([0.0])
                    row.append(f"{mm.mean():.4f}")
                writer.writerow(row)
        print(f"Measurements written to {csv_path}")
    else:
        stats = process_file(args.input, args.output, args.pix2mm, args.min_area)
        _write_cell_csv(args.output.with_name(f"{args.output.stem}_cells.csv"), stats, args.pix2mm)
        print(f"{args.input.name}: {_diameter_summary(stats)} -> {args.output}")


if __name__ == "__main__":
    main()
