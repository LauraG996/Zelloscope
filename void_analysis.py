#!/usr/bin/env python3
"""Detect voids (dark pits/pores) in phenolic material micrographs and report
count, size, and spatial distribution.

Voids show up as small patches darker than their immediate surroundings, on
top of a naturally grainy/speckled material texture. Detection (see
detect_void_mask) blurs away that fine grain, estimates a smooth local
background with a large morphological closing (which bridges over -- erases
-- the voids), and Otsu-thresholds how far each pixel falls below that
background, restricted to the specimen itself (see specimen_mask) so any
dark mount/background the specimen was photographed against can't be
mistaken for voids. What survives is cleaned of speckle noise and split into
individual voids with cv2.connectedComponentsWithStats.

For each image, writes an annotated copy with void outlines, a per-void CSV
(id/area/diameter/centroid), and a grid-based spatial-distribution CSV
(void density per region of the image). Works on single files or a whole
directory; for a directory, also writes a void_summary.csv with one
aggregate row per image. Pass --plot to additionally render a size-histogram
and spatial-density heatmap PNG per image.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from straighten import IMAGE_EXTS

# Gaussian blur sigma (px) applied before background estimation, to average
# out the material's own fine grain texture without erasing real voids.
DEFAULT_BLUR_SIGMA = 3.0
# Size of the morphological closing kernel used to estimate the smooth local
# background, as a fraction of the image's shorter side. Needs to be larger
# than any void so closing fully bridges over it.
DEFAULT_BG_KERNEL_FRAC = 0.025
# Voids smaller than this (px^2, after speckle cleanup) are dropped as noise.
DEFAULT_MIN_AREA_PX = 6
# Spatial distribution grid (rows x cols) that the image is split into for
# the per-region void count / porosity breakdown.
DEFAULT_GRID_SIZE = 4

# Pixels darker than this (after a heavy blur) are considered outside the
# specimen entirely (e.g. a mount/background the specimen was photographed
# against), not material -- see specimen_mask. Real specimens in this data
# set never got this dark even inside a void, only the mount around them did.
DEFAULT_MOUNT_THRESHOLD = 50
# Blur sigma (px) used to find the specimen's own extent, well beyond
# DEFAULT_BLUR_SIGMA since this only needs to find the coarse mount/specimen
# boundary, not preserve individual voids.
DEFAULT_MOUNT_BLUR_SIGMA = 15.0


def specimen_mask(
    gray: np.ndarray,
    mount_threshold: float = DEFAULT_MOUNT_THRESHOLD,
    blur_sigma: float = DEFAULT_MOUNT_BLUR_SIGMA,
) -> np.ndarray:
    """Binary mask (255 = specimen) excluding any dark mount/background the specimen sits against.

    Heavily blurs away texture and voids, keeping only the coarse
    illumination level, then keeps the single largest region brighter than
    mount_threshold. Harmless when the specimen fills the whole frame (the
    common case in this data set) -- the mask then just comes back all-255.
    """
    heavy = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
    _, mask = cv2.threshold(heavy, mount_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.full(gray.shape, 255, np.uint8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def _otsu_threshold_within(values: np.ndarray, region: np.ndarray) -> float:
    """Otsu threshold computed only from values[region > 0], as a scalar."""
    t, _ = cv2.threshold(values[region > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def detect_void_mask(
    gray: np.ndarray,
    region: np.ndarray,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    bg_kernel_frac: float = DEFAULT_BG_KERNEL_FRAC,
) -> np.ndarray:
    """Binary mask (255 = void) of pixels darker than their local background, within region.

    Otsu-thresholds how far each pixel falls below a smoothed local
    background estimate (a large morphological closing, which fills in --
    erases -- anything smaller than the kernel, i.e. the voids themselves),
    then removes single-pixel speckle left over from the material's texture.
    The threshold is computed only from pixels inside region (see
    specimen_mask) so a dark mount/background around the specimen can't skew
    it, and the result is masked to region too.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)

    k = max(15, int(round(min(h, w) * bg_kernel_frac)) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, close_kernel)

    below_background = cv2.subtract(background, blurred)
    t = _otsu_threshold_within(below_background, region)
    mask = ((below_background > t) & (region > 0)).astype(np.uint8) * 255

    speckle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, speckle_kernel)


class Void:
    """One detected void: its pixel area, equivalent circular diameter, centroid, and bbox."""

    def __init__(self, void_id: int, area_px: float, centroid: tuple[float, float], bbox: tuple[int, int, int, int]):
        self.void_id = void_id
        self.area_px = area_px
        self.diameter_px = 2.0 * np.sqrt(area_px / np.pi)
        self.centroid = centroid
        self.bbox = bbox


def find_voids(mask: np.ndarray, min_area_px: float = DEFAULT_MIN_AREA_PX) -> tuple[list[Void], np.ndarray]:
    """Connected components of mask, filtered by min_area_px and sorted largest-first.

    Returns the void list plus a same-shaped label image (0 = background,
    matching each Void's void_id elsewhere) with dropped/filtered components
    zeroed out, for drawing/lookup convenience.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    voids = []
    kept_labels = np.zeros_like(labels)
    next_id = 1
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1  # largest area first
    for label in order:
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        bbox = tuple(int(v) for v in stats[label, :4])
        voids.append(Void(next_id, area, tuple(centroids[label]), bbox))
        kept_labels[labels == label] = next_id
        next_id += 1
    return voids, kept_labels


def size_summary(voids: list[Void], specimen_area_px: int, pix2mm: float | None = None) -> dict:
    """Aggregate count/size/porosity stats for a list of voids.

    porosity_pct is relative to specimen_area_px (the specimen itself, per
    specimen_mask), not the whole image -- images with a mount/background
    border around the specimen would otherwise read an artificially low
    porosity.
    """
    diameters = np.array([v.diameter_px for v in voids]) if voids else np.array([])
    total_area_px = float(sum(v.area_px for v in voids))

    summary = {
        "void_count": len(voids),
        "total_void_area_px": total_area_px,
        "porosity_pct": 100.0 * total_area_px / specimen_area_px,
        "diameter_px_mean": float(diameters.mean()) if voids else 0.0,
        "diameter_px_median": float(np.median(diameters)) if voids else 0.0,
        "diameter_px_std": float(diameters.std()) if voids else 0.0,
        "diameter_px_min": float(diameters.min()) if voids else 0.0,
        "diameter_px_max": float(diameters.max()) if voids else 0.0,
    }
    if pix2mm is not None:
        summary["total_void_area_mm2"] = total_area_px * pix2mm ** 2
        for key in ("mean", "median", "std", "min", "max"):
            summary[f"diameter_mm_{key}"] = summary[f"diameter_px_{key}"] * pix2mm
    return summary


def spatial_distribution(
    voids: list[Void], image_shape: tuple[int, int], grid_size: int = DEFAULT_GRID_SIZE,
) -> list[dict]:
    """Per-cell void count/area/porosity for a grid_size x grid_size split of the image.

    Shows where voids concentrate across the specimen rather than just an
    overall count -- e.g. a defect cluster in one corner reads very
    differently from the same count spread evenly.
    """
    h, w = image_shape
    cell_h, cell_w = h / grid_size, w / grid_size
    cells = {(r, c): {"count": 0, "area_px": 0.0} for r in range(grid_size) for c in range(grid_size)}

    for v in voids:
        cx, cy = v.centroid
        row = min(grid_size - 1, int(cy // cell_h))
        col = min(grid_size - 1, int(cx // cell_w))
        cells[(row, col)]["count"] += 1
        cells[(row, col)]["area_px"] += v.area_px

    cell_area_px = cell_h * cell_w
    rows = []
    for (row, col), cell in sorted(cells.items()):
        rows.append({
            "grid_row": row,
            "grid_col": col,
            "void_count": cell["count"],
            "void_area_px": cell["area_px"],
            "porosity_pct": 100.0 * cell["area_px"] / cell_area_px,
        })
    return rows


def draw_void_contours(image: np.ndarray, labels: np.ndarray, void_count: int) -> np.ndarray:
    """Outline each void in red and label the total count in the corner."""
    output = image.copy()
    mask = np.where(labels > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale = max(image.shape[:2]) / 1500
    thickness = max(2, int(round(2 * scale)))
    cv2.drawContours(output, contours, -1, (0, 0, 255), thickness, cv2.LINE_AA)

    text = f"voids: {void_count}"
    font_scale = max(0.6, 1.2 * scale)
    font_thickness = max(1, int(round(2 * scale)))
    pos = (int(15 * scale), int(40 * scale))
    cv2.putText(output, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
    cv2.putText(output, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    return output


def plot_distributions(
    voids: list[Void], image_shape: tuple[int, int], spatial_rows: list[dict], dst: Path,
) -> None:
    """Save a size histogram + spatial density heatmap PNG for one image, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_size = int(round(len(spatial_rows) ** 0.5))
    heatmap = np.zeros((grid_size, grid_size))
    for row in spatial_rows:
        heatmap[row["grid_row"], row["grid_col"]] = row["porosity_pct"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    diameters = [v.diameter_px for v in voids]
    ax1.hist(diameters, bins=30, color="#4477AA", edgecolor="white")
    ax1.set_xlabel("Void diameter (px)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Size distribution (n={len(voids)})")

    im = ax2.imshow(heatmap, cmap="inferno")
    ax2.set_title("Spatial distribution (porosity % per cell)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.colorbar(im, ax=ax2, label="porosity %")

    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def process_file(
    src: Path,
    out_dir: Path,
    pix2mm: float | None = None,
    min_area_px: float = DEFAULT_MIN_AREA_PX,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    bg_kernel_frac: float = DEFAULT_BG_KERNEL_FRAC,
    grid_size: int = DEFAULT_GRID_SIZE,
    make_plots: bool = False,
) -> dict:
    """Detect voids in src, write the annotated image + per-void and spatial CSVs to out_dir.

    Returns the aggregate size_summary dict (see size_summary) for src.
    """
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {src}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    region = specimen_mask(gray)
    # Pull the region in from its own (possibly ragged) edge a little, so
    # that edge itself -- e.g. where a mount border was cut away -- can't be
    # mistaken for a void.
    border_px = max(5, int(round(min(gray.shape) * 0.003)) | 1)
    border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_px * 2 + 1,) * 2)
    region = cv2.erode(region, border_kernel)

    mask = detect_void_mask(gray, region, blur_sigma, bg_kernel_frac)
    voids, labels = find_voids(mask, min_area_px)
    specimen_area_px = int(np.count_nonzero(region))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    annotated = draw_void_contours(image, labels, len(voids))
    cv2.imwrite(str(out_dir / src.name), annotated)

    with open(out_dir / f"{stem}_voids.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["void_id", "area_px", "diameter_px", "centroid_x_px", "centroid_y_px",
                   "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        if pix2mm is not None:
            header += ["area_mm2", "diameter_mm"]
        writer.writerow(header)
        for v in voids:
            row = [v.void_id, f"{v.area_px:.1f}", f"{v.diameter_px:.2f}",
                   f"{v.centroid[0]:.1f}", f"{v.centroid[1]:.1f}", *v.bbox]
            if pix2mm is not None:
                row += [f"{v.area_px * pix2mm ** 2:.4f}", f"{v.diameter_px * pix2mm:.3f}"]
            writer.writerow(row)

    spatial_rows = spatial_distribution(voids, gray.shape, grid_size)
    with open(out_dir / f"{stem}_spatial.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grid_row", "grid_col", "void_count", "void_area_px", "porosity_pct"])
        for row in spatial_rows:
            writer.writerow([row["grid_row"], row["grid_col"], row["void_count"],
                              f"{row['void_area_px']:.1f}", f"{row['porosity_pct']:.2f}"])

    if make_plots:
        plot_distributions(voids, gray.shape, spatial_rows, out_dir / f"{stem}_distribution.png")

    return size_summary(voids, specimen_area_px, pix2mm)


def _describe(name: str, summary: dict) -> str:
    text = (
        f"{name}: {summary['void_count']} voids, porosity {summary['porosity_pct']:.2f}%, "
        f"diameter mean {summary['diameter_px_mean']:.1f}px "
        f"(median {summary['diameter_px_median']:.1f}, "
        f"range {summary['diameter_px_min']:.1f}-{summary['diameter_px_max']:.1f})"
    )
    if "diameter_mm_mean" in summary:
        text += f" [{summary['diameter_mm_mean']:.3f}mm mean]"
    return text


def _summary_csv_header(pix2mm: float | None) -> list[str]:
    header = ["filename", "void_count", "total_void_area_px", "porosity_pct",
              "diameter_px_mean", "diameter_px_median", "diameter_px_std",
              "diameter_px_min", "diameter_px_max"]
    if pix2mm is not None:
        header += ["total_void_area_mm2", "diameter_mm_mean", "diameter_mm_median",
                   "diameter_mm_std", "diameter_mm_min", "diameter_mm_max"]
    return header


def _summary_csv_row(name: str, summary: dict, pix2mm: float | None) -> list[str]:
    row = [name, summary["void_count"], f"{summary['total_void_area_px']:.1f}",
           f"{summary['porosity_pct']:.2f}", f"{summary['diameter_px_mean']:.2f}",
           f"{summary['diameter_px_median']:.2f}", f"{summary['diameter_px_std']:.2f}",
           f"{summary['diameter_px_min']:.2f}", f"{summary['diameter_px_max']:.2f}"]
    if pix2mm is not None:
        row += [f"{summary['total_void_area_mm2']:.4f}", f"{summary['diameter_mm_mean']:.3f}",
                f"{summary['diameter_mm_median']:.3f}", f"{summary['diameter_mm_std']:.3f}",
                f"{summary['diameter_mm_min']:.3f}", f"{summary['diameter_mm_max']:.3f}"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image file or directory")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument(
        "--pix2mm", type=float, default=None,
        help="Millimeters per pixel, to also report void size/area in mm/mm^2",
    )
    parser.add_argument(
        "--min-area-px", type=float, default=DEFAULT_MIN_AREA_PX,
        help=f"Drop detected voids smaller than this many pixels^2, as noise (default: {DEFAULT_MIN_AREA_PX})",
    )
    parser.add_argument(
        "--blur-sigma", type=float, default=DEFAULT_BLUR_SIGMA,
        help=f"Gaussian blur sigma (px) used to smooth out material grain texture before "
             f"detection (default: {DEFAULT_BLUR_SIGMA})",
    )
    parser.add_argument(
        "--bg-kernel-frac", type=float, default=DEFAULT_BG_KERNEL_FRAC,
        help="Background-estimation morphological kernel size, as a fraction of the "
             f"image's shorter side (default: {DEFAULT_BG_KERNEL_FRAC})",
    )
    parser.add_argument(
        "--grid-size", type=int, default=DEFAULT_GRID_SIZE,
        help=f"Split each image into an N x N grid for the spatial distribution CSV "
             f"(default: {DEFAULT_GRID_SIZE})",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Also save a <name>_distribution.png (size histogram + spatial heatmap) per image; "
             "requires matplotlib",
    )
    args = parser.parse_args()

    files = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        sys.exit(f"No images found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for src in files:
        summary = process_file(
            src, args.output, args.pix2mm, args.min_area_px,
            args.blur_sigma, args.bg_kernel_frac, args.grid_size, args.plot,
        )
        print(_describe(src.name, summary))
        summary_rows.append((src.name, summary))

    if len(files) > 1:
        csv_path = args.output / "void_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_summary_csv_header(args.pix2mm))
            for name, summary in summary_rows:
                writer.writerow(_summary_csv_row(name, summary, args.pix2mm))
        print(f"Summary written to {csv_path}")


if __name__ == "__main__":
    main()
