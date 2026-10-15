#!/usr/bin/env python3
"""Track how a foam specimen's shape changes across a series of images.

The foam is expected to keep its relative shape: it may shift, rotate and
shrink/swell uniformly, but should not distort. This script separates those
two kinds of change. Each frame's silhouette is registered onto a reference
frame (the first image, unless --reference is given) with a similarity
transform -- translation, rotation and one uniform scale -- fitted by ICP on
the object's outer and inner (hole) contours. What the similarity cannot
explain is the real shape change, reported as:

  * anisotropy / shear: from an additional affine fit, how far the change is
    from uniform (e.g. shrinking more in height than in width);
  * local deviation: for every point on the reference contour, the signed
    distance (after the similarity alignment) to the frame's contour --
    positive where the foam locally sticks out further than a uniform scale
    would predict, negative where it has pulled in;
  * IoU of the aligned silhouettes.

Contour points within a margin of the image border are ignored, so a
specimen cut off by the frame edge (like the horseshoe legs) doesn't bias
the fit. Writes one overlay per frame (reference contour mapped onto the
frame, coloured by local deviation) plus shape_tracking.csv.
"""
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from straighten import IMAGE_EXTS, largest_foreground_mask

# Registration runs on a downsampled mask: contours are long, smooth curves,
# so a ~1000px working size keeps sub-pixel (full-res) accuracy from the
# least-squares fit over thousands of points while staying fast.
WORK_SIZE = 1000
# Contour points closer than this (px, working scale) to the image border
# are dropped: they're where the object is clipped by the frame, not a real edge.
BORDER_MARGIN = 6
ICP_ITERATIONS = 60
# Correspondences farther than this multiple of the median distance are
# treated as outliers (dust, a clipped region, a genuinely deformed spot)
# and excluded from the fit, so local defects don't drag the global fit.
ICP_TRIM_FACTOR = 3.0
ICP_CONVERGENCE_PX = 1e-3
# Colour scale for the overlay: deviations at or beyond this many working-
# scale px saturate to full red/blue.
DEVIATION_COLOR_RANGE_PX = 4.0


@dataclass
class ShapeChange:
    scale: float              # uniform linear scale vs. reference (1.0 = same size)
    rotation_deg: float
    dx_px: float
    dy_px: float
    anisotropy: float         # ratio of principal stretches from the affine fit (1.0 = uniform)
    shear_deg: float          # affine shear angle (0 = none)
    iou: float                # silhouette overlap after similarity alignment
    mean_abs_dev_px: float    # mean |local deviation| along the contour, full-res px
    max_out_px: float         # largest outward deviation (locally swollen)
    max_in_px: float          # largest inward deviation (locally pulled in), <= 0
    rms_dev_pct: float        # RMS deviation as % of the reference's size


@dataclass
class Silhouette:
    mask: np.ndarray          # working-scale binary mask
    points: np.ndarray        # (N, 2) float contour points, border-clipped ones removed
    factor: float             # full-res px per working px
    size: float               # sqrt(foam area), working px -- a length scale for %


def load_silhouette(path: Path) -> tuple[np.ndarray, Silhouette]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    factor = max(gray.shape) / WORK_SIZE
    small = cv2.resize(gray, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
    mask = largest_foreground_mask(small)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    points = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)
    h, w = mask.shape
    inside = (
        (points[:, 0] >= BORDER_MARGIN) & (points[:, 0] < w - BORDER_MARGIN)
        & (points[:, 1] >= BORDER_MARGIN) & (points[:, 1] < h - BORDER_MARGIN)
    )
    points = points[inside]
    if len(points) < 50:
        raise ValueError(f"Too little object contour found in {path.name}")
    size = float(np.sqrt(np.count_nonzero(mask)))
    return image, Silhouette(mask=mask, points=points, factor=factor, size=size)


class NearestEdge:
    """Nearest-contour-point lookup for one silhouette, via a labelled distance transform."""

    def __init__(self, sil: Silhouette):
        h, w = sil.mask.shape
        edges = np.full((h, w), 255, np.uint8)
        pts = sil.points.astype(int)
        edges[pts[:, 1], pts[:, 0]] = 0
        _, labels = cv2.distanceTransformWithLabels(
            edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
        )
        # Labels number the zero pixels in row-major scan order.
        zy, zx = np.nonzero(edges == 0)
        self.label_to_xy = np.column_stack([zx, zy]).astype(np.float64)
        self.labels = labels
        self.shape = (h, w)

    def query(self, pts: np.ndarray) -> np.ndarray:
        h, w = self.shape
        xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        return self.label_to_xy[self.labels[yi, xi] - 1]


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (Umeyama) mapping src -> dst, as a 2x3 matrix."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    sign = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ sign @ vt
    scale = np.trace(np.diag(sig) @ sign) / (s ** 2).sum(1).mean()
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = mu_d - m[:, :2] @ mu_s
    return m


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = np.column_stack([src, np.ones(len(src))])
    sol, *_ = np.linalg.lstsq(a, dst, rcond=None)
    return sol.T


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:, :2].T + m[:, 2]


def icp(ref: Silhouette, target: NearestEdge, fit, init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a transform mapping ref.points onto the target contour. Returns (matrix, inlier mask)."""
    m = init
    inliers = np.ones(len(ref.points), bool)
    for _ in range(ICP_ITERATIONS):
        moved = _apply(m, ref.points)
        nearest = target.query(moved)
        dist = np.linalg.norm(nearest - moved, axis=1)
        inliers = dist <= ICP_TRIM_FACTOR * max(np.median(dist), 0.5)
        new_m = fit(ref.points[inliers], nearest[inliers])
        step = np.abs(_apply(new_m, ref.points) - moved).max()
        m = new_m
        if step < ICP_CONVERGENCE_PX:
            break
    return m, inliers


# Multi-start grid for the similarity ICP. A single start from the centroid
# can lock into a wrong minimum where the reference slides along a
# frame-clipped feature (e.g. the horseshoe's legs) and the cap is discarded
# as outliers; trying a few rotations/scales and keeping the best fit avoids it.
INIT_ROTATIONS_DEG = np.arange(-8, 9, 2)
INIT_SCALES = (0.95, 1.0, 1.05, 1.1)
# Per-point distance cap (working px) when scoring a fit, so a clipped or
# truly deformed region can't dominate the choice between starts.
FIT_SCORE_CAP_PX = 20.0


def _centroid(mask: np.ndarray) -> np.ndarray:
    mo = cv2.moments(mask, binaryImage=True)
    return np.array([mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]])


def _initial_guesses(ref: Silhouette, frame: Silhouette):
    """Similarity starts: rotate/scale the reference about its contour centre, placed on the frame's."""
    shift = _centroid(frame.mask) - _centroid(ref.mask)
    c_ref = ref.points.mean(0)
    for deg in INIT_ROTATIONS_DEG:
        r = np.radians(deg)
        for s in INIT_SCALES:
            a = s * np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
            yield np.hstack([a, (c_ref + shift - a @ c_ref)[:, None]])


def fit_similarity(ref: Silhouette, target: NearestEdge, frame: Silhouette) -> np.ndarray:
    best_score, best_m = np.inf, None
    for init in _initial_guesses(ref, frame):
        m, _ = icp(ref, target, _fit_similarity, init)
        moved = _apply(m, ref.points)
        score = np.minimum(np.linalg.norm(target.query(moved) - moved, axis=1), FIT_SCORE_CAP_PX).mean()
        if score < best_score:
            best_score, best_m = score, m
    return best_m


def signed_deviation(ref: Silhouette, frame: Silhouette, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map reference contour into the frame; return (mapped points, signed distance in working px).

    Positive = the frame's foam extends past where the uniformly-scaled
    reference edge would be (locally swollen); negative = pulled in.
    """
    moved = _apply(m, ref.points)
    h, w = frame.mask.shape
    edges = np.full((h, w), 255, np.uint8)
    contours, _ = cv2.findContours(frame.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(edges, contours, -1, 0, 1)
    dist = cv2.distanceTransform(edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    xi = np.clip(np.rint(moved[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(moved[:, 1]).astype(int), 0, h - 1)
    d = dist[yi, xi].astype(np.float64)
    in_foam = frame.mask[yi, xi] > 0
    return moved, np.where(in_foam, d, -d)


def silhouette_iou(ref: Silhouette, frame: Silhouette, m: np.ndarray) -> float:
    """IoU of the aligned silhouettes, restricted to where both images have real data."""
    h, w = frame.mask.shape
    warped = cv2.warpAffine(ref.mask, m, (w, h), flags=cv2.INTER_NEAREST)
    # Only compare inside the region covered by the (warped) reference frame,
    # shrunk by the border margin so frame-edge clipping isn't counted.
    valid = cv2.warpAffine(np.full(ref.mask.shape, 255, np.uint8), m, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((2 * BORDER_MARGIN + 1,) * 2, np.uint8))
    valid[:BORDER_MARGIN], valid[-BORDER_MARGIN:] = 0, 0
    valid[:, :BORDER_MARGIN], valid[:, -BORDER_MARGIN:] = 0, 0
    a, b = (warped > 0) & (valid > 0), (frame.mask > 0) & (valid > 0)
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def compare(ref: Silhouette, frame: Silhouette) -> tuple[ShapeChange, np.ndarray, np.ndarray]:
    target = NearestEdge(frame)
    sim = fit_similarity(ref, target, frame)
    aff, _ = icp(ref, target, _fit_affine, sim)

    a = sim[:, :2]
    scale = float(np.sqrt(abs(np.linalg.det(a))))
    rotation = float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))

    # Affine: polar-decompose into rotation * stretch; the stretch's
    # eigenvalue ratio is the anisotropy, and the angle between the mapped
    # unit axes (minus 90 deg) is the shear.
    _, sv, _ = np.linalg.svd(aff[:, :2])
    anisotropy = float(sv[0] / sv[1])
    ex, ey = aff[:, 0], aff[:, 1]
    cos_xy = ex @ ey / (np.linalg.norm(ex) * np.linalg.norm(ey))
    shear = float(90.0 - np.degrees(np.arccos(np.clip(cos_xy, -1, 1))))

    moved, dev = signed_deviation(ref, frame, sim)
    f = frame.factor
    change = ShapeChange(
        scale=scale,
        rotation_deg=rotation,
        dx_px=float(sim[0, 2] * f),
        dy_px=float(sim[1, 2] * f),
        anisotropy=anisotropy,
        shear_deg=shear,
        iou=silhouette_iou(ref, frame, sim),
        mean_abs_dev_px=float(np.abs(dev).mean() * f),
        max_out_px=float(max(dev.max(), 0.0) * f),
        max_in_px=float(min(dev.min(), 0.0) * f),
        rms_dev_pct=float(np.sqrt((dev ** 2).mean()) / (ref.size * scale) * 100),
    )
    return change, moved, dev


def _deviation_color(d: float) -> tuple[int, int, int]:
    """Blue (pulled in) -> green (matches) -> red (swollen), in BGR."""
    t = float(np.clip(d / DEVIATION_COLOR_RANGE_PX, -1, 1))
    if t >= 0:
        return (0, int(255 * (1 - t)), int(255 * t))
    return (int(255 * -t), int(255 * (1 + t)), 0)


def draw_overlay(image: np.ndarray, moved: np.ndarray, dev: np.ndarray, factor: float,
                 change: ShapeChange, label: str) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    radius = max(2, int(round(w / 600)))
    for (x, y), d in zip(moved * factor, dev):
        cv2.circle(out, (int(round(x)), int(round(y))), radius, _deviation_color(d), -1, cv2.LINE_AA)

    lines = [
        label,
        f"scale {change.scale * 100:.2f}%  (area {change.scale ** 2 * 100:.2f}%)",
        f"rotation {change.rotation_deg:+.2f} deg  shift ({change.dx_px:+.0f}, {change.dy_px:+.0f}) px",
        f"anisotropy {change.anisotropy:.4f}  shear {change.shear_deg:+.2f} deg",
        f"IoU {change.iou:.4f}  RMS dev {change.rms_dev_pct:.2f}%",
        f"local dev: mean {change.mean_abs_dev_px:.1f}px, "
        f"+{change.max_out_px:.0f} / {change.max_in_px:.0f} px",
        "red = sticks out, blue = pulled in (vs. uniform scale)",
    ]
    font_scale = w / 1600
    thick = max(1, int(round(font_scale * 2)))
    line_h = int(40 * font_scale)
    for i, text in enumerate(lines):
        org = (int(20 * font_scale), int((i + 1) * line_h + 10))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick * 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


CSV_HEADER = [
    "filename", "scale_pct", "area_pct", "rotation_deg", "dx_px", "dy_px",
    "anisotropy", "shear_deg", "iou", "rms_dev_pct", "mean_abs_dev_px", "max_out_px", "max_in_px",
]


def csv_row(name: str, c: ShapeChange) -> list[str]:
    return [
        name, f"{c.scale * 100:.3f}", f"{c.scale ** 2 * 100:.3f}", f"{c.rotation_deg:.3f}",
        f"{c.dx_px:.1f}", f"{c.dy_px:.1f}", f"{c.anisotropy:.5f}", f"{c.shear_deg:.3f}",
        f"{c.iou:.5f}", f"{c.rms_dev_pct:.3f}", f"{c.mean_abs_dev_px:.2f}",
        f"{c.max_out_px:.1f}", f"{c.max_in_px:.1f}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Directory of images (processed in filename order)")
    parser.add_argument("output", type=Path, help="Output directory for overlays and shape_tracking.csv")
    parser.add_argument(
        "--reference", type=Path, default=None,
        help="Reference image to compare against (default: first image in the directory)",
    )
    args = parser.parse_args()

    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"No images found in {args.input}")
    ref_path = args.reference or files[0]
    _, ref = load_silhouette(ref_path)
    print(f"Reference: {ref_path.name}")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "shape_tracking.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for src in files:
            image, frame = load_silhouette(src)
            change, moved, dev = compare(ref, frame)
            overlay = draw_overlay(image, moved, dev, frame.factor, change, f"{src.name} vs {ref_path.name}")
            cv2.imwrite(str(args.output / src.name), overlay)
            writer.writerow(csv_row(src.name, change))
            print(
                f"{src.name}: scale {change.scale * 100:.2f}%, rot {change.rotation_deg:+.2f} deg, "
                f"anisotropy {change.anisotropy:.4f}, IoU {change.iou:.4f}, "
                f"RMS dev {change.rms_dev_pct:.2f}%"
            )
    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
