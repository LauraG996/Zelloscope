#!/usr/bin/env python3
"""Compare horseshoe foam specimens against an ideal geometric shape.

The ideal shape is a simple parametric model: an outer outline and an inner
hole, each a downward-tapering "U" -- two straight legs joined across the top
by rounded corners -- and both symmetric about the same vertical axis.
Mathematically each is an isosceles triangle (apex pointing down, out of
frame) with its top corners rounded to a radius r.

  1. The model is fitted freely to every image, giving that image's own
     dimensions (top width, leg taper, corner radius, wall thickness, ...).
  2. The ideal is the median of those fits, written to ideal_shape.json --
     or, with --ideal, read from a JSON file you supply or edit, so the
     target can be your intended geometry instead of the samples' median.
  3. The ideal is then placed on every image allowing only position,
     rotation and one uniform scale (the foam may shrink/swell and sit
     differently, but should keep its shape). Everything else is a change:
     the overlay draws the ideal outline in cyan and colours the real
     outline red where the foam sticks out past it, blue where it falls
     short. Each image's own dimensions are listed next to the ideal's.

Because the model is left-right symmetric, a specimen flipped over between
shots needs no special handling. Contour points on the image border (where
the legs leave the frame) are ignored. Writes one overlay per image plus
ideal_shape.csv.
"""
import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from straighten import IMAGE_EXTS, largest_foreground_mask

# Fitting runs on a downsampled mask (long smooth contours keep sub-pixel
# accuracy over thousands of points); reported values are full-res px.
WORK_SIZE = 1000
# Contour points this close (working px) to the image border are where the
# specimen is clipped by the frame, not a real edge, and are dropped.
BORDER_MARGIN = 6
# Contours shorter than this (working px) are dark specks inside the foam
# or stray dust, not the specimen's outline or hole.
MIN_CONTOUR_LENGTH = 100
# Robust-loss scale (working px): residuals much beyond this -- dust, a
# chipped spot -- count roughly linearly instead of quadratically.
FIT_LOSS_SCALE_PX = 2.0
# Starting rotations (deg) tried for the free fit; the best one is kept.
INIT_ROTATIONS_DEG = (-12, -6, 0, 6, 12)
# Overlay colour scale: deviations at/beyond this (full-res px) saturate.
DEVIATION_COLOR_RANGE_PX = 60.0


@dataclass
class ShapeParams:
    """Model shape, in px, in the model's own frame: x right from the symmetry
    axis, y down from the outer outline's top line.

    Each outline is an isosceles triangle pointing down, with top corners at
    (+-w, t), legs tapering inward by angle a (rad) from vertical, and its
    corners rounded by r. The rounding is an inset-then-offset: the visible
    outline lies r outside that triangle.
    """
    outer_w: float
    outer_a: float
    outer_r: float
    inner_t: float
    inner_w: float
    inner_a: float
    inner_r: float

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)])

    @classmethod
    def from_array(cls, v) -> "ShapeParams":
        return cls(*map(float, v))

    def scaled(self, k: float) -> "ShapeParams":
        v = self.to_array()
        lengths = np.array([1, 0, 1, 1, 1, 0, 1], bool)  # angles don't scale
        v[lengths] *= k
        return ShapeParams.from_array(v)


@dataclass
class Dimensions:
    """Human-readable dimensions derived from ShapeParams (px, degrees)."""
    outer_max_width: float      # widest point of the outer outline (level with its corner centres)
    hole_max_width: float       # widest point of the hole
    top_wall: float             # top cap thickness, on the symmetry axis
    side_wall: float            # leg thickness, at the hole's top level
    outer_taper_deg: float      # each outer leg's lean inward from vertical
    hole_taper_deg: float
    outer_corner_r: float
    hole_corner_r: float


def dimensions(p: ShapeParams) -> Dimensions:
    def leg_x(w, a, r, t, y):
        # Outer edge of a leg at depth y: the triangle's leg line pushed out by r.
        return w - (y - t) * np.tan(a) + r / np.cos(a)

    y = p.inner_t - p.inner_r  # hole's top line
    gap = leg_x(p.outer_w, p.outer_a, p.outer_r, 0.0, y) - leg_x(p.inner_w, p.inner_a, p.inner_r, p.inner_t, y)
    return Dimensions(
        outer_max_width=2 * (p.outer_w + p.outer_r),
        hole_max_width=2 * (p.inner_w + p.inner_r),
        top_wall=p.inner_t - p.inner_r + p.outer_r,
        side_wall=gap * np.cos((p.outer_a + p.inner_a) / 2),
        outer_taper_deg=np.degrees(p.outer_a),
        hole_taper_deg=np.degrees(p.inner_a),
        outer_corner_r=p.outer_r,
        hole_corner_r=p.inner_r,
    )


def _sd_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Exact signed distance from points p (N,2) to triangle abc; negative inside."""
    e = [b - a, c - b, a - c]
    v = [p - a, p - b, p - c]
    s = np.sign(e[0][0] * e[2][1] - e[0][1] * e[2][0])
    dist2, inside = [], []
    for ei, vi in zip(e, v):
        h = np.clip((vi @ ei) / (ei @ ei), 0, 1)
        pq = vi - h[:, None] * ei
        dist2.append((pq ** 2).sum(1))
        inside.append(s * (vi[:, 0] * ei[1] - vi[:, 1] * ei[0]))
    d2 = np.min(dist2, axis=0)
    sgn = np.min(inside, axis=0)
    return -np.sqrt(d2) * np.sign(sgn)


def _sd_rounded_u(p: np.ndarray, w: float, a: float, r: float, t: float) -> np.ndarray:
    apex_y = t + w / np.tan(a)
    tri = _sd_triangle(p, np.array([-w, t]), np.array([w, t]), np.array([0.0, apex_y]))
    return tri - r


def sd_foam(p: np.ndarray, s: ShapeParams) -> np.ndarray:
    """Signed distance (model px) to the foam region: inside the outer outline, outside the hole."""
    outer = _sd_rounded_u(p, s.outer_w, s.outer_a, s.outer_r, 0.0)
    hole = _sd_rounded_u(p, s.inner_w, s.inner_a, s.inner_r, s.inner_t)
    return np.maximum(outer, -hole)


@dataclass
class Pose:
    """Model -> image: q = scale * R(theta) @ p + (tx, ty)."""
    tx: float
    ty: float
    theta: float
    scale: float = 1.0

    def to_model(self, q: np.ndarray) -> np.ndarray:
        c, s = np.cos(-self.theta), np.sin(-self.theta)
        d = (q - [self.tx, self.ty]) / self.scale
        return np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]])


def residuals(points: np.ndarray, pose: Pose, shape: ShapeParams) -> np.ndarray:
    """Signed distance (image px) of each contour point to the posed model's outline."""
    return sd_foam(pose.to_model(points), shape) * pose.scale


def load_contour(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (full-res BGR image, working-scale contour points, full-res px per working px)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    factor = max(gray.shape) / WORK_SIZE
    small = cv2.resize(gray, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
    mask = largest_foreground_mask(small)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c.reshape(-1, 2) for c in contours if len(c) >= MIN_CONTOUR_LENGTH]).astype(np.float64)
    h, w = mask.shape
    keep = (
        (pts[:, 0] >= BORDER_MARGIN) & (pts[:, 0] < w - BORDER_MARGIN)
        & (pts[:, 1] >= BORDER_MARGIN) & (pts[:, 1] < h - BORDER_MARGIN)
    )
    pts = pts[keep]
    if len(pts) < 100:
        raise ValueError(f"Too little object outline found in {path.name}")
    return image, pts, factor


def _initial_shape(points: np.ndarray) -> tuple[float, float, ShapeParams]:
    """Rough model + top-centre position from the contour's bounding box."""
    x0, y0 = points.min(0)
    x1, _ = points.max(0)
    width = x1 - x0
    shape = ShapeParams(
        outer_w=0.32 * width, outer_a=np.radians(10), outer_r=0.18 * width,
        inner_t=0.45 * width, inner_w=0.18 * width, inner_a=np.radians(10), inner_r=0.08 * width,
    )
    return (x0 + x1) / 2, y0 + shape.outer_r, shape


# Bounds keep the model physical: tapers between ~0.5 and 40 deg (a taper of
# 0 would put the triangle's apex at infinity), radii and widths positive.
_SHAPE_LO = [1, np.radians(0.5), 1, 1, 1, np.radians(0.5), 1]
_SHAPE_HI = [np.inf, np.radians(40), np.inf, np.inf, np.inf, np.radians(40), np.inf]


def fit_free(points: np.ndarray) -> tuple[Pose, ShapeParams, float]:
    """Fit pose (no scale) and all shape parameters to one image. Returns (pose, shape, RMS px)."""
    cx, cy, shape0 = _initial_shape(points)
    best = None
    for deg in INIT_ROTATIONS_DEG:
        x0 = np.concatenate([[cx, cy, np.radians(deg)], shape0.to_array()])

        def f(x):
            return residuals(points, Pose(x[0], x[1], x[2]), ShapeParams.from_array(x[3:]))

        res = least_squares(
            f, x0, loss="soft_l1", f_scale=FIT_LOSS_SCALE_PX,
            bounds=([-np.inf, -np.inf, -np.pi / 4] + _SHAPE_LO, [np.inf, np.inf, np.pi / 4] + _SHAPE_HI),
        )
        if best is None or res.cost < best.cost:
            best = res
    x = best.x
    pose, shape = Pose(x[0], x[1], x[2]), ShapeParams.from_array(x[3:])
    rms = float(np.sqrt(np.mean(residuals(points, pose, shape) ** 2)))
    return pose, shape, rms


def fit_pose(points: np.ndarray, shape: ShapeParams, init: Pose) -> Pose:
    """Place a fixed shape on the image: fit only position, rotation and uniform scale."""
    def f(x):
        return residuals(points, Pose(*x), shape)

    res = least_squares(
        f, [init.tx, init.ty, init.theta, init.scale], loss="soft_l1", f_scale=FIT_LOSS_SCALE_PX,
        bounds=([-np.inf, -np.inf, -np.pi / 4, 0.5], [np.inf, np.inf, np.pi / 4, 2.0]),
    )
    return Pose(*res.x)


def model_outline(shape: ShapeParams, pose: Pose, image_shape: tuple[int, int], factor: float):
    """Outline of the posed model as full-res polylines, rasterised on the working grid.

    Stretches along the image border are left out: there the model is just
    cut off by the frame (as the real specimen is), not a real edge.
    """
    h, w = (int(np.ceil(n / factor)) for n in image_shape)
    ys, xs = np.mgrid[0:h, 0:w]
    q = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    inside = (sd_foam(pose.to_model(q), shape) <= 0).reshape(h, w).astype(np.uint8) * 255
    contours, _ = cv2.findContours(inside, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    runs = []
    for c in contours:
        c = c.reshape(-1, 2)
        on_border = (
            (c[:, 0] < BORDER_MARGIN) | (c[:, 0] >= w - BORDER_MARGIN)
            | (c[:, 1] < BORDER_MARGIN) | (c[:, 1] >= h - BORDER_MARGIN)
        )
        # Split into runs of consecutive off-border points.
        breaks = np.flatnonzero(np.diff(np.concatenate([[1], on_border.astype(int), [1]])))
        for start, stop in zip(breaks[::2], breaks[1::2]):
            if stop - start > 1:
                runs.append((c[start:stop].astype(np.float64) * factor).astype(np.int32))
    return runs


def _deviation_color(d: float) -> tuple[int, int, int]:
    """Blue (falls short of ideal) -> green (on it) -> red (sticks out), in BGR."""
    t = float(np.clip(d / DEVIATION_COLOR_RANGE_PX, -1, 1))
    if t >= 0:
        return (0, int(255 * (1 - t)), int(255 * t))
    return (int(255 * -t), int(255 * (1 + t)), 0)


DIM_LABELS = {
    "outer_max_width": "outer max width",
    "hole_max_width": "hole max width",
    "top_wall": "top wall",
    "side_wall": "side wall",
    "outer_taper_deg": "outer taper (deg)",
    "hole_taper_deg": "hole taper (deg)",
    "outer_corner_r": "outer corner radius",
    "hole_corner_r": "hole corner radius",
}


def draw_overlay(image, points_full, dev_full, outline, pose: Pose, own: Dimensions, ideal: Dimensions,
                 rms_full: float, label: str) -> np.ndarray:
    out = image.copy()
    w = out.shape[1]
    lw = max(2, int(round(w / 700)))
    cv2.polylines(out, outline, False, (255, 255, 0), lw, cv2.LINE_AA)
    for (x, y), d in zip(points_full, dev_full):
        cv2.circle(out, (int(round(x)), int(round(y))), lw, _deviation_color(d), -1, cv2.LINE_AA)

    lines = [
        label,
        f"vs ideal: scale {pose.scale * 100:.2f}%  rotation {np.degrees(pose.theta):+.2f} deg  "
        f"RMS deviation {rms_full:.1f}px",
        "cyan = ideal shape; outline red = sticks out, blue = falls short",
        "                       this image    ideal(scaled)   change",
    ]
    own_d, ideal_d = asdict(own), asdict(ideal)
    for key, name in DIM_LABELS.items():
        is_angle = key.endswith("_deg")
        mine, target = own_d[key], ideal_d[key] * (1 if is_angle else pose.scale)
        change = f"{mine - target:+8.2f} deg" if is_angle else f"{(mine / target - 1) * 100:+7.1f} %"
        lines.append(f"{name:<22}{mine:>10.1f}{target:>14.1f}   {change}")

    font_scale = w / 2200
    thick = max(1, int(round(font_scale * 2)))
    line_h = int(42 * font_scale)
    for i, text in enumerate(lines):
        org = (int(20 * font_scale), int((i + 1) * line_h + 10))
        font = cv2.FONT_HERSHEY_SIMPLEX if i < 3 else cv2.FONT_HERSHEY_PLAIN
        fs = font_scale if i < 3 else font_scale * 1.6
        cv2.putText(out, text, org, font, fs, (0, 0, 0), thick * 3, cv2.LINE_AA)
        cv2.putText(out, text, org, font, fs, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Directory of images")
    parser.add_argument("output", type=Path, help="Output directory for overlays, ideal_shape.json and ideal_shape.csv")
    parser.add_argument(
        "--ideal", type=Path, default=None,
        help="JSON file with the ideal ShapeParams in full-res px/radians (e.g. an edited "
             "ideal_shape.json from an earlier run). Default: median of all images' fits",
    )
    args = parser.parse_args()

    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"No images found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Free fit per image (working px).
    fits = []
    for src in files:
        image, points, factor = load_contour(src)
        pose, shape, rms = fit_free(points)
        fits.append((src, image, points, factor, pose, shape))
        print(f"{src.name}: model fit RMS {rms * factor:.1f}px")

    # 2. Ideal shape, kept in full-res px on disk.
    factor = fits[0][3]
    if args.ideal:
        ideal_full = ShapeParams(**json.loads(args.ideal.read_text()))
        print(f"Ideal shape: {args.ideal}")
    else:
        median = np.median([shape.to_array() for *_, shape in fits], axis=0)
        ideal_full = ShapeParams.from_array(median).scaled(factor)
        ideal_path = args.output / "ideal_shape.json"
        ideal_path.write_text(json.dumps(asdict(ideal_full), indent=2) + "\n")
        print(f"Ideal shape (median of all fits) written to {ideal_path}")
    ideal_dims = dimensions(ideal_full)

    # 3. Place the ideal on each image; report deviation and dimension changes.
    csv_path = args.output / "ideal_shape.csv"
    dim_keys = list(DIM_LABELS)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "scale_pct", "rotation_deg", "rms_dev_px", "max_out_px", "max_in_px"]
            + dim_keys + [f"{k}_ideal" for k in dim_keys]
        )
        for src, image, points, factor, free_pose, shape in fits:
            ideal = ideal_full.scaled(1 / factor)
            pose = fit_pose(points, ideal, Pose(free_pose.tx, free_pose.ty, free_pose.theta, 1.0))
            dev = residuals(points, pose, ideal) * factor
            rms = float(np.sqrt(np.mean(dev ** 2)))
            own = dimensions(shape.scaled(factor))
            outline = model_outline(ideal, pose, image.shape[:2], factor)
            overlay = draw_overlay(image, points * factor, dev, outline, pose, own, ideal_dims, rms, src.name)
            cv2.imwrite(str(args.output / src.name), overlay)

            own_d, ideal_d = asdict(own), asdict(ideal_dims)
            writer.writerow(
                [src.name, f"{pose.scale * 100:.2f}", f"{np.degrees(pose.theta):.2f}", f"{rms:.1f}",
                 f"{max(dev.max(), 0):.1f}", f"{min(dev.min(), 0):.1f}"]
                + [f"{own_d[k]:.2f}" for k in dim_keys]
                + [f"{ideal_d[k] * (1 if k.endswith('_deg') else pose.scale):.2f}" for k in dim_keys]
            )
            print(
                f"{src.name}: scale {pose.scale * 100:.2f}%, rot {np.degrees(pose.theta):+.2f} deg, "
                f"RMS deviation from ideal {rms:.1f}px"
            )
    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
