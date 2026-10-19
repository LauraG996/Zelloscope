#!/usr/bin/env python3
"""Check whether a product edge fits the hole of a horseshoe corner protector.

The product (e.g. a board of a given thickness) is pushed into the hole from
the open end. Where the hole is narrower than the product the foam must
compress -- that's what grips it; where it is wider the product is loose.

For each image:
  1. The specimen's symmetry axis is found by fitting ideal_shape.py's
     parametric model, and the foam mask is resampled so that axis is
     vertical (removes tilt from how the sample was placed).
  2. The hole width is measured perpendicular to the axis at every depth,
     from the top of the hole down to where the frame cuts it off.
  3. With --product-thickness, that width is compared to the product:
       squeeze  = product thickness - hole width  (positive = foam compressed)
     reported at the narrowest visible point, along with how long the
     contact zone (squeeze > 0) is and the clearance where the hole is widest.
     --min-squeeze / --max-squeeze turn that into a pass/fail.

Overlay: the product drawn as a band along the axis, red where it overlaps
foam (compression), with hole widths labelled at several depths. Also writes
fit_check.csv (one row per image) and fit_profiles.csv (width vs. depth).

Measurements are in mm when --pix2mm is given, otherwise px (and then
--product-thickness and the squeeze limits are in px too).

NOTE: the hole's mouth is usually outside the frame in these images; the
narrowest *visible* width is reported, and the true mouth may be narrower.
"""
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ideal_shape import fit_free, load_contour
from straighten import IMAGE_EXTS, largest_foreground_mask

# Margin (rectified px) kept above the specimen's top when resampling.
TOP_PAD = 200
# Hole rows whose ends lie this close (px) to the part of the rectified image
# that falls outside the original frame are not measured: there the hole is
# cut off by the frame, not by foam.
FRAME_EDGE_GUARD = 15
# Depths (fraction of measured hole length) at which widths are labelled.
LABEL_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Background blobs smaller than this (full-res px^2) inside the foam are dark
# specks in the material, not the hole; they're filled before measuring.
MAX_SPECK_AREA = 20000


@dataclass
class HoleProfile:
    top_v: int                 # rectified row of the hole's top, on the axis
    depths: np.ndarray         # px below the hole top
    widths: np.ndarray         # px, perpendicular to the axis
    lefts: np.ndarray          # rectified x of the hole's left wall per depth
    rights: np.ndarray


def rectify(image_shape, pose_full, top_offset: float) -> tuple[np.ndarray, tuple[int, int], int]:
    """Affine map rectified -> image (2x3), output size, and the axis column.

    Rectified coords: axis vertical at column x0, the specimen's top at row
    TOP_PAD. top_offset is how far the model origin lies below the specimen's
    top (the outer corner radius: the model's top line is inset by it).
    """
    h, w = image_shape
    c, s = np.cos(pose_full.theta), np.sin(pose_full.theta)
    rot = np.array([[c, -s], [s, c]])
    x0 = w // 2
    t = np.array([pose_full.tx, pose_full.ty])
    origin_v = TOP_PAD + top_offset
    m = np.hstack([rot, (t - rot @ [x0, origin_v])[:, None]])
    return m, (w, int(h + origin_v)), x0


def fill_specks(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask == 0).astype(np.uint8), connectivity=4)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < MAX_SPECK_AREA)
    out = mask.copy()
    out[np.isin(labels, small)] = 255
    return out


def measure_hole(mask: np.ndarray, m: np.ndarray, size: tuple[int, int], x0: int) -> HoleProfile:
    mask = fill_specks(mask)
    flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    # Outside the original frame counts as foam, so the hole can't leak into
    # the surrounding background through the cut-off bottom edge.
    rect = cv2.warpAffine(mask, m, size, flags=flags, borderValue=255)
    valid = cv2.warpAffine(np.full(mask.shape, 255, np.uint8), m, size, flags=flags, borderValue=0)
    near_edge = cv2.dilate(
        (valid == 0).astype(np.uint8), np.ones((2 * FRAME_EDGE_GUARD + 1,) * 2, np.uint8)
    ) > 0

    # Start at the first in-frame row: the padding above is outside the frame
    # and, filled as foam, would otherwise look like the top wall.
    column = (rect[:, x0] > 0) & (valid[:, x0] > 0)
    column[:np.flatnonzero(valid[:, x0])[0]] = False
    foam_rows = np.flatnonzero(column)
    if not len(foam_rows):
        raise ValueError("No foam on the symmetry axis")
    below = np.flatnonzero(~column[foam_rows[0]:]) + foam_rows[0]
    if not len(below):
        raise ValueError("No hole found below the top wall on the symmetry axis")
    top_v = int(below[0])

    _, labels = cv2.connectedComponents((rect == 0).astype(np.uint8), connectivity=4)
    hole = labels == labels[top_v, x0]

    depths, widths, lefts, rights = [], [], [], []
    for v in range(top_v, size[1]):
        xs = np.flatnonzero(hole[v])
        if not len(xs):
            break
        left, right = xs[0], xs[-1]
        if near_edge[v, max(left - 2, 0)] or near_edge[v, min(right + 2, size[0] - 1)]:
            break
        depths.append(v - top_v)
        widths.append(right - left + 1)
        lefts.append(left)
        rights.append(right)
    if not depths:
        raise ValueError("Hole is cut off by the frame right at its top")
    return HoleProfile(top_v, np.array(depths), np.array(widths, float), np.array(lefts), np.array(rights))


@dataclass
class FitResult:
    hole_max_width: float       # widest point of the hole (just below its rounded top corners)
    min_width: float
    min_width_depth: float
    measured_depth: float
    squeeze: float | None = None          # product - min width; > 0 = compressed
    max_clearance: float | None = None    # hole max width - product: play where the hole is widest
    contact_length: float | None = None   # length of hole (from the visible end) narrower than product
    passed: bool | None = None


def evaluate(p: HoleProfile, k: float, thickness: float | None,
             min_squeeze: float | None, max_squeeze: float | None) -> FitResult:
    """k converts px to the output unit (mm/px, or 1).

    Only the part of the hole from its widest point toward the mouth is
    considered: that's where the product sits. Above it the hole's rounded
    top corners narrow again, but the product never gets there.
    """
    i_max = int(np.argmax(p.widths))
    i_min = i_max + int(np.argmin(p.widths[i_max:]))
    r = FitResult(
        hole_max_width=p.widths.max() * k,  # widest point, just below the rounded top corners
        min_width=p.widths[i_min] * k,
        min_width_depth=p.depths[i_min] * k,
        measured_depth=p.depths[-1] * k,
    )
    if thickness is not None:
        r.squeeze = thickness - r.min_width
        r.max_clearance = r.hole_max_width - thickness
        r.contact_length = float(np.count_nonzero(p.widths[i_max:] * k < thickness)) * k
        ok = True
        if min_squeeze is not None:
            ok &= r.squeeze >= min_squeeze
        if max_squeeze is not None:
            ok &= r.squeeze <= max_squeeze
        r.passed = bool(ok)
    return r


def draw_overlay(image, m, size, x0, rect_mask, p: HoleProfile, r: FitResult, k: float,
                 thickness: float | None, unit: str, label: str) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    lw = max(2, int(round(w / 700)))
    to_image = lambda pts: (np.column_stack([pts, np.ones(len(pts))]) @ m.T).round().astype(np.int32)

    if thickness is not None:
        half = thickness / k / 2
        y_end = p.top_v + p.depths[-1]
        band = np.zeros((size[1], size[0]), np.uint8)
        cv2.rectangle(band, (int(x0 - half), p.top_v), (int(x0 + half), y_end), 255, -1)
        overlap = cv2.bitwise_and(band, rect_mask)
        overlap_img = cv2.warpAffine(overlap, m, (w, h), flags=cv2.INTER_NEAREST) > 0
        band_img = cv2.warpAffine(band, m, (w, h), flags=cv2.INTER_NEAREST) > 0
        tint = out.copy()
        tint[band_img] = (255, 200, 0)
        tint[overlap_img] = (0, 0, 255)
        out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)
        corners = np.array([[x0 - half, p.top_v], [x0 + half, p.top_v], [x0 + half, y_end], [x0 - half, y_end]])
        cv2.polylines(out, [to_image(corners)], True, (255, 200, 0), lw, cv2.LINE_AA)

    # Symmetry axis and width ticks.
    axis = to_image(np.array([[x0, p.top_v - 50], [x0, p.top_v + p.depths[-1]]]))
    cv2.line(out, tuple(axis[0]), tuple(axis[1]), (200, 200, 200), max(1, lw // 2), cv2.LINE_AA)
    font_scale = w / 2400
    thick = max(1, int(round(font_scale * 2)))
    for frac in LABEL_FRACTIONS:
        i = int(round(frac * (len(p.depths) - 1)))
        v = p.top_v + p.depths[i]
        a, b = to_image(np.array([[p.lefts[i], v], [p.rights[i] + 1, v]]))
        cv2.line(out, tuple(a), tuple(b), (0, 255, 255), lw, cv2.LINE_AA)
        text = f"{p.widths[i] * k:.2f} {unit}"
        org = (int(b[0] + 15), int(b[1] + 10))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick * 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thick, cv2.LINE_AA)

    lines = [label, f"hole max width {r.hole_max_width:.2f} {unit}, narrowest visible {r.min_width:.2f} {unit} "
                    f"at {r.min_width_depth:.2f} {unit} deep (measured {r.measured_depth:.2f} {unit})"]
    if thickness is not None:
        verdict = "" if r.passed is None else ("  PASS" if r.passed else "  FAIL")
        lines += [
            f"product {thickness:.2f} {unit}: squeeze {r.squeeze:+.2f} {unit} at narrowest, "
            f"contact over {r.contact_length:.2f} {unit}, max clearance {r.max_clearance:+.2f} {unit}{verdict}",
            "band = product, red = foam it must compress",
        ]
    lines.append("hole mouth may be outside the frame: narrowest *visible* width shown")
    for i, text in enumerate(lines):
        org = (int(20 * font_scale), int((i + 1) * 45 * font_scale + 10))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick * 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Image file or directory of images")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--pix2mm", type=float, default=None, help="Millimetres per pixel (report in mm)")
    parser.add_argument("--product-thickness", type=float, default=None,
                        help="Thickness of the product edge that goes into the hole (mm with --pix2mm, else px)")
    parser.add_argument("--min-squeeze", type=float, default=None,
                        help="Fail if the foam compresses the product by less than this at the narrowest point")
    parser.add_argument("--max-squeeze", type=float, default=None,
                        help="Fail if the foam compresses the product by more than this at the narrowest point")
    parser.add_argument("--at-depth", type=float, default=None,
                        help="Depth below the hole top at which to compare all images' hole width "
                             "(default: the deepest point visible in every image)")
    args = parser.parse_args()

    files = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"No images found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    k = args.pix2mm or 1.0
    unit = "mm" if args.pix2mm else "px"

    results = []
    profile_path = args.output / "fit_profiles.csv"
    with open(profile_path, "w", newline="") as fp:
        profiles = csv.writer(fp)
        profiles.writerow(["filename", f"depth_{unit}", f"width_{unit}"])
        for src in files:
            image, points, factor = load_contour(src)
            pose, shape, _ = fit_free(points)
            pose.tx, pose.ty = pose.tx * factor, pose.ty * factor
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = largest_foreground_mask(gray)
            m, size, x0 = rectify(gray.shape, pose, shape.outer_r * factor)
            prof = measure_hole(mask, m, size, x0)
            r = evaluate(prof, k, args.product_thickness, args.min_squeeze, args.max_squeeze)
            results.append((src, prof, r))

            rect_mask = cv2.warpAffine(fill_specks(mask), m, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
            overlay = draw_overlay(image, m, size, x0, rect_mask, prof, r, k,
                                   args.product_thickness, unit, src.name)
            cv2.imwrite(str(args.output / src.name), overlay)
            for d, wd in zip(prof.depths[::10], prof.widths[::10]):
                profiles.writerow([src.name, f"{d * k:.3f}", f"{wd * k:.3f}"])

            msg = f"{src.name}: hole max {r.hole_max_width:.2f}{unit}, narrowest visible {r.min_width:.2f}{unit}"
            if r.squeeze is not None:
                msg += f", squeeze {r.squeeze:+.2f}{unit}, contact {r.contact_length:.2f}{unit}"
                if r.passed is not None:
                    msg += ", PASS" if r.passed else ", FAIL"
            print(msg)

    # The narrowest *visible* width depends on where the frame happens to cut
    # each specimen off, so also compare every image at one shared depth.
    at_depth = args.at_depth if args.at_depth is not None else min(r.measured_depth for *_, r in results)
    print(f"Hole width {at_depth:.2f}{unit} below the hole top:")
    summary_path = args.output / "fit_check.csv"
    with open(summary_path, "w", newline="") as fs:
        summary = csv.writer(fs)
        summary.writerow([f"{c}_{unit}" if c not in ("filename", "pass") else c for c in (
            "filename", "hole_max_width", "width_at_depth", "squeeze_at_depth", "min_width", "min_width_depth",
            "measured_depth", "squeeze", "max_clearance", "contact_length", "pass")] + [f"at_depth_{unit}"])
        fmt = lambda v: "" if v is None else f"{v:.3f}"
        for src, prof, r in results:
            idx = np.searchsorted(prof.depths * k, at_depth)
            width_at = prof.widths[idx] * k if idx < len(prof.depths) else None
            squeeze_at = (args.product_thickness - width_at
                          if args.product_thickness is not None and width_at is not None else None)
            summary.writerow([src.name, fmt(r.hole_max_width), fmt(width_at), fmt(squeeze_at), fmt(r.min_width),
                              fmt(r.min_width_depth), fmt(r.measured_depth), fmt(r.squeeze), fmt(r.max_clearance),
                              fmt(r.contact_length), "" if r.passed is None else ("yes" if r.passed else "no"),
                              fmt(at_depth)])
            print(f"  {src.name}: " + ("not visible this deep" if width_at is None else
                  f"{width_at:.2f}{unit}" + ("" if squeeze_at is None else f" (squeeze {squeeze_at:+.2f}{unit})")))
    print(f"Results written to {summary_path} and {profile_path}")


if __name__ == "__main__":
    main()
