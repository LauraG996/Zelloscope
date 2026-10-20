"""Overlay images and console summaries."""
from pathlib import Path

import cv2
import numpy as np

from .measure import line_extents
from .pipeline import InspectionResult

OVERLAY_PX_PER_MM = 8.0
OVERLAY_MARGIN_MM = 8.0
# Raster resolution used to offset contours for the tolerance band.
BAND_PX_PER_MM = 20.0
MIN_BGR = (255, 200, 0)     # light blue
MAX_BGR = (0, 140, 255)     # orange


def offset_contours(poly: np.ndarray, d: float) -> list[np.ndarray]:
    """Contours of the region within the polygon grown (d > 0) or shrunk (d < 0) by |d| mm."""
    margin = abs(d) + 2
    lo = poly.min(axis=0) - margin
    size = np.ceil((poly.max(axis=0) + margin - lo) * BAND_PX_PER_MM).astype(int) + 1
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(mask, [np.round((poly - lo) * BAND_PX_PER_MM * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    if d >= 0:
        dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / BAND_PX_PER_MM
        region = dist <= d
    else:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / BAND_PX_PER_MM
        region = dist > -d
    cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2) / BAND_PX_PER_MM + lo for c in cs if len(c) > 2]


def _draw_band(img, to_px, contours, tol: float, thickness: int = 1) -> None:
    """Min (contour - tol) and max (contour + tol) lines around each reference contour."""
    for contour in contours:
        for d, colour in ((-tol, MIN_BGR), (tol, MAX_BGR)):
            polys = [to_px(c) for c in offset_contours(contour, d)]
            cv2.polylines(img, polys, True, colour, thickness, cv2.LINE_AA)


def _deviation_colour(d: float, tol: float) -> tuple[int, int, int]:
    """BGR: green at 0 -> red at |d| >= tol."""
    t = min(abs(d) / tol, 1.0) if tol > 0 else 1.0
    return 0, int(round(255 * (1 - t))), int(round(255 * t))


def draw_overlay(ref_outer: np.ndarray, ref_cavity: np.ndarray, res: InspectionResult, path: str | Path,
                 title: str) -> None:
    """Reference contour in white, part contour coloured green -> red by |deviation| up to the tolerance."""
    tol = res.data["contour_tol_mm"]
    pts = np.vstack([ref_outer, ref_cavity, res.outer, res.cavity])
    lo = pts.min(axis=0) - OVERLAY_MARGIN_MM
    size = np.ceil((pts.max(axis=0) + OVERLAY_MARGIN_MM - lo) * OVERLAY_PX_PER_MM).astype(int)
    size[1] += 110                                  # room for text
    to_px = lambda p: np.round((p - lo) * OVERLAY_PX_PER_MM + [0, 50]).astype(np.int32)

    img = np.full((size[1], size[0], 3), 30, np.uint8)
    foam = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(foam, [to_px(ref_outer)], 1)
    cv2.fillPoly(foam, [to_px(ref_cavity)], 0)
    img[foam > 0] = (90, 70, 40)
    _draw_band(img, to_px, (ref_outer, ref_cavity), tol)
    cv2.polylines(img, [to_px(ref_outer), to_px(ref_cavity)], True, (255, 255, 255), 2, cv2.LINE_AA)
    for name, contour in (("outer", res.outer), ("cavity", res.cavity)):
        for (x, y), d in zip(to_px(contour), res.deviations[name]):
            cv2.circle(img, (int(x), int(y)), 2, _deviation_colour(d, tol), -1)

    colour = (80, 220, 80) if res.passed else (60, 60, 255)
    cv2.putText(img, title, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    for i, line in enumerate([f"white = ideal, blue/orange = min/max (-/+{tol} mm)",
                              f"part: green->red = 0->{tol} mm deviation",
                              *res.reasons[:2]]):
        cv2.putText(img, line, (10, size[1] - 70 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def draw_reference(ref_outer: np.ndarray, ref_cavity: np.ndarray, m: dict, path: str | Path, title: str,
                   tol: float | None = None, parts: list[tuple[np.ndarray, np.ndarray]] = (),
                   px_per_mm: float = OVERLAY_PX_PER_MM) -> None:
    """The reference (ideal) shape, base at the bottom, with its main dimensions and a 10 mm scale bar.

    With `tol`, min/max lines are drawn at -/+tol around both contours: the band a part's outline must stay
    inside (for 99 % of its points) to pass the contour check. `parts` (already aligned to the reference)
    are drawn as thin grey outlines to show the spread.
    """
    pts = np.vstack([ref_outer, ref_cavity, *[p for pair in parts for p in pair]])
    lo = pts.min(axis=0) - [OVERLAY_MARGIN_MM + 14 + (tol or 0), OVERLAY_MARGIN_MM]
    size = np.ceil((pts.max(axis=0) + OVERLAY_MARGIN_MM - lo) * px_per_mm).astype(int)
    size += [0, 130]                                 # room for text
    to_px = lambda p: np.round((np.asarray(p, float) - lo) * px_per_mm + [0, 50]).astype(np.int32)
    pt = lambda x, y: tuple(int(v) for v in to_px([x, y]))

    img = np.full((size[1], size[0], 3), 30, np.uint8)
    foam = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(foam, [to_px(ref_outer)], 1)
    cv2.fillPoly(foam, [to_px(ref_cavity)], 0)
    img[foam > 0] = (150, 100, 40)
    for o, c in parts:
        cv2.polylines(img, [to_px(o), to_px(c)], True, (140, 140, 140), 1, cv2.LINE_AA)
    if tol:
        _draw_band(img, to_px, (ref_outer, ref_cavity), tol, thickness=2)
    cv2.polylines(img, [to_px(ref_outer), to_px(ref_cavity)], True, (255, 255, 255), 2, cv2.LINE_AA)

    yellow, font = (0, 220, 255), cv2.FONT_HERSHEY_SIMPLEX
    label = lambda text, p: cv2.putText(img, text, p, font, 0.5, yellow, 1, cv2.LINE_AA)
    # inner width at the widest cavity row
    y_w = ref_cavity[:, 1].max() - m["inner_width_max_height_mm"]
    cl, cr = (v[0] for v in line_extents(ref_cavity, np.array([y_w]), axis=1))
    cv2.line(img, pt(cl, y_w), pt(cr, y_w), yellow, 2, cv2.LINE_AA)
    label(f"W {m['inner_width_max']:.2f}", pt(cl + 1, y_w - 1))
    # base thickness on the centre line
    xc = (cl + cr) / 2
    _, yc = line_extents(ref_cavity, np.array([xc]), axis=0)
    cv2.line(img, pt(xc, yc[0]), pt(xc, yc[0] + m["base_thickness"]), yellow, 2, cv2.LINE_AA)
    label(f"T {m['base_thickness']:.2f}", pt(xc + 1, yc[0] + m["base_thickness"] / 2))
    # length along the axis, left of the part
    x_l = ref_outer[:, 0].min() - 5 - (tol or 0)
    y0, y1 = ref_outer[:, 1].min(), ref_outer[:, 1].max()
    cv2.line(img, pt(x_l, y0), pt(x_l, y1), yellow, 2, cv2.LINE_AA)
    label(f"L {m['length']:.2f}", pt(x_l - 13, (y0 + y1) / 2))
    # 10 mm scale bar
    sb = (int(10), size[1] - 75)
    cv2.line(img, sb, (sb[0] + int(round(10 * px_per_mm)), sb[1]), (255, 255, 255), 3)
    cv2.putText(img, "10 mm", (sb[0], sb[1] - 8), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(img, title, (10, 32), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    lines = [f"outer width {m['outer_width_max']:.2f}  walls {m['wall_left']:.2f}/{m['wall_right']:.2f}"
             f"  tip diff {m['tip_height_diff']:.2f}  (mm)"]
    legend = "white = ideal (mean of good parts)"
    if tol:
        legend += f", blue = min (-{tol} mm), orange = max (+{tol} mm)"
    if parts:
        legend += ", grey = parts"
    lines.append(legend)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, size[1] - 40 + 20 * i), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def summary_line(name: str, m: dict, dev: dict, suffix: str = "") -> str:
    return (f"{name}: L={m['length']:.2f} W={m['inner_width_max']:.2f} T={m['base_thickness']:.2f} "
            f"outerW={m['outer_width_max']:.2f} walls={m['wall_left']:.2f}/{m['wall_right']:.2f} "
            f"tipdiff={m['tip_height_diff']:.2f}  dev p99 outer/cav="
            f"{dev['outer']['p99_abs']:.2f}/{dev['cavity']['p99_abs']:.2f}{suffix}")
