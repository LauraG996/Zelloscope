"""Overlay images and console summaries."""
from pathlib import Path

import cv2
import numpy as np

from .pipeline import InspectionResult

OVERLAY_PX_PER_MM = 8.0
OVERLAY_MARGIN_MM = 8.0


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
    size[1] += 90                                   # room for text
    to_px = lambda p: np.round((p - lo) * OVERLAY_PX_PER_MM + [0, 50]).astype(np.int32)

    img = np.full((size[1], size[0], 3), 30, np.uint8)
    foam = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(foam, [to_px(ref_outer)], 1)
    cv2.fillPoly(foam, [to_px(ref_cavity)], 0)
    img[foam > 0] = (90, 70, 40)
    cv2.polylines(img, [to_px(ref_outer), to_px(ref_cavity)], True, (255, 255, 255), 2, cv2.LINE_AA)
    for name, contour in (("outer", res.outer), ("cavity", res.cavity)):
        for (x, y), d in zip(to_px(contour), res.deviations[name]):
            cv2.circle(img, (int(x), int(y)), 2, _deviation_colour(d, tol), -1)

    colour = (80, 220, 80) if res.passed else (60, 60, 255)
    cv2.putText(img, title, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    for i, line in enumerate([f"white = reference, dots green->red = 0->{tol} mm |deviation|",
                              *res.reasons[:2]]):
        cv2.putText(img, line, (10, size[1] - 60 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def summary_line(name: str, m: dict, dev: dict, suffix: str = "") -> str:
    return (f"{name}: L={m['length']:.2f} W={m['inner_width_max']:.2f} T={m['base_thickness']:.2f} "
            f"outerW={m['outer_width_max']:.2f} walls={m['wall_left']:.2f}/{m['wall_right']:.2f} "
            f"tipdiff={m['tip_height_diff']:.2f}  dev p99 outer/cav="
            f"{dev['outer']['p99_abs']:.2f}/{dev['cavity']['p99_abs']:.2f}{suffix}")
