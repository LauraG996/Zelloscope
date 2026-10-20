"""Signed distance of a part's contours to the reference contours (mm)."""
import cv2
import numpy as np


def signed_distances(points: np.ndarray, ref_contour: np.ndarray) -> np.ndarray:
    """Distance of each point to the reference polyline; + outside the reference region, - inside.

    For the outer contour + means more material than the reference; for the cavity + means a bigger cavity.
    """
    ref = ref_contour.astype(np.float32).reshape(-1, 1, 2)
    # pointPolygonTest is + inside; flip so that + = bigger than the reference.
    return -np.array([cv2.pointPolygonTest(ref, (float(x), float(y)), True) for x, y in points])


def stats(d: np.ndarray) -> dict:
    a = np.abs(d)
    r = lambda x: round(float(x), 3)
    return dict(max_abs=r(a.max()), p99_abs=r(np.percentile(a, 99)), rms=r(np.sqrt((d ** 2).mean())),
                mean_signed=r(d.mean()), min_signed=r(d.min()), max_signed=r(d.max()))


def contour_deviation(outer: np.ndarray, cavity: np.ndarray, ref_outer: np.ndarray, ref_cavity: np.ndarray):
    """Returns (stats per contour, raw signed distances per contour)."""
    raw = {"outer": signed_distances(outer, ref_outer), "cavity": signed_distances(cavity, ref_cavity)}
    return {k: stats(v) for k, v in raw.items()}, raw


def worst_p99(dev_stats: dict) -> float:
    return max(s["p99_abs"] for s in dev_stats.values())
