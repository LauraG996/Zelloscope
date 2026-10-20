"""Canonical mm frame, rigid transforms, rasterising and IoU alignment to a reference.

Canonical frame: mm, origin at the outer area centroid, principal axis vertical, image-style y axis
(pointing down) with the base at the bottom (+y) and the tips at the top (-y).
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import AlignmentParams

# fillPoly fractional bits: render polygons with 1/16 px precision.
_SHIFT = 4


def resample(poly: np.ndarray, step: float) -> np.ndarray:
    """Resample a closed polyline to (roughly) uniform arc-length spacing."""
    closed = np.vstack([poly, poly[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(8, int(round(s[-1] / step)))
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def _moments(poly: np.ndarray) -> dict:
    return cv2.moments(poly.astype(np.float32))


def canonicalize(outer: np.ndarray, cavity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put an (outer, cavity) pair in mm into the canonical frame (a proper rotation, never a mirror)."""
    m = _moments(outer)
    if m["m00"] == 0:
        raise ValueError("degenerate outer contour")
    cen = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    cov = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / m["m00"]
    _, vecs = np.linalg.eigh(cov)
    ax = vecs[:, 1]                         # major axis -> new y
    perp = np.array([ax[1], -ax[0]])        # -> new x; det([perp, ax]) = +1
    rot = np.column_stack([perp, ax])
    o, c = (outer - cen) @ rot, (cavity - cen) @ rot
    # Base at the bottom: the cavity is widest near the base, so its area centroid lies on that side.
    mc = _moments(c)
    if mc["m01"] / mc["m00"] < (c[:, 1].min() + c[:, 1].max()) / 2:
        o, c = -o, -c                       # 180 deg rotation
    return o, c


def transform(pts: np.ndarray, angle_deg: float = 0.0, dx: float = 0.0, dy: float = 0.0,
              mirror: bool = False) -> np.ndarray:
    p = pts * [-1.0, 1.0] if mirror else pts
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return p @ rot.T + [dx, dy]


@dataclass(frozen=True)
class Canvas:
    """Square raster of the mm frame, origin in the middle."""
    res: float          # px per mm
    size_mm: float

    @property
    def n(self) -> int:
        return int(round(self.size_mm * self.res))

    def to_px(self, pts_mm: np.ndarray) -> np.ndarray:
        """Pixel-centre coordinates (float)."""
        return pts_mm * self.res + self.n / 2

    def to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        return (pts_px - self.n / 2) / self.res

    def render(self, poly_mm: np.ndarray) -> np.ndarray:
        mask = np.zeros((self.n, self.n), np.uint8)
        pts = np.round(self.to_px(poly_mm) * (1 << _SHIFT)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1, lineType=cv2.LINE_8, shift=_SHIFT)
        return mask

    def contour_mm(self, mask: np.ndarray) -> np.ndarray:
        """Largest external contour of a mask, back in mm."""
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            raise ValueError("empty mask")
        return self.to_mm(max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 0.0


@dataclass
class Alignment:
    angle_deg: float
    dx_mm: float
    dy_mm: float
    mirrored: bool
    score: float

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return transform(pts, self.angle_deg, self.dx_mm, self.dy_mm, self.mirrored)

    def to_dict(self) -> dict:
        return dict(rot_deg=round(self.angle_deg, 3), dx_mm=round(self.dx_mm, 3), dy_mm=round(self.dy_mm, 3),
                    mirrored=self.mirrored, match_score=round(self.score, 4))


class _Scorer:
    def __init__(self, canvas: Canvas, ref_outer: np.ndarray, ref_cavity: np.ndarray, cavity_weight: float):
        self.canvas = canvas
        ro, rc = canvas.render(ref_outer).astype(bool), canvas.render(ref_cavity).astype(bool)
        self.ref_foam, self.ref_cav = ro & ~rc, rc
        self.w = cavity_weight

    def __call__(self, outer: np.ndarray, cavity: np.ndarray, p: tuple) -> float:
        o = self.canvas.render(transform(outer, *p)).astype(bool)
        c = self.canvas.render(transform(cavity, *p)).astype(bool)
        return iou(o & ~c, self.ref_foam) + self.w * iou(c, self.ref_cav)


def _descend(score, p: list, s: float, steps) -> tuple[list, float]:
    """Greedy coordinate search over (angle, dx, dy) with shrinking steps."""
    for step_a, step_t in steps:
        improved = True
        while improved:
            improved = False
            for i, st in ((0, step_a), (1, step_t), (2, step_t)):
                for sgn in (-1, 1):
                    q = list(p)
                    q[i] += sgn * st
                    sq = score(q)
                    if sq > s + 1e-7:
                        p, s, improved = q, sq, True
    return p, s


def align(outer: np.ndarray, cavity: np.ndarray, ref_outer: np.ndarray, ref_cavity: np.ndarray,
          params: AlignmentParams) -> Alignment:
    """Mirror + rotation + translation of a canonical (outer, cavity) pair that maximises IoU with the reference.

    Both shapes are already in the canonical frame, so only small residual rotations are expected; the 180 deg
    start covers a wrongly picked base end. Search is coarse-to-fine on two canvas resolutions.
    """
    coarse = _Scorer(Canvas(params.coarse_res_px_per_mm, params.canvas_size_mm), ref_outer, ref_cavity,
                     params.cavity_weight)
    fine = _Scorer(Canvas(params.canvas_res_px_per_mm, params.canvas_size_mm), ref_outer, ref_cavity,
                   params.cavity_weight)

    best = None
    for mirror in (False, True):
        for a0 in (0.0, 180.0):
            sc = lambda q, m=mirror: coarse(outer, cavity, (*q, m))
            p, s = _descend(sc, [a0, 0.0, 0.0], sc([a0, 0.0, 0.0]), [(4.0, 2.0), (1.0, 0.5)])
            if best is None or s > best[2]:
                best = (p, mirror, s)
    p, mirror, _ = best
    sf = lambda q: fine(outer, cavity, (*q, mirror))
    p, s = _descend(sf, p, sf(p), [(0.5, 0.2), (0.1, 0.05)])
    return Alignment(float(p[0]), float(p[1]), float(p[2]), mirror, float(s))
