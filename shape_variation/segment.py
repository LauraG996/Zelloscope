"""Image -> outer + cavity contours in mm, in a canonical frame, within a common window from the base.

Canonical frame (same convention as profile_check): mm, y pointing down, profile axis vertical, base at the
bottom (outer base at y = 0, tips towards negative y), x = 0 on the cavity midline.

The rig images show only the base and part of the legs (the frame cuts the legs off), so every part is compared
within a window that reaches `depth` mm up from the outer base; the window is closed by a straight cut line.
With a field of view that shows the whole profile, `depth=None` uses the full outline.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from profile_check.segmentation import foam_mask as hsv_foam_mask

from .io import ImageRecord


class SegmentationError(RuntimeError):
    pass


@dataclass
class Prepared:
    """One image in the canonical raster, before cutting the window."""
    record: ImageRecord
    foam: np.ndarray            # bool, canonical raster
    cavity: np.ndarray          # bool, canonical raster
    valid: np.ndarray           # bool: inside the original image frame
    res: float                  # canonical raster px per mm
    base_row: float             # row of the outer base
    axis_col: float             # column of the cavity midline
    visible_depth_mm: float     # how far up from the base the part is fully inside the frame
    full_view: bool             # whole profile (tips included) inside the frame
    image_to_canvas: np.ndarray = field(repr=False)   # 2x3 affine, original image px -> canonical raster px

    def to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        return (np.asarray(pts_px, float) - [self.axis_col, self.base_row]) / self.res

    def image_px_to_mm(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float)
        return self.to_mm(pts @ self.image_to_canvas[:, :2].T + self.image_to_canvas[:, 2])


@dataclass
class Shape:
    name: str
    outer: np.ndarray           # closed polygon (mm, canonical)
    cavity: np.ndarray          # closed polygon (mm, canonical)
    outer_curve: np.ndarray     # n_outer points with correspondence: window = open curve left cut -> base ->
    cavity_curve: np.ndarray    # right cut; full = closed curve starting at the tip on the axis
    depth_mm: float | None      # window depth used (None = full outline)
    record: ImageRecord = field(repr=False)
    prepared: Prepared = field(repr=False)
    holes: list = field(default_factory=list, repr=False)   # generic mode: every hole (dense, mm), largest first


def _odd(n: float) -> int:
    return max(3, int(round(n)) | 1)


def _largest(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return np.zeros(mask.shape, bool)
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def foam_mask(img: np.ndarray, px_per_mm: float, seg: dict, spec_seg) -> np.ndarray:
    """Bool mask of the foam cut face (largest blob, small pores filled)."""
    if seg["method"] == "hsv":
        foam = hsv_foam_mask(img, spec_seg, px_per_mm) > 0
    elif seg["method"] == "otsu":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = cv2.GaussianBlur(gray, (0, 0), max(0.5, seg["blur_mm"] * px_per_mm))
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(seg["cleanup_kernel_mm"] * px_per_mm),) * 2)
        th = cv2.morphologyEx(cv2.morphologyEx(th, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        foam = _largest(th > 0)
    else:
        raise ValueError(f"unknown segmentation method {seg['method']!r}")
    if not foam.any():
        raise SegmentationError("no foam found")
    # Fill pores: background blobs inside the foam that are too small to be the cavity.
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~foam).astype(np.uint8), connectivity=4)
    max_pore = seg["max_pore_area_mm2"] * px_per_mm ** 2
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    for i in range(1, n):
        if i not in border and stats[i, cv2.CC_STAT_AREA] < max_pore:
            foam[labels == i] = True
    return foam


def cavity_mask(foam: np.ndarray) -> np.ndarray:
    """Background inside the foam's convex hull: works for a closed cavity, an open mouth and legs cut by the
    frame alike (the hull closes the mouth / the frame edge). Largest such region."""
    pts = cv2.findNonZero(foam.astype(np.uint8))
    hull = np.zeros(foam.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    cav = _largest((hull > 0) & ~foam)
    if cav.sum() < 0.02 * foam.sum():
        raise SegmentationError("no cavity found")
    return cav


def _axis_angle(cav: np.ndarray) -> float:
    """Angle (rad) of the cavity's major axis, image coordinates."""
    ys, xs = np.nonzero(cav)
    cov = np.cov(np.vstack([xs, ys]).astype(float))
    w, v = np.linalg.eigh(cov)
    ax = v[:, 1]
    return float(np.arctan2(ax[1], ax[0]))


def _midline_tilt(cav: np.ndarray, res: float) -> float:
    """Residual tilt (rad) of the cavity midline in a roughly vertical cavity: slope of the row midpoints."""
    rows = np.nonzero(cav.any(axis=1))[0]
    if len(rows) < 10:
        return 0.0
    lo, hi = rows.min(), rows.max()
    span = hi - lo
    sel = rows[(rows > lo + 0.1 * span) & (rows < hi - 0.1 * span)]
    mids = np.array([np.nonzero(cav[r])[0][[0, -1]].mean() for r in sel])
    slope = np.polyfit(sel, mids, 1)[0]       # dx/dy
    return float(np.arctan(slope))


def _canvas_transform(shape_hw, angle: float, centre, scale: float):
    """Affine (2x3) that rotates by -angle about centre and scales, plus the canvas size holding the whole frame."""
    a = -angle
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * scale
    h, w = shape_hw
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    moved = (corners - centre) @ rot.T
    lo, hi = moved.min(axis=0) - 4, moved.max(axis=0) + 4
    m = np.hstack([rot, (-centre @ rot.T - lo)[:, None]])
    size = tuple(int(np.ceil(v)) for v in hi - lo)
    return m, size


def _warp(mask: np.ndarray, m: np.ndarray, size) -> np.ndarray:
    return cv2.warpAffine(mask.astype(np.float32), m, size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 0.5


def prepare(record: ImageRecord, cfg) -> Prepared:
    if record.mm_per_px is None:
        raise SegmentationError(f"unknown scale: {record.scale_source}")
    img = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
    if img is None:
        raise SegmentationError(f"cannot read {record.path}")
    seg = cfg["segmentation"]
    px_per_mm = 1.0 / record.mm_per_px
    # Work on a 2x-oversampled copy of the canonical resolution (fast, and well below the cell size).
    mask_res = min(px_per_mm, 2 * seg["work_px_per_mm"])
    f = mask_res / px_per_mm
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    foam = foam_mask(small, mask_res, seg, cfg.spec.segmentation)
    cav = cavity_mask(foam)

    res = seg["work_px_per_mm"]
    centre = np.array(np.nonzero(cav)[::-1], float).mean(axis=1)
    angle = _axis_angle(cav) - np.pi / 2            # rotate the major axis onto the vertical
    for _ in range(2):
        m, size = _canvas_transform(foam.shape, angle, centre, res / mask_res)
        cav_c = _warp(cav, m, size)
        angle += _midline_tilt(cav_c, res)
    m, size = _canvas_transform(foam.shape, angle, centre, res / mask_res)
    foam_c, cav_c = _warp(foam, m, size), _warp(cav, m, size)
    valid_c = _warp(np.ones(foam.shape, bool), m, size)

    # Base at the bottom: the cavity is widest near the base, so its area centroid lies on that side.
    rows = np.nonzero(cav_c.any(axis=1))[0]
    cy = np.nonzero(cav_c)[0].mean()
    if cy < (rows.min() + rows.max()) / 2:
        angle += np.pi
        m, size = _canvas_transform(foam.shape, angle, centre, res / mask_res)
        foam_c, cav_c = _warp(foam, m, size), _warp(cav, m, size)
        valid_c = _warp(np.ones(foam.shape, bool), m, size)

    # Where does the frame cut the part off? Rows (from the base up) where foam meets the frame edge.
    margin = max(1, int(round(seg["frame_margin_mm"] * res)))
    invalid = cv2.dilate((~valid_c).astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
    foam_rows = np.nonzero(foam_c.any(axis=1))[0]
    base_row, top_row = float(foam_rows.max()), float(foam_rows.min())
    touching = np.nonzero((foam_c & invalid).any(axis=1))[0]
    full_view = len(touching) == 0
    if full_view:
        visible = (base_row - top_row) / res
    else:
        if touching.max() >= base_row - 2 * res:
            raise SegmentationError("the base is cut off by the image frame")
        visible = (base_row - touching.max() - 1) / res

    rows = np.nonzero(cav_c.any(axis=1))[0]
    mid_rows = rows[rows < rows.max() - 2 * res]           # skip the rounded cavity bottom
    axis_col = float(np.median([np.nonzero(cav_c[r])[0][[0, -1]].mean() for r in mid_rows])) \
        if len(mid_rows) else float(np.nonzero(cav_c)[1].mean())

    # image px -> canonical px: full-res image px -> mask px (scale f) -> canvas (m)
    to_canvas = m.copy()
    to_canvas[:, :2] = m[:, :2] * (f if f < 1 else 1.0)
    return Prepared(record, foam_c, cav_c, valid_c, res, base_row, axis_col, visible, full_view, to_canvas)


def _contour(mask: np.ndarray, smooth_px: float = 0.0) -> np.ndarray:
    """Largest external contour; optionally smoothed along its length (removes the pixel staircase, which
    otherwise makes arc-length resampling slide points by a different amount at every rotation)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        raise SegmentationError("empty region")
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if smooth_px > 0 and len(c) > 8:
        c = gaussian_filter1d(c, smooth_px, axis=0, mode="wrap")
    return c


def resample_open(curve: np.ndarray, n: int) -> np.ndarray:
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, curve[:, 0]), np.interp(t, s, curve[:, 1])])


def resample_closed(curve: np.ndarray, n: int, start: int) -> np.ndarray:
    c = np.roll(curve, -start, axis=0)
    return resample_open(np.vstack([c, c[:1]]), n + 1)[:-1]


def _tip_start(curve: np.ndarray) -> np.ndarray:
    """Closed curve starting at its topmost point near the axis (x ~ 0), going down the left side first."""
    near = np.abs(curve[:, 0]) <= max(1.0, 0.1 * np.ptp(curve[:, 0]))
    idx = np.nonzero(near)[0] if near.any() else np.arange(len(curve))
    start = idx[np.argmin(curve[idx, 1])]
    c = np.roll(curve, -start, axis=0)
    if c[min(5, len(c) - 1), 0] > c[0, 0]:      # heading right -> reverse (keep the start point first)
        c = np.roll(c[::-1], 1, axis=0)
    return c


def extract(prep: Prepared, depth_mm: float | None, n_outer: int, n_cavity: int, smooth_mm: float = 0.3) -> Shape:
    """Cut the window (or take the full outline) and build correspondence curves."""
    foam, cav = prep.foam, prep.cavity
    if depth_mm is not None:
        if depth_mm > prep.visible_depth_mm + 1e-6:
            raise SegmentationError(f"window {depth_mm:.1f} mm deeper than the visible {prep.visible_depth_mm:.1f} mm")
        cut_row = prep.base_row - depth_mm * prep.res
        rows = np.arange(foam.shape[0])[:, None]
        foam, cav = foam & (rows >= cut_row), cav & (rows >= cut_row)
    elif not prep.full_view:
        raise SegmentationError("full outline requested but the part is cut off by the frame")

    outer_px = _contour(foam | cav, smooth_mm * prep.res)
    cav_px = _contour(cav, smooth_mm * prep.res)
    outer, cavity = prep.to_mm(outer_px), prep.to_mm(cav_px)
    if depth_mm is not None:
        cut_mm = -depth_mm
        oc = resample_open(_open_chain(outer, cut_mm, prep.res), n_outer)
        cc = resample_open(_open_chain(cavity, cut_mm, prep.res), n_cavity)
    else:
        oc = resample_closed(_tip_start(outer), n_outer, 0)
        cc = resample_closed(_tip_start(cavity), n_cavity, 0)
    return Shape(prep.record.name, outer, cavity, oc, cc, depth_mm, prep.record, prep)


def _open_chain(contour_mm: np.ndarray, cut_y: float, res: float) -> np.ndarray:
    """The part of a window contour that is not the cut line: the path between the leftmost and rightmost
    points on the cut that passes through the base, ordered left -> right."""
    on_cut = np.nonzero(contour_mm[:, 1] <= cut_y + 0.5 / res)[0]
    if len(on_cut) < 2:
        raise SegmentationError("contour does not reach the window cut")
    i_l = on_cut[np.argmin(contour_mm[on_cut, 0])]
    i_r = on_cut[np.argmax(contour_mm[on_cut, 0])]
    n = len(contour_mm)
    a = contour_mm[np.arange(i_l, i_l + (i_r - i_l) % n + 1) % n]          # i_l -> i_r forwards
    b = contour_mm[np.arange(i_r, i_r + (i_l - i_r) % n + 1) % n][::-1]    # i_l -> i_r backwards
    return a if a[:, 1].max() >= b[:, 1].max() else b
