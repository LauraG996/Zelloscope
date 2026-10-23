"""Shape-agnostic mode (config `kind: generic`): any closed foam cross-section with any number of holes.

What changes compared with the U-profile mode:

* Segmentation: the part is the foam blob; holes are background regions fully enclosed by the foam (larger than
  a pore). The whole cross-section must be inside the image - there is no window.
* Canonical frame: mm, y down, origin at the area centroid, long axis horizontal (the smallest rotation that does
  this, so a part imaged upright stays upright).
* Correspondence: every contour (outer, then the holes) is resampled to a fixed number of points by arc length.
  Where each contour starts is not taken from the geometry of a particular product; it is found while aligning:
  each part is tried at `orientations` rotations (4 = every 90 degrees) and mirrored, the start point of every
  contour is moved to the point nearest the target's start point, and the best rigid fit wins.
* Measurements: plain geometry (area, width, height, convexity, hole size and position, walls around each hole)
  instead of named product dimensions. Mode names say which of these change and where the outline moves most.
* Wall thickness: at each point of the outer outline, the diameter of the largest circle touching the outline there
  that fits inside the foam (position = mm along the outline).

Symmetric parts: if the outline looks the same rotated or mirrored (a rectangle), "left" and "right" are only
defined by small details such as where the hole sits, so a part with the hole shifted left can be matched to one
shifted right. If the parts are always imaged the same way up, set `shape_model.orientations: 1` and
`allow_mirror: false`: left/right and up/down then keep their meaning.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from profile_check.measure import line_extents

from .io import ImageRecord
from .segment import (
    Prepared,
    SegmentationError,
    Shape,
    _canvas_transform,
    _contour,
    _warp,
    foam_mask,
    resample_open,
)
from .shape_model import Aligned, _kabsch

# ---------------- segmentation -----------------

def holes_mask(foam: np.ndarray) -> np.ndarray:
    """Background regions completely enclosed by the foam (pores were already filled by foam_mask)."""
    n, labels = cv2.connectedComponents((~foam).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    return (labels > 0) & ~np.isin(labels, border)


def _principal_angle(mask: np.ndarray) -> float:
    """Angle (rad, image coordinates) of the long axis, folded into [-90, 90) degrees."""
    ys, xs = np.nonzero(mask)
    w, v = np.linalg.eigh(np.cov(np.vstack([xs, ys]).astype(float)))
    ax = v[:, 1]
    a = float(np.arctan2(ax[1], ax[0]))
    return (a + np.pi / 2) % np.pi - np.pi / 2


def prepare_generic(record: ImageRecord, cfg) -> Prepared:
    if record.mm_per_px is None:
        raise SegmentationError(f"unknown scale: {record.scale_source}")
    img = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
    if img is None:
        raise SegmentationError(f"cannot read {record.path}")
    seg = cfg["segmentation"]
    px_per_mm = 1.0 / record.mm_per_px
    mask_res = min(px_per_mm, 2 * seg["work_px_per_mm"])
    f = mask_res / px_per_mm
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    foam = foam_mask(small, mask_res, seg, cfg.spec.segmentation)
    margin = max(1, int(round(seg["frame_margin_mm"] * mask_res)))
    if foam[:margin].any() or foam[-margin:].any() or foam[:, :margin].any() or foam[:, -margin:].any():
        raise SegmentationError("the part touches the image frame; the generic mode needs the whole "
                                "cross-section in view")
    holes = holes_mask(foam)
    filled = foam | holes

    res = seg["work_px_per_mm"]
    centre = np.array(np.nonzero(filled)[::-1], float).mean(axis=1)
    angle = _principal_angle(filled)                     # rotate the long axis onto the horizontal
    m, size = _canvas_transform(foam.shape, angle, centre, res / mask_res)
    foam_c, holes_c = _warp(foam, m, size), _warp(holes, m, size)
    valid_c = _warp(np.ones(foam.shape, bool), m, size)
    ys, xs = np.nonzero(foam_c | holes_c)
    cy, cx = float(ys.mean()), float(xs.mean())
    to_canvas = m.copy()
    to_canvas[:, :2] = m[:, :2] * (f if f < 1 else 1.0)
    # base_row / axis_col are the canonical origin: here the area centroid
    return Prepared(record, foam_c, holes_c, valid_c, res, cy, cx, float(np.ptp(ys)) / res, True, to_canvas)


def _ccw(c: np.ndarray) -> np.ndarray:
    """Same winding for every contour (positive shoelace area in x-right, y-down coordinates)."""
    x, y = c[:, 0], c[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return c if area >= 0 else c[::-1]


def extract_generic(prep: Prepared, smooth_mm: float = 0.3) -> Shape:
    """Outer outline and every hole as dense closed polygons (canonical mm), largest hole first."""
    sm = smooth_mm * prep.res
    outer = _ccw(prep.to_mm(_contour(prep.foam | prep.cavity, sm)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(prep.cavity.astype(np.uint8), connectivity=8)
    order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]) if n > 1 else []
    holes = [_ccw(prep.to_mm(_contour(labels == i, sm))) for i in order]
    cav = holes[0] if holes else np.zeros((0, 2))
    return Shape(prep.record.name, outer, cav, outer, cav, None, prep.record, prep, holes=holes)


# ---------------- correspondence + alignment -----------------

def resample_from(contour: np.ndarray, n: int, start_xy) -> np.ndarray:
    """Closed contour resampled to n points by arc length, starting at the point nearest `start_xy`."""
    i = int(np.argmin(((contour - start_xy) ** 2).sum(axis=1)))
    c = np.roll(contour, -i, axis=0)
    return resample_open(np.vstack([c, c[:1]]), n + 1)[:-1]


def split_points(pts: np.ndarray, sizes) -> list[np.ndarray]:
    return np.split(pts, np.cumsum(sizes)[:-1])


def _match_holes(holes: list[np.ndarray], targets: list[np.ndarray]) -> list[np.ndarray]:
    if len(holes) != len(targets):
        raise SegmentationError(f"{len(holes)} hole(s), the other parts have {len(targets)}")
    if len(holes) < 2:
        return holes
    a = np.array([h.mean(axis=0) for h in holes])
    b = np.array([t.mean(axis=0) for t in targets])
    _, col = linear_sum_assignment(((b[:, None] - a[None]) ** 2).sum(-1))
    return [holes[j] for j in col]


def _rot(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


DENSE = 4            # candidate start points per correspondence point


def _fit_errors(pts: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Residual sum of squares of the best rotation + translation of each candidate pts[s] onto the centred
    target (Kabsch, no reflection), for all candidates at once."""
    pc = pts - pts.mean(axis=1, keepdims=True)
    h = np.einsum("snk,nl->skl", pc, target)
    u, sv, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.transpose(vt, (0, 2, 1)) @ np.transpose(u, (0, 2, 1))))
    return (pc ** 2).sum(axis=(1, 2)) + (target ** 2).sum() - 2 * (sv[:, 0] + d * sv[:, 1])


def _align_candidate(contours, target, tgt, sizes, lin, rounds: int = 2):
    """For one starting pose `lin` (rotation / mirror): choose the start point of every contour by searching
    +-1/8 of its length for the best rigid fit of all points, then fit. Returns (rms error, points, lin, off)."""
    cur = [_ccw(c @ lin.T) for c in contours]
    cur = [cur[0]] + _match_holes(cur[1:], tgt[1:])
    dense = [resample_open(np.vstack([c, c[:1]]), n * DENSE + 1)[:-1] for c, n in zip(cur, sizes)]
    starts = [int(np.argmin(((d - t[0]) ** 2).sum(axis=1))) for d, t in zip(dense, tgt)]
    take = lambda j, s0: dense[j][(s0 + DENSE * np.arange(sizes[j])[None]) % len(dense[j])]  # (S, n, 2)
    for _ in range(rounds):
        for j in range(len(dense)):
            w = len(dense[j]) // 8
            shifts = (starts[j] + np.arange(-w, w + 1))[:, None]
            parts = [take(i, np.array([[starts[i]]])) for i in range(len(dense))]
            parts[j] = take(j, shifts)
            cand = np.concatenate([np.broadcast_to(q, (len(shifts),) + q.shape[1:]) for q in parts], axis=1)
            starts[j] = int(shifts[int(np.argmin(_fit_errors(cand, target))), 0])
    p = np.vstack([take(j, np.array([[starts[j]]]))[0] for j in range(len(dense))])
    c = p.mean(axis=0)
    r = _kabsch(p - c, target)
    q = (p - c) @ r.T
    return float(np.sqrt(((q - target) ** 2).sum(axis=1).mean())), q, r @ lin, -c @ r.T


def _candidates(orientations: int, allow_mirror: bool) -> list[tuple[bool, int]]:
    return [(m, k) for m in ((False, True) if allow_mirror else (False,)) for k in range(max(1, orientations))]


def _align(contours, target, sizes, orientations, cands) -> tuple[Aligned, tuple[bool, int]]:
    tgt = split_points(target, sizes)
    best = None
    for mirrored, k in cands:
        lin = _rot(2 * np.pi * k / max(1, orientations)) @ (np.diag([-1.0, 1.0]) if mirrored else np.eye(2))
        err, q, lin2, off = _align_candidate(contours, target, tgt, sizes, lin)
        if best is None or err < best[0]:
            best = (err, q, mirrored, lin2, off, (mirrored, k))
    _, q, mirrored, lin, off, cand = best
    rot = lin @ (np.diag([-1.0, 1.0]) if mirrored else np.eye(2))       # q = (M x) @ rot.T + off
    return Aligned(q, mirrored, rot, -off @ rot), cand


def align_generic(contours: list[np.ndarray], target: np.ndarray, sizes, orientations: int = 4,
                  allow_mirror: bool = True) -> Aligned:
    """Rigid fit (rotation, translation, optional mirror; no scaling) of a part's dense contours to a centred
    target with `sizes` points per contour. The coarse orientation (`orientations` starting poses, mirrored or
    not) and the start point of every contour are searched; the best fit of all points wins."""
    return _align(contours, target, sizes, orientations, _candidates(orientations, allow_mirror))[0]


def generalized_procrustes_generic(parts: list[list[np.ndarray]], sizes, orientations: int = 4,
                                   allow_mirror: bool = True, iterations: int = 10,
                                   tol: float = 1e-4) -> tuple[np.ndarray, list[Aligned]]:
    """Iteratively align all parts (lists of dense contours, outer first) to their mean. Every starting pose is
    tried in the first pass; later passes only refine each part's best pose (the mean barely moves)."""
    first = parts[0]
    start = [c[np.argmax(c[:, 0])] for c in first]                  # rightmost point of each contour
    mean = np.vstack([resample_from(c, n, s) for c, n, s in zip(first, sizes, start)])
    mean = mean - mean.mean(axis=0)
    ref0 = mean.copy()
    every = _candidates(orientations, allow_mirror)
    cands = [every] * len(parts)
    for it in range(iterations):
        res = [_align(p, mean, sizes, orientations, c) for p, c in zip(parts, cands)]
        cands = [[cand] for _, cand in res]
        new = np.mean([al.points for al, _ in res], axis=0)
        new -= new.mean(axis=0)
        new = new @ _kabsch(new, ref0).T
        done = np.sqrt(((new - mean) ** 2).sum(axis=1).mean()) < tol
        mean = new
        if done:
            break
    # orient the mean by its own long axis (area moments of the outer outline): the first part's small tilt
    # would otherwise carry over into every width / height measured in the aligned frame
    mo = cv2.moments(mean[: sizes[0]].astype(np.float32))
    ang = 0.5 * np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])
    mean = mean @ _rot(-ang).T
    aligned = [_align(p, mean, sizes, orientations, every)[0] for p in parts]
    return mean, aligned


# ---------------- plain geometric description -----------------

def _area(c: np.ndarray) -> float:
    return abs(cv2.contourArea(c.astype(np.float32)))


def _hole_prefix(j: int, n_holes: int) -> str:
    return "hole" if n_holes == 1 else f"hole{j + 1}"


def describe_generic(points: np.ndarray, sizes) -> dict:
    """Geometry of one aligned shape (y down): size, convexity, and per hole its size, position relative to
    the outer centre and the wall on each side (measured along the lines through the hole's centre), all in the
    part's own principal axes."""
    # measure in the part's own principal axes (a small residual tilt of the alignment would otherwise read as
    # extra width and height); the aligned frame only decides which way is left / up
    mo = cv2.moments(points[: sizes[0]].astype(np.float32))
    ang = 0.5 * np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])
    ang = (ang + np.pi / 4) % (np.pi / 2) - np.pi / 4                  # smallest turn onto an axis
    outer, *holes = split_points(points @ _rot(-ang).T, sizes)
    hull = cv2.convexHull(outer.astype(np.float32)).reshape(-1, 2)
    oc = outer.mean(axis=0)
    out = dict(area_mm2=_area(outer) - sum(_area(h) for h in holes),
               outer_width=float(np.ptp(outer[:, 0])), outer_height=float(np.ptp(outer[:, 1])),
               perimeter_mm=float(np.hypot(*np.diff(np.vstack([outer, outer[:1]]), axis=0).T).sum()),
               convexity=_area(outer) / max(_area(hull), 1e-9))
    for j, h in enumerate(holes):
        pre = _hole_prefix(j, len(holes))
        hc = h.mean(axis=0)
        out[f"{pre}_width"], out[f"{pre}_height"] = float(np.ptp(h[:, 0])), float(np.ptp(h[:, 1]))
        out[f"{pre}_x"], out[f"{pre}_y"] = float(hc[0] - oc[0]), float(hc[1] - oc[1])
        (olo,), (ohi,) = line_extents(outer, np.array([hc[1]]), axis=1)
        (hlo,), (hhi,) = line_extents(h, np.array([hc[1]]), axis=1)
        out[f"{pre}_wall_left"], out[f"{pre}_wall_right"] = float(hlo - olo), float(ohi - hhi)
        (otop,), (obot,) = line_extents(outer, np.array([hc[0]]), axis=0)
        (htop,), (hbot,) = line_extents(h, np.array([hc[0]]), axis=0)
        out[f"{pre}_wall_top"], out[f"{pre}_wall_bottom"] = float(htop - otop), float(obot - hbot)
    return {k: round(float(v), 4 if k == "convexity" else 3) for k, v in out.items()}


def feature_words(key: str) -> tuple[str, str, str, float]:
    """(unit, increase, decrease, weight) for a generic feature; weight makes changes comparable (~mm)."""
    fixed = {"area_mm2": ("mm²", "more foam in the cross-section", "less foam in the cross-section", 0.05),
             "outer_width": ("mm", "part wider", "part narrower", 1.0),
             "outer_height": ("mm", "part taller", "part flatter", 1.0),
             "perimeter_mm": ("mm", "longer outline", "shorter outline", 0.5),
             "convexity": ("", "sides straighter", "sides more hollow", 20.0)}
    if key in fixed:
        return fixed[key]
    pre, _, what = key.partition("_")
    name = "hole" if pre == "hole" else f"hole {pre[4:]}"
    words = {"width": (f"{name} wider", f"{name} narrower"), "height": (f"{name} taller", f"{name} shorter"),
             "x": (f"{name} further right", f"{name} further left"), "y": (f"{name} lower", f"{name} higher"),
             "wall_left": (f"wall left of the {name} thicker", f"wall left of the {name} thinner"),
             "wall_right": (f"wall right of the {name} thicker", f"wall right of the {name} thinner"),
             "wall_top": (f"wall above the {name} thicker", f"wall above the {name} thinner"),
             "wall_bottom": (f"wall below the {name} thicker", f"wall below the {name} thinner")}
    up, down = words.get(what, (f"{key} larger", f"{key} smaller"))
    return "mm", up, down, 1.0


def _where(p: np.ndarray, centre: np.ndarray) -> str:
    """Compass word for a point relative to the centre (y down, so negative y is the top)."""
    dx, dy = p - centre
    ang = np.degrees(np.arctan2(-dy, dx)) % 360
    names = ["right", "top right", "top", "top left", "left", "bottom left", "bottom", "bottom right"]
    return names[int(((ang + 22.5) % 360) // 45)]


def outline_motion(lo: np.ndarray, hi: np.ndarray, sizes) -> dict:
    """Largest outward/inward movement (mm, along the outline normal) from shape lo to shape hi, and where."""
    best = None
    centre = split_points(lo, sizes)[0].mean(axis=0)
    for j, (a, b) in enumerate(zip(split_points(lo, sizes), split_points(hi, sizes))):
        tan = np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)
        nrm = np.column_stack([tan[:, 1], -tan[:, 0]])                # outward for the CCW (y-down) winding
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        d = ((b - a) * nrm).sum(axis=1)
        i = int(np.argmax(np.abs(d)))
        if best is None or abs(d[i]) > abs(best[0]):
            best = (float(d[i]), j, a[i])
    d, j, p = best
    contour = "outer outline" if j == 0 else ("hole" if len(sizes) == 2 else f"hole {j}")
    if j == 0:
        direction = "out" if d > 0 else "in"
    else:                                   # a hole's outward normal points into the foam
        direction = "into the foam (hole larger)" if d > 0 else "towards the hole centre (hole smaller)"
    return dict(mm=round(abs(d), 3), text=f"{contour} moves {direction} most at the {_where(p, centre)} "
                                          f"({abs(d):.2f} mm)")


def describe_mode_generic(model, i: int, n_sigma: float, sizes) -> dict:
    lo = describe_generic(model.mode_shape(i, -n_sigma), sizes)
    hi = describe_generic(model.mode_shape(i, n_sigma), sizes)
    deltas = {k: round(hi[k] - lo[k], 4) for k in lo if np.isfinite(hi[k] - lo[k])}
    weight = lambda k: abs(deltas[k]) * feature_words(k)[3]
    ranked = sorted(deltas, key=weight, reverse=True)
    parts = []
    for k in ranked[:2]:
        if weight(k) < 0.05:
            break
        unit, up, down, _ = feature_words(k)
        parts.append(f"{up if deltas[k] > 0 else down} ({abs(deltas[k]):.2f} {unit})".replace(" )", ")"))
    motion = outline_motion(model.mode_shape(i, -n_sigma), model.mode_shape(i, n_sigma), sizes)
    name = "; ".join(parts) if parts else motion["text"]
    rms = float(np.sqrt(model.variance[i] / len(model.mean)))
    return dict(mode=i + 1, name=name, where=motion["text"], deltas={k: deltas[k] for k in ranked},
                rms_mm_per_sd=round(rms, 3), explained=round(float(model.explained[i]), 4))


# ---------------- wall thickness along the outline -----------------

def thickness_on_outline(points: np.ndarray, sizes, smooth_pts: float = 1.0, upsample: int = 4) -> np.ndarray:
    """Local wall thickness (mm) at each point of the outer correspondence curve: the diameter of the largest
    circle that touches the outline at that point and fits inside the foam (it stops at the hole, or at the
    far side of a solid section). Found by bisection on the circle radius along the inward normal."""
    curves = split_points(points, sizes)
    outer = curves[0]
    n = len(outer)
    tan = np.roll(outer, -2, axis=0) - np.roll(outer, 2, axis=0)
    inward = np.column_stack([-tan[:, 1], tan[:, 0]])                 # CCW in y-down coordinates
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-9)
    # boundary points, upsampled along every edge so the circle cannot slip between points
    f = np.arange(upsample)[:, None] / upsample
    bnd = np.vstack([(c[:, None] + f[None] * (np.roll(c, -1, axis=0) - c)[:, None]).reshape(-1, 2)
                     for c in curves])
    tree = cKDTree(bnd)
    lo = np.zeros(n)
    hi = np.full(n, float(np.ptp(points, axis=0).max()))
    own = 1e-6 + 0.5 * np.linalg.norm(np.roll(outer, -1, axis=0) - outer, axis=1) / upsample
    for _ in range(30):
        r = (lo + hi) / 2
        d, _ = tree.query(outer + inward * r[:, None])
        fits = d >= r - own                                            # nothing closer than the point itself
        lo, hi = np.where(fits, r, lo), np.where(fits, hi, r)
    out = 2 * lo
    if smooth_pts > 0:
        out = gaussian_filter1d(out, smooth_pts, mode="wrap")
    return out
