"""Point-distribution shape model: Procrustes alignment (rotation + translation, optional mirror, no scaling)
and PCA of the aligned outer + cavity curves."""
from dataclasses import dataclass, field

import numpy as np

from profile_check.measure import line_extents, measure


@dataclass
class Aligned:
    points: np.ndarray        # (n_outer + n_cavity, 2), aligned frame (mm)
    mirrored: bool
    rotation: np.ndarray      # 2x2, applied after mirroring and centring
    shift: np.ndarray         # centroid subtracted before rotation

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Map other points of the same part (canonical mm) into the aligned frame."""
        p = np.asarray(pts, float) * ([-1.0, 1.0] if self.mirrored else [1.0, 1.0])
        return (p - self.shift) @ self.rotation.T


def mirror_curve(c: np.ndarray, closed: bool) -> np.ndarray:
    """Mirror x and reverse the order, so point i still means the same place (left stays left)."""
    m = c * [-1.0, 1.0]
    return np.roll(m[::-1], 1, axis=0) if closed else m[::-1]


def _kabsch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rotation R minimising |x @ R.T - y| for centred x, y (no reflection)."""
    u, _, vt = np.linalg.svd(x.T @ y)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, d]) @ u.T


def _fit(p: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    c = p.mean(axis=0)
    r = _kabsch(p - c, target)
    q = (p - c) @ r.T
    return q, r, c, float(np.sqrt(((q - target) ** 2).sum(axis=1).mean()))


def align_to(outer: np.ndarray, cavity: np.ndarray, target: np.ndarray, closed: bool,
             allow_mirror: bool = True) -> Aligned:
    """Align one part's correspondence curves to a centred target (n_outer + n_cavity points)."""
    best = None
    for mirrored in ((False, True) if allow_mirror else (False,)):
        o = mirror_curve(outer, closed) if mirrored else outer
        cc = mirror_curve(cavity, closed) if mirrored else cavity
        q, r, c, err = _fit(np.vstack([o, cc]), target)
        if best is None or err < best[0]:
            # shift is the centroid in the (mirrored) canonical frame: un-mirror x for apply()
            best = (err, Aligned(q, mirrored, r, c))
    return best[1]


def generalized_procrustes(curves: list[tuple[np.ndarray, np.ndarray]], closed: bool, allow_mirror: bool = True,
                           iterations: int = 10, tol: float = 1e-4) -> tuple[np.ndarray, list[Aligned]]:
    """Iteratively align all parts to their mean. The mean keeps the orientation of the first part."""
    first = np.vstack(curves[0])
    mean = first - first.mean(axis=0)
    ref0 = mean.copy()
    for _ in range(iterations):
        aligned = [align_to(o, c, mean, closed, allow_mirror) for o, c in curves]
        new = np.mean([a.points for a in aligned], axis=0)
        new -= new.mean(axis=0)
        new = new @ _kabsch(new, ref0).T            # keep the reference orientation (no drift)
        done = np.sqrt(((new - mean) ** 2).sum(axis=1).mean()) < tol
        mean = new
        if done:
            break
    aligned = [align_to(o, c, mean, closed, allow_mirror) for o, c in curves]
    return mean, aligned


@dataclass
class ShapeModel:
    mean: np.ndarray                 # (P, 2)
    components: np.ndarray           # (m, 2P) unit vectors
    variance: np.ndarray             # (m,) mm^2
    scores: np.ndarray               # (n, m)
    n_outer: int
    closed: bool
    n_keep: int
    explained: np.ndarray = field(init=False)

    def __post_init__(self):
        total = self.variance.sum()
        self.explained = self.variance / total if total > 0 else np.zeros_like(self.variance)

    def shape(self, b: np.ndarray) -> np.ndarray:
        """Shape for mode weights b (mm along each component)."""
        b = np.asarray(b, float)
        return self.mean + (b @ self.components[: len(b)]).reshape(-1, 2)

    def mode_shape(self, i: int, n_sigma: float) -> np.ndarray:
        b = np.zeros(i + 1)
        b[i] = n_sigma * np.sqrt(self.variance[i])
        return self.shape(b)

    def project(self, points: np.ndarray, k: int | None = None) -> np.ndarray:
        k = self.n_keep if k is None else k
        return (points - self.mean).reshape(-1) @ self.components[:k].T

    def residual_rms(self, points: np.ndarray, k: int | None = None) -> float:
        """RMS distance (mm) between a shape and its reconstruction from the first k modes."""
        k = self.n_keep if k is None else k
        rec = self.shape(self.project(points, k))
        return float(np.sqrt(((points - rec) ** 2).sum(axis=1).mean()))

    def split(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pts[: self.n_outer], pts[self.n_outer:]


def fit_pca(aligned: np.ndarray, n_outer: int, closed: bool, variance_keep: float) -> ShapeModel:
    """aligned: (n, P, 2). Keeps the fewest modes explaining `variance_keep` of the variance."""
    n = len(aligned)
    x = aligned.reshape(n, -1)
    mean = x.mean(axis=0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    m = max(1, min(n - 1, len(s)))
    var = s[:m] ** 2 / max(1, n - 1)
    comps = vt[:m]
    # Sign convention: make each component's largest coordinate positive (stable across runs).
    signs = np.sign(comps[np.arange(m), np.abs(comps).argmax(axis=1)])
    comps = comps * signs[:, None]
    scores = xc @ comps.T
    cum = np.cumsum(var) / var.sum() if var.sum() > 0 else np.ones(m)
    n_keep = int(np.searchsorted(cum, variance_keep - 1e-12) + 1)
    return ShapeModel(mean.reshape(-1, 2), comps, var, scores, n_outer, closed, min(n_keep, m))


def bootstrap_stability(aligned: np.ndarray, model: ShapeModel, n_boot: int, seed: int = 0) -> dict:
    """How reproducible each kept mode is: |cosine| between the mode and the best-matching mode of PCA on
    resampled parts (1 = identical direction). Also the spread of the explained-variance fractions."""
    rng = np.random.default_rng(seed)
    n = len(aligned)
    k = model.n_keep
    cos, expl = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        x = aligned[idx].reshape(n, -1)
        x = x - x.mean(axis=0)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        c = np.abs(model.components[:k] @ vt[: max(k, 1) + 2].T)
        cos.append(c.max(axis=1))
        v = s ** 2
        expl.append((v / v.sum())[:k] if len(v) >= k else np.pad(v / v.sum(), (0, k - len(v))))
    cos, expl = np.array(cos), np.array(expl)
    return dict(
        cos_median=np.median(cos, axis=0).round(3).tolist(), cos_p05=np.percentile(cos, 5, axis=0).round(3).tolist(),
        explained_p05=np.percentile(expl, 5, axis=0).round(3).tolist(),
        explained_p95=np.percentile(expl, 95, axis=0).round(3).tolist(), n_boot=len(cos))


# ---------------- describing shapes / modes in plain terms -----------------

FEATURES = {
    # name: (unit, what an increase means, what a decrease means)
    "inner_width_max": ("mm", "cavity wider near the base", "cavity narrower near the base"),
    "cavity_width_mid": ("mm", "legs further apart half-way up", "legs closer together half-way up"),
    "cavity_width_top": ("mm", "legs open further at the top", "legs close in at the top"),
    "base_thickness": ("mm", "base thicker", "base thinner"),
    "wall_left": ("mm", "left wall thicker", "left wall thinner"),
    "wall_right": ("mm", "right wall thicker", "right wall thinner"),
    "outer_width_max": ("mm", "part wider overall", "part narrower overall"),
    "lean_left_deg": ("deg", "left leg leans further inward", "left leg more upright"),
    "lean_right_deg": ("deg", "right leg leans further inward", "right leg more upright"),
    "wall_asym": ("mm", "walls more unequal", "walls more equal"),
    "tip_height_diff": ("mm", "tips more uneven in height", "tips more even in height"),
}
# Only meaningful when the whole profile (tips) is visible.
FULL_VIEW_ONLY = {"tip_height_diff"}


def _lean(curve: np.ndarray, side: str, y_lo: float, y_hi: float) -> float:
    """Inward lean (deg) of one side of an open/closed curve between heights y_lo..y_hi (y down)."""
    sel = (curve[:, 1] >= y_lo) & (curve[:, 1] <= y_hi) & ((curve[:, 0] < 0) if side == "left" else (curve[:, 0] > 0))
    if sel.sum() < 5:
        return float("nan")
    slope = np.polyfit(-curve[sel, 1], curve[sel, 0], 1)[0]       # dx per mm upwards
    return float(np.degrees(np.arctan(slope if side == "left" else -slope)))


def describe(points: np.ndarray, n_outer: int, mparams, closed: bool = False) -> dict:
    """Plain geometric features of one shape in the aligned frame (y down, base at the bottom).
    `closed` = full outline; for a window (tips outside the frame) tip features are left out and
    "top" means the top of the window."""
    outer, cavity = points[:n_outer], points[n_outer:]
    x0 = (cavity[:, 0].min() + cavity[:, 0].max()) / 2
    outer, cavity = outer - [x0, 0], cavity - [x0, 0]
    m = measure(outer, cavity, mparams)
    top, bot = cavity[:, 1].min(), cavity[:, 1].max()
    h = bot - top
    out = {k: m[k] for k in ("inner_width_max", "base_thickness", "wall_left", "wall_right", "outer_width_max",
                             "wall_asym", "tip_height_diff")}
    for name, frac in (("cavity_width_mid", 0.5), ("cavity_width_top", 0.9)):
        lo, hi = line_extents(cavity, np.array([bot - frac * h]), axis=1)
        out[name] = float(hi[0] - lo[0])
    out["lean_left_deg"] = _lean(cavity, "left", top + 0.2 * h, bot - 0.3 * h)
    out["lean_right_deg"] = _lean(cavity, "right", top + 0.2 * h, bot - 0.3 * h)
    if not closed:
        for k in FULL_VIEW_ONLY:
            out.pop(k, None)
    return {k: round(float(v), 3) for k, v in out.items()}


def describe_mode(model: ShapeModel, i: int, n_sigma: float, mparams) -> dict:
    """Feature changes from mean - n_sigma*sd to mean + n_sigma*sd along mode i, largest first, and a name that
    describes the + direction (the two largest changes)."""
    lo = describe(model.mode_shape(i, -n_sigma), model.n_outer, mparams, model.closed)
    hi = describe(model.mode_shape(i, n_sigma), model.n_outer, mparams, model.closed)
    deltas = {k: round(hi[k] - lo[k], 3) for k in lo if np.isfinite(hi[k] - lo[k])}
    # rank: 1 deg of lean ~ 0.5 mm of width at these sizes
    weight = lambda k: abs(deltas[k]) * (0.5 if FEATURES[k][0] == "deg" else 1.0)
    ranked = sorted(deltas, key=weight, reverse=True)
    parts = []
    for k in ranked[:2]:
        if weight(k) < 0.05:
            break
        unit, up, down = FEATURES[k]
        parts.append(f"{up if deltas[k] > 0 else down} ({abs(deltas[k]):.2f} {unit})")
    name = "; ".join(parts) if parts else "very small change (below 0.05 mm)"
    # per-point RMS displacement of one standard deviation along this mode
    rms = float(np.sqrt(model.variance[i] / len(model.mean)))
    return dict(mode=i + 1, name=name, deltas={k: deltas[k] for k in ranked}, rms_mm_per_sd=round(rms, 3),
                explained=round(float(model.explained[i]), 4))
