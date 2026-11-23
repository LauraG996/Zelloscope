"""Transfer key points from a reference image to a new image of the same (non-rigidly deformed) part.

Pipeline
  1. RoMa dense matcher on downscaled images           -> thousands of correspondences
  2. keep matches on the part that agree with their neighbours, one per grid cell
  3. thin-plate-spline deformation field (+ outline snapping of part/hole contours)
  4. local NCC refinement of every key point at full resolution, template warped by the local Jacobian

Usage
  python keypoint_transfer.py REF NEW --points "1850,2200;1600,2550" [--out result.png] [--json result.json]

Requires: opencv-python scipy numpy torch pillow onnx onnxruntime  +  pip install git+https://github.com/Parskatt/RoMa.git
(ROMA_BACKEND=torch runs without onnx/onnxruntime, ~2x slower)
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

MATCH_SIDE = 448       # px, longest side of the images fed to the matcher / field fit
# Fast setting: one RoMa pass at 224 px, no refinement pass (the final full-resolution NCC refinement makes it redundant:
# 2.06x faster than 336 + 448 px on 80 dataset images at equal median / p90 error). Env overrides, e.g. ROMA_COARSE=336 ROMA_UPSAMPLE=448
# restores the previous setting. The coarse resolution must be a multiple of 14 (DINOv2 patch size).
COARSE_RES = int(os.environ.get("ROMA_COARSE", 224))      # RoMa coarse resolution; CPU time is roughly proportional to the resolutions
UPSAMPLE_RES = int(os.environ.get("ROMA_UPSAMPLE", 0))    # RoMa refinement pass resolution; 0 = no refinement pass (864 = RoMa default, ~4x slower)
BACKEND = os.environ.get("ROMA_BACKEND", "onnx")   # "onnx" = ONNX Runtime fp32 (roma_onnx.py), "torch" = plain PyTorch
NUM_THREADS = int(os.environ.get("ROMA_THREADS", 8))   # physical cores; hyper-threads gave no gain
N_MATCHES = 5000       # correspondences sampled from the matcher
GRID_CELLS = 50        # at most one correspondence per cell (keeps TPS fast and even)
TPS_SMOOTH = 1e-4      # stiffness of the deformation field (normalized coords). Larger = smoother
CONTOUR_WEIGHT = 0.5   # weight of outline correspondences relative to texture matches
CONTOUR_ITERS = 1      # rounds of outline snapping (a 2nd round changed nothing measurable)
FIT_ITERS = 2          # max outlier-removal iterations of one TPS fit (more changed nothing measurable)
REFINE_HALF = 25       # px, half size of the local template (full-res)
REFINE_SEARCH = 8      # px, local search after the field prediction (full-res)


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


# ---------------------------------------------------------------- geometry
def part_mask(img):
    """Binary mask of the part (largest bright component; holes stay out)."""
    L = max(img.shape)
    g = cv2.GaussianBlur(img, (0, 0), L / 800)
    _, m = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = int(L / 200) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lab == keep).astype(np.uint8)


def mask_contour(mask, step=None):
    """Outer outline + hole outlines, resampled to roughly `step` px spacing."""
    step = step or max(mask.shape) / 400
    cs, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    pts = [c[::max(1, int(step)), 0, :] for c in cs if len(c) > 50]
    return np.vstack(pts).astype(np.float64)


# ---------------------------------------------------------------- matcher
_ROMA = {}


def match_roma(ref, new, n=N_MATCHES, coarse=None, upsample=None):
    """RoMa dense matcher -> (k0, k1, certainty) in the pixels of the given images."""
    import torch
    from PIL import Image
    from romatch import roma_outdoor
    dev = "cpu"
    torch.set_num_threads(NUM_THREADS)
    key = (coarse or COARSE_RES, upsample or UPSAMPLE_RES)
    if key not in _ROMA:
        _ROMA[key] = roma_outdoor(device=dev, coarse_res=key[0], upsample_res=key[1] or key[0],
                                  amp_dtype=torch.float32, symmetric=False)   # symmetric=False halves the decoder work
        if not key[1]:
            _ROMA[key].upsample_preds = False                          # single coarse pass, no refinement pass
        if BACKEND == "onnx":
            import roma_onnx                    # fp32 ONNX Runtime for encoders + decoder refiners, reference features cached
            roma_onnx.install(_ROMA[key])
    pil = lambda g: Image.fromarray(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
    with torch.no_grad():
        warp, cert = _ROMA[key].match(pil(ref), pil(new), device=dev)
        # RoMa's own sample() balances matches with a KDE (O(N^2) cdist, ~6 s on CPU); grid_subsample
        # does the spatial balancing later, so certainty-weighted sampling is enough here.
        m, c = warp.reshape(-1, 4), cert.reshape(-1)
        idx = torch.multinomial(c.clamp_min(1e-6) ** 2, min(n, len(c)), replacement=False)
        m, c = m[idx], c[idx]
        k0, k1 = _ROMA[key].to_pixel_coordinates(m, ref.shape[0], ref.shape[1], new.shape[0], new.shape[1])
    return k0.cpu().numpy().astype(np.float64), k1.cpu().numpy().astype(np.float64), c.cpu().numpy()


# ---------------------------------------------------------------- filtering
def _inside(mask, p):
    x = np.clip(np.round(p[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.round(p[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return mask[y, x] > 0


def filter_matches(k0, k1, c, mask0, mask1, k=12, rounds=3):
    """Keep matches on the part whose displacement agrees with their neighbours (smooth-field prior)."""
    dil = lambda m: cv2.dilate(m, np.ones((15, 15), np.uint8))
    ok = _inside(dil(mask0), k0) & _inside(dil(mask1), k1)
    k0, k1, c = k0[ok], k1[ok], c[ok]
    L = max(mask0.shape)
    for _ in range(rounds):
        if len(k0) < k + 2:
            break
        d = k1 - k0
        _, idx = cKDTree(k0).query(k0, k + 1)
        err = np.linalg.norm(d - np.median(d[idx[:, 1:]], axis=1), axis=1)
        tol = max(0.01 * L, 3 * 1.4826 * np.median(err))
        ok = err < tol
        if ok.all():
            break
        k0, k1, c = k0[ok], k1[ok], c[ok]
    return k0, k1, c


def grid_subsample(k0, k1, c, L, cells=GRID_CELLS):
    """Best-confidence correspondence per grid cell -> even coverage, bounded count."""
    cell = L / cells
    key = (k0[:, 1] // cell) * (cells + 1) + (k0[:, 0] // cell)
    order = np.lexsort((-c, key))
    _, first = np.unique(key[order], return_index=True)
    sel = order[first]
    return k0[sel], k1[sel], c[sel]


# ---------------------------------------------------------------- deformation field
def _tps_kernel(a, b):
    """Thin-plate-spline kernel r^2 log r (0 at r = 0) between two point sets."""
    d2 = cdist(a, b, "sqeuclidean")
    return 0.5 * d2 * np.log(np.where(d2 > 0, d2, 1.0))     # = r^2 log r, and 0 where r = 0


class TPSField:
    """Smooth map reference px -> new px (thin-plate spline on the displacement).

    Own implementation of what scipy's RBFInterpolator(kernel="thin_plate_spline", degree 1, per-point
    smoothing) computes: it lets fit_field build the kernel matrix once and reuse it across iterations."""

    def __init__(self, src, dst, L, smooth=TPS_SMOOTH, weights=None):
        self.L = float(L)
        sm = smooth if weights is None else smooth / np.asarray(weights, float)
        pn = np.asarray(src, float) / self.L
        self._set(pn, _tps_kernel(pn, pn), (np.asarray(dst, float) - src) / self.L, np.broadcast_to(sm, len(pn)))

    def _set(self, pn, K, disp, sm):
        """Solve [[K + diag(sm), P], [P^T, 0]] [a; b] = [disp; 0] for the points pn (normalised px)."""
        n = len(pn)
        self.pts = pn
        self.shift = (pn.max(0) + pn.min(0)) / 2
        self.scale = np.maximum((pn.max(0) - pn.min(0)) / 2, 1e-12)
        P = np.c_[np.ones(n), (pn - self.shift) / self.scale]
        A = np.zeros((n + 3, n + 3))
        A[:n, :n] = K
        A[np.arange(n), np.arange(n)] += sm
        A[:n, n:], A[n:, :n] = P, P.T
        rhs = np.zeros((n + 3, disp.shape[1]))
        rhs[:n] = disp
        sol = solve(A, rhs, assume_a="sym", overwrite_a=True, overwrite_b=True, check_finite=False)
        self.coef, self.poly = sol[:n], sol[n:]

    def _eval_n(self, pn):
        P = np.c_[np.ones(len(pn)), (pn - self.shift) / self.scale]
        return _tps_kernel(pn, self.pts) @ self.coef + P @ self.poly

    def __call__(self, p):
        p = np.atleast_2d(np.asarray(p, float))
        return p + self._eval_n(p / self.L) * self.L

    def jacobian(self, p, h=2.0):
        p = np.atleast_2d(np.asarray(p, float))
        ex, ey = np.array([h, 0.0]), np.array([0.0, h])
        jx = (self(p + ex) - self(p - ex)) / (2 * h)
        jy = (self(p + ey) - self(p - ey)) / (2 * h)
        return np.stack([jx, jy], axis=2)          # (N, 2, 2): column 0 = d/dx, column 1 = d/dy


def fit_field(src, dst, w, L, smooth=TPS_SMOOTH, iters=FIT_ITERS, k=3.0):
    """TPS fit with iterative residual-based outlier removal.
    The kernel matrix of all points is built once; each iteration only slices it and solves."""
    pn = src / float(L)
    K = _tps_kernel(pn, pn)
    disp = (dst - src) / float(L)
    sm = smooth / np.asarray(w, float)
    keep = np.ones(len(src), bool)
    for _ in range(iters):
        idx = np.flatnonzero(keep)
        f = TPSField.__new__(TPSField)
        f.L = float(L)
        f._set(pn[idx], K[np.ix_(idx, idx)], disp[idx], sm[idx])
        # residual at every point (kept or not) from the cached kernel rows: no new kernel evaluation
        P = np.c_[np.ones(len(pn)), (pn - f.shift) / f.scale]
        r = np.linalg.norm((K[:, idx] @ f.coef + P @ f.poly - disp) * float(L), axis=1)
        thr = max(0.002 * L, k * 1.4826 * np.median(r[keep]))
        new_keep = r < thr
        if (new_keep == keep).all():
            break
        keep = new_keep
    return f, keep, r


# ---------------------------------------------------------------- local refinement
def _subpix_peak(R, y, x):
    """Parabolic sub-pixel offset of the correlation peak at (y, x)."""
    def off(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (a - c) / d, -1, 1))
    ox = off(R[y, x - 1], R[y, x], R[y, x + 1]) if 0 < x < R.shape[1] - 1 else 0.0
    oy = off(R[y - 1, x], R[y, x], R[y + 1, x]) if 0 < y < R.shape[0] - 1 else 0.0
    return ox, oy


def refine_affine(ref, new, p, q, J, h=REFINE_HALF, s=REFINE_SEARCH):
    """Template from ref around p, resampled with the local Jacobian J so it is
    rotated/sheared like the new image, then NCC-matched around q."""
    H, W = new.shape
    qi = np.round(q).astype(int)
    if not (h + s <= qi[0] < W - h - s and h + s <= qi[1] < H - h - s):
        return np.asarray(q, float), np.nan, False
    u = np.arange(-h, h + 1, dtype=np.float64)
    ux, uy = np.meshgrid(u, u)
    off = np.stack([ux, uy], -1) @ np.linalg.inv(J).T          # new-frame offsets -> ref offsets
    tpl = cv2.remap(ref, (p[0] + off[..., 0]).astype(np.float32), (p[1] + off[..., 1]).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    if tpl.std() < 1e-3:
        return np.asarray(q, float), np.nan, False
    win = new[qi[1] - h - s:qi[1] + h + s + 1, qi[0] - h - s:qi[0] + h + s + 1].astype(np.float32)
    R = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(R)
    at_edge = mx in (0, 2 * s) or my in (0, 2 * s)              # peak clipped by the search window
    ox, oy = _subpix_peak(R, my, mx)
    shift = np.array([mx + ox - s, my + oy - s])
    # aperture problem: on a straight edge keep only the shift across the edge
    gx, gy = cv2.Sobel(tpl, cv2.CV_32F, 1, 0), cv2.Sobel(tpl, cv2.CV_32F, 0, 1)
    T = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]])
    ev, evec = np.linalg.eigh(T)
    if ev[0] < 0.05 * ev[1]:
        nrm = evec[:, 1]
        shift = (shift @ nrm) * nrm
    return qi + shift, float(score), at_edge


# ---------------------------------------------------------------- main
def map_points(ref, new, ref_pts, refine=True, verbose=True, match_side=None, coarse=None, upsample=None):
    """Map `ref_pts` (x, y in full-res reference pixels) into `new`.

    Matching and field fitting run on images downscaled to MATCH_SIDE; the field is then evaluated
    at full resolution and every point is refined by NCC on the full-res images.
    Returns a dict with refined points ("new_pts"), field prediction ("pred") and diagnostics.
    """
    t0 = time.time()
    ref_pts = np.atleast_2d(np.asarray(ref_pts, float))
    s0, s1 = (min(1.0, (match_side or MATCH_SIDE) / max(g.shape)) for g in (ref, new))   # per-image downscale factors
    small = lambda g, s: cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    ref_s, new_s = small(ref, s0), small(new, s1)
    L = max(ref_s.shape + new_s.shape)
    m0, m1 = part_mask(ref_s), part_mask(new_s)
    t_prep = time.time() - t0

    k0, k1, c = match_roma(ref_s, new_s, coarse=coarse, upsample=upsample)
    t_match = time.time() - t0 - t_prep
    n_raw = len(k0)
    k0, k1, c = filter_matches(k0, k1, c, m0, m1)
    k0, k1, c = grid_subsample(k0, k1, c, L)
    if len(k0) < 20:
        raise RuntimeError(f"only {len(k0)} usable matches")
    field, keep, res = fit_field(k0, k1, np.ones(len(k0)), L)

    # snap the outlines (outer edge + hole) to each other through the current field
    c0, c1 = mask_contour(m0), mask_contour(m1)
    tree1 = cKDTree(c1)
    for md in np.linspace(0.02, 0.006, CONTOUR_ITERS) * L:
        d, j = tree1.query(field(c0))
        ok = d < md
        src = np.vstack([k0, c0[ok]])
        dst = np.vstack([k1, c1[j[ok]]])
        w = np.concatenate([np.ones(len(k0)), np.full(ok.sum(), CONTOUR_WEIGHT)])
        field, keep, res = fit_field(src, dst, w, L)
    n_contour = int(ok.sum())
    contour_frac = float(np.mean(tree1.query(field(c0))[0] < 0.006 * L))   # ref outline landing on the new outline

    pts_s = ref_pts * s0
    pred = field(pts_s) / s1
    Js = field.jacobian(pts_s)                                  # downscaled ref px -> downscaled new px
    J = Js * (s0 / s1)                                          # full-res ref px -> full-res new px
    detJ = np.linalg.det(Js)                                    # local area change, free of the downscaling
    rot = np.degrees(np.arctan2(Js[:, 1, 0] - Js[:, 0, 1], Js[:, 0, 0] + Js[:, 1, 1]))

    # support: inlier texture matches near each point
    tex_in = keep[:len(k0)]
    nb = cKDTree(k0[tex_in]).query_ball_point(pts_s, 0.05 * L)
    support = np.array([len(i) for i in nb])
    t_field = time.time() - t0 - t_prep - t_match

    out, scores, edge = pred.copy(), np.full(len(pred), np.nan), np.zeros(len(pred), bool)
    if refine:
        # bring the reference to the new image's scale first (prefiltered), so templates are not aliased
        rho = float(np.clip(np.sqrt(np.median(np.abs(np.linalg.det(J)))), 0.05, 1.0))
        ref_r = cv2.resize(ref, None, fx=rho, fy=rho, interpolation=cv2.INTER_AREA) if rho < 1 else ref
        for i, (p, q) in enumerate(zip(ref_pts, pred)):
            out[i], scores[i], edge[i] = refine_affine(ref_r, new, p * rho, q, J[i] / rho)
    t_total = time.time() - t0

    flags = []
    for i in range(len(ref_pts)):
        f = []
        if not np.isfinite(scores[i]) or scores[i] < 0.6:
            f.append("low NCC")
        if edge[i]:
            f.append("peak at search edge")
        if detJ[i] < 0.5 or detJ[i] > 2:
            f.append("odd detJ")
        if support[i] < 5:
            f.append("few matches nearby")
        flags.append(f)

    if verbose:
        print(f"matches: {n_raw} raw -> {len(k0)} filtered, {tex_in.sum()} TPS inliers, {n_contour} outline pairs | "
              f"time {t_total:.1f}s (prep {t_prep:.1f}, match {t_match:.1f}, field {t_field:.1f}, "
              f"refine {t_total - t_prep - t_match - t_field:.1f})")
        print(f"{'#':>3} {'ref x':>8} {'ref y':>8}  {'new x':>9} {'new y':>9}  {'NCC':>5} "
              f"{'loc.rot':>7} {'detJ':>5} {'support':>7}  flags")
        for i in range(len(ref_pts)):
            print(f"{i:>3} {ref_pts[i,0]:>8.1f} {ref_pts[i,1]:>8.1f}  {out[i,0]:>9.2f} {out[i,1]:>9.2f}  "
                  f"{scores[i]:>5.2f} {rot[i]:>7.2f} {detJ[i]:>5.2f} {support[i]:>7d}  {', '.join(flags[i])}")

    return dict(new_pts=out, pred=pred, scores=scores, peak_at_edge=edge, local_rot_deg=rot, detJ=detJ,
                support=support, flags=flags, seconds=t_total, n_inliers=int(tex_in.sum()),
                contour_frac=contour_frac, mean_cert=float(c.mean()), n_filtered=len(k0),
                tps_res=float(np.median(res[:len(k0)][tex_in]) / L))


# ---------------------------------------------------------------- crop + PCA orientation
CANVAS = 96            # px, canvas for comparing silhouettes
PCA_WINDOW = 30        # deg, IoU refinement window around each PCA angle
PCA_STEP = 2           # deg, step of that refinement
SIFT_SIDE = 640         # px, longest side of the images used for the SIFT orientation estimate
SIFT_MIN_INLIERS = 6   # fewer RANSAC inliers than this = SIFT orientation unreliable, use the PCA one
SIFT_SCALE = (0.4, 2.5)  # plausible scale of the SIFT similarity (valid fits: 0.85-1.83 on the dataset; degenerate RANSAC fits collapse to ~0)
CROP_MARGIN = 0.03     # fraction of the part size kept around the crop


def _small(img, side=384):
    s = min(1.0, side / max(img.shape))
    return (cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img), s


def _bbox(mask, margin=CROP_MARGIN):
    ys, xs = np.nonzero(mask)
    m = margin * max(xs.max() - xs.min(), ys.max() - ys.min())
    return (max(0, int(xs.min() - m)), max(0, int(ys.min() - m)),
            min(mask.shape[1], int(xs.max() + m) + 1), min(mask.shape[0], int(ys.max() + m) + 1))


def _centroid_norm(mask):
    """3x3 matrix putting the mask centroid at the canvas centre with unit-area scale."""
    m = cv2.moments(mask, binaryImage=True)
    cx, cy, sc = m["m10"] / m["m00"], m["m01"] / m["m00"], (CANVAS * 0.42) / np.sqrt(m["m00"])
    return np.array([[sc, 0, CANVAS / 2 - sc * cx], [0, sc, CANVAS / 2 - sc * cy], [0, 0, 1.]]), (cx, cy)


def _principal_angle(mask):
    """Direction (deg, image coords, x towards y) of the principal axis of the mask (PCA of the pixel coordinates)."""
    m = cv2.moments(mask, binaryImage=True)
    return np.degrees(0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))


def pca_orientation(mref, mnew):
    """Align the part silhouettes with PCA: rotate the new mask so its principal axis matches the reference's.
    PCA fixes the axis only up to 180 deg and roughly; both options are refined by silhouette IoU (milliseconds).
    Returns (iou, flip, angle_deg, G); G (3x3, mask px) maps new px -> reference-like orientation.
    Mirrored images are not supported."""
    Cr, _ = _centroid_norm(mref)
    ref_c = cv2.warpAffine(mref.astype(np.float32), Cr[:2], (CANVAS, CANVAS)) > 0.5
    Cn, (cx, cy) = _centroid_norm(mnew)
    mf = mnew.astype(np.float32)
    base = (_principal_angle(mnew) - _principal_angle(mref)) % 360
    best = None
    # PCA gives the axis up to 180 deg and only roughly (perspective, near-round parts): both options are
    # refined over a small angle window by silhouette IoU (a few ms), the best is kept
    for ang0 in (base, base + 180):
        for ang in ang0 + np.arange(-PCA_WINDOW, PCA_WINDOW + 1e-6, PCA_STEP):
            G = np.vstack([cv2.getRotationMatrix2D((cx, cy), ang, 1.0), [0, 0, 1]])
            a = cv2.warpAffine(mf, (Cn @ G)[:2], (CANVAS, CANVAS)) > 0.5
            iou = float((a & ref_c).sum() / max((a | ref_c).sum(), 1))
            if best is None or iou > best[0]:
                best = (iou, 0, round(float(ang % 360), 1), G)
    return best


_SIFT = {}


def _sift_features(img, mask_small):
    """CLAHE-normalised SIFT keypoints/descriptors of the part (<= SIFT_SIDE px)."""
    s = min(1.0, SIFT_SIDE / max(img.shape))
    g = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    m = cv2.resize(mask_small, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) * 255
    return cv2.SIFT_create(nfeatures=3000).detectAndCompute(g, m)


def sift_orientation(ref, new, mref, mnew):
    """Rotation that brings the new part into the reference orientation, from SIFT matches + RANSAC similarity.
    Uses the part's texture (not only its outline), so it also resolves (near-)symmetric silhouettes.
    Returns (n_inliers, G) with G a 3x3 rotation about the new mask centroid (mask px), or (0, None)."""
    key = (ref.shape, hash(ref[::97, ::97].tobytes()))
    if key not in _SIFT:                                       # the reference is the same for many images: cached
        if len(_SIFT) > 16: _SIFT.clear()
        _SIFT[key] = _sift_features(ref, mref)
    k0, d0 = _SIFT[key]
    k1, d1 = _sift_features(new, mnew)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return 0, None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d0, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 6:
        return 0, None
    p1 = np.float32([k1[g.queryIdx].pt for g in good])
    p0 = np.float32([k0[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(p1, p0, method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=2000, confidence=0.99)
    if M is None:
        return 0, None
    scale = np.sqrt(abs(np.linalg.det(M[:, :2])))
    if not SIFT_SCALE[0] <= scale <= SIFT_SCALE[1]:             # degenerate fit (e.g. all matches collapsed onto one point)
        return 0, None
    R = M[:, :2] / scale
    _, (cx, cy) = _centroid_norm(mnew)
    c = np.array([cx, cy])
    G = np.eye(3)
    G[:2, :2], G[:2, 2] = R, c - R @ c
    return int(inl.sum()), G


def estimate_orientation(ref, new, mref, mnew):
    """(iou, flip, angle_deg, G, method). SIFT rotation when it is well supported, else PCA of the silhouettes."""
    n, G = sift_orientation(ref, new, mref, mnew)
    pca = pca_orientation(mref, mnew)
    if G is not None and n >= SIFT_MIN_INLIERS:
        Cr, _ = _centroid_norm(mref)
        Cn, _ = _centroid_norm(mnew)
        a = cv2.warpAffine(mnew.astype(np.float32), (Cn @ G)[:2], (CANVAS, CANVAS)) > 0.5
        r = cv2.warpAffine(mref.astype(np.float32), Cr[:2], (CANVAS, CANVAS)) > 0.5
        iou = float((a & r).sum() / max((a | r).sum(), 1))
        return iou, 0, round(float(np.degrees(np.arctan2(G[1, 0], G[0, 0]))), 1), G, "sift(%d)" % n
    return pca + ("pca",)


def _aligned_crop(new, G_small, s):
    """Warp `new` with G (given in downscaled mask px, scale s), then crop to the part.
    Returns the crop and the 3x3 matrix M: original new px -> crop px."""
    S = np.diag([s, s, 1.0])
    G = np.linalg.inv(S) @ G_small @ S
    h, w = new.shape
    corners = np.c_[[0, w, w, 0], [0, 0, h, h], np.ones(4)] @ G.T
    x0, y0 = np.floor(corners[:, :2].min(0)).astype(int)
    x1, y1 = np.ceil(corners[:, :2].max(0)).astype(int)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.]])
    full = cv2.warpAffine(new, (T @ G)[:2], (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR)
    sm, s2 = _small(full)
    bx0, by0, bx1, by1 = (int(v / s2) for v in _bbox(part_mask(sm)))
    crop = full[by0:by1, bx0:bx1]
    C = np.array([[1, 0, -bx0], [0, 1, -by0], [0, 0, 1.]])
    return crop, C @ T @ G


def map_points_oriented(ref, new, ref_pts, verbose=True, **kw):
    """Crop both images to the part, align the new image to the reference orientation (SIFT rotation, PCA of the
    silhouettes as fallback; any in-plane rotation), then run RoMa once on the aligned crops."""
    t0 = time.time()
    ref_pts = np.atleast_2d(np.asarray(ref_pts, float))
    ref_s, sr = _small(ref)
    new_s, sn = _small(new)
    mref, mnew = part_mask(ref_s), part_mask(new_s)
    rx0, ry0, rx1, ry1 = (int(v / sr) for v in _bbox(mref))
    ref_c, ref_off = ref[ry0:ry1, rx0:rx1], np.array([rx0, ry0], float)
    cand = estimate_orientation(ref, new, mref, mnew)
    t_orient = time.time() - t0

    def run(c):
        crop, M = _aligned_crop(new, c[3], sn)
        res = map_points(ref_c, crop, ref_pts - ref_off, verbose=verbose, **kw)
        Minv = np.linalg.inv(M).T
        for key in ("new_pts", "pred"):
            res[key] = (np.c_[res[key], np.ones(len(ref_pts))] @ Minv)[:, :2]
        res["orientation"] = f"{c[4]} rot={c[2]}"
        res["silhouette_iou"] = c[0]
        return res

    res = run(cand)                                              # a single RoMa pass
    res["seconds"] = time.time() - t0
    res["seconds_orient"] = t_orient
    return res


# ---------------------------------------------------------------- visual check
_COLORS = [(60, 60, 255), (60, 200, 255), (255, 160, 40), (230, 80, 230), (80, 220, 80), (255, 255, 80),
           (40, 120, 255), (200, 120, 255)]                           # BGR, one per distance / point
PRED_COLOR, GT_COLOR = (0, 0, 255), (0, 220, 0)                       # red = placed by the pipeline, green = ground truth


def _text(im, s, org, scale=0.6, color=(255, 255, 255)):
    """Text on a dark box, readable on any background (an outline drawn with thicker text misaligns on OpenCV 5)."""
    x, y = int(org[0]), int(org[1])
    t = 2 if scale >= 0.6 else 1
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
    x0, y0, x1, y1 = max(x - 3, 0), max(y - h - 4, 0), min(x + int(w * 1.1) + 4, im.shape[1]), min(y + base + 2, im.shape[0])
    if x1 > x0 and y1 > y0:
        im[y0:y1, x0:x1] = (im[y0:y1, x0:x1] * 0.3).astype(im.dtype)
    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, t, cv2.LINE_AA)


def _cross(im, p, color, r=7, t=2):
    x, y = int(round(p[0])), int(round(p[1]))
    cv2.line(im, (x - r, y), (x + r, y), color, t, cv2.LINE_AA)
    cv2.line(im, (x, y - r), (x, y + r), color, t, cv2.LINE_AA)


def _zoom(img, c, half, tile):
    """tile x tile px view of img around c (full-res px), half = half window in full-res px. Returns (tile, map)."""
    k = tile / (2.0 * half)
    M = np.array([[k, 0, tile / 2 - k * c[0]], [0, k, tile / 2 - k * c[1]]])
    t = cv2.warpAffine(img, M, (tile, tile), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR), (lambda p: (np.asarray(p, float) * k + M[:, 2]))


def draw_points(ref, new, ref_pts, new_pts, gt=None, pairs=None, names=None, title=None,
                panel_height=900, zoom_half=40, tile=220):
    """Picture of a transfer result.

    Top:    reference image with the reference points | new image with the placed points (red) and, if `gt` is given,
            the ground-truth points (green crosses, joined to the placed point by a yellow line).
    Bottom: zoomed full-resolution tiles around every point (reference row, new-image row), +-zoom_half px,
            so sub-pixel to ~40 px errors are visible. Tile captions give the error in px when `gt` is given.
    pairs:  list of (i, j) point indices joined by a line (e.g. the D1..D3 distances).
    Returns a BGR image (save with cv2.imwrite)."""
    ref_pts = np.atleast_2d(np.asarray(ref_pts, float))
    new_pts = np.atleast_2d(np.asarray(new_pts, float))
    gt = None if gt is None else np.atleast_2d(np.asarray(gt, float))
    n = len(ref_pts)
    names = names or [str(i) for i in range(n)]
    col = [_COLORS[(i // 2 if pairs else i) % len(_COLORS)] for i in range(n)]

    def panel(img, pts, is_new):
        s = panel_height / img.shape[0]
        im = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        P = pts * s
        for i, j in (pairs or []):
            cv2.line(im, tuple(np.round(P[i]).astype(int)), tuple(np.round(P[j]).astype(int)), col[i], 1, cv2.LINE_AA)
        if is_new and gt is not None:
            G = gt * s
            for i in range(n):
                cv2.line(im, tuple(np.round(G[i]).astype(int)), tuple(np.round(P[i]).astype(int)), (0, 255, 255), 1, cv2.LINE_AA)
                _cross(im, G[i], GT_COLOR, 9, 2)
        for i in range(n):
            cv2.circle(im, tuple(np.round(P[i]).astype(int)), 8, PRED_COLOR if is_new else col[i], 2, cv2.LINE_AA)
            _text(im, names[i], P[i] + (11, -8), 0.6, col[i])
        _text(im, "new image (red = placed" + (", green = truth)" if gt is not None else ")") if is_new else "reference",
              (10, 28), 0.8)
        return im

    top = [panel(ref, ref_pts, False), panel(new, new_pts, True)]
    top = np.hstack([top[0], np.zeros((panel_height, 6, 3), np.uint8), top[1]])

    rows = []
    for which in ("ref", "new"):
        tiles = []
        for i in range(n):
            if which == "ref":
                t, f = _zoom(ref, ref_pts[i], zoom_half, tile)
                cv2.circle(t, tuple(np.round(f(ref_pts[i])).astype(int)), 6, col[i], 2, cv2.LINE_AA)
                cap = f"{names[i]} reference"
            else:
                c = gt[i] if gt is not None else new_pts[i]
                t, f = _zoom(new, c, zoom_half, tile)
                if gt is not None:
                    _cross(t, f(gt[i]), GT_COLOR, 10, 2)
                cv2.circle(t, tuple(np.round(f(new_pts[i])).astype(int)), 6, PRED_COLOR, 2, cv2.LINE_AA)
                cap = f"{names[i]} err {np.linalg.norm(new_pts[i] - gt[i]):.1f}px" if gt is not None else f"{names[i]} placed"
            cv2.rectangle(t, (0, 0), (tile - 1, tile - 1), col[i], 2)
            _text(t, cap, (6, 20), 0.5)
            tiles.append(t)
        rows.append(np.hstack(tiles))
    _text(rows[0], f"zoom +-{zoom_half}px", (6, tile - 8), 0.45, (200, 200, 200))

    W = max(top.shape[1], rows[0].shape[1])
    padw = lambda im: cv2.copyMakeBorder(im, 0, 0, 0, W - im.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
    parts = [padw(top), padw(rows[0]), padw(rows[1])]
    if title:
        bar = np.full((36, W, 3), 40, np.uint8)
        _text(bar, title, (10, 25), 0.7)
        parts.insert(0, bar)
    return np.vstack(parts)


def draw_overlay(ref, new, ref_pts, res, gt=None, **kw):
    """draw_points for a result dict of map_points / map_points_oriented."""
    return draw_points(ref, new, ref_pts, res["new_pts"], gt=gt, **kw)


def parse_points(text):
    return [tuple(float(v) for v in pt.replace(",", " ").split()) for pt in text.replace("(", "").replace(")", "").split(";")
            if pt.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--points", required=True, help='reference points "x,y;x,y;..." in full-res pixels')
    ap.add_argument("--out", help="save a side-by-side overlay image")
    ap.add_argument("--json", help="save the inferred points + diagnostics")
    a = ap.parse_args()
    pts = parse_points(a.points)
    ref, new = load_gray(a.ref), load_gray(a.new)
    res = map_points(ref, new, pts)
    if a.out:
        cv2.imwrite(a.out, draw_overlay(ref, new, pts, res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}, f, indent=1)
