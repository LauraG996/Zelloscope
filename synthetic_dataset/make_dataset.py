"""Synthetic key-point / key-distance test set generated from the reference shapes.

Inputs   shapes/shape*.png   and   shapes/distance.txt (hand-picked point pairs per shape,
         full-resolution pixel coordinates; header "6:" = reference, "6_2:" = that sub-image).
Output   dataset_out/raw/            transformed images (PNG, lossless)
         dataset_out/plotted/        same images with the transformed key distances drawn
         dataset_out/reference_points.csv     key points/distances on the (un-transformed) base images
         dataset_out/transformed_points.csv   key points/distances of every transformed image
         dataset_out/base_keypoints_check.jpg overview of the base images with their key distances

How the key points stay exact
  Every geometric operation is a coordinate mapping with a forward function f (input -> output
  coordinates, used for the key points) and an inverse g (output -> input, used by cv2.remap for
  the pixels).  Affine/perspective/twirl have closed forms; ripple is solved by fixed-point
  iteration.  All geometry of an image is chained and the pixels are resampled ONCE; the script
  asserts g(f(p)) == p for every key point.  Photometric operations (hue, gamma, colour cast, contrast, blur, then
  vignette, brightness, noise, salt&pepper, JPEG - in that order) are applied afterwards and do not move anything.
  By default the exposure-changing ones (gamma, colour cast, contrast, vignette, brightness) are switched off so
  every image keeps the exposure of its original; --vary-exposure turns them back on.  Their random numbers are
  drawn either way, so geometry and the remaining effects are identical with and without the flag.

Scale (mm per pixel)
  The originals are taken to be MM_PER_PX (--mm-per-px, default 0.014019 = raw Zelloscope frame).
  Every output PNG gets that image's own scale written into it as a `ZelloscopeData` iTXt chunk
  (base64 JSON {"p2mm": ...}, the format Zelloscope itself writes), so zs_auto_measure picks it up
  as "image metadata".  The value is the mean local scale at the key points: sqrt|det J| of the
  whole geometric chain.  Stretch / shear / perspective / bend make the real scale direction- and
  position-dependent, so a single p2mm is approximate; transformed_points.csv gives, per distance,
  the true length (Dn_mm, from the original image) and the length p2mm predicts (Dn_mm_p2mm).
  --strength 0 removes the direction-dependent warps (rotation, zoom and crop remain), which
  makes p2mm exact.

Repeatability: image i of shape n uses numpy Generator([SEED, n, i]) - independent of run order.
Bases: shapes with sub-images (shape6, shape6_2 / shape8_1..8_7) use every base equally often
(image i uses base i % n_bases).  Key points of bases without hand-picked points are transferred
from the reference by similarity-ICP on the outer contours (approximate, flagged in the CSVs).
"""
import argparse, base64, csv, glob, json, os, re, struct, zlib
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20250601
BASE_SIZE = 1536          # square working canvas the originals are fitted into
OUT_SIZE = 768            # size of the output images
SS = 2                    # supersampling factor of the single resampling step
EDGE_MARGIN = 5           # key points must lie at least this many px inside the output image
STRENGTH = 0.3            # global distortion knob: scales the magnitude of every warp (1.0 = very strong)
VARY_EXPOSURE = False     # True (--vary-exposure): random gamma/colour cast/contrast/vignette/brightness per image
MIN_P95, MIN_SPREAD = 45, 30   # visibility guard: 95th-percentile grey level / (p95 - p5) of the output
COLORS = [(60, 60, 235), (80, 200, 60), (40, 215, 245), (235, 140, 40), (220, 80, 220),   # BGR: D1 red, D2 green,
          (230, 230, 60), (40, 140, 255), (160, 160, 255), (150, 255, 150), (255, 255, 255)]  # D3 yellow, ... (cycled)
MM_PER_PX = 0.014019      # scale of the original images in shapes/ (raw Zelloscope frame, 3800x3804 px)


# ----------------------------------------------------------------------------- scale
def local_scale(ops, P, h=0.5):
    """sqrt|det J| of the whole forward chain at each point of P (n, 2): output px per input px."""
    def f(X):
        for op in ops:
            X = op.f(X)
        return X
    ex, ey = np.array([h, 0.]), np.array([0., h])
    jx = (f(P + ex) - f(P - ex)) / (2 * h)
    jy = (f(P + ey) - f(P - ey)) / (2 * h)
    return np.sqrt(np.abs(jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]))


def write_png_scale(path, p2mm):
    """Insert a ZelloscopeData iTXt chunk (base64 JSON {"p2mm": mm per pixel}) right after IHDR."""
    data = open(path, 'rb').read()
    text = base64.b64encode(json.dumps({'p2mm': p2mm, 'source': 'make_dataset.py'}).encode())
    body = b'ZelloscopeData\0' + b'\0\0' + b'\0' + b'\0' + text      # uncompressed, no language/translation
    chunk = struct.pack('>I', len(body)) + b'iTXt' + body + struct.pack('>I', zlib.crc32(b'iTXt' + body))
    end_ihdr = 8 + 8 + 13 + 4                                         # signature + IHDR(len,type,data,crc)
    with open(path, 'wb') as f:
        f.write(data[:end_ihdr] + chunk + data[end_ihdr:])


def read_png_scale(path):
    """p2mm from the ZelloscopeData chunk, or None (used to check what was written)."""
    data, i = open(path, 'rb').read(), 8
    while i < len(data):
        n, kind = struct.unpack('>I4s', data[i:i + 8])
        body = data[i + 8:i + 8 + n]
        if kind == b'iTXt' and body.startswith(b'ZelloscopeData\0'):
            return json.loads(base64.b64decode(body.split(b'\0', 5)[-1]))['p2mm']
        if kind == b'IDAT':
            return None
        i += n + 12
    return None


# ----------------------------------------------------------------------------- input handling
def parse_distances(path):
    """-> {(shape_no, sub or None): [(p1, p2), ...]}   (header 'N:' / 'N.' / 'N_k:')"""
    out, cur = {}, None
    for line in open(path):
        line = line.strip()
        h = re.fullmatch(r'(\d+)(?:_(\d+))?\s*[:.]', line)
        if h:
            cur = (int(h.group(1)), int(h.group(2)) if h.group(2) else None)
            out[cur] = []
            continue
        pts = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', line)
        if cur is not None and len(pts) == 2:
            out[cur].append(tuple(tuple(float(v) for v in p) for p in pts))
    return out


def discover(src):
    """-> {shape_no: [(stem, path, is_reference), ...]} reference first."""
    groups = {}
    for f in sorted(glob.glob(os.path.join(src, 'shape*.png'))):
        stem = os.path.splitext(os.path.basename(f))[0]
        m = re.fullmatch(r'shape(\d+)(?:_(\d+))?', stem)
        n, sub = int(m.group(1)), int(m.group(2)) if m.group(2) else None
        groups.setdefault(n, []).append((sub or 1, stem, f, sub in (None, 1)))
    return {n: [(s, p, r) for _, s, p, r in sorted(v, key=lambda t: (not t[3], t[0]))]
            for n, v in groups.items()}


class Base:
    """One source image fitted (centred, uniform scale) into the BASE_SIZE canvas."""

    def __init__(self, stem, path):
        im = cv2.imread(path)
        h, w = im.shape[:2]
        s = BASE_SIZE / max(h, w)
        nw, nh = round(w * s), round(h * s)
        self.stem, self.orig_wh = stem, (w, h)
        self.sx, self.sy = nw / w, nh / h
        self.ox, self.oy = (BASE_SIZE - nw) // 2, (BASE_SIZE - nh) // 2
        self.contour, self.valid, self.bg, self.bg_std = outer_contour(im)   # full-res coords, bg colour/noise
        fill = np.random.default_rng(0).normal(0, 2 * self.bg_std, (BASE_SIZE, BASE_SIZE, 3)) + self.bg
        self.canvas = np.clip(fill, 0, 255).astype(np.uint8)       # padding looks like the real background
        self.canvas[self.oy:self.oy + nh, self.ox:self.ox + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        self.kp = None                                        # (n_distances, 2, 2) full-res coordinates
        self.source, self.reg = None, {}

    def to_base(self, P):
        P = np.asarray(P, float)
        return np.stack([(P[..., 0] + .5) * self.sx - .5 + self.ox, (P[..., 1] + .5) * self.sy - .5 + self.oy], -1)


def outer_contour(im, scale=0.25):
    """Outer contour of the foam piece (full-res coords), a mask of points NOT cut by the frame,
    and the background colour (BGR median) / noise std measured away from the piece."""
    sm = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (0, 0), 3)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ndi.binary_opening(g > 0.6 * t, structure=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)).astype(bool))
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    m = lab == 1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1))))
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    h, w = m.shape
    ok = (c[:, 0] > 5) & (c[:, 0] < w - 6) & (c[:, 1] > 5) & (c[:, 1] < h - 6)
    bad = ndi.maximum_filter1d((~ok).astype(np.uint8), 2 * int(0.03 * len(c)) + 1, mode='wrap')
    far = ~ndi.binary_dilation(m, iterations=12)                 # background pixels away from the piece
    bgpix = sm[far].astype(float) if far.sum() > 100 else sm.reshape(-1, 3).astype(float)
    return (c + .5) / scale - .5, bad == 0, np.median(bgpix, 0), float(max(bgpix.std(0).mean(), 1.0))


# ----------------------------------------------------------------------------- key-point transfer
def _umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sv, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.array([1, -1 if np.linalg.det(U) * np.linalg.det(Vt) < 0 else 1])
    R = U @ np.diag(d) @ Vt
    s = (sv * d).sum() / S.var(0).sum()
    return s, R, md - s * R @ ms


def register(ref, var):
    """Similarity ICP of the reference contour onto the variant contour (cropped stretches
    ignored, 10 % worst matches trimmed).  Returns f(P) mapping reference px -> variant px."""
    A, B = ref.contour[ref.valid][::3], var.contour[var.valid][::3]
    tb = cKDTree(B)
    cA, cB = A.mean(0), B.mean(0)
    best = None
    for ang in np.deg2rad(np.arange(-40, 41, 10)):
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        e = tb.query((A - cA) @ R.T + cB)[0].mean()
        if best is None or e < best[0]:
            best = (e, R)
    R = best[1]
    s, t = 1.0, cB - cA @ R.T
    for _ in range(80):
        P = s * A @ R.T + t
        d1, j = tb.query(P)
        d2, i = cKDTree(P).query(B)
        src, dst, d = np.r_[A, A[i]], np.r_[B[j], B], np.r_[d1, d2]
        keep = d <= np.quantile(d, 0.9)
        s, R, t = _umeyama(src[keep], dst[keep])
    res = float(tb.query(s * A @ R.T + t)[0].mean())
    var.reg = dict(reg_residual_px=res, reg_scale=float(s), reg_rot_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))
    return lambda P: s * np.asarray(P, float) @ R.T + t


def assign_keypoints(groups, manual, src):
    bases = {}
    for n, items in groups.items():
        ref = None
        for stem, path, is_ref in items:
            b = Base(stem, path)
            sub = None if is_ref else int(stem.split('_')[1])
            if (n, sub) in manual or (is_ref and (n, 1) in manual):
                pairs = manual.get((n, sub)) or manual[(n, 1)]
                b.kp, b.source = np.array(pairs, float).reshape(-1, 2, 2), 'manual'
            if is_ref:
                ref = b
                assert b.kp is not None and len(b.kp), f'no key distances for reference of shape {n} in distance.txt'
            bases.setdefault(n, []).append(b)
        for b in bases[n]:
            if b.kp is None:
                b.kp, b.source = register(ref, b)(ref.kp.reshape(-1, 2)).reshape(-1, 2, 2), 'registered'
            assert len(b.kp) == len(ref.kp), \
                f'{b.stem}: {len(b.kp)} distances in distance.txt, but its reference has {len(ref.kp)}'
    return bases


# ----------------------------------------------------------------------------- geometry ops
class Op:
    """f: input -> output coordinates (key points), g: output -> input (pixel resampling).
    `H` is set for ops that are 3x3 homographies so that neighbours can be merged."""

    def __init__(self, name, desc, f, g, H=None):
        self.name, self.desc, self.f, self.g, self.H = name, desc, f, g, H


def _hom(H, P):
    X = np.c_[P.reshape(-1, 2), np.ones(P.size // 2)] @ H.T
    return (X[:, :2] / X[:, 2:3]).reshape(P.shape)


def hom_op(name, desc, H):
    Hi = np.linalg.inv(H)
    return Op(name, desc, lambda P: _hom(H, P), lambda Q: _hom(Hi, Q), H)


def about(M2, c):
    """2x2 linear map applied about point c, as a 3x3 matrix."""
    T = np.eye(3)
    T[:2, :2] = M2
    T[:2, 2] = np.asarray(c) - M2 @ np.asarray(c)
    return T


def rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def twirl_op(c, R, phi):
    c = np.asarray(c, float)

    def turn(P, sign):
        d = P - c
        r = np.linalg.norm(d, axis=-1)
        th = sign * phi * np.clip(1 - r / R, 0, None) ** 2        # r is preserved -> exact inverse
        co, si = np.cos(th), np.sin(th)
        return c + np.stack([co * d[..., 0] - si * d[..., 1], si * d[..., 0] + co * d[..., 1]], -1)
    return Op('twirl', f'twirl(R={R:.0f},phi={phi:.2f})', lambda P: turn(P, 1), lambda Q: turn(Q, -1))


def compile_g(ops):
    """Single output->input map for the whole chain (adjacent homographies merged)."""
    segs, M = [], None
    for op in ops:
        if op.H is not None:
            M = op.H if M is None else op.H @ M
        else:
            if M is not None:
                segs.append(np.linalg.inv(M)); M = None
            segs.append(op.g)
    if M is not None:
        segs.append(np.linalg.inv(M))

    def g(Q):
        for s in reversed(segs):
            Q = _hom(s, Q) if isinstance(s, np.ndarray) else s(Q)
        return Q
    return g


def radial_op(name, desc, c, h):
    """Radial remap about c: output radius r -> input radius h(r) (h elementwise, increasing).
    Forward (key points) solves h(r) = r_p by bisection, so it is exact to ~1e-12."""
    c = np.asarray(c, float)

    def g(Q):
        d = Q - c
        r = np.linalg.norm(d, axis=-1, keepdims=True)
        return c + d * (h(r) / np.maximum(r, 1e-9))

    def f(P):
        d = P - c
        rp = np.linalg.norm(d, axis=-1)
        lo, hi = np.zeros_like(rp), rp * 4 + 1
        for _ in range(70):
            mid = (lo + hi) / 2
            below = h(mid) < rp
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return c + d * (((lo + hi) / 2) / np.maximum(rp, 1e-9))[..., None]
    return Op(name, desc, f, g)


def wave_op(modes, name='ripple'):
    """Sum of plane waves  q -> q + sum_k amp_k sin(2 pi u_k.q / lam_k + ph_k) v_k  (output -> input).
    Amplitudes are scaled so the Lipschitz bound is <= 0.6, which makes the key-point solve a
    contraction.  1 mode = ripple, 2-3 modes = elastic-like warp."""
    lip = sum(a * 2 * np.pi / l for _, _, a, l, _ in modes)
    k = min(1.0, 0.6 / lip) if lip > 0 else 1.0       # lip == 0 at --strength 0
    modes = [(u, v, a * k, l, p) for u, v, a, l, p in modes]

    def disp(Q):
        return sum(a * np.sin(2 * np.pi * (Q @ u) / l + p)[..., None] * v for u, v, a, l, p in modes)

    def f(P):
        Q = P.copy()
        for _ in range(120):
            Q = P - disp(Q)
        return Q
    amp = sum(m[2] for m in modes)
    return Op(name, f'{name}(n={len(modes)},amp={amp:.1f})', f, lambda Q: Q + disp(Q))


def bend_op(c, B, S):
    """Parabolic bend along the local x axis: input y = output y + B*S*((x-c)/S)^2 (exact both ways)."""
    def off(x):
        return B * S * ((x - c) / S) ** 2
    return Op('bend', f'bend({B:+.2f})',
              lambda P: np.stack([P[..., 0], P[..., 1] - off(P[..., 0])], -1),
              lambda Q: np.stack([Q[..., 0], Q[..., 1] + off(Q[..., 0])], -1))


def unit(a):
    return np.array([np.cos(a), np.sin(a)])


def sample_geometry(rng, kp_base, contour_base):
    """Draw one random chain of geometric ops and return (ops, key points in output px) or None."""
    S, c = OUT_SIZE, (OUT_SIZE - 1) / 2
    k = BASE_SIZE
    ops = [hom_op('to_out', '', np.array([[S / k, 0, .5 * S / k - .5], [0, S / k, .5 * S / k - .5], [0, 0, 1]]))]
    rng.random(); rng.random()      # flips removed (were flip_h p=.5, flip_v p=.3); the draws are kept so
                                    # every other transformation stays identical to the earlier dataset
    if rng.random() < .9:
        a = rng.uniform(-np.pi, np.pi) if rng.random() < .25 else rng.uniform(-np.pi / 6, np.pi / 6)
        ops.append(hom_op('rotate', f'rotate({np.degrees(a):.1f})', about(rot2(a), (c, c))))
    if rng.random() < .6:
        r = np.exp(rng.uniform(np.log(.6), np.log(1.65)) * STRENGTH)
        a = rng.uniform(0, np.pi)                       # stretch direction
        ops.append(hom_op('aspect', f'aspect({r:.2f}@{np.degrees(a):.0f})',
                          about(rot2(a) @ np.diag([np.sqrt(r), 1 / np.sqrt(r)]) @ rot2(-a), (c, c))))
    if rng.random() < .45:
        sh = rng.uniform(-.3, .3) * STRENGTH
        ops.append(hom_op('shear', f'shear({sh:.2f})', about(np.array([[1, sh], [0, 1]]), (c, c))))
    if rng.random() < .85:
        z, t = rng.uniform(.6, 1.0), rng.uniform(-.1, .1, 2) * S
        H = about(np.eye(2) * z, (c, c)); H[:2, 2] += t
        ops.append(hom_op('zoom_shift', f'zoom({z:.2f})', H))
    if rng.random() < .5:
        src = np.float32([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]])
        H = cv2.getPerspectiveTransform(src, src + rng.uniform(-.14, .14, (4, 2)).astype(np.float32) * STRENGTH * S)
        ops.append(hom_op('perspective', 'perspective', H.astype(float)))
    nl = set(rng.choice(['bend', 'lens', 'bulge', 'twirl', 'wave'], size=int(rng.choice([0, 1, 2], p=[.3, .55, .15])),
                        replace=False))               # at most two non-linear distortions per image
    if 'bend' in nl:                                     # bend (in a random frame)
        a, B = rng.uniform(0, np.pi), rng.choice([-1, 1]) * rng.uniform(.06, .2) * STRENGTH * 2
        ops += [hom_op('rot', '', about(rot2(a), (c, c))), bend_op(c, B, S), hom_op('rot', '', about(rot2(-a), (c, c)))]
    if 'lens' in nl:                                     # lens barrel / pincushion
        kk = rng.choice([-1, 1]) * rng.uniform(.12, .4) * STRENGTH
        R = .75 * S
        ops.append(radial_op('lens', f'lens({kk:+.2f})', c + rng.uniform(-.1, .1, 2) * S,
                             lambda r, kk=kk, R=R: r * (1 + kk * (r / R) ** 2)))
    if 'bulge' in nl:                                    # local bulge / pinch
        s, R = rng.choice([-1, 1]) * rng.uniform(.3, .7) * STRENGTH * 1.3, rng.uniform(.25, .5) * S
        ops.append(radial_op('bulge', f'bulge({s:+.2f},R={R:.0f})', rng.uniform(.3, .7, 2) * S,
                             lambda r, s=s, R=R: r * (1 - s * np.clip(1 - r / R, 0, None) ** 2)))
    if 'twirl' in nl:
        ops.append(twirl_op(rng.uniform(.3, .7, 2) * S, rng.uniform(.25, .55) * S,
                            rng.choice([-1, 1]) * rng.uniform(.8, 3.0) * STRENGTH))
    if 'wave' in nl:                                     # ripple (1 wave) or elastic-like (2-3 waves)
        nm = 1 if rng.random() < .6 else int(rng.integers(2, 4))
        modes = []
        for _ in range(nm):
            tu = rng.uniform(0, np.pi)
            tv = tu + rng.choice([np.pi / 2, 0.0]) + rng.uniform(-.3, .3)
            modes.append((unit(tu), unit(tv), rng.uniform(.01, .04) * STRENGTH * S, rng.uniform(.08, .3) * S,
                          rng.uniform(0, 2 * np.pi)))
        ops.append(wave_op(modes, 'ripple' if nm == 1 else 'elastic'))
    P = kp_base.reshape(-1, 2).copy()
    for op in ops:
        P = op.f(P)
    cropped = False
    if rng.random() < .3:                                # mild crop: window that keeps all key points
        lo, hi = P.min(0) - EDGE_MARGIN * 2, P.max(0) + EDGE_MARGIN * 2
        zmax = min(1.12, S / (hi - lo).max())
        if zmax > 1.03 and lo.min() >= 0 and hi.max() <= S:
            cropped = True
            z = rng.uniform(1.03, zmax)
            w = S / z
            x0 = rng.uniform(max(0, hi[0] - w), min(lo[0], S - w))
            y0 = rng.uniform(max(0, hi[1] - w), min(lo[1], S - w))
            H = np.array([[z, 0, .5 * z - x0 * z - .5], [0, z, .5 * z - y0 * z - .5], [0, 0, 1]])
            ops.append(hom_op('crop', f'crop(x{z:.2f})', H))
            P = ops[-1].f(P)
    ok = np.isfinite(P).all() and P.min() >= EDGE_MARGIN and P.max() <= S - 1 - EDGE_MARGIN
    if ok:                                               # the piece itself must stay in view (a crop may cost <= 4 %)
        C = contour_base
        for op in ops:
            C = op.f(C)
        inside = ((C >= 0) & (C <= S - 1)).all(1).mean()
        ok = bool(np.isfinite(C).all() and inside >= (.96 if cropped else .995))
    if ok:                                               # reject chains whose inverse does not reproduce the points
        ok = float(np.abs(compile_g(ops)(P) - kp_base.reshape(-1, 2)).max()) < 1e-3
    return (ops, P.reshape(-1, 2, 2)) if ok and len(ops) > 1 else None


def geometry(rng, kp_base, contour_base):
    for _ in range(300):
        r = sample_geometry(rng, kp_base, contour_base)
        if r:
            return r
    S, c = OUT_SIZE, (OUT_SIZE - 1) / 2                  # fallback: plain fit
    ops = [hom_op('to_out', '', np.array([[S / BASE_SIZE, 0, .5 * S / BASE_SIZE - .5],
                                          [0, S / BASE_SIZE, .5 * S / BASE_SIZE - .5], [0, 0, 1]]))]
    return ops, ops[0].f(kp_base.reshape(-1, 2)).reshape(-1, 2, 2)


def warp(b, ops, rng):
    """Resample once; regions outside the source are filled with the base's own background
    colour plus matching noise (instead of black)."""
    g = compile_g(ops)
    n = OUT_SIZE * SS
    q = (np.arange(n) + .5) / SS - .5
    Q = np.stack(np.meshgrid(q, q), -1).reshape(-1, 2)
    G = g(Q).reshape(n, n, 2).astype(np.float32)
    big = cv2.remap(b.canvas, G[..., 0], G[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=tuple(float(v) for v in b.bg))
    inside = cv2.remap(np.full(b.canvas.shape[:2], 255, np.uint8), G[..., 0], G[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = cv2.resize(big, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    a = cv2.resize(inside, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    img += rng.normal(0, b.bg_std, img.shape).astype(np.float32) * (1 - a)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8), g


# ----------------------------------------------------------------------------- photometric ops
def photometric(img, rng):
    """hue, gamma, colour cast, contrast, blur, then vignette, brightness, noise, salt&pepper, JPEG
    (random subset, fixed order; the sensor-style effects come last). Returns img, list of descriptions."""
    log = []
    S = img.shape[0]
    if rng.random() < .7:
        d = int(rng.integers(-35, 36))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + d) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        log.append(f'hue({2 * d:+d}deg)')
    x = img.astype(np.float32)
    if rng.random() < .4:
        gm = np.exp(rng.uniform(np.log(.6), np.log(1.7)))
        if VARY_EXPOSURE:
            x = 255 * (x / 255) ** gm
            log.append(f'gamma({gm:.2f})')
    if rng.random() < .4:
        gain = rng.uniform(.75, 1.25, 3).astype(np.float32)
        if VARY_EXPOSURE:
            x = x * gain
            log.append('colorcast(' + ','.join(f'{v:.2f}' for v in gain) + ')')
    if rng.random() < .8:
        ct = rng.uniform(.7, 1.5)
        if VARY_EXPOSURE:
            x = (x - 128) * ct + 128
            log.append(f'contrast({ct:.2f})')
    if rng.random() < .4:
        sg = rng.uniform(.6, 2.5)
        x = cv2.GaussianBlur(x, (0, 0), sg)
        log.append(f'blur({sg:.1f})')
    elif rng.random() < .15:
        n, a = int(rng.integers(5, 22)) | 1, rng.uniform(0, 180)
        ker = np.zeros((n, n), np.float32)
        cv2.line(ker, (0, n // 2), (n - 1, n // 2), 1, 1)
        M = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), a, 1)
        ker = cv2.warpAffine(ker, M, (n, n))
        x = cv2.filter2D(x, -1, ker / ker.sum())
        log.append(f'motionblur({n},{a:.0f})')
    if rng.random() < .35:
        s = rng.uniform(.3, .6)
        if VARY_EXPOSURE:
            yy, xx = np.mgrid[:S, :S].astype(np.float32)
            r2 = ((xx - S / 2) ** 2 + (yy - S / 2) ** 2) / (S / 2) ** 2 / 2
            x = x * (1 - s * r2)[..., None]
            log.append(f'vignette({s:.2f})')
    if rng.random() < .8:
        br = rng.uniform(-40, 45)
        if VARY_EXPOSURE:
            x = x + br
            log.append(f'brightness({br:+.0f})')
    if rng.random() < .7:
        sg = rng.uniform(4, 22)
        x = x + rng.normal(0, sg, x.shape).astype(np.float32)
        log.append(f'noise({sg:.1f})')
    x = np.clip(x, 0, 255)
    if rng.random() < .2:
        amt = rng.uniform(.001, .01)
        m = rng.random(x.shape[:2])
        x[m < amt / 2] = 0
        x[m > 1 - amt / 2] = 255
        log.append(f'saltpepper({amt:.3f})')
    img = x.astype(np.uint8)
    if rng.random() < .35:
        q = int(rng.integers(15, 56))
        img = cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
        log.append(f'jpeg({q})')
    return img, log


# ----------------------------------------------------------------------------- drawing
def draw_distances(img, kp, scale=1.0, label=True):
    out = img.copy()
    for i, (a, b) in enumerate(kp):
        pa, pb = tuple(int(round(v * scale)) for v in a), tuple(int(round(v * scale)) for v in b)
        cv2.line(out, pa, pb, COLORS[i % len(COLORS)], 2, cv2.LINE_AA)
        for p in (pa, pb):
            cv2.circle(out, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, p, 5, (0, 0, 0), 1, cv2.LINE_AA)
        if label:
            m = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            txt = f'D{i + 1} {np.linalg.norm(np.asarray(a) - np.asarray(b)):.0f}px'
            cv2.putText(out, txt, m, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, txt, m, cv2.FONT_HERSHEY_SIMPLEX, .5, COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------- main
def main():
    global STRENGTH, VARY_EXPOSURE
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100, help='images per shape')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--strength', type=float, default=STRENGTH, help='distortion strength (default %(default)s)')
    ap.add_argument('--vary-exposure', action='store_true',
                    help='random gamma / colour cast / contrast / vignette / brightness per image (off by default)')
    ap.add_argument('--out', default=os.path.join(HERE, 'dataset_out'))
    ap.add_argument('--shapes', type=int, nargs='*', help='only these shape numbers')
    ap.add_argument('--mm-per-px', type=float, default=MM_PER_PX,
                    help='scale of the original images in shapes/ (default %(default)s)')
    a = ap.parse_args()
    STRENGTH, VARY_EXPOSURE = a.strength, a.vary_exposure
    src = os.path.join(HERE, 'shapes')
    raw_dir, plot_dir = os.path.join(a.out, 'raw'), os.path.join(a.out, 'plotted')
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)

    manual = parse_distances(os.path.join(src, 'distance.txt'))
    groups = {n: v for n, v in discover(src).items() if not a.shapes or n in a.shapes}
    bases = assign_keypoints(groups, manual, src)

    # reference csv + visual check of every base
    n_dist = max(len(b.kp) for bl in bases.values() for b in bl)   # shapes with fewer distances get empty cells

    def kp_cols(prefix=''):
        return [f'{prefix}D{i}_{c}' for i in range(1, n_dist + 1) for c in ('x1', 'y1', 'x2', 'y2', 'len')]

    def kp_row(kp):
        row = []
        for p, q in kp:
            row += [round(p[0], 2), round(p[1], 2), round(q[0], 2), round(q[1], 2), round(float(np.linalg.norm(p - q)), 2)]
        return row + [''] * 5 * (n_dist - len(kp))

    tiles = []
    with open(os.path.join(a.out, 'reference_points.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['shape', 'base_image', 'keypoint_source', 'orig_width', 'orig_height'] + kp_cols()
                   + ['reg_residual_px', 'reg_scale', 'reg_rot_deg'])
        for n, bl in bases.items():
            for b in bl:
                w.writerow([n, b.stem + '.png', b.source, *b.orig_wh] + kp_row(b.kp)
                           + [round(b.reg.get(k, np.nan), 3) for k in ('reg_residual_px', 'reg_scale', 'reg_rot_deg')])
                t = draw_distances(b.canvas, b.to_base(b.kp), label=False)
                t = cv2.resize(t, (384, 384), interpolation=cv2.INTER_AREA)
                cv2.putText(t, f'{b.stem} [{b.source}]', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(a.out, 'base_keypoints_check.jpg'),
                np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # transformed set
    rows, max_err = [], 0.0
    for n, bl in bases.items():
        for i in range(a.n):
            b = bl[i % len(bl)]
            rng = np.random.default_rng([a.seed, n, i])
            kp_base = b.to_base(b.kp)
            ops, kp_out = geometry(rng, kp_base, b.to_base(b.contour[b.valid][::6]))
            img, g = warp(b, ops, rng)
            err = float(np.abs(g(kp_out.reshape(-1, 2)) - kp_base.reshape(-1, 2)).max())   # g(f(p)) == p
            assert err < 0.05, (n, i, err)
            max_err = max(max_err, err)
            warped = img
            for _ in range(25):                          # redraw (seeded) until the piece is still visible
                img, plog = photometric(warped, rng)
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                p5, p95 = np.percentile(g[::4, ::4], [5, 95])
                if p95 >= MIN_P95 and p95 - p5 >= MIN_SPREAD:
                    break
            name = f'shape{n}_{i:03d}.png'
            if not cv2.imwrite(os.path.join(raw_dir, name), img):
                raise IOError(f'could not write {os.path.join(raw_dir, name)} (file locked?)')
            mm_base = a.mm_per_px / b.sx                        # mm per base-canvas px
            p2mm = mm_base / float(np.mean(local_scale(ops, kp_base.reshape(-1, 2))))
            write_png_scale(os.path.join(raw_dir, name), p2mm)
            assert abs(read_png_scale(os.path.join(raw_dir, name)) - p2mm) < 1e-12
            true_mm = [float(np.linalg.norm(p - q)) * a.mm_per_px for p, q in b.kp]
            pred_mm = [float(np.linalg.norm(p - q)) * p2mm for p, q in kp_out]
            cv2.imwrite(os.path.join(plot_dir, name.replace('.png', '.jpg')), draw_distances(img, kp_out),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            desc = ';'.join([o.desc for o in ops if o.desc] + plog)
            rows.append([name, n, b.stem + '.png', b.source, f'{a.seed}-{n}-{i}', desc] + kp_row(kp_out)
                        + [f'{p2mm:.6g}'] + [x for t, pr in zip(true_mm, pred_mm) for x in (round(t, 4), round(pr, 4))]
                        + [''] * 2 * (n_dist - len(true_mm)))
        print(f'shape{n}: {a.n} images, bases {[x.stem for x in bl]}')
    with open(os.path.join(a.out, 'transformed_points.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'shape', 'base_image', 'keypoint_source', 'seed', 'ops'] + kp_cols()
                   + ['p2mm'] + [f'D{d}_{c}' for d in range(1, n_dist + 1) for c in ('mm', 'mm_p2mm')])
        w.writerows(rows)
    print(f'done: {len(rows)} images in {a.out}; max |g(f(p))-p| = {max_err:.2e} px')


if __name__ == '__main__':
    main()
