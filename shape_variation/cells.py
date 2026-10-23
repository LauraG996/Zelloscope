"""Foam cells: marker-controlled watershed (ported from cell_segmentation.py on the brave-hamilton branch, with
its pixel parameters converted to mm), per-cell size/shape/orientation, and maps in the aligned shape frame."""
import warnings

import cv2
import numpy as np
from skimage.measure import regionprops

from .segment import Prepared


def segment_cells(gray: np.ndarray, mask: np.ndarray, px_per_mm: float, p: dict) -> np.ndarray:
    """Watershed labels: -1 walls, 1 outside the foam, >= 2 one label per cell."""
    sigma = p["seed_smooth_mm"] * px_per_mm
    sep = max(3, int(round(p["seed_min_separation_mm"] * px_per_mm)))
    inverted = 255 - cv2.GaussianBlur(gray, (0, 0), sigma)
    local_max = cv2.dilate(inverted, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sep, sep)))
    seeds = ((inverted == local_max) & (mask > 0)).astype(np.uint8) * 255
    seeds = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, seed_labels = cv2.connectedComponents(seeds)

    markers = np.zeros(gray.shape, np.int32)
    markers[mask == 0] = 1
    markers[seed_labels > 0] = seed_labels[seed_labels > 0] + 1
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced = cv2.GaussianBlur(clahe.apply(gray), (0, 0), max(0.3, p["wall_smooth_mm"] * px_per_mm))
    cv2.watershed(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), markers)
    return markers


def cell_labels(img: np.ndarray, prep: Prepared, p: dict):
    """Watershed on the image downsampled to p["work_px_per_mm"].

    Returns (labels, small image, scale factor f from full-res px to small px, small px per mm, kept labels,
    edge labels, foam mask, cavity mask) - masks in small-image px. Kept = not touching the foam edge, not below
    the minimum area and not larger than `merged_area_factor` x the median cell area (several cells merged).
    At the surfaces the watershed does not split the first cell layer (it makes strips along the surface), so
    the surface layer is measured with surface_intercepts() instead."""
    px_per_mm_full = 1.0 / prep.record.mm_per_px
    f = min(1.0, p["work_px_per_mm"] / px_per_mm_full)
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    res = px_per_mm_full * f

    # foam / cavity masks in this image's pixels: canonical masks mapped back through the inverse transform
    m = prep.image_to_canvas.copy()
    m[:, :2] /= f                                      # small-image px -> canvas px
    inv = cv2.invertAffineTransform(m)
    size = (gray.shape[1], gray.shape[0])
    foam = cv2.warpAffine(prep.foam.astype(np.uint8), inv, size, flags=cv2.INTER_NEAREST) > 0
    cavity = cv2.warpAffine(prep.cavity.astype(np.uint8), inv, size, flags=cv2.INTER_NEAREST) > 0
    mask = cv2.erode(foam.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    labels = segment_cells(gray, mask, res, p)
    edge = set(np.unique(labels[cv2.dilate((labels == 1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0]))
    areas = np.bincount(np.where(labels >= 2, labels, 0).ravel())
    cell_areas = areas[2:][areas[2:] > 0]
    min_area = p["min_area_mm2"] * res * res
    max_area = p.get("merged_area_factor", 4.0) * float(np.median(cell_areas)) if len(cell_areas) else np.inf
    kept = {lab for lab in range(2, len(areas)) if min_area <= areas[lab] <= max_area and lab not in edge}
    return labels, small, f, res, kept, edge, foam, cavity


def measure_cells(img: np.ndarray, prep: Prepared, p: dict) -> dict:
    """Cells of one image: centroid (canonical mm), size, elongation, orientation, and their relation to the
    nearest surface - distance, which surface (outer or cavity), whether the cell touches it, and its extent
    across the surface vs into the foam (flattening > 1 = squashed against the surface)."""
    labels, _, f, res, kept, edge, foam, cavity = cell_labels(img, prep, p)
    outside = ~foam & ~cavity
    # distance (px) to the outer surface and to the cavity surface; the image border is not a surface
    d_out = cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, 5) if outside.any() \
        else np.full(foam.shape, np.inf, np.float32)
    d_cav = cv2.distanceTransform((~cavity).astype(np.uint8), cv2.DIST_L2, 5) if cavity.any() \
        else np.full(foam.shape, np.inf, np.float32)
    g = {name: (cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=5), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=5))
         for name, d in (("outer", d_out), ("cavity", d_cav))}

    keys = ("x", "y", "area_mm2", "diameter_mm", "elongation", "angle_rad", "surface_dist_mm", "at_surface",
            "thick_mm", "wide_mm", "flattening")
    out = {k: [] for k in keys}
    surface = []
    for r in regionprops(np.where(labels >= 2, labels, 0)):
        if r.label not in kept:
            continue
        cy, cx = r.centroid
        iy, ix = int(round(cy)), int(round(cx))
        do, dc = float(d_out[iy, ix]), float(d_cav[iy, ix])
        which = "outer" if do <= dc else "cavity"
        gx, gy = g[which][0][iy, ix], g[which][1][iy, ix]
        norm = np.hypot(gx, gy)
        nvec = np.array([gx, gy]) / norm if norm > 0 else np.array([0.0, 1.0])     # (x, y), away from surface
        tvec = np.array([-nvec[1], nvec[0]])
        xy = r.coords[:, ::-1].astype(float)                                        # (x, y)
        cov = np.cov(xy.T) if len(xy) > 2 else np.eye(2)
        thick = 4.0 * np.sqrt(max(nvec @ cov @ nvec, 1e-9)) / res                  # ~ full width of an ellipse
        wide = 4.0 * np.sqrt(max(tvec @ cov @ tvec, 1e-9)) / res
        out["x"].append(cx / f)
        out["y"].append(cy / f)
        out["area_mm2"].append(r.area / res ** 2)
        out["diameter_mm"].append(r.equivalent_diameter_area / res)
        out["elongation"].append(r.axis_major_length / max(r.axis_minor_length, 1e-6))
        # skimage orientation: angle between the row axis and the major axis -> angle from the x axis
        out["angle_rad"].append(np.pi / 2 - r.orientation)
        out["surface_dist_mm"].append(min(do, dc) / res)
        out["at_surface"].append(r.label in edge)
        out["thick_mm"].append(thick)
        out["wide_mm"].append(wide)
        out["flattening"].append(wide / thick)
        surface.append(which)
    cells = {k: np.asarray(v, bool if k == "at_surface" else float) for k, v in out.items()}
    cells["surface"] = np.asarray(surface, dtype="<U6")
    xy = prep.image_px_to_mm(np.column_stack([cells["x"], cells["y"]])) if len(cells["x"]) else np.zeros((0, 2))
    cells["x_mm"], cells["y_mm"] = xy[:, 0], xy[:, 1]
    # orientation in the canonical frame: rotate by the image->canvas rotation
    rot = np.arctan2(prep.image_to_canvas[1, 0], prep.image_to_canvas[0, 0])
    cells["angle_rad"] = cells["angle_rad"] - rot if len(cells["angle_rad"]) else cells["angle_rad"]
    return cells


# Distance-from-surface bins (mm) for the surface profiles; "core" = cells further than CORE_FROM_MM in.
SURFACE_BINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)   # deeper, lines that stop at the midline bias chords short
CORE_FROM_MM = 2.0


def wall_map(small: np.ndarray, foam: np.ndarray, res: float) -> np.ndarray:
    """Ridge strength of the bright cell walls (Sato tubeness at ~0.025-0.05 mm), 0 outside the foam."""
    from skimage.filters import sato
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    sig = [s * res / 40.0 for s in (1.0, 1.5, 2.0)]
    ridge = sato(gray.astype(np.float32) / 255.0, sigmas=sig, black_ridges=False).astype(np.float32)
    ridge[~foam] = 0
    return ridge


# Wall-detection sensitivities tried when calibrating against the core (peak height, in units of the 90th
# percentile of the wall map; prominence = 0.75 x height).
SENSITIVITY_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)


def _walls(profile: np.ndarray, step_mm: float, scale: float, height: float) -> np.ndarray:
    """Indices of cell walls along a sampled line: ridge peaks at least 0.15 mm apart, above `height` and with a
    prominence of 0.75 x `height` (both in units of `scale`)."""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(profile, distance=max(1, int(round(0.15 / step_mm))), height=height * scale,
                       prominence=0.75 * height * scale)
    return pk


def _sample_lines(img: np.ndarray, prep: Prepared, p: dict, depth_mm: float | None, levels, max_depth_mm: float,
                  spacing_mm: float, min_run_mm: float = 2.0) -> dict:
    """Wall-map profiles along lines straight in from each surface (every `spacing_mm` along it, up to
    `max_depth_mm` or halfway to the other surface) and along lines parallel to each surface at the depths in
    `levels` (mm). Lines are in small-image px; `to_canvas` maps them to the canonical raster."""
    from scipy.ndimage import gaussian_filter1d
    px_per_mm_full = 1.0 / prep.record.mm_per_px
    f = min(1.0, p["work_px_per_mm"] / px_per_mm_full)
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    res = px_per_mm_full * f
    m = prep.image_to_canvas.copy()
    m[:, :2] /= f
    inv = cv2.invertAffineTransform(m)
    size = (small.shape[1], small.shape[0])
    foam = cv2.warpAffine(prep.foam.astype(np.uint8), inv, size, flags=cv2.INTER_NEAREST) > 0
    cavity = cv2.warpAffine(prep.cavity.astype(np.uint8), inv, size, flags=cv2.INTER_NEAREST) > 0
    # usable area: inside the window (if any, with 1 mm margin) and away from the image border
    if depth_mm is not None:
        rows = np.arange(prep.foam.shape[0])[:, None]
        win = np.broadcast_to(rows >= prep.base_row - (depth_mm - 1.0) * prep.res, prep.foam.shape)
        usable = cv2.warpAffine(win.astype(np.uint8) * 255, inv, size, flags=cv2.INTER_NEAREST) > 0
    else:
        usable = np.ones(foam.shape, bool)
    border = int(round(1.0 * res))
    usable[:border], usable[-border:], usable[:, :border], usable[:, -border:] = False, False, False, False

    ridge = wall_map(small, foam, res)
    scale = float(np.percentile(ridge[foam], 90)) if foam.any() else 1.0
    outside = ~foam & ~cavity
    dist = {"outer": cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, 5),
            "cavity": cv2.distanceTransform((~cavity).astype(np.uint8), cv2.DIST_L2, 5)}
    step = 1.0 / res
    n_steps = int(max_depth_mm * res)

    # ---- sample all lines once: (surface, kind, level or None, profile, xs, ys)
    lines = []
    for surf, region in (("outer", foam | cavity), ("cavity", cavity)):
        other = dist["cavity" if surf == "outer" else "outer"]
        d_here = dist[surf]
        cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # the outer surface is the largest outline; every hole is a cavity surface
        cs = sorted(cs, key=cv2.contourArea, reverse=True)[:1] if surf == "outer" else list(cs)
        for c in cs:
            c = c.reshape(-1, 2).astype(float)
            if len(c) < 20:
                continue
            cs_s = gaussian_filter1d(c, res * 1.0, axis=0, mode="wrap")         # direction smoothed over ~1 mm
            tan = np.gradient(cs_s, axis=0)
            nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
            t = np.arange(n_steps, dtype=np.float32)
            for i in range(0, len(c), max(1, int(round(spacing_mm * res)))):
                x0, y0 = c[i]
                nv = nrm[i]
                px_, py_ = int(round(x0 + nv[0] * 0.5 * res)), int(round(y0 + nv[1] * 0.5 * res))
                if not (0 <= py_ < foam.shape[0] and 0 <= px_ < foam.shape[1]) or not foam[py_, px_]:
                    nv = -nv                                                    # point into the foam
                xs, ys = x0 + nv[0] * t, y0 + nv[1] * t
                ok = (xs >= 0) & (ys >= 0) & (xs < foam.shape[1] - 1) & (ys < foam.shape[0] - 1)
                xs, ys = xs[ok], ys[ok]
                xi, yi = xs.astype(int), ys.astype(int)
                if len(xi) < 10 or not usable[yi[0], xi[0]]:
                    continue
                # stop where the line leaves the foam / usable area or gets closer to the other surface
                good = foam[yi, xi] & usable[yi, xi] & (d_here[yi, xi] <= other[yi, xi] + 1)
                good[:3] = True
                stop = len(good) if good.all() else int(np.argmin(good))
                if stop < 10:
                    continue
                prof = cv2.remap(ridge, xs[:stop].reshape(1, -1).astype(np.float32),
                                 ys[:stop].reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR).ravel()
                lines.append((surf, "normal", None, prof, xs[:stop], ys[:stop]))
        for level in levels:
            iso = (d_here >= level * res).astype(np.uint8)
            cs2, _ = cv2.findContours(iso, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for c2 in cs2:
                c2 = c2.reshape(-1, 2)
                xi, yi = c2[:, 0], c2[:, 1]
                on = (np.abs(d_here[yi, xi] - level * res) < 1.0) & foam[yi, xi] & usable[yi, xi] & \
                     (d_here[yi, xi] < other[yi, xi])
                idx = np.nonzero(on)[0]
                if len(idx) < 20:
                    continue
                for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                    if len(run) < int(min_run_mm * res):
                        continue
                    pts = c2[run].astype(float)
                    s_ = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
                    u = np.arange(0, s_[-1], 1.0)
                    xs = np.interp(u, s_, pts[:, 0]).astype(np.float32)
                    ys = np.interp(u, s_, pts[:, 1]).astype(np.float32)
                    prof = cv2.remap(ridge, xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR).ravel()
                    lines.append((surf, "tangent", level, prof, xs, ys))

    return dict(lines=lines, step=step, scale=scale, res=res, to_canvas=m)


def surface_intercepts(img: np.ndarray, prep: Prepared, p: dict, depth_mm: float | None,
                       core_diameter_mm: float | None = None, max_depth_mm: float = 3.5,
                       spacing_mm: float = 0.25, keep_points: bool = False) -> dict:
    """Cell size against depth below each surface by the line-intercept method (as in ASTM D3576):
    walls are counted along lines straight in from the surface (chord = how deep the cells are) and along
    lines parallel to the surface at fixed depths (chord = how wide they are). Flattening = wide / deep.

    The wall-detection sensitivity is calibrated per image: with `core_diameter_mm` (mean diameter of the
    watershed cells, which are reliable in the core) it is chosen so that the mean chord in the core
    (>= CORE_FROM_MM from the surface) equals pi/4 x that diameter, the mean chord of a round cell.
    Returns per surface the binned profile (deep_mm, wide_mm, flattening) and the chosen sensitivity."""
    edges = np.asarray(SURFACE_BINS)
    levels = (edges[:-1] + edges[1:]) / 2
    smp = _sample_lines(img, prep, p, depth_mm, levels, max_depth_mm, spacing_mm)
    lines, step, scale, res = smp["lines"], smp["step"], smp["scale"], smp["res"]

    def chords(height):
        res_ = {"outer": ([], [], []), "cavity": ([], [], [])}               # normal, tangent, wall points
        for surf, kind, level, prof, xs, ys in lines:
            pk = _walls(prof, step, scale, height)
            if kind == "normal":
                res_[surf][0].extend(((a_ + b_) / 2 * step, (b_ - a_) * step) for a_, b_ in zip(pk[:-1], pk[1:]))
                res_[surf][2].extend(zip(xs[pk], ys[pk]))
            else:
                res_[surf][1].extend((level, float(b_ - a_) * step) for a_, b_ in zip(pk[:-1], pk[1:]))
        return res_

    def core_chord(r_):
        c_ = [ch for surf in r_ for (d_, ch) in r_[surf][0] + r_[surf][1] if d_ >= CORE_FROM_MM]
        return float(np.mean(c_)) if c_ else np.nan

    if core_diameter_mm:
        target = np.pi / 4 * core_diameter_mm
        trials = [(h, chords(h)) for h in SENSITIVITY_GRID]
        height, best = min(trials, key=lambda hr: abs(core_chord(hr[1]) - target) if np.isfinite(core_chord(hr[1]))
                           else np.inf)
    else:
        target, height = float("nan"), 0.6
        best = chords(height)

    out = {"bin_mid_mm": levels, "res": res, "sensitivity": height, "core_chord_mm": core_chord(best),
           "core_chord_target_mm": target}
    for surf in ("outer", "cavity"):
        nc = np.array(best[surf][0]).reshape(-1, 2)
        tc = np.array(best[surf][1]).reshape(-1, 2)
        idx_n, idx_t = np.digitize(nc[:, 0], edges) - 1, np.digitize(tc[:, 0], edges) - 1
        deep, wide, n_n, n_t = [], [], [], []
        for i in range(len(edges) - 1):
            bn, bt = nc[idx_n == i, 1], tc[idx_t == i, 1]
            n_n.append(len(bn))
            n_t.append(len(bt))
            deep.append(float(np.mean(bn)) if len(bn) >= 10 else np.nan)
            wide.append(float(np.mean(bt)) if len(bt) >= 10 else np.nan)
        deep, wide = np.array(deep), np.array(wide)
        out[surf] = dict(deep_mm=deep, wide_mm=wide, flattening=wide / deep, n_normal=np.array(n_n),
                         n_tangent=np.array(n_t))
        if keep_points:
            out[surf]["wall_points"] = np.array(best[surf][2]).reshape(-1, 2)
    return out


def flattening_samples(img: np.ndarray, prep: Prepared, p: dict, depth_mm: float | None, height: float,
                       spacing_mm: float = 0.25, level_step_mm: float = 0.25, max_depth_mm: float = 8.0) -> dict:
    """Wall crossings for a flattening map: sample points (canonical mm) along lines straight in from the
    nearest surface ("normal") and parallel to it ("tangent"), covering the whole foam, each with a flag for
    whether a wall was found there. Walls per mm along the normal divided by walls per mm along the tangent,
    per area, is how flattened the cells are against the nearest surface (1 = round). `height` is the
    calibrated wall sensitivity (surface_intercepts()["sensitivity"])."""
    levels = np.arange(level_step_mm / 2, max_depth_mm, level_step_mm)
    smp = _sample_lines(img, prep, p, depth_mm, levels, max_depth_mm, spacing_mm, min_run_mm=1.0)
    out = {}
    m = smp["to_canvas"]
    for kind in ("normal", "tangent"):
        xs, ys, walls = [], [], []
        for surf, k, level, prof, lx, ly in smp["lines"]:
            if k != kind:
                continue
            flag = np.zeros(len(prof), bool)
            flag[_walls(prof, smp["step"], smp["scale"], height)] = True
            xs.append(lx[:len(prof)])
            ys.append(ly[:len(prof)])
            walls.append(flag)
        if xs:
            px = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
            canvas = px @ m[:, :2].T + m[:, 2]
            out[kind] = dict(xy_mm=prep.to_mm(canvas), wall=np.concatenate(walls), step_mm=smp["step"])
        else:
            out[kind] = dict(xy_mm=np.zeros((0, 2)), wall=np.zeros(0, bool), step_mm=smp["step"])
    return out


def flattening_grid(samples: list[dict], aligned: list, bin_mm: float, extent, min_walls: int = 20) -> tuple:
    """Average flattening per square over parts: (walls per mm along the normal) / (walls per mm along the
    tangent) in each `bin_mm` square of the aligned frame. Squares with fewer than `min_walls` walls counted in
    either direction are left empty. Returns (grid, extent for imshow, walls counted)."""
    x0, x1, y0, y1 = extent
    nx, ny = max(1, int(np.ceil((x1 - x0) / bin_mm))), max(1, int(np.ceil((y1 - y0) / bin_mm)))
    acc = {k: (np.zeros((ny, nx)), np.zeros((ny, nx))) for k in ("normal", "tangent")}     # walls, length
    for smp, al in zip(samples, aligned):
        for kind in ("normal", "tangent"):
            d = smp[kind]
            if not len(d["wall"]):
                continue
            xy = al.apply(d["xy_mm"])
            ix = np.clip(((xy[:, 0] - x0) / bin_mm).astype(int), 0, nx - 1)
            iy = np.clip(((xy[:, 1] - y0) / bin_mm).astype(int), 0, ny - 1)
            np.add.at(acc[kind][0], (iy, ix), d["wall"].astype(float))
            np.add.at(acc[kind][1], (iy, ix), d["step_mm"])
    (wn, ln), (wt, lt) = acc["normal"], acc["tangent"]
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = (wn / ln) / (wt / lt)
    grid[(wn < min_walls) | (wt < min_walls)] = np.nan
    return grid, (x0, x0 + nx * bin_mm, y0 + ny * bin_mm, y0), np.minimum(wn, wt)


def surface_features(si: dict) -> dict:
    """Per-part numbers from surface_intercepts: the first layer (0-0.5 mm) and the core (>= CORE_FROM_MM)."""
    feats = {"wall_sensitivity": si["sensitivity"]}
    mids = si["bin_mid_mm"]
    core = mids >= CORE_FROM_MM
    for surf in ("outer", "cavity"):
        d = si[surf]
        for name, sel in (("layer", slice(0, 1)), ("core", core)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN (surface not in view)
                feats[f"{surf}_{name}_flattening"] = round(float(np.nanmean(d["flattening"][sel])), 4)
                feats[f"{surf}_{name}_deep_mm"] = round(float(np.nanmean(d["deep_mm"][sel])), 4)
                feats[f"{surf}_{name}_wide_mm"] = round(float(np.nanmean(d["wide_mm"][sel])), 4)
        # depth of the flattened zone: first bin whose flattening is within 10 % of round
        fl = d["flattening"]
        below = np.nonzero(np.isfinite(fl) & (fl < 1.1))[0]
        feats[f"{surf}_flattened_depth_mm"] = round(float(mids[below[0]]), 3) if len(below) else float("nan")
    return feats


# Overlay colours (BGR): core cells, surface-layer cells, dropped cells, foam outline, cavity outline, window cut.
KEPT_BGR, SURFACE_BGR, DROPPED_BGR, FOAM_BGR, CAVITY_BGR, CUT_BGR = (0, 230, 255), (255, 230, 60), \
    (200, 80, 255), (60, 220, 60), (0, 140, 255), (255, 255, 255)


def draw_segmentation(img: np.ndarray, prep: Prepared, p: dict, depth_mm: float | None,
                      with_cells: bool = True, walls: bool = True, generic: bool = False) -> tuple[np.ndarray, dict]:
    """The image (at the cell working resolution) with the foam and cavity outlines, the window cut line and
    every cell boundary: measured cells in yellow, cells dropped (touching the edge, merged, too small, or outside
    the window) in violet. With `walls`, the cell walls found by the surface intercepts are marked in cyan. Also returns the zoom-in points (small-image px) for the base centre and a leg."""
    if with_cells:
        labels, small, f, res, kept, edge, _, _ = cell_labels(img, prep, p)
    else:
        px_per_mm_full = 1.0 / prep.record.mm_per_px
        f = min(1.0, p["work_px_per_mm"] / px_per_mm_full)
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img.copy()
        res = px_per_mm_full * f
    out = small.copy()
    m = prep.image_to_canvas.copy()
    m[:, :2] /= f
    inv = cv2.invertAffineTransform(m)                       # canvas px -> small-image px
    to_img = lambda pts: (np.asarray(pts, float) @ inv[:, :2].T + inv[:, 2])
    canvas_of_mm = lambda q: np.asarray(q, float) * prep.res + [prep.axis_col, prep.base_row]
    th = max(1, int(round(res / 20)))                        # line width ~0.05 mm

    if with_cells:
        cell = labels >= 2
        # cells outside the window count as dropped
        if depth_mm is not None:
            rows = np.arange(prep.foam.shape[0])[:, None]
            win = (prep.foam & (rows >= prep.base_row - depth_mm * prep.res)).astype(np.uint8)
            win_img = cv2.warpAffine(win, inv, (small.shape[1], small.shape[0]), flags=cv2.INTER_NEAREST) > 0
            ys, xs = np.nonzero(cell)
            lab = labels[ys, xs]
            n = labels.max() + 1
            inside = np.bincount(lab, weights=win_img[ys, xs], minlength=n) / np.maximum(np.bincount(lab, minlength=n), 1)
            kept = {k for k in kept if inside[k] > 0.5}
        kept_mask = np.isin(labels, list(kept)) if kept else np.zeros_like(cell)
        for mask, colour in ((cell & ~kept_mask, DROPPED_BGR), (kept_mask, KEPT_BGR)):
            edges = cv2.morphologyEx(np.where(mask, labels, 0).astype(np.float32), cv2.MORPH_GRADIENT,
                                     np.ones((3, 3), np.uint8)) > 0
            out[edges & (mask | (labels == -1))] = colour

    for region, colour in ((prep.foam | prep.cavity, FOAM_BGR), (prep.cavity, CAVITY_BGR)):
        cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in (sorted(cs, key=cv2.contourArea)[-1:] if colour == FOAM_BGR else cs):
            c = c.reshape(-1, 2)
            cv2.polylines(out, [np.round(to_img(c)).astype(np.int32)], True, colour, 2 * th, cv2.LINE_AA)
    if with_cells and walls:
        areas = np.bincount(np.where(labels >= 2, labels, 0).ravel())
        kept_areas = np.array([areas[k] for k in kept if k < len(areas)], float)
        core_d = float(np.mean(2 * np.sqrt(kept_areas / np.pi)) / res) if len(kept_areas) else None
        si = surface_intercepts(img, prep, p, depth_mm, core_diameter_mm=core_d, keep_points=True)
        for surf in ("outer", "cavity"):
            for x, y in si[surf].get("wall_points", np.zeros((0, 2))):
                cv2.circle(out, (int(round(x)), int(round(y))), max(1, th), SURFACE_BGR, -1)
    if depth_mm is not None:
        cut = canvas_of_mm([[-60, -depth_mm], [60, -depth_mm]])
        a, b = np.round(to_img(cut)).astype(int)
        cv2.line(out, tuple(a), tuple(b), CUT_BGR, 2 * th, cv2.LINE_AA)

    if generic:
        # zoom points: the thickest wall (deepest point of the foam) and the outline point farthest from the centre
        # (a corner), moved 3 mm into the foam
        dist = cv2.distanceTransform(prep.foam.astype(np.uint8), cv2.DIST_L2, 5)
        r_, c_ = np.unravel_index(int(np.argmax(dist)), dist.shape)
        cs, _ = cv2.findContours((prep.foam | prep.cavity).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        o = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
        far = o[np.argmax(np.hypot(o[:, 0] - prep.axis_col, o[:, 1] - prep.base_row))]
        towards = np.array([prep.axis_col, prep.base_row]) - far
        corner = far + towards / max(np.linalg.norm(towards), 1e-9) * 3.0 * prep.res
        return out, {"thickest wall": to_img([[c_, r_]])[0], "corner": to_img([corner])[0], "res": res}
    # zoom points: base centre (6 mm up from the outer base) and the middle of the left leg at 2/3 depth
    base_pt = to_img(canvas_of_mm([[0, -6]]))[0]
    d = 0.66 * (depth_mm if depth_mm is not None else prep.visible_depth_mm)
    row = int(round(prep.base_row - d * prep.res))
    cols = np.nonzero(prep.foam[row])[0] if 0 <= row < prep.foam.shape[0] else np.zeros(0, int)
    left = cols[cols < prep.axis_col]
    leg_pt = to_img([[left.mean() if len(left) else prep.axis_col, row]])[0]
    return out, dict(base=base_pt, leg=leg_pt, res=res)


def overlay_panel(img: np.ndarray, prep: Prepared, p: dict, depth_mm: float | None, title: str,
                  zoom_mm: float = 12.0, generic: bool = False) -> np.ndarray:
    """Whole image plus two zoomed crops (base and leg; generic: thickest wall and a corner), with a legend."""
    full, pts = draw_segmentation(img, prep, p, depth_mm, generic=generic)
    res = pts["res"]
    h = 900
    left = cv2.resize(full, (int(full.shape[1] * h / full.shape[0]), h), interpolation=cv2.INTER_AREA)
    crops = []
    half = int(zoom_mm * res / 2)
    for key in [k for k in pts if k != "res"]:
        x, y = np.round(pts[key]).astype(int)
        x0, y0 = max(0, x - half), max(0, y - half)
        c = full[y0:y0 + 2 * half, x0:x0 + 2 * half]
        c = cv2.resize(c, (h // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, f"{key} ({zoom_mm:g} x {zoom_mm:g} mm)", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        crops.append(c)
    panel = np.hstack([left, np.vstack(crops)])
    bar = np.full((70, panel.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    x = 12
    legend = ((("foam outline", FOAM_BGR), ("holes", CAVITY_BGR)) if generic else
              (("foam outline", FOAM_BGR), ("cavity", CAVITY_BGR), ("window cut", CUT_BGR))) + \
        (("cells", KEPT_BGR), ("walls on surface lines", SURFACE_BGR),
         ("dropped (edge / merged)" if generic else "dropped (edge / merged / outside window)", DROPPED_BGR))
    for text, colour in legend:
        cv2.rectangle(bar, (x, 44), (x + 18, 58), colour, -1)
        cv2.putText(bar, text, (x + 24, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        x += 34 + 9 * len(text)
    return np.vstack([bar, panel])


def _order(angles: np.ndarray) -> float:
    """Nematic order of orientations (0 = random, 1 = all parallel)."""
    return float(np.abs(np.mean(np.exp(2j * angles)))) if len(angles) else float("nan")


def cell_features(cells: dict, outer: np.ndarray, cavity: np.ndarray) -> dict:
    """Per-part cell features (aligned frame, y down, base at the bottom): overall and base vs legs."""
    x, y = cells["x_al"], cells["y_al"]
    if len(x) == 0:
        return {}
    holes = cavity if isinstance(cavity, list) else [cavity]           # generic mode: a list of holes
    regions = {"all": np.ones(len(x), bool)}
    if not isinstance(cavity, list):                                    # U profile: base vs legs
        cav_bottom = cavity[:, 1].max()
        regions.update(base=y > cav_bottom, legs=y <= cav_bottom)
    area = abs(cv2.contourArea(outer.astype(np.float32))) - sum(abs(cv2.contourArea(h.astype(np.float32)))
                                                                 for h in holes if len(h) >= 3)
    feats = dict(cell_count=int(len(x)), cell_density_per_mm2=round(len(x) / area, 4) if area > 0 else float("nan"))
    for name, sel in regions.items():
        if sel.sum() < 5:
            continue
        feats[f"cell_diam_{name}"] = round(float(np.mean(cells["diameter_mm"][sel])), 4)
        feats[f"cell_elong_{name}"] = round(float(np.mean(cells["elongation"][sel])), 4)
        feats[f"cell_order_{name}"] = round(_order(cells["angle_al"][sel]), 4)
    feats["cell_diam_cv"] = round(float(np.std(cells["diameter_mm"]) / np.mean(cells["diameter_mm"])), 4)

    return feats


def _foam_area_per_bin(foam_polys, bin_mm: float, x0: float, y0: float, nx: int, ny: int,
                       sub: int = 10) -> np.ndarray:
    """Total foam area (mm^2) in each bin summed over parts, by rasterising each part at `sub` px per bin."""
    res = sub / bin_mm
    area = np.zeros((ny, nx))
    for outer, cavity in foam_polys:
        m = np.zeros((ny * sub, nx * sub), np.uint8)
        to_px = lambda q: np.round((q - [x0, y0]) * res * 16).astype(np.int32)
        cv2.fillPoly(m, [to_px(outer)], 1, cv2.LINE_8, shift=4)
        for h in (cavity if isinstance(cavity, list) else [cavity]):
            if len(h) >= 3:
                cv2.fillPoly(m, [to_px(h)], 0, cv2.LINE_8, shift=4)
        area += m.reshape(ny, sub, nx, sub).sum(axis=(1, 3)) / res ** 2
    return area


def binned_map(all_cells: list[dict], key: str, bin_mm: float, extent, foam_polys=None) -> tuple[np.ndarray, tuple]:
    """Mean of `key` per bin over all parts (aligned frame). extent = (x0, x1, y0, y1). For key "density" the
    count is divided by the foam area in each bin (from `foam_polys`), so bins on the outline are not diluted."""
    x0, x1, y0, y1 = extent
    nx, ny = max(1, int(np.ceil((x1 - x0) / bin_mm))), max(1, int(np.ceil((y1 - y0) / bin_mm)))
    acc, cnt = np.zeros((ny, nx)), np.zeros((ny, nx))
    for c in all_cells:
        if key == "density":
            vals = np.ones(len(c["x_al"]))
        else:
            vals = c[key]
        ix = np.clip(((c["x_al"] - x0) / bin_mm).astype(int), 0, nx - 1)
        iy = np.clip(((c["y_al"] - y0) / bin_mm).astype(int), 0, ny - 1)
        np.add.at(acc, (iy, ix), vals)
        np.add.at(cnt, (iy, ix), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if key == "density":
            area = (_foam_area_per_bin(foam_polys, bin_mm, x0, y0, nx, ny) if foam_polys is not None
                    else np.full((ny, nx), len(all_cells) * bin_mm * bin_mm))
            out = acc / area
            # bins that are mostly outside the foam give noisy densities
            out[(cnt == 0) | (area < 0.3 * len(all_cells) * bin_mm * bin_mm)] = np.nan
        else:
            out = acc / cnt
    return out, (x0, x0 + nx * bin_mm, y0 + ny * bin_mm, y0)
