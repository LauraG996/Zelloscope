"""Run the whole analysis on a set of images and collect everything the report needs."""
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import cells as cellmod
from . import generic as gen
from .config import AnalysisConfig
from .correlate import forest_importance, spearman_table
from .io import ImageRecord, discover
from .monitor import i_chart, spe_leave_one_out, spe_limit, t2_leave_one_out, t2_limit
from .relaxation import early_prediction_check, fit_pooled, settle_time
from .segment import SegmentationError, Shape, extract, prepare
from .shape_model import (ShapeModel, align_to, bootstrap_stability, describe, describe_mode, fit_pca,
                          generalized_procrustes)
from .thickness import common_grid, on_grid, thickness_profile

KEY_DIMS = ["inner_width_max", "base_thickness", "outer_width_max", "wall_left", "wall_right",
            "cavity_width_mid", "lean_left_deg", "lean_right_deg"]
GENERIC_KEY_DIMS = ["area_mm2", "outer_width", "outer_height", "hole_width", "hole_height", "hole_x", "hole_y",
                    "hole1_width", "hole1_height", "hole1_x", "hole1_y"]
MAX_CHART_MODES = 3


@dataclass
class Analysis:
    cfg: AnalysisConfig
    records: list[ImageRecord]
    skipped: list[tuple[str, str]]
    shapes: list[Shape]
    depth_mm: float | None
    closed: bool
    aligned: list = field(default_factory=list)
    model: ShapeModel | None = None
    modes: list[dict] = field(default_factory=list)
    stability: dict = field(default_factory=dict)
    table: pd.DataFrame | None = None
    t2: dict = field(default_factory=dict)
    spe: dict = field(default_factory=dict)
    thickness_grid: np.ndarray | None = None
    thickness: np.ndarray | None = None
    charts: dict = field(default_factory=dict)
    relaxation: dict | str = "not run"
    cells: dict | str = "not run"
    correlations: dict | str = "not run"
    not_done: list[str] = field(default_factory=list)
    collect_next: list[str] = field(default_factory=list)
    kind: str = "u_profile"
    key_dims: list[str] = field(default_factory=lambda: list(KEY_DIMS))
    shape_features: list[str] = field(default_factory=list)     # columns that describe the shape itself


def _window(ts: dt.datetime | None, cfg) -> dict | None:
    pw = cfg.get("process_windows")
    if not pw or ts is None or str(ts.date()) != str(pw.get("date")):
        return None
    hm = ts.strftime("%H:%M")
    for w in pw["windows"]:
        if (w["start"] is None or hm >= w["start"]) and (w["end"] is None or hm < w["end"]):
            return w
    return None


def process_window(ts: dt.datetime | None, cfg) -> str | None:
    w = _window(ts, cfg)
    return w["label"] if w else None


def process_settings(ts: dt.datetime | None, cfg) -> dict:
    """Settings given on the part's process window (e.g. feed_rate_kg_h): every key except start/end/label."""
    w = _window(ts, cfg) or {}
    return {k: v for k, v in w.items() if k not in ("start", "end", "label")}


def choose_depth(preps, cfg) -> tuple[float | None, list]:
    """Window depth for the data set: config number, 'full', or auto (deepest depth visible in every image).
    In auto mode an image showing much less of the part than the others (the part touches the side of the frame)
    would shrink the window for all of them; such images are returned as (prep, reason) to be skipped instead."""
    d = cfg["window"]["depth_mm"]
    if d == "full":
        bad = [p.record.name for p in preps if not p.full_view]
        if bad:
            raise SegmentationError(f"window 'full' needs the whole profile in every image; cut off in {bad}")
        return None, []
    if d == "auto":
        if all(p.full_view for p in preps):
            return None, []
        depths = np.array([p.visible_depth_mm for p in preps])
        floor = cfg["window"]["auto_min_fraction"] * float(np.median(depths))
        short = [(p, f"only {p.visible_depth_mm:.1f} mm of the part is visible (the others show at least "
                     f"{floor:.1f} mm; the part touches the side of the image) - would shrink the window for all")
                 for p in preps if p.visible_depth_mm < floor]
        kept = depths[depths >= floor]
        return float(kept.min() - cfg["window"]["auto_margin_mm"]), short
    return float(d), []


def _aligned_polygons(shape: Shape, al):
    """Outer and cavity polygons in the aligned frame; in the generic mode the cavity is the list of holes."""
    if shape.holes:
        return al.apply(shape.outer), [al.apply(h) for h in shape.holes]
    return al.apply(shape.outer), al.apply(shape.cavity)


def _generic_shapes(preps, skipped) -> list[Shape]:
    """Generic mode: outlines of every part; parts whose number of holes differs from most parts are skipped."""
    from collections import Counter
    shapes = []
    for p in preps:
        try:
            shapes.append(gen.extract_generic(p))
        except SegmentationError as e:
            skipped.append((p.record.path.name, str(e)))
    if not shapes:
        return shapes
    n_holes = Counter(len(s.holes) for s in shapes).most_common(1)[0][0]
    for s in [s for s in shapes if len(s.holes) != n_holes]:
        skipped.append((s.record.path.name, f"{len(s.holes)} hole(s); most parts have {n_holes}"))
        shapes.remove(s)
    return shapes


def run(inputs, cfg: AnalysisConfig, with_cells: bool = False, reference_names=None, progress=print) -> Analysis:
    generic = cfg["kind"] == "generic"
    records = discover(inputs, tuple(cfg["exclude"]))
    skipped, preps = [], []
    for r in records:
        if r.mm_per_px is None:
            skipped.append((r.path.name, f"scale unknown - {r.scale_source}"))
            continue
        try:
            preps.append(gen.prepare_generic(r, cfg) if generic else prepare(r, cfg))
            progress(f"  segmented {r.path.name}" + ("" if generic else
                     f" (visible {preps[-1].visible_depth_mm:.1f} mm from the base)"))
        except SegmentationError as e:
            skipped.append((r.path.name, str(e)))
    if len(preps) < 3:
        raise SegmentationError(f"need at least 3 usable images, have {len(preps)}; skipped: {skipped}")

    sm = cfg["shape_model"]
    if generic:
        depth, shapes = None, _generic_shapes(preps, skipped)
        sizes = (sm["n_outer"],) + (sm["n_cavity"],) * (len(shapes[0].holes) if shapes else 0)
    else:
        depth, short = choose_depth(preps, cfg)
        for p, reason in short:
            skipped.append((p.record.path.name, reason))
            preps.remove(p)
        shapes = []
        for p in preps:
            try:
                shapes.append(extract(p, depth, sm["n_outer"], sm["n_cavity"]))
            except SegmentationError as e:
                skipped.append((p.record.path.name, str(e)))
        sizes = (sm["n_outer"], sm["n_cavity"])
    if len(shapes) < 3:
        raise SegmentationError(f"need at least 3 usable parts, have {len(shapes)}; skipped: {skipped}")
    closed = depth is None
    a = Analysis(cfg, records, skipped, shapes, depth, closed, kind=cfg["kind"])

    # --- shape model
    if generic:
        mean, aligned = gen.generalized_procrustes_generic([[s.outer] + s.holes for s in shapes], sizes,
                                                           sm["orientations"], sm["allow_mirror"])
    else:
        mean, aligned = generalized_procrustes([(s.outer_curve, s.cavity_curve) for s in shapes], closed,
                                               sm["allow_mirror"])
    pts = np.array([al.points for al in aligned])
    model = fit_pca(pts, sm["n_outer"], closed, sm["variance_keep"], sizes)
    a.aligned, a.model = aligned, model
    mp = cfg.spec.measurement
    n_modes = min(model.n_keep, len(model.variance))
    a.modes = [gen.describe_mode_generic(model, i, sm["mode_sigma"], sizes) if generic else
               describe_mode(model, i, sm["mode_sigma"], mp) for i in range(n_modes)]
    a.stability = bootstrap_stability(pts, model, sm["bootstrap"]) if len(shapes) >= 4 else {}

    # --- per-part table: time, dims, spec, scores
    rows = []
    for s, al, sc in zip(shapes, aligned, model.scores):
        o, c = _aligned_polygons(s, al)
        feats = gen.describe_generic(al.points, sizes) if generic else describe(al.points, sm["n_outer"], mp, closed)
        a.shape_features = list(feats)
        row = dict(name=s.name, image=str(s.record.path), timestamp=s.record.timestamp,
                   process_window=process_window(s.record.timestamp, cfg), mirrored=al.mirrored,
                   visible_depth_mm=round(s.prepared.visible_depth_mm, 2), **feats)
        if closed and not generic:
            row["length"] = round(float(o[:, 1].max() - o[:, 1].min()), 3)
        for k, lim in cfg.spec.spec.items():
            if k in row:
                row[f"{k}_ok"] = bool(lim.contains(row[k]))
        row["spec_ok"] = all(row[f"{k}_ok"] for k in cfg.spec.spec if f"{k}_ok" in row)
        for i in range(model.n_keep):
            row[f"PC{i + 1}"] = round(float(sc[i]), 4)
        row.update(process_settings(s.record.timestamp, cfg))
        row.update(s.record.meta)              # parts.csv wins over the window settings
        rows.append(row)
    table = pd.DataFrame(rows)

    # --- anomalies: T^2 in the kept modes (leave-one-out) + residual not explained by them
    mon = cfg["monitoring"]
    ref_mask = np.ones(len(shapes), bool) if reference_names is None else table["name"].isin(reference_names).to_numpy()
    t2 = t2_leave_one_out(model.scores, model.n_keep, mon["t2_alpha"], ref_mask)
    spe_vals = spe_leave_one_out(pts, t2["k"])
    # each part's limit from the other reference parts, so an outlier does not raise its own limit
    spe_lims = np.array([spe_limit(spe_vals[ref_mask & (np.arange(len(pts)) != i)], mon["t2_alpha"])
                         for i in range(len(pts))])
    spe_lim = spe_limit(spe_vals[ref_mask], mon["t2_alpha"])
    table["T2"], table["T2_limit"] = t2["t2"].round(3), t2["limit"].round(3)
    table["shape_abnormal"] = t2["abnormal"]
    table["residual_rms_mm"] = np.sqrt(spe_vals / pts.shape[1]).round(4)
    table["SPE"], table["SPE_limit"] = spe_vals.round(3), spe_lims.round(3)
    table["new_deformation"] = spe_vals > spe_lims
    table["main_mode"] = [f"PC{int(np.argmax(np.abs(c))) + 1}" for c in t2["contributions"]]
    a.t2, a.spe = t2, dict(values=spe_vals, limit=spe_lim)

    # --- wall thickness
    th = cfg["thickness"]
    if generic:
        # along the outer outline: position = mm along the mean outline from its start point
        a.thickness = np.array([gen.thickness_on_outline(al.points, sizes)
                                for al in aligned])
        o = model.mean[: sizes[0]]
        a.thickness_grid = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(o, axis=0).T))])
        with np.errstate(all="ignore"):
            table["thickness_min"] = np.nanmin(a.thickness, axis=1).round(3)
            table["thickness_median"] = np.nanmedian(a.thickness, axis=1).round(3)
            table["thickness_max"] = np.nanmax(a.thickness, axis=1).round(3)
        a.shape_features += ["thickness_min", "thickness_median", "thickness_max"]
        a.key_dims = [k for k in GENERIC_KEY_DIMS if k in table] + ["thickness_min"]
    else:
        profiles = []
        for s, al in zip(shapes, aligned):
            o, c = _aligned_polygons(s, al)
            profiles.append(thickness_profile(o, c, th["raster_px_per_mm"], closed))
        grid = common_grid(profiles, th["grid_step_mm"], th["end_trim_mm"])
        a.thickness_grid = grid
        a.thickness = np.array([on_grid(p, grid, th["end_trim_mm"]) for p in profiles])
        if len(grid):
            i0 = int(np.argmin(np.abs(grid)))
            table["thickness_base_centre"] = a.thickness[:, i0].round(3)

    # --- control charts (time order = table order)
    series = {f"PC{i + 1}": table[f"PC{i + 1}"] for i in range(min(MAX_CHART_MODES, model.n_keep))}
    series.update({k: table[k] for k in a.key_dims if k in table})
    a.charts = {k: dict(values=v.to_numpy(float), **i_chart(v.to_numpy(float), mon["spc_sigma"], mon["run_length"],
                                                             mon["trend_length"]))
                for k, v in series.items()}
    a.table = table

    # --- relaxation (needs the same part imaged at several times since extrusion)
    a.relaxation = _relaxation(table, cfg)

    # --- cells
    if with_cells:
        a.cells = _cells(a)
        if isinstance(a.cells, dict):
            feats = pd.DataFrame(a.cells["features"])
            a.table = table = pd.concat([table, feats], axis=1)

    # --- correlations
    a.correlations = _correlations(a)
    _what_is_missing(a, with_cells)
    return a


def _relaxation(table: pd.DataFrame, cfg) -> dict | str:
    need = {"part_id", "time_since_extrusion_min"}
    if not need <= set(table.columns):
        return ("no part_id / time_since_extrusion_min in parts.csv: every image is treated as a separate, "
                "fresh cut")
    t = table.dropna(subset=list(need)).copy()
    t["time_since_extrusion_min"] = t["time_since_extrusion_min"].astype(float)
    counts = t.groupby("part_id").size()
    parts = counts[counts >= cfg["relaxation"]["min_images_per_part"]].index
    if len(parts) < 2:
        return (f"fewer than 2 parts with >= {cfg['relaxation']['min_images_per_part']} images at different "
                f"times since extrusion")
    out = {}
    for target in [c for c in table.columns if c.startswith("PC")][:MAX_CHART_MODES] + ["inner_width_max",
                                                                                        "base_thickness"]:
        series = {p: (g["time_since_extrusion_min"].to_numpy(), g[target].to_numpy(float))
                  for p, g in t[t["part_id"].isin(parts)].groupby("part_id")}
        try:
            fit = fit_pooled(series)
        except (ValueError, RuntimeError) as e:
            out[target] = dict(error=str(e))
            continue
        fit["settle_min"] = settle_time(fit["tau"], cfg["relaxation"]["settle_fraction"])
        fit["early_prediction"] = early_prediction_check(series)
        fit["series"] = {p: (s[0].tolist(), s[1].tolist()) for p, s in series.items()}
        out[target] = fit
    return out


def _cells(a: Analysis) -> dict | str:
    p = a.cfg["cells"]
    per_part, features, profiles, flat_samples = [], [], [], []
    for s, al in zip(a.shapes, a.aligned):
        img = cv2.imread(str(s.record.path), cv2.IMREAD_COLOR)
        c = cellmod.measure_cells(img, s.prepared, p)
        if a.depth_mm is not None:
            keep = c["y_mm"] >= -a.depth_mm
            c = {k: v[keep] for k, v in c.items()}
        xy = al.apply(np.column_stack([c["x_mm"], c["y_mm"]])) if len(c["x_mm"]) else np.zeros((0, 2))
        c["x_al"], c["y_al"] = xy[:, 0], xy[:, 1]
        rot = np.arctan2(al.rotation[1, 0], al.rotation[0, 0])
        ang = (np.pi - c["angle_rad"]) if al.mirrored else c["angle_rad"]
        c["angle_al"] = ang + rot
        o, cav = _aligned_polygons(s, al)
        per_part.append(c)
        feats = cellmod.cell_features(c, o, cav)
        # surface layer by line intercepts, calibrated so that the core agrees with the cell outlines
        core_d = float(np.mean(c["diameter_mm"])) if len(c["diameter_mm"]) else None
        si = cellmod.surface_intercepts(img, s.prepared, p, a.depth_mm, core_diameter_mm=core_d)
        feats.update(cellmod.surface_features(si))
        features.append(feats)
        profiles.append(si)
        flat_samples.append(cellmod.flattening_samples(img, s.prepared, p, a.depth_mm, si["sensitivity"]))
    if not any(features):
        return "no cells detected"
    pts = np.vstack([al.apply(s.outer) for s, al in zip(a.shapes, a.aligned)])     # the outline holds every hole
    extent = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
    foam_polys = [_aligned_polygons(s, al) for s, al in zip(a.shapes, a.aligned)]
    maps = {key: cellmod.binned_map(per_part, key, p["map_bin_mm"], extent, foam_polys)
            for key in ("diameter_mm", "density", "elongation")}
    flat = cellmod.flattening_grid(flat_samples, a.aligned, p["flat_bin_mm"], extent)
    return dict(features=features, maps=maps, n_cells=[len(c["x_al"]) for c in per_part], per_part=per_part,
                profiles=profiles, flat_grid=flat)


def _correlations(a: Analysis) -> dict | str:
    t = a.table
    dims = ("area_mm2", "outer_width", "outer_height") if a.kind == "generic" else \
        ("inner_width_max", "base_thickness", "outer_width_max")
    targets = t[[c for c in t.columns if c.startswith("PC")][:MAX_CHART_MODES] + [k for k in dims if k in t] + ["T2"]]
    known = set(targets.columns) | set(a.key_dims) | set(a.shape_features) | {"name", "image", "timestamp", "mirrored", "visible_depth_mm",
                                                    "spec_ok", "T2_limit", "shape_abnormal", "residual_rms_mm",
                                                    "SPE", "SPE_limit", "new_deformation", "main_mode", "part_id",
                                                    "notes", "length", "wall_asym", "cavity_width_top",
                                                    "tip_height_diff", "thickness_base_centre", "process_window"}
    feats = {}
    for c in t.columns:
        if c in known or c.endswith("_ok") or c.startswith("PC"):
            continue
        v = pd.to_numeric(t[c], errors="coerce")
        if v.notna().sum() >= 6 and v.nunique() >= 2:
            feats[c] = v
    if t["timestamp"].notna().all():
        ts = pd.to_datetime(t["timestamp"])
        feats["time_of_day_h"] = ts.dt.hour + ts.dt.minute / 60
        if ts.dt.date.nunique() > 1:
            feats["day_index"] = pd.Series(pd.factorize(ts.dt.date)[0], index=t.index)
    if t["process_window"].notna().sum() >= 6 and t["process_window"].nunique() >= 2:
        feats["process_window_no"] = t["process_window"].map(
            {w["label"]: i + 1 for i, w in enumerate(a.cfg["process_windows"]["windows"])})
    if not feats:
        return "no cell or process features to correlate with"
    fdf = pd.DataFrame(feats)
    table = spearman_table(fdf, targets)
    forests = {}
    for target in [c for c in targets.columns if c.startswith("PC")][:2]:
        r = forest_importance(fdf, targets[target])
        if r:
            forests[target] = r
    return dict(table=table, forests=forests, features=list(fdf.columns), n=len(t))


def _what_is_missing(a: Analysis, with_cells: bool) -> None:
    n = len(a.shapes)
    if a.skipped:
        a.not_done.append(f"{len(a.skipped)} image(s) skipped (see the data section): " +
                          "; ".join(f"{n_}: {r}" for n_, r in a.skipped[:5]) + (" ..." if len(a.skipped) > 5 else ""))
    if not a.closed:
        a.not_done.append(f"Tips, mouth, length and tip overlap are outside the field of view: shapes are compared "
                          f"only within {a.depth_mm:.1f} mm of the base.")
        a.collect_next.append("Images with the whole cross-section in view (the wider field of view) - needed for "
                              "length, tip overlap/curl and leg closing at the tips.")
    if isinstance(a.relaxation, str):
        a.not_done.append(f"Time-dependent shape change (Step 5): {a.relaxation}.")
        a.collect_next.append("Re-image the same marked parts at fixed times after extrusion (e.g. 0, 10, 30, 60 min, "
                              "4 h, 24 h, 7 days), with part_id and time_since_extrusion_min in parts.csv.")
    proc_cols = [c for c in a.table.columns if c in ("line_speed_m_min", "melt_temp_c", "batch")]
    if not proc_cols:
        a.not_done.append("No process data (line speed, temperatures, batch); only the configured process-change "
                          "time windows are used.")
        a.collect_next.append("Process settings per image in parts.csv: line speed, melt/die temperature, cooling, "
                              "batch / raw-material lot, and the time of each process change.")
    if not with_cells:
        a.not_done.append("Cell structure not analysed (run with --cells).")
    if n < 30:
        a.not_done.append(f"Only {n} parts: modes, control limits and anomaly limits are uncertain "
                          f"(see the bootstrap stability); treat them as provisional.")
        a.collect_next.append("At least 30 parts spread over several days/batches for stable control limits, "
                              "and NMC's list of approved (good) parts to use as the reference set.")
    spec_fail = int((~a.table["spec_ok"]).sum())
    if spec_fail and a.kind == "u_profile":
        a.not_done.append(f"{spec_fail} of {n} parts are outside the drawing limits as measured here. "
                          "TODO(NMC): confirm where the 24 mm width and 12 mm thickness are measured before "
                          "treating these as rejects.")
    elif spec_fail:
        a.not_done.append(f"{spec_fail} of {n} parts are outside the limits in the config (spec).")
    if a.kind == "generic" and not a.cfg.spec.spec:
        a.not_done.append("No spec limits configured: parts are only compared with each other (add 'spec:' with "
                          "limits on the measured features, e.g. hole_width, to check against a drawing).")


# ---------------- saved model for scoring new parts -----------------

def save_model(a: Analysis, path: str | Path) -> None:
    m = a.model
    k = a.t2["k"]
    np.savez_compressed(
        path, mean=m.mean, components=m.components, variance=m.variance, n_outer=m.n_outer, closed=m.closed,
        n_keep=m.n_keep, ref_scores=m.scores[:, :k], t2_k=k, spe_ref=a.spe["values"],  # leave-one-out SPEs
        sizes=np.array(m.sizes), kind=a.kind,
        depth_mm=np.nan if a.depth_mm is None else a.depth_mm,
        meta=json.dumps(dict(created=dt.datetime.now().isoformat(timespec="seconds"), config=a.cfg.source,
                             parts=a.table["name"].tolist(), modes=[md["name"] for md in a.modes])))


def score_images(model_path: str | Path, inputs, cfg: AnalysisConfig) -> pd.DataFrame:
    """Score new images against a saved model: mode scores, T^2 and residual against the reference parts."""
    z = np.load(model_path, allow_pickle=False)
    closed = bool(z["closed"])
    depth = None if np.isnan(float(z["depth_mm"])) else float(z["depth_mm"])
    kind = str(z["kind"]) if "kind" in z else "u_profile"
    if kind != cfg["kind"]:
        raise ValueError(f"the model was built with kind '{kind}', the config says '{cfg['kind']}'")
    sizes = tuple(int(v) for v in z["sizes"]) if "sizes" in z else ()
    model = ShapeModel(z["mean"], z["components"], z["variance"], np.zeros((0, len(z["variance"]))),
                       int(z["n_outer"]), closed, int(z["n_keep"]), sizes)
    ref, k = z["ref_scores"], int(z["t2_k"])
    mu, inv = ref.mean(axis=0), np.linalg.pinv(np.atleast_2d(np.cov(ref, rowvar=False)))
    alpha = cfg["monitoring"]["t2_alpha"]
    limit, s_lim = t2_limit(len(ref), k, alpha), spe_limit(z["spe_ref"], alpha)
    rows = []
    for r in discover(inputs, tuple(cfg["exclude"])):
        row = dict(name=r.name, timestamp=r.timestamp)
        sm = cfg["shape_model"]
        try:
            if r.mm_per_px is None:
                raise SegmentationError(f"scale unknown - {r.scale_source}")
            if kind == "generic":
                s = gen.extract_generic(gen.prepare_generic(r, cfg))
                al = gen.align_generic([s.outer] + s.holes, model.mean, model.sizes, sm["orientations"],
                                       sm["allow_mirror"])
            else:
                s = extract(prepare(r, cfg), depth, int(z["n_outer"]), len(z["mean"]) - int(z["n_outer"]))
                al = align_to(s.outer_curve, s.cavity_curve, model.mean, closed, sm["allow_mirror"])
        except SegmentationError as e:
            rows.append(dict(row, error=str(e)))
            continue
        b = model.project(al.points, max(k, model.n_keep))
        d = b[:k] - mu
        t2 = float(d @ inv @ d)
        spe = float(((al.points - model.shape(model.project(al.points, k))) ** 2).sum())
        feats = gen.describe_generic(al.points, model.sizes) if kind == "generic" else \
            describe(al.points, model.n_outer, cfg.spec.measurement, closed)
        row.update(feats, mirrored=al.mirrored, T2=round(t2, 3), T2_limit=round(limit, 3), shape_abnormal=t2 > limit,
                   SPE=round(spe, 3), SPE_limit=round(s_lim, 3), new_deformation=spe > s_lim)
        for i, v in enumerate(b[: model.n_keep]):
            row[f"PC{i + 1}"] = round(float(v), 4)
        for key, lim in cfg.spec.spec.items():
            if key in feats:
                row[f"{key}_ok"] = bool(lim.contains(feats[key]))
        rows.append(row)
    return pd.DataFrame(rows)
