"""Figures (matplotlib) and a self-contained HTML report for an Analysis."""
import base64
import datetime as dt
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .pipeline import KEY_DIMS, Analysis  # noqa: E402

# Reference data-viz palette (light): categorical slots, status, surfaces, ramps.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ef8f8e", "#b52d2c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
    "lines.linewidth": 2, "legend.frameon": False,
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _draw_shape(ax, pts, n_outer, closed, **kw):
    o, c = pts[:n_outer], pts[n_outer:]
    for i, curve in enumerate((o, c)):
        if closed:
            curve = np.vstack([curve, curve[:1]])
        ax.plot(curve[:, 0], curve[:, 1], **(kw if i == 0 else {k: v for k, v in kw.items() if k != "label"}))


def _shape_axes(ax):
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("mm")
    ax.grid(False)


# ---------------- figures -----------------

def fig_overlay(a: Analysis) -> str:
    m = a.model
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    for al in a.aligned:
        _draw_shape(ax, al.points, m.n_outer, m.closed, color=MUTED, lw=0.8, alpha=0.6)
    _draw_shape(ax, m.mean, m.n_outer, m.closed, color=INK, lw=2, label="mean shape")
    ax.plot([], [], color=MUTED, lw=0.8, label="individual parts")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    _shape_axes(ax)
    return _png(fig)


def _exaggeration(a: Analysis) -> float:
    """Visual magnification of mode displacements so a +-2 sd change of ~0.3 mm is visible on a 50 mm part."""
    m, n_sigma = a.model, a.cfg["shape_model"]["mode_sigma"]
    biggest = max(np.abs(m.mode_shape(i, n_sigma) - m.mean).max() for i in range(len(a.modes)))
    return float(max(1, np.floor(4.0 / max(biggest, 1e-6))))


def fig_modes(a: Analysis) -> str:
    m, n_sigma = a.model, a.cfg["shape_model"]["mode_sigma"]
    k = len(a.modes)
    ex = _exaggeration(a)
    show = lambda i, sgn: m.mean + ex * (m.mode_shape(i, sgn * n_sigma) - m.mean)
    cols = min(3, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.9 * rows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= k:
            ax.axis("off")
            continue
        _draw_shape(ax, m.mean, m.n_outer, m.closed, color=MUTED, lw=1.2, label="mean")
        _draw_shape(ax, show(i, -1), m.n_outer, m.closed, color=BLUE, lw=2, label=f"mean - {n_sigma:g} sd")
        _draw_shape(ax, show(i, 1), m.n_outer, m.closed, color=ORANGE, lw=2, label=f"mean + {n_sigma:g} sd")
        ax.set_title(f"Mode {i + 1} - {100 * m.explained[i]:.0f} % of variation", loc="left")
        _shape_axes(ax)
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, fontsize=8)
    if ex > 1:
        fig.suptitle(f"Displacements drawn {ex:g}x larger than real so the change is visible "
                     f"(the numbers in the table are real).", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    return _png(fig)


def fig_variance(a: Analysis) -> str:
    ex = a.model.explained
    n = len(ex)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    x = np.arange(1, n + 1)
    ax.bar(x, 100 * ex, color=BLUE, width=0.6, label="this mode")
    ax.step(x, 100 * np.cumsum(ex), where="mid", color=ORANGE, label="cumulative")
    ax.axhline(100 * a.cfg["shape_model"]["variance_keep"], color=MUTED, lw=1, ls="--")
    ax.axvline(a.model.n_keep + 0.5, color=MUTED, lw=1)
    ax.text(a.model.n_keep + 0.6, 50, f"kept: {a.model.n_keep} modes", color=INK2)
    ax.set_xticks(x)
    ax.set_xlabel("mode")
    ax.set_ylabel("% of shape variance")
    ax.set_ylim(0, 105)
    ax.legend(loc="center right")
    return _png(fig)


def fig_thickness(a: Analysis) -> str:
    g, t = a.thickness_grid, a.thickness
    names = list(a.table["name"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2.6 + 0.22 * len(names) + 2.6),
                                   gridspec_kw=dict(height_ratios=[2.6, 0.22 * len(names) + 1]))
    mean = np.nanmean(t, axis=0)
    ax1.fill_between(g, np.nanmin(t, axis=0), np.nanmax(t, axis=0), color="#b7d3f6", lw=0, label="range of parts")
    ax1.plot(g, mean, color=BLUE, label="mean")
    ax1.axvline(0, color=MUTED, lw=1)
    ax1.set_ylabel("wall thickness (mm)")
    ax1.set_xlabel("position along the U from the base centre (mm); left leg < 0 < right leg")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    dev = t - mean
    lim = max(0.2, float(np.nanmax(np.abs(dev))))
    im = ax2.imshow(dev, aspect="auto", cmap=DIV, vmin=-lim, vmax=lim, interpolation="nearest",
                    extent=(g[0], g[-1], len(names) - 0.5, -0.5))
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([n[-8:] for n in names], fontsize=7)
    ax2.set_xlabel("position (mm)")
    ax2.set_title("difference from the mean thickness, parts in time order (top = first)", loc="left")
    ax2.grid(False)
    cb = fig.colorbar(im, ax=ax2, pad=0.01)
    cb.set_label("mm (red thicker, blue thinner)")
    fig.tight_layout()
    return _png(fig)


def _shade_windows(ax, labels):
    """Grey bands for consecutive parts in the same process window."""
    if labels is None or all(l is None for l in labels):
        return
    start, band = 0, 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] is not None:
                if band % 2 == 0:
                    ax.axvspan(start - 0.5, i - 0.5, color="#f0efec", lw=0)
                band += 1
                ax.text((start + i - 1) / 2, 1.0, str(labels[start]).replace("Process change ", "proc. "),
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=MUTED)
            start = i


def fig_control_charts(a: Analysis) -> str:
    keys = list(a.charts)
    cols = 2
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.3 * rows), squeeze=False)
    labels = [None if (isinstance(v, float) or v is None) else v for v in a.table["process_window"].tolist()]
    names = [n[-8:] for n in a.table["name"]]
    x = np.arange(len(names))
    for ax, key in zip(axes.ravel(), keys):
        c = a.charts[key]
        _shade_windows(ax, labels)
        ax.plot(x, c["values"], color=BLUE, marker="o", ms=4, lw=1.5)
        ax.axhline(c["cl"], color=MUTED, lw=1)
        for lim in (c["ucl"], c["lcl"]):
            ax.axhline(lim, color=MUTED, lw=1, ls="--")
        bad = sorted({v["index"] for v in c["violations"]})
        if bad:
            ax.plot(x[bad], np.asarray(c["values"])[bad], ls="none", marker="o", ms=8, mfc="none", mec=CRITICAL,
                    mew=2)
        unit = "deg" if key.endswith("_deg") else ("" if key.startswith("PC") or key == "T2" else "mm")
        ax.set_title(f"{key} {('(' + unit + ')') if unit else ''}", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=6)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle("Individuals charts: centre line (solid), +-3 sigma limits (dashed); red ring = SPC signal",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    return _png(fig)


def fig_anomaly(a: Analysis) -> str:
    t = a.table.sort_values("T2", ascending=True)
    ratio = (t["T2"] / t["T2_limit"]).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(t) + 1.2))
    colors = [CRITICAL if r > 1 else BLUE for r in ratio]
    ax.barh(range(len(t)), ratio, color=colors, height=0.6)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.text(1.02, len(t) - 0.6, "limit", color=INK2, fontsize=8)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{n[-8:]}{'  ABNORMAL' if r > 1 else ''}" for n, r in zip(t["name"], ratio)], fontsize=7)
    ax.set_xlabel("T² / limit (shape distance from the other parts; > 1 = abnormal)")
    return _png(fig)


def fig_cell_maps(a: Analysis) -> str:
    maps = a.cells["maps"]
    titles = {"diameter_mm": "mean cell diameter (mm)", "density": "cells per mm²",
              "elongation": "mean cell elongation (major/minor)"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (key, (grid, ext)) in zip(axes, maps.items()):
        im = ax.imshow(grid, cmap=SEQ, extent=ext, interpolation="nearest")
        _draw_shape(ax, a.model.mean, a.model.n_outer, a.model.closed, color=INK, lw=1)
        ax.set_aspect("equal")
        ax.set_title(titles[key], loc="left")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _png(fig)


def fig_relaxation(a: Analysis) -> str | None:
    fits = {k: v for k, v in a.relaxation.items() if "tau" in v}
    if not fits:
        return None
    fig, axes = plt.subplots(1, len(fits), figsize=(4 * len(fits), 3), squeeze=False)
    for ax, (key, f) in zip(axes[0], fits.items()):
        tt = np.linspace(0, max(max(s[0]) for s in f["series"].values()), 200)
        for p, (t, y) in f["series"].items():
            ax.plot(t, y, "o", color=BLUE, ms=4)
            ax.plot(tt, f["final"][p] + f["amplitude"] * np.exp(-tt / f["tau"]), color=MUTED, lw=1)
        ax.axvline(f["settle_min"], color=ORANGE, ls="--", lw=1)
        ax.set_title(f"{key}: tau = {f['tau']:.0f} min", loc="left")
        ax.set_xlabel("minutes since extrusion")
    fig.tight_layout()
    return _png(fig)


# ---------------- HTML -----------------

def _img(b64: str, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{b64}">'


def _table(df, cols=None, max_rows=200) -> str:
    df = df if cols is None else df[[c for c in cols if c in df.columns]]
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.head(max_rows).iterrows():
        cells = []
        for v in r:
            if isinstance(v, (bool, np.bool_)):
                v = "yes" if v else "no"
            elif isinstance(v, float):
                v = "" if np.isnan(v) else f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
            elif isinstance(v, (dt.datetime,)):
                v = v.strftime("%Y-%m-%d %H:%M")
            cells.append(f"<td>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def summary_points(a: Analysis) -> list[str]:
    t, m = a.table, a.model
    n = len(t)
    pts = []
    days = sorted({ts.date().isoformat() for ts in t["timestamp"] if ts is not None and ts == ts})
    view = (f"within {a.depth_mm:.1f} mm of the base (the tips are outside the frame)" if not a.closed
            else "over the whole outline")
    pts.append(f"{n} parts analysed{(' from ' + ', '.join(days)) if days else ''}; "
               f"{len(a.skipped)} image(s) skipped. Shapes are compared {view}.")
    kept = 100 * m.explained[: m.n_keep].sum()
    pts.append(f"{m.n_keep} modes describe {kept:.0f} % of the shape variation. The largest:")
    for md in a.modes[:3]:
        stab = ""
        if a.stability:
            c = a.stability["cos_median"][md["mode"] - 1]
            stab = " (reproducible)" if c >= 0.9 else " (not yet reproducible with this many parts)" if c < 0.7 \
                else " (fairly reproducible)"
        pts.append(f"&nbsp;&nbsp;Mode {md['mode']} ({100 * md['explained']:.0f} %){stab}: {html.escape(md['name'])}.")
    for k in ("inner_width_max", "base_thickness"):
        if k in t:
            lim = a.cfg.spec.spec.get(k)
            out = int((~t[f"{k}_ok"]).sum()) if f"{k}_ok" in t else 0
            pts.append(f"{k.replace('_', ' ')}: median {t[k].median():.2f} mm (range {t[k].min():.2f}-{t[k].max():.2f})"
                       + (f"; {out} of {n} outside {lim.min}-{lim.max} mm as measured here" if lim else "") + ".")
    if a.thickness is not None and len(a.thickness_grid):
        mean = np.nanmean(a.thickness, axis=0)
        sd = np.nanstd(a.thickness, axis=0)
        i0 = int(np.argmin(np.abs(a.thickness_grid)))
        j = int(np.nanargmax(sd))
        pts.append(f"Wall thickness: {mean[i0]:.2f} mm at the base centre, {np.nanmin(mean):.2f}-{np.nanmax(mean):.2f} mm "
                   f"along the U; parts differ most at {a.thickness_grid[j]:+.0f} mm from the base centre "
                   f"(sd {sd[j]:.2f} mm).")
    signals = [(k, sorted({v['rule'] for v in c['violations']}), len({v['index'] for v in c['violations']}))
               for k, c in a.charts.items() if c["violations"]]
    if signals:
        pts.append("Drift / SPC signals: " + "; ".join(f"{k} ({', '.join(r)}; {c} point(s))" for k, r, c in signals)
                   + ".")
    else:
        pts.append("No SPC signals on the charted modes and dimensions.")
    ab = t[t["shape_abnormal"] | t["new_deformation"]]
    if len(ab):
        items = []
        for _, r in ab.iterrows():
            why = []
            if r["shape_abnormal"]:
                md = int(r["main_mode"][2:]) - 1
                why.append(f"unusual amount of mode {md + 1}" + (f" ({a.modes[md]['name'].split(';')[0]})"
                                                                 if md < len(a.modes) else ""))
            if r["new_deformation"]:
                why.append("a deformation not seen in the other parts")
            items.append(f"{r['name']}: {', '.join(why)}")
        pts.append("Abnormal shapes: " + html.escape("; ".join(items)) + ".")
    else:
        pts.append("No part has an abnormal shape relative to the others.")
    if isinstance(a.correlations, dict) and len(a.correlations["table"]):
        top = a.correlations["table"].iloc[0]
        sig = (a.correlations["table"]["q"] < 0.05).sum()
        pts.append(f"Strongest link: {top['feature']} vs {top['target']} (Spearman rho {top['rho']:+.2f}, n {top['n']}); "
                   f"{sig} link(s) survive the multiple-testing correction (q < 0.05). Exploratory only.")
    return pts


CSS = """
:root{--surface:#fcfcfb;--bg:#f4f3f0;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--crit:#d03b3b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--bg:#111110;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#9a998f;--line:#383835}}
:root[data-theme="dark"]{--surface:#1a1a19;--bg:#111110;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#383835}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:36px 0 8px;border-top:1px solid var(--line);padding-top:20px}
.sub{color:var(--ink2);margin:0 0 20px}
section.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 20px}
img{max-width:100%;height:auto;border-radius:6px;background:#fcfcfb}
.tw{overflow-x:auto} table{border-collapse:collapse;font-size:12px;margin:8px 0}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600} td:first-child,th:first-child{text-align:left}
ul.sum li{margin:4px 0} .warn{color:var(--crit)} .note{color:var(--ink2);font-size:13px}
details{margin:8px 0} summary{cursor:pointer;color:var(--ink2)}
"""


def render_html(a: Analysis) -> str:
    t = a.table
    parts = []
    add = parts.append
    add(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>Shape Variation Report</title><style>{CSS}</style>"
        f"</head><body><main>")
    add(f"<h1>Shape variation - {html.escape(a.cfg['product'])}</h1>")
    add(f"<p class='sub'>Generated {dt.datetime.now():%Y-%m-%d %H:%M} from {len(a.records)} image(s); "
        f"config {html.escape(a.cfg.source)}</p>")

    add("<h2>Summary</h2><section class='card'><ul class='sum'>")
    for p in summary_points(a):
        add(f"<li>{p}</li>")
    add("</ul><p class='note'>With few parts every number here is provisional: modes, limits and correlations "
        "will move as more parts are added.</p></section>")

    add("<h2>Data</h2>")
    used = t[["name", "timestamp", "process_window", "mirrored", "visible_depth_mm"]].copy()
    used["scale (mm/px)"] = [s.record.mm_per_px for s in a.shapes]
    used["scale from"] = [s.record.scale_source for s in a.shapes]
    add(_table(used))
    if a.skipped:
        add("<p class='warn'>Skipped images:</p><ul>" +
            "".join(f"<li>{html.escape(n)}: {html.escape(r)}</li>" for n, r in a.skipped) + "</ul>")

    add("<h2>Shape model</h2>")
    add("<p>All parts aligned to their mean (rotation and position, mirrored where the other cut end was imaged; "
        "no scaling, so size differences stay in the model). Each mode is one independent way the shape varies; "
        "the plots show the mean shape and the shape at -2 and +2 standard deviations along the mode.</p>")
    add(_img(fig_overlay(a), "all aligned parts with the mean shape"))
    add(_img(fig_modes(a), "mode plots"))
    add(_img(fig_variance(a), "explained variance per mode"))
    rows = []
    for md in a.modes:
        r = dict(mode=md["mode"], explained=f"{100 * md['explained']:.1f} %", name=md["name"],
                 rms_mm_per_sd=md["rms_mm_per_sd"])
        if a.stability:
            i = md["mode"] - 1
            r["stability (median |cos|)"] = a.stability["cos_median"][i]
            r["explained 90% interval"] = (f"{100 * a.stability['explained_p05'][i]:.0f}-"
                                           f"{100 * a.stability['explained_p95'][i]:.0f} %")
        rows.append(r)
    add(_table(pd.DataFrame(rows)))
    add("<details><summary>Feature changes per mode (from -2 sd to +2 sd)</summary>" +
        _table(pd.DataFrame([dict(mode=md["mode"], **md["deltas"]) for md in a.modes])) + "</details>")
    add("<p class='note'>Stability: |cos| between each mode and the matching mode of PCA on resampled parts "
        "(bootstrap). Near 1 = the same mode every time; below ~0.7 = the mode is not yet pinned down by the data.</p>")

    add("<h2>Wall thickness</h2>")
    if a.thickness is not None and len(a.thickness_grid):
        add("<p>Thickness = 2 x the distance from the medial axis (skeleton) of the foam to its nearest edge, along "
            "the U from the left end through the base to the right end.</p>")
        add(_img(fig_thickness(a), "wall thickness profile and heat map"))
    else:
        add("<p>Not available.</p>")

    add("<h2>Drift (control charts)</h2>")
    add("<p>Parts in time order. Limits from the parts themselves (individuals chart, sigma from the moving range). "
        "Signals: a point beyond the limits, 8 in a row on one side, or 6 steadily rising/falling. Grey bands and "
        "labels (proc. N) mark the configured process-change windows. PCn = shape mode n.</p>")
    add(_img(fig_control_charts(a), "control charts"))

    add("<h2>Anomalies</h2>")
    add(f"<p>T² = distance of a part from the others in the first {a.t2['k']} modes, each part compared with "
        f"all <em>other</em> parts (Hotelling limit at alpha = {a.cfg['monitoring']['t2_alpha']}). "
        "Residual (SPE) = shape change the model built without that part cannot explain - a new kind of "
        "deformation. The hard spec check is listed alongside and is independent of both.</p>")
    add(_img(fig_anomaly(a), "anomaly ranking"))
    add(_table(t.sort_values("T2", ascending=False),
               ["name", "T2", "T2_limit", "shape_abnormal", "main_mode", "SPE", "SPE_limit", "new_deformation",
                "spec_ok", "inner_width_max", "base_thickness"] + (["length"] if a.closed else [])))

    add("<h2>Cell structure</h2>")
    if isinstance(a.cells, dict):
        add(f"<p>{sum(a.cells['n_cells'])} cells in {len(a.cells['n_cells'])} parts (cells touching the foam edge are "
            "left out, so density reads slightly low in the outermost bins). Maps average all parts in the "
            "aligned frame.</p>")
        add(_img(fig_cell_maps(a), "cell maps"))
        cell_cols = [c for c in t.columns if c.startswith("cell_")]
        add(_table(t, ["name"] + cell_cols))
    else:
        add(f"<p>{html.escape(str(a.cells))}</p>")

    add("<h2>Links to cells / process</h2>")
    if isinstance(a.correlations, dict):
        add(f"<p>Spearman rank correlations (n = {a.correlations['n']}), sorted by strength. q = p-value corrected for "
            "testing many pairs (Benjamini-Hochberg); only q &lt; 0.05 is worth following up, and with this little "
            "data even that needs confirming on new parts.</p>")
        add(_table(a.correlations["table"].head(25)))
        for target, f in a.correlations["forests"].items():
            imp = ", ".join(f"{k} {v:.3f}" for k, v in f["importance"][:5])
            add(f"<p class='note'>Random forest for {target}: leave-one-out R² = {f['loo_r2']} "
                f"(&le; 0 means no predictive value); top features by permutation importance: {html.escape(imp)}.</p>")
    else:
        add(f"<p>{html.escape(str(a.correlations))}</p>")

    add("<h2>Shape change after extrusion</h2>")
    if isinstance(a.relaxation, dict):
        rel = fig_relaxation(a)
        if rel:
            add(_img(rel, "relaxation fits"))
        rows = [dict(target=k, tau_min=round(v["tau"], 1), tau_se=round(v["tau_se"], 1),
                     settles_after_min=round(v["settle_min"], 0), amplitude=round(v["amplitude"], 3),
                     r2=round(v["r2"], 3), early_pred_rmse=round(v["early_prediction"]["rmse"], 3),
                     no_model_rmse=round(v["early_prediction"]["rmse_without_model"], 3))
                for k, v in a.relaxation.items() if "tau" in v]
        add(_table(pd.DataFrame(rows)))
    else:
        add(f"<p>Not possible with this data: {html.escape(a.relaxation)}.</p>")

    add("<h2>What could not be done, and what to collect next</h2><section class='card'><p><strong>Not done / "
        "limitations</strong></p><ul>")
    for s in a.not_done:
        add(f"<li>{html.escape(s)}</li>")
    add("</ul><p><strong>Collect next</strong></p><ul>")
    for s in a.collect_next:
        add(f"<li>{html.escape(s)}</li>")
    add("</ul></section>")

    add("<h2>All parts</h2>")
    cols = ["name", "timestamp", "process_window"] + [k for k in KEY_DIMS if k in t] + \
           [c for c in t.columns if c.startswith("PC")] + ["T2", "spec_ok"]
    add(_table(t, cols))
    add("</main></body></html>")
    return "\n".join(parts)
