"""Figures (matplotlib) and a self-contained HTML report for an Analysis."""
import base64
import datetime as dt
import html
import io
import warnings
from pathlib import Path

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
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
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


def fig_surface(a: Analysis) -> str | None:
    """Cell flattening and size (wall-to-wall distance) against depth below the outer / cavity surface."""
    profs = a.cells.get("profiles") if isinstance(a.cells, dict) else None
    if not profs:
        return None
    x = profs[0]["bin_mid_mm"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for surf, colour, name in (("outer", BLUE, "below the outer surface"), ("cavity", ORANGE, "below the cavity")):
        for ax, key in ((axes[0], "flattening"), (axes[1], None)):
            if key:
                vals = np.array([pr[surf][key] for pr in profs])
            else:
                vals = np.array([pr[surf]["deep_mm"] for pr in profs])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(vals, axis=0)
                lo, hi = np.nanmin(vals, axis=0), np.nanmax(vals, axis=0)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.15, lw=0)
            ax.plot(x, mean, color=colour, marker="o", ms=4, label=name)
            if key is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    wide = np.nanmean(np.array([pr[surf]["wide_mm"] for pr in profs]), axis=0)
                ax.plot(x, wide, color=colour, ls="--", lw=1.5)
    axes[0].axhline(1.0, color=MUTED, lw=1, ls="--")
    axes[0].text(x[-1], 1.0, "round ", va="bottom", ha="right", color=MUTED, fontsize=8)
    axes[0].set_title("flattening (width along the surface / depth into the foam)", loc="left")
    axes[1].set_title("cell depth (solid) and width (dashed), mm wall to wall", loc="left")
    for ax in axes:
        ax.set_xlabel("depth below the surface (mm)")
    axes[0].legend(loc="upper right")
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

FONTS = ("https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@600;700&"
         "family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap")

CSS = """
/* Layout: one reading column; findings first (tiles + two finding cards), evidence sections below,
   figures sit on light 'paper' panels in both themes because the charts are rendered on paper. */
:root{
  --bg:#f2f4f7; --surface:#ffffff; --paper:#fcfcfb; --ink:#11151b; --ink2:#47505d; --muted:#778190;
  --paper-ink:#11151b; --paper-ink2:#47505d;  /* text on the always-light figure paper */
  --line:#dce1e8; --accent:#1c5cab; --accent-soft:#e2ecf8; --crit:#c23434; --crit-soft:#fbe9e9;
  --warn:#8a5a00; --warn-soft:#fdf1d6; --ok:#0b7a0b; --ok-soft:#e3f4e3; --bar:#2a78d6;
  --display:"Barlow Semi Condensed","Arial Narrow",system-ui,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0e1115; --surface:#161a20; --ink:#e7ebf0; --ink2:#b2bbc7; --muted:#8a94a3; --line:#2a313b;
  --accent:#86b6ef; --accent-soft:#18283c; --crit:#ef8585; --crit-soft:#3a1d1f; --warn:#f2c35a;
  --warn-soft:#352a12; --ok:#6fcf6f; --ok-soft:#163018; --bar:#3987e5; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0e1115; --surface:#161a20; --ink:#e7ebf0; --ink2:#b2bbc7; --muted:#8a94a3; --line:#2a313b;
  --accent:#86b6ef; --accent-soft:#18283c; --crit:#ef8585; --crit-soft:#3a1d1f; --warn:#f2c35a;
  --warn-soft:#352a12; --ok:#6fcf6f; --ok-soft:#163018; --bar:#3987e5; color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 var(--body)}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:10;background:var(--surface)}
.wrap{max-width:1120px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px))}
header.hero{background:var(--surface);border-bottom:1px solid var(--line)}
header.hero .wrap{padding-block:32px 24px;display:grid;gap:14px}
.eyebrow{font:500 12px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
h1{font:700 clamp(30px,5vw,46px)/1.05 var(--display);margin:0;letter-spacing:.005em;text-wrap:balance}
h1 .sub{display:block;font:500 17px/1.4 var(--body);color:var(--ink2);margin-top:8px;letter-spacing:0}
.specs{display:flex;flex-wrap:wrap;gap:8px}
.spec{font:500 13px/1 var(--mono);border:1px solid var(--line);border-radius:4px;padding:7px 10px;color:var(--ink2);
  background:var(--bg)}
.spec b{color:var(--ink);font-weight:500}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--ink2);font-size:13px}
.meta span{white-space:nowrap}
.badge{display:inline-flex;align-items:center;gap:6px;font:600 12px/1 var(--body);text-transform:uppercase;
  letter-spacing:.06em;color:var(--warn);background:var(--warn-soft);border-radius:999px;padding:6px 10px;
  justify-self:start}
nav.toc{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:color-mix(in srgb,var(--bg) 92%,transparent);
  backdrop-filter:blur(6px);border-bottom:1px solid var(--line)}
nav.toc .wrap{display:flex;gap:4px;overflow-x:auto;padding-block:8px;scrollbar-width:none}
nav.toc a{flex:none;color:var(--ink2);text-decoration:none;font-size:13px;padding:5px 10px;border-radius:6px}
nav.toc a:hover,nav.toc a:focus-visible{background:var(--accent-soft);color:var(--accent);outline:none}
main .wrap{padding-block:28px 72px;display:grid;gap:44px}
section{display:grid;gap:16px;min-width:0;scroll-margin-top:60px}
h2{font:700 26px/1.15 var(--display);margin:0;text-wrap:balance}
h3{font:600 15px/1.3 var(--body);margin:0}
p{margin:0} .lead{color:var(--ink2);max-width:72ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;display:grid;gap:4px;min-width:0}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.tile .v{font:600 30px/1.1 var(--display);font-variant-numeric:tabular-nums}
.tile .v small{font:500 15px var(--body);color:var(--ink2);margin-left:4px}
.tile .d{font-size:13px;color:var(--ink2)}
@media (max-width:560px){.tile:last-child:nth-child(odd){grid-column:1/-1}}
.tile.crit{border-left:4px solid var(--crit)} .tile.warn{border-left:4px solid var(--warn)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:18px 20px;display:grid;
  gap:14px;align-content:start;min-width:0}
.mode{display:grid;grid-template-columns:64px 1fr;gap:4px 12px;align-items:center}
.mode .n{font:500 12px var(--mono);color:var(--muted)}
.mode .bar{height:8px;background:var(--accent-soft);border-radius:4px;overflow:hidden}
.mode .bar i{display:block;height:100%;background:var(--bar);border-radius:4px}
.mode .txt{grid-column:2;font-size:14px}
.mode .txt b{font-variant-numeric:tabular-nums}
.flag{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;align-items:baseline;padding-bottom:12px;
  border-bottom:1px solid var(--line)}
.flag:last-child{border-bottom:0;padding-bottom:0}
.flag .t{font:500 15px var(--mono)}
.flag .why{grid-column:2;font-size:14px;color:var(--ink2)}
.pill{display:inline-flex;align-items:center;gap:4px;font:600 11px/1 var(--body);letter-spacing:.04em;
  text-transform:uppercase;border-radius:999px;padding:4px 8px;white-space:nowrap}
.pill.crit{color:var(--crit);background:var(--crit-soft)} .pill.warn{color:var(--warn);background:var(--warn-soft)}
.pill.ok{color:var(--ok);background:var(--ok-soft)} .pill.info{color:var(--accent);background:var(--accent-soft)}
.callout{background:var(--warn-soft);border-radius:8px;padding:16px 20px;display:grid;gap:6px}
.callout h3{color:var(--warn)}
.callout p{max-width:80ch}
ul.plain{margin:0;padding-left:18px;display:grid;gap:6px}
figure{margin:0;background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:12px;display:grid;
  gap:10px;min-width:0}
figure img{display:block;width:100%;height:auto}
figure.narrow{max-width:560px}
figcaption{font-size:13px;color:var(--paper-ink2)}
figcaption b{color:var(--paper-ink)}
.figs2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;align-items:start}
.tw{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:7px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
tbody tr:last-child td{border-bottom:0}
th{font-weight:600;font-size:12px;color:var(--ink2);background:var(--bg);position:sticky;top:0}
td{font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums}
td.txt{font-family:var(--body);font-size:13px;white-space:normal;min-width:220px;text-align:left}
td:first-child,th:first-child{text-align:left}
tr.hl td{background:var(--crit-soft)}
details{background:var(--surface);border:1px solid var(--line);border-radius:8px}
details>summary{cursor:pointer;padding:12px 16px;font-weight:600;font-size:14px;color:var(--ink2);list-style:none}
details>summary::before{content:"+";display:inline-block;width:18px;color:var(--accent);font-family:var(--mono)}
details[open]>summary::before{content:"\\2212"}
details>summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:8px}
details>.tw{border:0;border-top:1px solid var(--line);border-radius:0 0 8px 8px}
.note{font-size:13px;color:var(--muted);max-width:80ch}
.cols2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:12px}
footer .wrap{padding-block:20px 32px}
a{color:var(--accent)}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
html{scroll-behavior:smooth}
"""

LABELS = {
    "name": "Part", "timestamp": "Time", "process_window": "Process window", "mirrored": "Mirrored",
    "visible_depth_mm": "Visible depth (mm)", "T2": "T²", "T2_limit": "T² limit",
    "shape_abnormal": "Shape", "main_mode": "Main mode", "SPE": "Residual", "SPE_limit": "Residual limit",
    "new_deformation": "New deformation", "spec_ok": "Spec", "inner_width_max": "Inner width (mm)",
    "base_thickness": "Base (mm)", "outer_width_max": "Outer width (mm)", "wall_left": "Left wall (mm)",
    "wall_right": "Right wall (mm)", "cavity_width_mid": "Cavity mid (mm)", "lean_left_deg": "Left lean (°)",
    "lean_right_deg": "Right lean (°)", "length": "Length (mm)", "feature": "Feature", "target": "Target",
    "rho": "ρ", "p": "p", "q": "q", "n": "n",
    "outer_layer_flattening": "Outer: flattening", "outer_layer_deep_mm": "Outer: cell depth (mm)",
    "outer_layer_wide_mm": "Outer: cell width (mm)", "outer_flattened_depth_mm": "Outer: flat zone (mm)",
    "cavity_layer_flattening": "Cavity: flattening", "cavity_layer_deep_mm": "Cavity: cell depth (mm)",
    "cavity_layer_wide_mm": "Cavity: cell width (mm)", "cavity_flattened_depth_mm": "Cavity: flat zone (mm)",
    "outer_core_flattening": "Core flattening (outer side)", "cavity_core_flattening": "Core flattening (cavity side)",
    "wall_sensitivity": "Wall sensitivity",
}


def _label(c: str) -> str:
    return LABELS.get(c, c.replace("_", " "))


def _pill(kind: str, text: str) -> str:
    glyph = {"crit": "▲", "warn": "●", "ok": "✓", "info": "•"}[kind]
    return f'<span class="pill {kind}" aria-label="{html.escape(text)}">{glyph} {html.escape(text)}</span>'


def _fmt(col: str, v):
    """Cell HTML: flags become labelled pills, numbers are compact, times short."""
    if col in ("shape_abnormal",):
        return _pill("crit", "Abnormal") if v else _pill("ok", "Normal")
    if col == "new_deformation":
        return _pill("crit", "Yes") if v else "–"
    if col == "spec_ok" or col.endswith("_ok"):
        return _pill("ok", "In spec") if v else _pill("warn", "Out")
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else (f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}")
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%d %H:%M")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return html.escape(str(v))


def _table(df, cols=None, max_rows=500, highlight=None, text_cols=()) -> str:
    df = df if cols is None else df[[c for c in cols if c in df.columns]]
    head = "".join(f"<th>{html.escape(_label(str(c)))}</th>" for c in df.columns)
    body = []
    for i, (_, r) in enumerate(df.head(max_rows).iterrows()):
        cls = ' class="hl"' if highlight is not None and bool(highlight.iloc[i]) else ""
        cells = "".join(f'<td{" class=txt" if c in text_cols else ""}>{_fmt(c, v)}</td>' for c, v in r.items())
        body.append(f"<tr{cls}>{cells}</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


class _Figures:
    def __init__(self):
        self.n = 0

    def __call__(self, b64: str, title: str, caption: str, narrow: bool = False) -> str:
        self.n += 1
        return (f'<figure{" class=narrow" if narrow else ""}><img alt="{html.escape(title)}" '
                f'src="data:image/png;base64,{b64}"><figcaption><b>Figure {self.n}. {html.escape(title)}.</b> '
                f'{caption}</figcaption></figure>')


def _spec_tag(name: str, lim) -> str:
    lo, hi = lim.nominal - lim.min, lim.max - lim.nominal
    tol = f"±{lo:g}" if abs(lo - hi) < 1e-9 else f"−{lo:g}/+{hi:g}"
    return f'<span class="spec">{html.escape(_label(name).split(" (")[0])} <b>{lim.nominal:g} {tol}</b> mm</span>'


def _stability_pill(a: Analysis, i: int) -> str:
    if not a.stability:
        return ""
    c = a.stability["cos_median"][i]
    return _pill("ok", "stable") if c >= 0.9 else _pill("info", "fairly stable") if c >= 0.7 else \
        _pill("warn", "not yet stable")


def _flag_reasons(a: Analysis, r) -> list[str]:
    why = []
    if r["shape_abnormal"]:
        md = int(r["main_mode"][2:]) - 1
        name = a.modes[md]["name"].split(";")[0] if md < len(a.modes) else ""
        why.append(f"unusual amount of mode {md + 1}" + (f" ({name})" if name else ""))
    if r["new_deformation"]:
        why.append("a deformation the other parts don't have")
    for k, lim in a.cfg.spec.spec.items():
        if f"{k}_ok" in r and not r[f"{k}_ok"]:
            why.append(f"{_label(k).split(' (')[0].lower()} {r[k]:.2f} mm (spec {lim.min:g}–{lim.max:g})")
    return why


def _head(a: Analysis) -> str:
    return (f"<title>{html.escape(a.cfg['product'].replace('NMC ', '').split(' 0-')[0])} Shape Variation</title>\n"
            f'<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'<link rel="preconnect" href="https://fonts.googleapis.com">'
            f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            f'<link rel="stylesheet" href="{FONTS}">\n<style>{CSS}</style>\n')


def render_html(a: Analysis, standalone: bool = True) -> str:
    """The report. standalone=False leaves out <html>/<head>/<body> (for hosts that wrap the page themselves)."""
    t, m = a.table, a.model
    n = len(t)
    fig = _Figures()
    out = []
    add = out.append
    days = sorted({ts.date().isoformat() for ts in t["timestamp"] if ts is not None and ts == ts})
    flagged = t[t["shape_abnormal"] | t["new_deformation"]]
    spec_out = int((~t["spec_ok"]).sum())
    kept = 100 * m.explained[: m.n_keep].sum()

    # ---- header
    add('<header class="hero"><div class="wrap">')
    add('<div class="eyebrow">Zelloscope · shape-variation report</div>')
    add(f'<h1>{html.escape(a.cfg["product"])}<span class="sub">Cut-end cross-sections of blue PE-LD foam '
        f'U-profiles: how the whole shape varies from part to part.</span></h1>')
    add('<div class="specs">' + "".join(_spec_tag(k, v) for k, v in a.cfg.spec.spec.items()) +
        f'<span class="spec">Contour tolerance <b>{a.cfg.spec.contour_tolerance_mm:g}</b> mm</span></div>')
    add('<div class="meta">' +
        f'<span>{n} parts · {", ".join(days) if days else "no timestamps"}</span>' +
        (f'<span>Window {a.depth_mm:.1f} mm from the base</span>' if not a.closed else '<span>Full outline</span>') +
        f'<span>Generated {dt.datetime.now():%Y-%m-%d %H:%M}</span></div>')
    if n < 30:
        add(f'<span class="badge">● Provisional · only {n} parts</span>')
    add('</div></header>')

    sections = [("summary", "Summary"), ("shape", "Shape model"), ("thickness", "Wall thickness"),
                ("drift", "Drift"), ("anomalies", "Anomalies"), ("cells", "Cells"), ("links", "Links"),
                ("extrusion", "After extrusion"), ("next", "Next steps"), ("data", "Data")]
    add('<nav class="toc" aria-label="Sections"><div class="wrap">' +
        "".join(f'<a href="#{i}">{html.escape(l)}</a>' for i, l in sections) + '</div></nav>')
    add('<main><div class="wrap">')

    # ---- summary
    add('<section id="summary"><h2>Summary</h2><div class="tiles">')
    add(f'<div class="tile"><span class="k">Parts analysed</span><span class="v">{n}</span>'
        f'<span class="d">{len(a.skipped)} image(s) skipped</span></div>')
    add(f'<div class="tile"><span class="k">Shape modes</span><span class="v">{m.n_keep}</span>'
        f'<span class="d">cover {kept:.0f} % of the variation</span></div>')
    add(f'<div class="tile{" crit" if len(flagged) else ""}"><span class="k">Abnormal shapes</span>'
        f'<span class="v">{len(flagged)}<small>of {n}</small></span><span class="d">vs. the other parts</span></div>')
    add(f'<div class="tile{" warn" if spec_out else ""}"><span class="k">Outside drawing limits</span>'
        f'<span class="v">{spec_out}<small>of {n}</small></span><span class="d">as measured here</span></div>')
    add('<div class="tile"><span class="k">Compared over</span><span class="v">'
        + (f'{a.depth_mm:.1f}<small>mm</small></span><span class="d">from the base; tips out of frame</span>'
           if not a.closed else 'full<small>outline</small></span><span class="d">tips included</span>') + '</div>')
    add('</div><div class="grid2">')

    add('<div class="card"><h3>How the shape varies</h3>')
    for md in a.modes[:4]:
        i = md["mode"] - 1
        add(f'<div class="mode"><span class="n">Mode {md["mode"]}</span><span class="bar"><i style="width:'
            f'{100 * md["explained"]:.1f}%"></i></span><span class="txt"><b>{100 * md["explained"]:.0f} %</b> · '
            f'{html.escape(md["name"])} {_stability_pill(a, i)}</span></div>')
    add('<p class="note">Mode = one independent way the shape changes. "Stable" = the same mode comes back when the '
        'parts are resampled.</p></div>')

    add('<div class="card"><h3>Parts to look at</h3>')
    if len(flagged):
        for _, r in flagged.iterrows():
            ts = r["timestamp"]
            when = ts.strftime("%H:%M") if isinstance(ts, dt.datetime) else ""
            pw = r.get("process_window")
            add(f'<div class="flag"><span class="t">{when}</span><span>{_pill("crit", "Abnormal")} '
                f'<span class="note">{html.escape(r["name"])}{(" · " + html.escape(str(pw))) if isinstance(pw, str) else ""}'
                f'</span></span><span class="why">{html.escape("; ".join(_flag_reasons(a, r)))}</span></div>')
    else:
        add('<p>No part has an abnormal shape compared with the others.</p>')
    add('</div></div>')

    if spec_out:
        rows = []
        for k in ("inner_width_max", "base_thickness"):
            if k in t:
                lim = a.cfg.spec.spec[k]
                rows.append(f"{_label(k).split(' (')[0]}: median {t[k].median():.2f} mm, range "
                            f"{t[k].min():.2f}–{t[k].max():.2f} mm (drawing {lim.min:g}–{lim.max:g} mm)")
        add('<div class="callout"><h3>Open question for NMC</h3><p>' + html.escape(
            f"{spec_out} of {n} parts are outside the drawing limits as measured here. "
            "Before treating them as rejects, NMC needs to confirm where the 24 mm width and the 12 mm "
            "thickness are measured.") + '</p><p class="note" style="color:inherit">' +
            "<br>".join(html.escape(x) for x in rows) + '</p></div>')

    signals = [(k, sorted({v['rule'] for v in c['violations']}), len({v['index'] for v in c['violations']}))
               for k, c in a.charts.items() if c["violations"]]
    if signals:
        add('<div class="card"><h3>Drift signals</h3><ul class="plain">' + "".join(
            f'<li><b>{html.escape(_label(k))}</b>: {html.escape(", ".join(r))} ({c} point{"s" if c > 1 else ""})</li>'
            for k, r, c in signals) + '</ul></div>')
    add('</section>')

    # ---- shape model
    add('<section id="shape"><h2>Shape model</h2>')
    add('<p class="lead">Every part is aligned to the mean shape by rotation and position, and mirrored where the '
        'other cut end was imaged. There is no scaling, so size differences stay in the model. Each mode is one '
        'independent way the shape varies.</p>')
    add('<div class="figs2">')
    add(fig(fig_overlay(a), "All parts on the mean shape", "Grey: each aligned part. Black: the mean. "
            "Base at the bottom, legs up."))
    add(fig(fig_variance(a), "Share of the variation per mode", f"Bars: each mode. Line: running total. "
            f"Modes up to the dashed line ({100 * a.cfg['shape_model']['variance_keep']:.0f} %) are kept."))
    add('</div>')
    add(fig(fig_modes(a), "The modes", "Grey: mean shape. Blue and orange: the mean moved 2 standard deviations "
            "each way along the mode. Displacements are magnified so the change is visible; the table gives the "
            "real sizes."))
    rows = []
    for md in a.modes:
        i = md["mode"] - 1
        r = {"Mode": md["mode"], "Share": f"{100 * md['explained']:.1f} %", "What changes (+2 sd)": md["name"],
             "RMS per sd (mm)": md["rms_mm_per_sd"]}
        if a.stability:
            r["Stability"] = a.stability["cos_median"][i]
            r["Share, 90 % range"] = (f"{100 * a.stability['explained_p05'][i]:.0f}–"
                                      f"{100 * a.stability['explained_p95'][i]:.0f} %")
        rows.append(r)
    add(_table(pd.DataFrame(rows), text_cols=("What changes (+2 sd)",)))
    add('<details><summary>Feature changes per mode, from −2 sd to +2 sd</summary>' +
        _table(pd.DataFrame([dict(mode=md["mode"], **md["deltas"]) for md in a.modes])) + '</details>')
    add('<p class="note">Stability is the median |cos| between a mode and its match when the parts are resampled '
        '(bootstrap). Near 1: the same mode every time. Below about 0.7: the data does not pin the mode down yet.</p>')
    add('</section>')

    # ---- thickness
    add('<section id="thickness"><h2>Wall thickness</h2>')
    if a.thickness is not None and len(a.thickness_grid):
        mean = np.nanmean(a.thickness, axis=0)
        sd = np.nanstd(a.thickness, axis=0)
        i0 = int(np.argmin(np.abs(a.thickness_grid)))
        j = int(np.nanargmax(sd))
        add(f'<p class="lead">Measured along the middle of the foam (the medial axis) from the left leg, through the '
            f'base, to the right leg. Mean {mean[i0]:.2f} mm at the base centre and {np.nanmin(mean):.2f}–'
            f'{np.nanmax(mean):.2f} mm along the U. Parts differ most {abs(a.thickness_grid[j]):.0f} mm from the '
            f'base centre on the {"right" if a.thickness_grid[j] > 0 else "left"} (sd {sd[j]:.2f} mm).</p>')
        add(fig(fig_thickness(a), "Thickness along the U", "Top: mean and range of all parts. Bottom: each part's "
                "difference from the mean, parts in time order; red is thicker, blue thinner."))
    else:
        add('<p>Not available.</p>')
    add('</section>')

    # ---- drift
    add('<section id="drift"><h2>Drift</h2>')
    add('<p class="lead">Control charts of the main modes and dimensions, parts in time order. Limits come from the '
        'parts themselves. A signal is a point beyond the limits, 8 in a row on one side of the centre line, or '
        '6 steadily rising or falling. Shaded bands mark the process-change windows (proc. N).</p>')
    add(fig(fig_control_charts(a), "Control charts", "Solid: centre line. Dashed: ±3 sigma limits. "
            "Red ring: a signal. PC<i>n</i> is shape mode <i>n</i>."))
    add('</section>')

    # ---- anomalies
    add('<section id="anomalies"><h2>Anomalies</h2>')
    add(f'<p class="lead">T² measures how far a part is from all the other parts in the first {a.t2["k"]} modes; '
        f'above its limit (α = {a.cfg["monitoring"]["t2_alpha"]}) the shape is abnormal. The residual catches a '
        'deformation that a model built without the part cannot explain. The spec check is independent of both.</p>')
    add(fig(fig_anomaly(a), "Distance from the other parts", "Each part's T² divided by its limit; "
            "above 1 the shape is abnormal.", narrow=True))
    ts = t.sort_values("T2", ascending=False)
    add(_table(ts, ["name", "T2", "T2_limit", "shape_abnormal", "main_mode", "SPE", "SPE_limit", "new_deformation",
                    "inner_width_max", "base_thickness", "spec_ok"] + (["length"] if a.closed else []),
               highlight=(ts["shape_abnormal"] | ts["new_deformation"])))
    add('</section>')

    # ---- cells
    add('<section id="cells"><h2>Cell structure</h2>')
    if isinstance(a.cells, dict):
        add(f'<p class="lead">{sum(a.cells["n_cells"]):,} cells in {len(a.cells["n_cells"])} parts, including the '
            'layer touching the surfaces. Patches much larger than a cell (several cells merged, mostly where the '
            'walls are hidden at the surface) are left out. The maps average all parts in the aligned frame.</p>')
        add(fig(fig_cell_maps(a), "Cell maps", "Mean cell diameter, cells per mm² and elongation "
                "(long axis / short axis) in 2 mm bins, with the mean outline."))
        surf_fig = fig_surface(a)
        layer_cols = [c for c in ("outer_layer_flattening", "outer_layer_deep_mm", "outer_layer_wide_mm",
                                  "outer_flattened_depth_mm", "cavity_layer_flattening", "cavity_layer_deep_mm",
                                  "cavity_layer_wide_mm", "cavity_flattened_depth_mm", "outer_core_flattening",
                                  "cavity_core_flattening", "wall_sensitivity") if c in t]
        if surf_fig:
            mean_of = lambda col: t[col].mean() if col in t else float("nan")
            add(f'<h3>Cells at the surfaces</h3><p class="lead">Measured by counting cell walls along lines '
                f'straight in from each surface and along lines parallel to it (line-intercept method, as in ASTM '
                f'D3576). In the first 0.5 mm the cells are {mean_of("outer_layer_flattening"):.2f}\u00d7 wider than '
                f'deep below the outer surface and {mean_of("cavity_layer_flattening"):.2f}\u00d7 below the cavity; '
                f'in the core about {mean_of("outer_core_flattening"):.2f}\u00d7. The flattened zone reaches about '
                f'{mean_of("outer_flattened_depth_mm"):.1f} mm (outer) and {mean_of("cavity_flattened_depth_mm"):.1f} '
                f'mm (cavity) into the foam.</p>'
                '<p class="note">The cavity value for the first 0.5 mm is less certain: in these images the cut '
                'face is blurred along the cavity edge, so some walls there are missed and the cells read wider '
                'than they are. Check with <code>python -m shape_variation overlay</code>; the outer surface is '
                'sharp.</p>')
            add(fig(surf_fig, "Cells against depth below the surface", "Points: mean over parts in 0.5 mm "
                    "steps (wider steps deeper in); bands: range of parts. Wall-detection sensitivity is set per "
                    "image so that, in the core, the wall-to-wall distance matches the cell outlines (\u03c0/4 \u00d7 "
                    "the mean cell diameter)."))
            add(_table(t, ["name"] + layer_cols))
        cell_cols = [c for c in t.columns if c.startswith("cell_")]
        add('<details><summary>Other cell features per part</summary>' + _table(t, ["name"] + cell_cols) +
            '</details>')
    else:
        add(f'<p class="lead">{html.escape(str(a.cells))}. Run <code>analyze --cells</code> to include it.</p>')
    add('</section>')

    # ---- links
    add('<section id="links"><h2>Links to cells and process</h2>')
    if isinstance(a.correlations, dict):
        tab = a.correlations["table"]
        sig = int((tab["q"] < 0.05).sum()) if len(tab) else 0
        add(f'<p class="lead">Spearman rank correlations between cell or process features and the shape '
            f'(n = {a.correlations["n"]}). q corrects for testing many pairs; {sig} link(s) have q &lt; 0.05. '
            'With this little data, treat every link as a lead to check on new parts.</p>')
        add(_table(tab.head(15), highlight=tab.head(15)["q"] < 0.05))
        for target, f in a.correlations["forests"].items():
            imp = ", ".join(f"{k} ({v:.3f})" for k, v in f["importance"][:4])
            verdict = "no predictive value" if f["loo_r2"] <= 0 else "some predictive value"
            add(f'<p class="note">Random forest for {html.escape(target)}: leave-one-out R² = {f["loo_r2"]} '
                f'({verdict}). Most used features: {html.escape(imp)}.</p>')
    else:
        add(f'<p class="lead">{html.escape(str(a.correlations))}.</p>')
    add('</section>')

    # ---- extrusion
    add('<section id="extrusion"><h2>Shape change after extrusion</h2>')
    if isinstance(a.relaxation, dict):
        rel = fig_relaxation(a)
        if rel:
            add(fig(rel, "Relaxation fits", "Points: repeated images of the same parts. Lines: the fitted "
                    "exponential. Dashed: time to 95 % settled."))
        rows = [{"Measure": k, "τ (min)": round(v["tau"], 1), "±": round(v["tau_se"], 1),
                 "Settled after (min)": round(v["settle_min"], 0), "Change": round(v["amplitude"], 3),
                 "R²": round(v["r2"], 3), "Early-image error": round(v["early_prediction"]["rmse"], 3),
                 "Error without model": round(v["early_prediction"]["rmse_without_model"], 3)}
                for k, v in a.relaxation.items() if "tau" in v]
        add(_table(pd.DataFrame(rows)))
    else:
        add(f'<p class="lead">Not possible with this data: {html.escape(a.relaxation)}. The model is ready; it needs '
            'the same marked parts imaged several times after extrusion.</p>')
    add('</section>')

    # ---- next steps
    add('<section id="next"><h2>What could not be done, and what to collect next</h2><div class="cols2">')
    add('<div class="card"><h3>Not possible yet</h3><ul class="plain">' +
        "".join(f"<li>{html.escape(s)}</li>" for s in a.not_done) + '</ul></div>')
    add('<div class="card"><h3>Collect next</h3><ul class="plain">' +
        "".join(f"<li>{html.escape(s)}</li>" for s in a.collect_next) + '</ul></div>')
    add('</div></section>')

    # ---- data
    add('<section id="data"><h2>Data</h2>')
    used = t[["name", "timestamp", "process_window", "mirrored", "visible_depth_mm"]].copy()
    used["Scale (mm/px)"] = [s.record.mm_per_px for s in a.shapes]
    used["Scale from"] = [Path(s.record.scale_source).name if s.record.scale_source != "png-metadata"
                          else "PNG metadata" for s in a.shapes]
    add(_table(used))
    if a.skipped:
        add('<div class="callout"><h3>Skipped images</h3><ul class="plain">' + "".join(
            f"<li>{html.escape(nm)}: {html.escape(r)}</li>" for nm, r in a.skipped) + '</ul></div>')
    cols = ["name", "timestamp", "process_window"] + [k for k in KEY_DIMS if k in t] + \
           [c for c in t.columns if c.startswith("PC")] + ["T2", "spec_ok"]
    add('<details><summary>All measurements per part</summary>' + _table(t, cols) + '</details>')
    add('</section>')

    add('</div></main>')
    add(f'<footer><div class="wrap">Zelloscope shape_variation · config {html.escape(Path(a.cfg.source).name)} '
        f'· spec {html.escape(Path(a.cfg.spec.source).name)}</div></footer>')
    body = "\n".join(out)
    if not standalone:
        return _head(a) + body
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1,viewport-fit=cover'>"
            + _head(a) + "</head><body>" + body + "</body></html>")
