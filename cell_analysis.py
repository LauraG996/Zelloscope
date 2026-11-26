#!/usr/bin/env python3
"""Analyse foam-cell morphology from a micrograph and its cell segmentation.

Inputs
------
image         The original colour micrograph (e.g. blue foam strip on a dark
              background, next to the yellow holder).
segmentation  A same-sized image where cell walls are drawn as dark lines on a
              white background (the "line drawing" output of a segmentation tool).

What it does
------------
1. Turns the line drawing into labelled regions (every closed white area
   bounded by dark lines becomes one region).
2. Decides which regions are foam cells by their mean colour in the original
   image (blue hue by default). Background and holder regions are dropped.
3. Flags "edge" cells, which touch the foam boundary or the image border and are
   therefore probably cut off, and leaves them out of the statistics by default.
4. Measures every cell: area, equivalent diameter, perimeter, Feret diameters,
   fitted-ellipse axes, aspect ratio, orientation, circularity, solidity, number
   of neighbours, and position.
5. Writes:
     <out>/cell_data.xlsx    per-cell table, summary sheet and binned profiles
     <out>/cell_data.csv     the per-cell table as CSV
     <out>/overview.png      overlays plus histograms (a one-page dashboard)
     <out>/overlay_*.png     full-resolution overlays coloured by a metric
   and prints a short summary to the terminal.

Example
-------
    python cell_analysis.py foam.jpg foam_seg.png --um-per-px 2.5 -o results/

Folder mode: pass two folders instead of two files. Each image is matched to
the segmentation with the same file name (extension and suffixes such as
"_seg"/"_mask" ignored). Every pair gets its own subfolder, plus a combined
all_images.xlsx (one summary row per image and every cell), all_cells.csv and
trends.png (metrics per image, in file-name order):

    python cell_analysis.py videocompressed/images videocompressed/Segmentation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from skimage.measure import label, regionprops  # noqa: E402
from skimage.segmentation import expand_labels, find_boundaries  # noqa: E402

# Pixels darker than this in the segmentation image are treated as cell wall.
# The wall lines are anti-aliased, so a high threshold keeps them closed.
DEFAULT_WALL_THRESHOLD = 230
# Regions smaller than this (px) are fragments or line artefacts, not cells.
DEFAULT_MIN_AREA_PX = 20
# Foam-region colour gate in OpenCV HSV (H in 0..179). Blue is about 100-130.
DEFAULT_HUE_RANGE = (95, 135)
DEFAULT_MIN_SATURATION = 80
# How far (px) labels are grown into the walls to find adjacent cells.
NEIGHBOUR_GROW_PX = 3


# ---------------------------------------------------------------- loading
def _read_on_white(path: Path) -> np.ndarray | None:
    """Read any image as 8-bit BGR. Transparent pixels (PNG alpha) become white,
    so a line drawing saved with a transparent background still reads as dark
    lines on white. 16-bit images are scaled down to 8-bit."""
    # np.fromfile + imdecode also works with non-ASCII Windows paths.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[..., 3:4].astype(float) / 255
        return (img[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return img


def load_images(image_path: Path, seg_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = _read_on_white(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    seg_bgr = _read_on_white(seg_path)
    if seg_bgr is None:
        raise ValueError(f"Could not read segmentation: {seg_path}")
    seg = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)
    if np.median(seg) < 128:
        # Mostly dark: assume light lines on a black background and invert.
        print(f"note: {seg_path.name} looks like light lines on dark; inverting it")
        seg = 255 - seg
    if seg.shape != image.shape[:2]:
        print(f"note: resizing segmentation {seg.shape[::-1]} -> {image.shape[1::-1]}")
        seg = cv2.resize(seg, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    return image, seg


def label_regions(seg: np.ndarray, wall_threshold: int) -> np.ndarray:
    """Label every white area enclosed by dark wall lines (4-connectivity so
    regions cannot leak through diagonal gaps in 1-px lines)."""
    return label(seg >= wall_threshold, connectivity=1)


# ---------------------------------------------------------------- classification
def classify_foam_regions(
    labels: np.ndarray,
    image: np.ndarray,
    hue_range: tuple[int, int],
    min_saturation: int,
    min_area: int,
    max_area: int | None,
) -> np.ndarray:
    """Return a boolean array indexed by label id: True where the region is a foam cell."""
    n = labels.max()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n + 1).astype(float)
    safe = np.maximum(counts, 1)
    mean_h = np.bincount(flat, weights=hsv[..., 0].ravel(), minlength=n + 1) / safe
    mean_s = np.bincount(flat, weights=hsv[..., 1].ravel(), minlength=n + 1) / safe

    is_foam = (
        (mean_h >= hue_range[0])
        & (mean_h <= hue_range[1])
        & (mean_s >= min_saturation)
        & (counts >= min_area)
    )
    if max_area is not None:
        is_foam &= counts <= max_area
    is_foam[0] = False  # label 0 = wall pixels
    return is_foam


# ---------------------------------------------------------------- adjacency
def neighbour_info(labels: np.ndarray, is_foam: np.ndarray) -> tuple[dict[int, set[int]], np.ndarray]:
    """Find adjacent regions across walls.

    Returns (neighbours of every foam cell among foam cells, edge flag per label).
    A cell is an edge cell if it borders a non-foam region or the image border.
    """
    grown = expand_labels(labels, NEIGHBOUR_GROW_PX)
    pairs = []
    for a, b in ((grown[:, :-1], grown[:, 1:]), (grown[:-1, :], grown[1:, :])):
        diff = (a != b) & (a > 0) & (b > 0)
        pairs.append(np.stack([a[diff], b[diff]], axis=1))
    pairs = np.unique(np.concatenate(pairs), axis=0)

    neighbours: dict[int, set[int]] = {}
    is_edge = np.zeros(labels.max() + 1, dtype=bool)
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            if not is_foam[x]:
                continue
            if is_foam[y]:
                neighbours.setdefault(int(x), set()).add(int(y))
            else:
                is_edge[x] = True

    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    is_edge[border] = True
    # Expanded labels that reach the image border also count.
    border_g = np.unique(np.concatenate([grown[0], grown[-1], grown[:, 0], grown[:, -1]]))
    is_edge[border_g] = True
    return neighbours, is_edge


# ---------------------------------------------------------------- measurement
def measure_cells(
    labels: np.ndarray,
    image: np.ndarray,
    is_foam: np.ndarray,
    is_edge: np.ndarray,
    neighbours: dict[int, set[int]],
    scale: float | None,
) -> pd.DataFrame:
    """One row per foam cell. Lengths are in px, plus µm when a scale is given."""
    foam_labels = np.where(is_foam, np.arange(is_foam.size), 0)[labels]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows = []
    for p in regionprops(foam_labels, intensity_image=gray):
        rr, cc = p.coords[:, 0], p.coords[:, 1]
        major = p.axis_major_length
        minor = p.axis_minor_length
        perim = p.perimeter
        # skimage orientation: angle between row axis and major axis. Convert
        # to degrees from the horizontal image axis, in [0, 180).
        orient = (90.0 - np.degrees(p.orientation)) % 180.0
        rows.append(
            {
                "cell_id": p.label,
                "centroid_x_px": p.centroid[1],
                "centroid_y_px": p.centroid[0],
                "area_px": p.area,
                "equiv_diameter_px": p.equivalent_diameter_area,
                "perimeter_px": perim,
                "major_axis_px": major,
                "minor_axis_px": minor,
                "feret_max_px": p.feret_diameter_max,
                "bbox_width_px": p.bbox[3] - p.bbox[1],
                "bbox_height_px": p.bbox[2] - p.bbox[0],
                "aspect_ratio": major / minor if minor > 0 else np.nan,
                "orientation_deg": orient,
                "circularity": 4 * np.pi * p.area / perim**2 if perim > 0 else np.nan,
                "solidity": p.solidity,
                "eccentricity": p.eccentricity,
                "n_neighbours": len(neighbours.get(p.label, ())),
                "is_edge_cell": bool(is_edge[p.label]),
                "mean_gray": p.intensity_mean,
                "mean_r": rgb[rr, cc, 0].mean(),
                "mean_g": rgb[rr, cc, 1].mean(),
                "mean_b": rgb[rr, cc, 2].mean(),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["circularity"] = df["circularity"].clip(upper=1.0)

    if scale:
        length_cols = [c for c in df.columns if c.endswith("_px") and not c.startswith("area")]
        for c in length_cols:
            df[c.replace("_px", "_um")] = df[c] * scale
        df["area_um2"] = df["area_px"] * scale**2
    return df


# ---------------------------------------------------------------- summary
def unit_cols(scale: float | None) -> tuple[str, str, str]:
    """(area column, diameter column, length unit label)"""
    if scale:
        return "area_um2", "equiv_diameter_um", "µm"
    return "area_px", "equiv_diameter_px", "px"


def summarise(df: pd.DataFrame, labels: np.ndarray, is_foam: np.ndarray, scale: float | None) -> pd.DataFrame:
    inner = df[~df["is_edge_cell"]]
    area_col, diam_col, unit = unit_cols(scale)
    foam_mask = is_foam[labels]
    # Foam footprint = cells plus the walls between them (closing fills walls).
    footprint = cv2.morphologyEx(
        foam_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    ).astype(bool)
    wall_fraction = 1 - foam_mask.sum() / max(footprint.sum(), 1)
    footprint_area = footprint.sum() * (scale**2 if scale else 1)

    def stats(name, col, u=""):
        s = inner[col].dropna()
        return [
            (f"{name} mean{u}", s.mean()),
            (f"{name} median{u}", s.median()),
            (f"{name} std{u}", s.std()),
            (f"{name} P10{u}", s.quantile(0.1)),
            (f"{name} P90{u}", s.quantile(0.9)),
        ]

    area_u = f" [{unit}²]"
    rows = [
        ("Regions in segmentation", int(labels.max())),
        ("Foam cells (total)", len(df)),
        ("Edge cells (excluded from stats)", int(df["is_edge_cell"].sum())),
        ("Interior cells (used for stats)", len(inner)),
        (f"Foam footprint area{area_u}", footprint_area),
        (f"Cell density [cells per mm²]" if scale else "Cell density [cells per 1e6 px²]",
         len(df) / footprint_area * 1e6),
        ("Wall (strut) area fraction", wall_fraction),
        *stats("Area", area_col, area_u),
        *stats("Equivalent diameter", diam_col, f" [{unit}]"),
        *stats("Aspect ratio", "aspect_ratio"),
        *stats("Circularity", "circularity"),
        *stats("Solidity", "solidity"),
        *stats("Neighbours", "n_neighbours"),
        ("Dominant orientation [deg from horizontal]", dominant_orientation(inner)),
        ("Orientation order (0 = random, 1 = all aligned)", orientation_order(inner)),
        ("Area polydispersity (std/mean)", inner[area_col].std() / inner[area_col].mean()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _axial_vector(df: pd.DataFrame) -> complex:
    """Mean of exp(2iθ) weighted by elongation; θ is axial (θ ≡ θ+180)."""
    if df.empty:
        return 0j
    theta = np.radians(df["orientation_deg"].to_numpy())
    w = (df["aspect_ratio"].to_numpy() - 1).clip(min=0)
    if w.sum() == 0:
        return 0j
    return np.sum(w * np.exp(2j * theta)) / w.sum()


def dominant_orientation(df: pd.DataFrame) -> float:
    return float((np.degrees(np.angle(_axial_vector(df))) / 2) % 180)


def orientation_order(df: pd.DataFrame) -> float:
    return float(abs(_axial_vector(df)))


def spatial_profiles(df: pd.DataFrame, scale: float | None, bins: int = 12) -> dict[str, pd.DataFrame]:
    """Binned mean of key metrics along x and along y (gradients across the strip)."""
    inner = df[~df["is_edge_cell"]]
    area_col, diam_col, _ = unit_cols(scale)
    out = {}
    for axis in ("x", "y"):
        pos = inner[f"centroid_{axis}_px"]
        edges = np.linspace(pos.min(), pos.max(), bins + 1)
        grp = inner.groupby(pd.cut(pos, edges, include_lowest=True), observed=False)
        prof = grp.agg(
            n_cells=("cell_id", "size"),
            mean_area=(area_col, "mean"),
            mean_diameter=(diam_col, "mean"),
            mean_aspect_ratio=("aspect_ratio", "mean"),
            mean_circularity=("circularity", "mean"),
        ).reset_index(drop=True)
        prof.insert(0, f"{axis}_center_px", (edges[:-1] + edges[1:]) / 2)
        out[axis] = prof
    return out


# ---------------------------------------------------------------- figures
def colour_overlay(
    image: np.ndarray, labels: np.ndarray, df: pd.DataFrame, metric: str, cmap: str,
    vmin: float | None = None, vmax: float | None = None,
) -> np.ndarray:
    """RGB image with each cell filled by its metric value, edge cells grey."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float) / 255
    out = rgb * 0.35
    values = np.full(labels.max() + 1, np.nan)
    values[df["cell_id"].to_numpy()] = df[metric].to_numpy()
    vals = values[labels]
    lo = np.nanpercentile(df[metric], 2) if vmin is None else vmin
    hi = np.nanpercentile(df[metric], 98) if vmax is None else vmax
    norm = np.clip((vals - lo) / max(hi - lo, 1e-9), 0, 1)
    coloured = plt.get_cmap(cmap)(np.nan_to_num(norm))[..., :3]
    has = ~np.isnan(vals)
    out[has] = coloured[has] * 0.85 + rgb[has] * 0.15

    edge_ids = df.loc[df["is_edge_cell"], "cell_id"].to_numpy()
    edge_mask = np.isin(labels, edge_ids)
    out[edge_mask] = 0.55
    out[find_boundaries(labels, mode="inner") & (labels > 0)] *= 0.5
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def orientation_overlay(image: np.ndarray, labels: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Hue = orientation (cyclic), brightness = elongation."""
    vals_o = np.full(labels.max() + 1, np.nan)
    vals_a = np.full(labels.max() + 1, np.nan)
    vals_o[df["cell_id"]] = df["orientation_deg"]
    vals_a[df["cell_id"]] = df["aspect_ratio"]
    o, a = vals_o[labels], vals_a[labels]
    has = ~np.isnan(o)
    hsv = np.zeros(labels.shape + (3,), np.uint8)
    hsv[..., 0] = np.nan_to_num(o).astype(np.uint8)  # 0..179 matches OpenCV hue
    hsv[..., 1] = 230
    hsv[..., 2] = (np.clip((np.nan_to_num(a, nan=1) - 1) / 1.5, 0, 1) * 200 + 55).astype(np.uint8)
    col = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    base = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) * 0.35).astype(np.uint8)
    base[has] = col[has]
    return base


def save_overlays(image, labels, df, scale, out_dir: Path) -> dict[str, np.ndarray]:
    area_col, diam_col, _ = unit_cols(scale)
    overlays = {
        "diameter": colour_overlay(image, labels, df, diam_col, "viridis"),
        "aspect_ratio": colour_overlay(image, labels, df, "aspect_ratio", "magma"),
        "circularity": colour_overlay(image, labels, df, "circularity", "cividis"),
        "neighbours": colour_overlay(image, labels, df, "n_neighbours", "coolwarm", 3, 9),
        "orientation": orientation_overlay(image, labels, df),
    }
    for name, img in overlays.items():
        cv2.imwrite(str(out_dir / f"overlay_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return overlays


def crop_to_foam(img: np.ndarray, df: pd.DataFrame, pad: int = 40) -> np.ndarray:
    x0 = max(int(df["centroid_x_px"].min()) - pad, 0)
    x1 = int(df["centroid_x_px"].max()) + pad
    y0 = max(int(df["centroid_y_px"].min()) - pad, 0)
    y1 = int(df["centroid_y_px"].max()) + pad
    return img[y0:y1, x0:x1]


def overview_figure(image, labels, df, overlays, profiles, summary, scale, out_path: Path, title: str):
    inner = df[~df["is_edge_cell"]]
    area_col, diam_col, unit = unit_cols(scale)

    fig = plt.figure(figsize=(18, 15), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    panels = [
        ("Original", crop_to_foam(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), df)),
        (f"Equivalent diameter [{unit}]", crop_to_foam(overlays["diameter"], df)),
        ("Aspect ratio (major/minor)", crop_to_foam(overlays["aspect_ratio"], df)),
        ("Orientation (hue) × elongation (brightness)", crop_to_foam(overlays["orientation"], df)),
    ]
    for i, (t, img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_title(t, fontsize=10)
        ax.axis("off")

    def hist(ax, col, xlabel, bins=40, discrete=False):
        data = inner[col].dropna()
        if discrete:
            vc = data.value_counts().sort_index()
            ax.bar(vc.index, vc.values, color="#3b6fb6")
        else:
            ax.hist(data, bins=bins, color="#3b6fb6", edgecolor="white", linewidth=0.4)
        ax.axvline(data.median(), color="#d1495b", lw=1.5, label=f"median {data.median():.3g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cells")
        ax.legend(fontsize=8, frameon=False)

    hist(fig.add_subplot(gs[1, 0]), diam_col, f"Equivalent diameter [{unit}]")
    hist(fig.add_subplot(gs[1, 1]), area_col, f"Area [{unit}²]")
    hist(fig.add_subplot(gs[1, 2]), "aspect_ratio", "Aspect ratio")
    hist(fig.add_subplot(gs[1, 3]), "circularity", "Circularity (1 = circle)")
    hist(fig.add_subplot(gs[2, 0]), "n_neighbours", "Number of neighbours", discrete=True)
    hist(fig.add_subplot(gs[2, 1]), "solidity", "Solidity (area / convex area)")

    # Orientation rose (axial: mirror θ and θ+180)
    ax = fig.add_subplot(gs[2, 2], projection="polar")
    theta = np.radians(np.concatenate([inner["orientation_deg"], inner["orientation_deg"] + 180]))
    weights = np.concatenate([inner["aspect_ratio"] - 1] * 2).clip(min=0)
    counts, edges = np.histogram(theta, bins=36, range=(0, 2 * np.pi), weights=weights)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#3b6fb6", alpha=0.8)
    ax.set_yticklabels([])
    ax.set_title(f"Orientation rose (elongation-weighted)\n"
                 f"dominant {dominant_orientation(inner):.0f}°, order {orientation_order(inner):.2f}",
                 fontsize=10)

    ax = fig.add_subplot(gs[2, 3])
    ax.scatter(inner[diam_col], inner["aspect_ratio"], s=3, alpha=0.35, color="#3b6fb6")
    ax.set_xlabel(f"Equivalent diameter [{unit}]")
    ax.set_ylabel("Aspect ratio")
    ax.set_title("Size vs elongation", fontsize=10)

    for i, axis in enumerate(("x", "y")):
        prof = profiles[axis]
        ax = fig.add_subplot(gs[3, i])
        ax.plot(prof[f"{axis}_center_px"], prof["mean_diameter"], "o-", color="#3b6fb6", label="diameter")
        ax.set_xlabel(f"{axis} position [px]")
        ax.set_ylabel(f"mean diameter [{unit}]", color="#3b6fb6")
        ax2 = ax.twinx()
        ax2.plot(prof[f"{axis}_center_px"], prof["mean_aspect_ratio"], "s--", color="#d1495b")
        ax2.set_ylabel("mean aspect ratio", color="#d1495b")
        ax.set_title(f"Profile along {axis}", fontsize=10)

    ax = fig.add_subplot(gs[3, 2:])
    ax.axis("off")
    show = summary.copy()
    show["value"] = show["value"].map(lambda v: f"{v:,.4g}" if isinstance(v, float) else f"{v:,}")
    hidden = ("std", "P10", "P90", "Solidity", "Regions in")
    show = show[~show["metric"].str.contains("|".join(hidden))]
    tbl = ax.table(cellText=show.values, colLabels=["Metric", "Value"], loc="center",
                   cellLoc="left", colWidths=[0.7, 0.3])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.15)

    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- excel
def write_excel(path: Path, df: pd.DataFrame, summary: pd.DataFrame, profiles: dict, params: dict):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        df.round(4).to_excel(xw, sheet_name="cells", index=False)
        profiles["x"].round(4).to_excel(xw, sheet_name="profile_x", index=False)
        profiles["y"].round(4).to_excel(xw, sheet_name="profile_y", index=False)
        pd.DataFrame(list(params.items()), columns=["parameter", "value"]).to_excel(
            xw, sheet_name="parameters", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 55)


# ---------------------------------------------------------------- per image
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# Common suffixes/prefixes segmentation tools add to the original file name.
SEG_AFFIXES = ("_seg", "-seg", "_segmentation", "-segmentation", "_mask", "-mask",
               "_masks", "_labels", "_label", "_outlines", "_outline", "_cp_outlines",
               "_lines", "_cells", "seg_", "mask_")


def analyse_one(image_path: Path, seg_path: Path, out: Path, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full analysis on one image/segmentation pair and write its outputs to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    image, seg = load_images(image_path, seg_path)
    labels = label_regions(seg, args.wall_threshold)

    if args.all_regions:
        counts = np.bincount(labels.ravel())
        max_area = args.max_area or int(0.01 * labels.size)
        is_foam = (counts >= args.min_area) & (counts <= max_area)
        is_foam[0] = False
    else:
        is_foam = classify_foam_regions(labels, image, tuple(args.hue), args.min_saturation,
                                        args.min_area, args.max_area)
    if not is_foam.any():
        raise ValueError("No foam cells found. Check --hue / --min-saturation or try --all-regions.")

    neighbours, is_edge = neighbour_info(labels, is_foam)
    df = measure_cells(labels, image, is_foam, is_edge, neighbours, args.um_per_px)
    is_edge_saved = df["is_edge_cell"].copy()
    if args.include_edge:
        df["is_edge_cell"] = False
    summary = summarise(df, labels, is_foam, args.um_per_px)
    profiles = spatial_profiles(df, args.um_per_px)
    df["is_edge_cell"] = is_edge_saved

    params = {k: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()}
    params.update(image=str(image_path), segmentation=str(seg_path))
    write_excel(out / "cell_data.xlsx", df, summary, profiles, params)
    df.to_csv(out / "cell_data.csv", index=False)
    overlays = save_overlays(image, labels, df, args.um_per_px, out)
    overview_figure(image, labels, df, overlays, profiles, summary, args.um_per_px,
                    out / "overview.png", f"Foam cell analysis: {image_path.name}")
    return df, summary


# ---------------------------------------------------------------- folder mode
def _normalise(stem: str) -> str:
    s = stem.lower()
    changed = True
    while changed:
        changed = False
        for a in SEG_AFFIXES:
            if s.endswith(a) and len(s) > len(a):
                s, changed = s[: -len(a)], True
            elif s.startswith(a) and len(s) > len(a):
                s, changed = s[len(a):], True
    return s


def pair_files(image_dir: Path, seg_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Match each image to a segmentation by file name.

    Tries, in order: identical stem; stem after stripping common affixes such as
    "_seg"/"_mask"; a unique segmentation whose name contains the image stem.
    """
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    segs = sorted(p for p in seg_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    by_stem = {p.stem.lower(): p for p in segs}
    by_norm: dict[str, list[Path]] = {}
    for p in segs:
        by_norm.setdefault(_normalise(p.stem), []).append(p)

    pairs, unmatched = [], []
    for img in images:
        key = img.stem.lower()
        match = by_stem.get(key)
        if match is None and len(by_norm.get(_normalise(img.stem), [])) == 1:
            match = by_norm[_normalise(img.stem)][0]
        if match is None:
            cands = [p for p in segs if key in p.stem.lower()]
            match = cands[0] if len(cands) == 1 else None
        if match is None:
            unmatched.append(img)
        else:
            pairs.append((img, match))
    return pairs, unmatched


def run_folder(image_dir: Path, seg_dir: Path, out: Path, args) -> None:
    pairs, unmatched = pair_files(image_dir, seg_dir)
    print(f"Found {len(pairs)} image/segmentation pairs in {image_dir} and {seg_dir}")
    for img in unmatched:
        print(f"  no segmentation found for {img.name} (skipped)")
    if not pairs:
        raise SystemExit("Nothing to do: no image could be matched to a segmentation by file name.")

    all_cells, summaries, failed = [], [], []
    for i, (img, seg) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {img.name}  <->  {seg.name}")
        try:
            df, summary = analyse_one(img, seg, out / img.stem, args)
        except Exception as exc:  # keep going on bad frames
            print(f"    FAILED: {exc}")
            failed.append((img.name, seg.name, str(exc)))
            continue
        df.insert(0, "image", img.name)
        all_cells.append(df)
        row = {"image": img.name, "segmentation": seg.name}
        row.update(dict(zip(summary["metric"], summary["value"])))
        summaries.append(row)

    if summaries:
        per_image = pd.DataFrame(summaries)
        cells = pd.concat(all_cells, ignore_index=True)
        with pd.ExcelWriter(out / "all_images.xlsx", engine="openpyxl") as xw:
            per_image.to_excel(xw, sheet_name="per_image_summary", index=False)
            cells.round(4).to_excel(xw, sheet_name="all_cells", index=False)
            if failed:
                pd.DataFrame(failed, columns=["image", "segmentation", "error"]).to_excel(
                    xw, sheet_name="failed", index=False)
            for ws in xw.book.worksheets:
                ws.freeze_panes = "B2"
        cells.to_csv(out / "all_cells.csv", index=False)
        trend_figure(per_image, args.um_per_px, out / "trends.png")
    print(f"\nDone: {len(summaries)} analysed, {len(failed)} failed, {len(unmatched)} unmatched.")
    print(f"Combined results in {out / 'all_images.xlsx'}")


def trend_figure(per_image: pd.DataFrame, scale: float | None, path: Path) -> None:
    """Key metrics per image, in file-name order (e.g. frames of a video)."""
    unit = "µm" if scale else "px"
    metrics = [
        ("Interior cells (used for stats)", "Interior cells"),
        (f"Equivalent diameter median [{unit}]", f"Median diameter [{unit}]"),
        ("Aspect ratio median", "Median aspect ratio"),
        ("Circularity median", "Median circularity"),
        ("Neighbours mean", "Mean neighbours"),
        ("Dominant orientation [deg from horizontal]", "Dominant orientation [°]"),
    ]
    metrics = [(c, t) for c, t in metrics if c in per_image]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(8, 0.25 * len(per_image)), 2.2 * len(metrics)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(per_image))
    for ax, (col, title) in zip(axes, metrics):
        ax.plot(x, per_image[col], "o-", color="#3b6fb6", ms=3)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.3)
    step = max(1, len(per_image) // 25)
    axes[-1].set_xticks(x[::step])
    axes[-1].set_xticklabels(per_image["image"].iloc[::step], rotation=60, ha="right", fontsize=7)
    fig.suptitle("Foam cell metrics per image", fontweight="bold")
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path, help="original micrograph, or a folder of them")
    ap.add_argument("segmentation", type=Path,
                    help="cell-wall line drawing (dark lines on white), or a folder of them")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output folder (default: <image>_cells/ next to the input)")
    ap.add_argument("--um-per-px", type=float, default=None, help="pixel size in µm to report physical units")
    ap.add_argument("--wall-threshold", type=int, default=DEFAULT_WALL_THRESHOLD)
    ap.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA_PX, help="min cell area [px]")
    ap.add_argument("--max-area", type=int, default=None, help="max cell area [px]")
    ap.add_argument("--hue", type=int, nargs=2, default=DEFAULT_HUE_RANGE, metavar=("MIN", "MAX"),
                    help="OpenCV hue range (0-179) of the foam colour; default blue")
    ap.add_argument("--min-saturation", type=int, default=DEFAULT_MIN_SATURATION)
    ap.add_argument("--all-regions", action="store_true",
                    help="skip colour filtering: treat every region (except huge ones) as a cell")
    ap.add_argument("--include-edge", action="store_true", help="keep edge cells in the statistics")
    args = ap.parse_args()

    if args.image.is_dir():
        if not args.segmentation.is_dir():
            raise SystemExit("When IMAGE is a folder, SEGMENTATION must be a folder too.")
        out = args.out or args.image.parent / "cell_results"
        run_folder(args.image, args.segmentation, out, args)
        return

    out = args.out or args.image.with_name(args.image.stem + "_cells")
    try:
        _, summary = analyse_one(args.image, args.segmentation, out, args)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    print(f"\nWrote results to {out}/")


if __name__ == "__main__":
    main()
