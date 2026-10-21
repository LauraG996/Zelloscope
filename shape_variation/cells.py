"""Foam cells: marker-controlled watershed (ported from cell_segmentation.py on the brave-hamilton branch, with
its pixel parameters converted to mm), per-cell size/shape/orientation, and maps in the aligned shape frame."""
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


def measure_cells(img: np.ndarray, prep: Prepared, p: dict) -> dict:
    """Cells of one image with centroids in the canonical frame (mm). Cells touching the foam edge are dropped."""
    px_per_mm_full = 1.0 / prep.record.mm_per_px
    f = min(1.0, p["work_px_per_mm"] / px_per_mm_full)
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    res = px_per_mm_full * f

    # foam mask in this image's pixels: canonical foam mask mapped back through the inverse transform
    m = prep.image_to_canvas.copy()
    m[:, :2] /= f                                      # small-image px -> canvas px
    inv = cv2.invertAffineTransform(m)
    mask = cv2.warpAffine(prep.foam.astype(np.uint8), inv, (gray.shape[1], gray.shape[0]),
                          flags=cv2.INTER_NEAREST) > 0
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    labels = segment_cells(gray, mask, res, p)
    edge = np.unique(labels[cv2.dilate((labels == 1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0])
    min_area = p["min_area_mm2"] * res * res
    out = {k: [] for k in ("x", "y", "area_mm2", "diameter_mm", "elongation", "angle_rad")}
    for r in regionprops(np.where(labels >= 2, labels, 0)):
        if r.label in edge or r.area < min_area:
            continue
        cy, cx = r.centroid
        out["x"].append(cx / f)
        out["y"].append(cy / f)
        out["area_mm2"].append(r.area / res ** 2)
        out["diameter_mm"].append(r.equivalent_diameter_area / res)
        out["elongation"].append(r.axis_major_length / max(r.axis_minor_length, 1e-6))
        # skimage orientation: angle between the row axis and the major axis -> angle from the x axis
        out["angle_rad"].append(np.pi / 2 - r.orientation)
    cells = {k: np.asarray(v, float) for k, v in out.items()}
    xy = prep.image_px_to_mm(np.column_stack([cells["x"], cells["y"]])) if len(cells["x"]) else np.zeros((0, 2))
    cells["x_mm"], cells["y_mm"] = xy[:, 0], xy[:, 1]
    # orientation in the canonical frame: rotate by the image->canvas rotation
    rot = np.arctan2(prep.image_to_canvas[1, 0], prep.image_to_canvas[0, 0])
    cells["angle_rad"] = cells["angle_rad"] - rot if len(cells["angle_rad"]) else cells["angle_rad"]
    return cells


def _order(angles: np.ndarray) -> float:
    """Nematic order of orientations (0 = random, 1 = all parallel)."""
    return float(np.abs(np.mean(np.exp(2j * angles)))) if len(angles) else float("nan")


def cell_features(cells: dict, outer: np.ndarray, cavity: np.ndarray, base_thickness: float,
                  skin_mm: float = 1.5) -> dict:
    """Per-part cell features overall and by region (aligned frame, y down, base at y ~ 0 at the bottom)."""
    x, y = cells["x_al"], cells["y_al"]
    if len(x) == 0:
        return {}
    o = outer.astype(np.float32).reshape(-1, 1, 2)
    dist_outer = np.array([abs(cv2.pointPolygonTest(o, (float(a), float(b)), True)) for a, b in zip(x, y)])
    cav_bottom = cavity[:, 1].max()
    regions = {
        "all": np.ones(len(x), bool),
        "base": y > cav_bottom,
        "legs": y <= cav_bottom,
        "skin": dist_outer < skin_mm,
        "core": dist_outer >= skin_mm,
    }
    area = abs(cv2.contourArea(outer.astype(np.float32))) - abs(cv2.contourArea(cavity.astype(np.float32)))
    feats = dict(cell_count=int(len(x)), cell_density_per_mm2=round(len(x) / area, 4) if area > 0 else float("nan"))
    for name, sel in regions.items():
        if sel.sum() < 5:
            continue
        d = cells["diameter_mm"][sel]
        feats[f"cell_diam_{name}"] = round(float(np.mean(d)), 4)
        feats[f"cell_elong_{name}"] = round(float(np.mean(cells["elongation"][sel])), 4)
        feats[f"cell_order_{name}"] = round(_order(cells["angle_al"][sel]), 4)
    feats["cell_diam_cv"] = round(float(np.std(cells["diameter_mm"]) / np.mean(cells["diameter_mm"])), 4)
    if "cell_diam_skin" in feats and "cell_diam_core" in feats:
        feats["cell_diam_skin_core_ratio"] = round(feats["cell_diam_skin"] / feats["cell_diam_core"], 4)
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
        cv2.fillPoly(m, [to_px(cavity)], 0, cv2.LINE_8, shift=4)
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
