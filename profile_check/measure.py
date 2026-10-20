"""Dimensional measurements (mm) of a profile in the canonical / reference-aligned frame.

Frame: y points down, base at the bottom (+y), tips at the top. Measurements use exact intersections of
horizontal / vertical lines with the contour polylines, so they do not depend on a raster grid.

TODO(NMC): the definitions below are our reading of the drawing; confirm where the 24 mm width and the
12 mm thickness are measured. Each measurement is a few lines here so it is easy to change.
"""
import numpy as np

from .config import Config, MeasurementParams


def line_extents(poly: np.ndarray, values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max crossing of a closed polygon with lines {coord[axis] = v} (NaN where a line misses it).

    axis=1: horizontal lines y=v -> x extents. axis=0: vertical lines x=v -> y extents.
    """
    p, q = poly, np.roll(poly, -1, axis=0)
    a, b = p[:, axis], q[:, axis]
    other = 1 - axis
    v = np.asarray(values, float)[:, None]
    hit = (v >= np.minimum(a, b)) & (v < np.maximum(a, b))
    denom = np.where(b == a, 1.0, b - a)
    x = p[:, other] + (v - a) / denom * (q[:, other] - p[:, other])
    lo = np.where(hit, x, np.inf).min(axis=1)
    hi = np.where(hit, x, -np.inf).max(axis=1)
    miss = ~hit.any(axis=1)
    lo[miss] = np.nan
    hi[miss] = np.nan
    return lo, hi


def measure(outer: np.ndarray, cavity: np.ndarray, params: MeasurementParams) -> dict:
    step = params.contour_step_mm

    # length: outer extent along the profile axis (base -> tips)
    length = outer[:, 1].max() - outer[:, 1].min()
    # outer width: outer extent perpendicular to the axis
    outer_width = outer[:, 0].max() - outer[:, 0].min()

    # inner width: widest cavity row
    cav_top, cav_bottom = cavity[:, 1].min(), cavity[:, 1].max()
    rows = np.arange(cav_top + step / 2, cav_bottom, step)
    cl, cr = line_extents(cavity, rows, axis=1)
    widths = cr - cl
    i = int(np.nanargmax(widths))
    y_w = rows[i]
    inner_width = widths[i]

    # walls at the widest cavity row
    ol, orr = line_extents(outer, np.array([y_w]), axis=1)
    wall_l, wall_r = cl[i] - ol[0], orr[0] - cr[i]

    # base thickness: centre line of the widest cavity row, cavity bottom -> outer base, median over a band
    xc = (cl[i] + cr[i]) / 2
    band = params.base_thickness_band_mm
    xs = xc + np.arange(-band, band + step / 2, step)
    _, yo = line_extents(outer, xs, axis=0)
    _, yc = line_extents(cavity, xs, axis=0)
    base_t = float(np.nanmedian(yo - yc))

    # tip heights: topmost outer point left / right of the centre line
    left, right = outer[outer[:, 0] < xc], outer[outer[:, 0] >= xc]
    tip_diff = abs(left[:, 1].min() - right[:, 1].min()) if len(left) and len(right) else float("nan")

    # inner width every N mm upwards from the cavity bottom
    d = np.arange(params.width_profile_step_mm, cav_bottom - cav_top, params.width_profile_step_mm)
    pl, pr = line_extents(cavity, cav_bottom - d, axis=1)
    profile = [[round(float(dd), 1), round(float(w), 3)] for dd, w in zip(d, pr - pl) if np.isfinite(w)]

    r = lambda x: round(float(x), 3)
    return dict(length=r(length), inner_width_max=r(inner_width), base_thickness=r(base_t),
                outer_width_max=r(outer_width), wall_left=r(wall_l), wall_right=r(wall_r),
                wall_asym=r(abs(wall_l - wall_r)), tip_height_diff=r(tip_diff),
                inner_width_max_height_mm=r(cav_bottom - y_w),
                inner_width_profile_from_cavity_bottom=profile)


def spec_check(measurements: dict, cfg: Config) -> dict:
    return {name: dict(value=measurements[name], min=lim.min, max=lim.max, nominal=lim.nominal,
                       ok=bool(lim.contains(measurements[name])))
            for name, lim in cfg.spec.items()}
