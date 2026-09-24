#!/usr/bin/env python3
"""Find the circular cutter marks left on a machined foam surface.

A rotating cutter leaves thin, dark, circular tracks across the specimen,
and each pass leaves a set of concentric tracks around its own center
(usually far outside the frame). Dark patches of foam along a track get
segmented and measured like pinholes even when there is no real hole there,
so this module finds the tracks so those detections can be flagged.

How: for a candidate center, every pixel's distance to it is binned into a
radial profile of the (background-flattened) image; concentric tracks around
the right center stack up into sharp dips in that profile, so the center is
the one whose profile has the most fine-scale structure. Around that
center the image is unwrapped to polar coordinates, where each track becomes
a near-straight line; smoothing along the track direction washes out the
foam texture but not the track, and a ridge filter across it picks out the
track itself. Pixels explained by one set of tracks are then masked out and
the search repeats, for surfaces cut in more than one pass.

Run directly to save a preview of the tracks found on an image:
    python cutter_arcs.py phenolic/2026-09-23_13-29-36.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Downscale factor for the center search (the radial profile only needs to
# resolve track spacing, not track width).
SEARCH_DOWNSCALE = 4

# Pixels at or below this gray level are the mount / outside the specimen.
BACKGROUND_MAX = 15

# Track length (full-res px, along the arc) that smoothing averages over:
# long enough to wash out foam texture, short enough that a track that
# isn't perfectly concentric with its center still stays in one place.
ALONG_TRACK_PX = 301

# A ridge must stay continuous for this long (px along the arc) to count as
# a track, which rejects short dark streaks in the foam itself.
MIN_TRACK_PX = 600

# Ridge strength threshold (gray levels below the local mean) for a track.
DEFAULT_RIDGE_THRESHOLD = 8.0

# Stop looking for more cutter passes once a center's profile structure
# drops below this fraction of the first (strongest) pass's.
MIN_PASS_SCORE_FRAC = 0.8

MAX_PASSES = 3

# Cutter-noise test for a segmented region (see region_features): a region
# counts as on a track if at least TRACK_FRAC_MIN of it lies within
# TRACK_BAND_PX of one (tracks come as close pairs with the noise between),
# and as having a real hole if at least CORE_FRAC_MIN of it is solid dark
# core -- pixels darker than DARK_REL x the local background that survive an
# opening of radius CORE_OPEN_RADIUS_PX, which strips the scattered dark
# specks of foam texture that make a track look darker. On the calibration
# image every region judged by eye as track noise had a core fraction of 0.00,
# and every real hole on a track 0.23 or more.
TRACK_BAND_PX = 41
TRACK_FRAC_MIN = 0.25
DARK_REL = 0.35
CORE_OPEN_RADIUS_PX = 6
CORE_FRAC_MIN = 0.10


def _flatten(gray: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """High-passed image (clipped so pinholes don't dominate) and its valid-pixel mask."""
    gray = gray.astype(np.float32)
    valid = gray > BACKGROUND_MAX
    hp = np.clip(gray - cv2.GaussianBlur(gray, (0, 0), sigma), -40, 40)
    hp[~valid] = 0
    return hp, valid


def _profile_score(hp: np.ndarray, weight: np.ndarray, center: tuple[float, float]) -> float:
    """Fine-scale structure (std after detrending) of the weighted radial profile around center."""
    h, w = hp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - center[0], yy - center[1])
    bins = (r - r.min()).astype(np.int32).ravel()
    sums = np.bincount(bins, (hp * weight).ravel())
    counts = np.bincount(bins, weight.ravel())
    profile = np.where(counts > 20, sums / np.maximum(counts, 1), 0.0)
    trend = np.convolve(profile, np.ones(31) / 31, "same")
    return float(np.std((profile - trend)[20:-20]))


def _search_center(hp: np.ndarray, weight: np.ndarray) -> tuple[tuple[float, float], float]:
    """Best track center (in hp's pixel coords): coarse grid, then a shrinking-step local search."""
    h, w = hp.shape
    # Cutter centers sit outside the frame, up to a few frame widths away.
    xs = np.linspace(-2.5 * w, 3.5 * w, 25)
    ys = np.linspace(-2.5 * h, 3.5 * h, 25)
    score, x, y = max((_profile_score(hp, weight, (x, y)), x, y) for x in xs for y in ys)

    # Pattern search (numpy only, no scipy): try the 8 neighbors at the current
    # step, move to the best if it improves, otherwise halve the step, down to 1 px.
    step = xs[1] - xs[0]
    while step >= 1.0:
        neighbors = [(x + dx * step, y + dy * step) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        best = max((_profile_score(hp, weight, v), *v) for v in neighbors)
        if best[0] > score:
            score, x, y = best
        else:
            step /= 2
    return (float(x), float(y)), float(score)


def _track_mask(hp: np.ndarray, valid: np.ndarray, center: tuple[float, float], threshold: float) -> np.ndarray:
    """Full-res bool mask of the tracks concentric with center."""
    h, w = hp.shape
    cx, cy = center
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    r_corners = np.hypot(corners[:, 0] - cx, corners[:, 1] - cy)
    t_corners = np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx)
    if np.ptp(t_corners) > np.pi:  # center inside the frame or angle wrap: not a cutter far away
        return np.zeros((h, w), bool)
    dx, dy = max(-cx, 0, cx - w), max(-cy, 0, cy - h)
    r_min, r_max = np.hypot(dx, dy) - 5, r_corners.max() + 5
    t_min, t_max = t_corners.min(), t_corners.max()
    n_r = int(r_max - r_min)
    n_t = int((t_max - t_min) * (r_min + r_max) / 2)  # ~1 px along the arc at mid radius

    # Polar image: rows = angle (along the track), cols = radius (across it).
    radius, theta = np.meshgrid(r_min + np.arange(n_r), t_min + (t_max - t_min) * np.arange(n_t) / n_t)
    map_x = (cx + radius * np.cos(theta)).astype(np.float32)
    map_y = (cy + radius * np.sin(theta)).astype(np.float32)
    polar = cv2.remap(hp, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    polar_valid = cv2.remap(valid.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
    del radius, theta, map_x, map_y

    along = cv2.blur(polar, (1, ALONG_TRACK_PX))
    coverage = cv2.blur(polar_valid, (1, ALONG_TRACK_PX))
    smoothed = np.where(coverage > 0.3, along / np.maximum(coverage, 1e-3), 0)
    ridge = (cv2.GaussianBlur(smoothed, (0, 0), sigmaX=4, sigmaY=0.1)
             - cv2.GaussianBlur(smoothed, (0, 0), sigmaX=25, sigmaY=0.1))
    tracks = ((ridge < -threshold) & (coverage > 0.3)).astype(np.uint8)
    tracks = cv2.morphologyEx(tracks, cv2.MORPH_OPEN, np.ones((MIN_TRACK_PX, 1), np.uint8))

    # Back to image coords.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    back_x = (np.hypot(xx - cx, yy - cy) - r_min).astype(np.float32)
    back_y = ((np.arctan2(yy - cy, xx - cx) - t_min) / (t_max - t_min) * n_t).astype(np.float32)
    return cv2.remap(tracks, back_x, back_y, cv2.INTER_NEAREST, borderValue=0).astype(bool)


def find_cutter_tracks(
    gray: np.ndarray, threshold: float = DEFAULT_RIDGE_THRESHOLD, max_passes: int = MAX_PASSES,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Mask of cutter tracks in a grayscale micrograph, plus each cutter pass's center (full-res px)."""
    small = cv2.resize(gray, None, fx=1 / SEARCH_DOWNSCALE, fy=1 / SEARCH_DOWNSCALE, interpolation=cv2.INTER_AREA)
    hp_small, valid_small = _flatten(small, 25)
    hp, valid = _flatten(gray, 60)
    weight = valid_small.astype(np.float32)

    mask = np.zeros(gray.shape, bool)
    centers = []
    first_score = None
    for _ in range(max_passes):
        center_small, score = _search_center(hp_small, weight)
        if first_score is None:
            first_score = score
        elif score < MIN_PASS_SCORE_FRAC * first_score:
            break
        center = (center_small[0] * SEARCH_DOWNSCALE, center_small[1] * SEARCH_DOWNSCALE)
        tracks = _track_mask(hp, valid, center, threshold)
        if not tracks.any():
            break
        centers.append(center)
        mask |= tracks
        # Hide this pass's tracks (and a margin) from the next center search.
        near = cv2.dilate(tracks.astype(np.uint8), np.ones((81, 81), np.uint8))
        near_small = cv2.resize(near, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        weight[near_small > 0] = 0
    return mask, centers


def dark_mask(gray: np.ndarray, rel: float) -> np.ndarray:
    """uint8 mask of pixels darker than rel x the local foam background (lightly smoothed)."""
    image = gray.astype(np.float32)
    # Local background from foam pixels only, so a big hole doesn't darken its own reference.
    foam = (image > 50).astype(np.float32)
    background = cv2.GaussianBlur(image * foam, (0, 0), 40) / np.maximum(cv2.GaussianBlur(foam, (0, 0), 40), 1e-3)
    return (cv2.GaussianBlur(image, (0, 0), 1.5) < rel * background).astype(np.uint8)


def region_features(gray: np.ndarray, labels: np.ndarray, tracks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-label (fraction within TRACK_BAND_PX of a track, fraction that is solid dark core).

    Indexed by label value, so feats[labels[y, x]] gives the region at (x, y).
    """
    dark = dark_mask(gray, DARK_REL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CORE_OPEN_RADIUS_PX + 1,) * 2)
    core = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    band = cv2.dilate(tracks.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TRACK_BAND_PX,) * 2))

    flat = labels.ravel()
    n = int(flat.max()) + 1
    area = np.maximum(np.bincount(flat, minlength=n), 1).astype(float)
    return np.bincount(flat, band.ravel(), n) / area, np.bincount(flat, core.ravel(), n) / area


def is_cutter_noise(track_frac: np.ndarray, core_frac: np.ndarray) -> np.ndarray:
    """On a cutter track with no solid dark core: a dark patch of foam along the track, not a hole."""
    return (track_frac >= TRACK_FRAC_MIN) & (core_frac < CORE_FRAC_MIN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Original (unsegmented) micrograph")
    parser.add_argument("--threshold", type=float, default=DEFAULT_RIDGE_THRESHOLD,
                        help=f"Ridge strength for a track, gray levels (default: {DEFAULT_RIDGE_THRESHOLD})")
    parser.add_argument("--output", type=Path, default=None, help="Preview path (default: <image>_cutter_tracks.png)")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        sys.exit(f"Could not read {args.image}")
    mask, centers = find_cutter_tracks(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), args.threshold)
    print(f"{args.image}: {len(centers)} cutter pass(es), tracks cover {100 * mask.mean():.2f}% of the image")
    for cx, cy in centers:
        print(f"  center ({cx:.0f}, {cy:.0f}) px")

    preview = image.copy()
    preview[mask] = (0.5 * preview[mask] + [0, 0, 127]).astype(np.uint8)
    dst = args.output or args.image.with_name(f"{args.image.stem}_cutter_tracks.png")
    cv2.imwrite(str(dst), preview)
    print(f"  wrote {dst}")


if __name__ == "__main__":
    main()
