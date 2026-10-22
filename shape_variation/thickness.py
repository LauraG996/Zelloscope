"""Wall thickness along the U (left end -> base -> right end) from the medial axis of the foam.

Thickness at a skeleton point = 2 x its distance to the nearest foam edge. Positions are arclength along the
skeleton in mm, measured from the base centre (negative = left leg, positive = right leg), so parts can be
compared on a common grid.
"""
from collections import deque

import cv2
import numpy as np
from skimage.morphology import medial_axis

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _bfs(skel: np.ndarray, start: tuple[int, int]) -> tuple[dict, tuple[int, int]]:
    prev, far, q = {start: None}, start, deque([start])
    while q:
        p = q.popleft()
        far = p
        for dy, dx in _NEIGHBOURS:
            n = (p[0] + dy, p[1] + dx)
            if 0 <= n[0] < skel.shape[0] and 0 <= n[1] < skel.shape[1] and skel[n] and n not in prev:
                prev[n] = p
                q.append(n)
    return prev, far


def main_path(skel: np.ndarray) -> np.ndarray:
    """Longest path through the largest piece of a tree-like skeleton (two BFS passes), as (row, col) pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skel.astype(np.uint8), connectivity=8)
    if n < 2:
        return np.zeros((0, 2), int)
    skel = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    pts = np.argwhere(skel)
    if len(pts) == 0:
        return np.zeros((0, 2), int)
    _, a = _bfs(skel, tuple(pts[0]))
    prev, b = _bfs(skel, a)
    path, p = [], b
    while p is not None:
        path.append(p)
        p = prev[p]
    return np.array(path[::-1])


def thickness_profile(outer: np.ndarray, cavity: np.ndarray, res: float, closed: bool,
                      smooth_mm: float = 0.5) -> dict:
    """Thickness (mm) along the medial axis of the foam between `outer` and `cavity` (aligned frame, mm).

    Returns positions (mm from the base centre along the skeleton, left negative) and thickness values.
    """
    pts = np.vstack([outer, cavity])
    lo = pts.min(axis=0) - 2
    size = np.ceil((pts.max(axis=0) + 2 - lo) * res).astype(int)
    to_px = lambda p: np.round((p - lo) * res * 16).astype(np.int32)
    foam = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(foam, [to_px(outer)], 1, cv2.LINE_8, shift=4)
    cv2.fillPoly(foam, [to_px(cavity)], 0, cv2.LINE_8, shift=4)
    if closed:
        # Touching tips make the foam a ring; open it at the tips (above the cavity, on the axis).
        x0 = (cavity[:, 0].min() + cavity[:, 0].max()) / 2
        col = int(round((x0 - lo[0]) * res))
        top = int(round((cavity[:, 1].min() - lo[1]) * res))
        foam[:top, max(0, col - 1): col + 2] = 0

    skel, dist = medial_axis(foam > 0, return_distance=True)
    path = main_path(skel)
    if len(path) < 10:
        return dict(position_mm=np.zeros(0), thickness_mm=np.zeros(0))
    xy = path[:, ::-1] / res + lo                       # (x, y) mm
    if xy[0, 0] > xy[-1, 0]:                            # left -> right
        path, xy = path[::-1], xy[::-1]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    t = 2.0 * dist[path[:, 0], path[:, 1]] / res
    # base centre: the path point closest to the cavity midline, on the base side
    x0 = (cavity[:, 0].min() + cavity[:, 0].max()) / 2
    below = xy[:, 1] > cavity[:, 1].max() - 1.0
    cand = np.nonzero(below)[0] if below.any() else np.arange(len(xy))
    i0 = cand[np.argmin(np.abs(xy[cand, 0] - x0))]
    pos = s - s[i0]
    if smooth_mm > 0:
        step = np.median(np.diff(s)) or 1.0 / res
        t = cv2.GaussianBlur(t.reshape(-1, 1).astype(np.float64), (1, 0), smooth_mm / step).ravel() \
            if len(t) > 3 else t
    return dict(position_mm=pos, thickness_mm=t)


def on_grid(profile: dict, grid: np.ndarray, end_trim_mm: float) -> np.ndarray:
    """Thickness interpolated onto a common position grid; NaN outside the trimmed range of this part."""
    pos, t = profile["position_mm"], profile["thickness_mm"]
    out = np.full(len(grid), np.nan)
    if len(pos) < 2:
        return out
    lo, hi = pos.min() + end_trim_mm, pos.max() - end_trim_mm
    sel = (grid >= lo) & (grid <= hi)
    out[sel] = np.interp(grid[sel], pos, t)
    return out


def common_grid(profiles: list[dict], step: float, end_trim_mm: float) -> np.ndarray:
    """Grid covering positions present (after trimming) in every part."""
    lo = max(p["position_mm"].min() + end_trim_mm for p in profiles if len(p["position_mm"]))
    hi = min(p["position_mm"].max() - end_trim_mm for p in profiles if len(p["position_mm"]))
    lo, hi = np.ceil(lo / step) * step, np.floor(hi / step) * step
    return np.arange(lo, hi + step / 2, step) if hi > lo else np.zeros(0)
