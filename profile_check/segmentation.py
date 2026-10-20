"""Segment the foam cross-section into an outer contour and a cavity contour (image pixels)."""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import SegmentationParams


class SegmentationError(RuntimeError):
    pass


@dataclass
class Segmentation:
    outer: np.ndarray        # Nx2 float, image px (x, y)
    cavity: np.ndarray       # Mx2 float, image px (x, y)
    bridge_mm: float         # 0 = cavity was closed; otherwise closing size used to bridge an open mouth


def _odd(n: float, minimum: int = 3) -> int:
    return max(minimum, int(round(n)) | 1)


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        raise SegmentationError("empty mask")
    return max(cs, key=cv2.contourArea)


def foam_mask(img: np.ndarray, params: SegmentationParams, px_per_mm: float) -> np.ndarray:
    """Binary (0/255) mask of the largest blue-foam blob."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    foam = ((h >= params.hue[0]) & (h <= params.hue[1]) &
            (s >= params.sat_min) & (v >= params.val_min)).astype(np.uint8) * 255

    k = _ellipse(_odd(params.cleanup_kernel_mm * px_per_mm))
    foam = cv2.morphologyEx(foam, cv2.MORPH_CLOSE, k)
    foam = cv2.morphologyEx(foam, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(foam, connectivity=8)
    if n < 2:
        raise SegmentationError("no foam found - check lighting / HSV thresholds")
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == big, 255, 0).astype(np.uint8)


def _exterior(closed: np.ndarray) -> np.ndarray:
    """Background region of a padded mask that is connected to the image border (bool)."""
    _, labels = cv2.connectedComponents((closed == 0).astype(np.uint8), connectivity=4)
    return labels == labels[0, 0]


def _cavity_from_exterior(foam: np.ndarray, exterior: np.ndarray, min_area: float) -> np.ndarray | None:
    """Largest non-foam region that is not part of the exterior, or None if it is too small."""
    cand = ((foam == 0) & ~exterior).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    if n < 2:
        return None
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[big, cv2.CC_STAT_AREA] < min_area:
        return None
    return np.where(labels == big, 255, 0).astype(np.uint8)


def segment(img: np.ndarray, params: SegmentationParams, px_per_mm: float) -> Segmentation:
    """Foam mask -> outer contour + cavity (largest hole).

    If the legs are open the cavity is connected to the background; it is then bridged by closing the
    foam mask with growing kernels until a hole appears. The closing only decides which background pixels
    are "outside": the cavity itself is taken from the un-closed foam mask, so its shape is not eroded.
    A glare spot inside the cavity is either not foam (low saturation) or a small separate blob that the
    largest-blob step discards, and the cavity's outer contour is used, so it cannot cut the cavity up.
    """
    foam = foam_mask(img, params, px_per_mm)
    max_bridge = max(params.bridge_close_mm, default=0.0)
    pad = int(np.ceil(max_bridge * px_per_mm)) + 4
    foam_p = cv2.copyMakeBorder(foam, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    filled_area = cv2.contourArea(_largest_contour(foam_p))
    min_area = params.min_cavity_area_frac * filled_area

    cavity, bridge_mm = _cavity_from_exterior(foam_p, _exterior(foam_p), min_area), 0.0
    if cavity is None:
        # Bridge on a downscaled mask: big elliptic kernels are slow at full resolution, and only the
        # mouth line depends on this resolution.
        scale = min(1.0, params.bridge_work_px_per_mm / px_per_mm)
        small = cv2.resize(foam_p, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small = np.where(small >= 128, 255, 0).astype(np.uint8)
        for mm in params.bridge_close_mm:
            closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, _ellipse(_odd(mm * px_per_mm * scale)))
            ext = cv2.resize(_exterior(closed).astype(np.uint8), (foam_p.shape[1], foam_p.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
            cavity = _cavity_from_exterior(foam_p, ext, min_area)
            if cavity is not None:
                bridge_mm = mm
                break
    if cavity is None:
        raise SegmentationError("no cavity found (legs open wider than the largest bridge_close_mm?)")

    cav_c = _largest_contour(cavity)
    outer_c = _largest_contour(cv2.bitwise_or(foam_p, cavity))
    off = np.array([pad, pad], float)
    return Segmentation(outer=outer_c.reshape(-1, 2).astype(float) - off,
                        cavity=cav_c.reshape(-1, 2).astype(float) - off,
                        bridge_mm=bridge_mm)
