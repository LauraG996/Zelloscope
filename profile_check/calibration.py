"""Image scale (px/mm): load/save a calibration file, or measure it from a checkerboard target image.

The scale is never defaulted - every run must name a calibration file or pass --px-per-mm explicitly.
"""
import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


class CalibrationError(RuntimeError):
    pass


def load_calibration(path: str | Path) -> float:
    path = Path(path)
    text = path.read_text()
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    try:
        px_per_mm = float(data["px_per_mm"])
    except (TypeError, KeyError, ValueError) as e:
        raise CalibrationError(f"{path}: expected a 'px_per_mm' number") from e
    if px_per_mm <= 0:
        raise CalibrationError(f"{path}: px_per_mm must be > 0")
    return px_per_mm


def save_calibration(path: str | Path, px_per_mm: float, **info) -> None:
    data = dict(px_per_mm=round(float(px_per_mm), 5), created=dt.datetime.now().isoformat(timespec="seconds"),
                **info)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2) + "\n")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False))


def px_per_mm_from_checkerboard(img: np.ndarray, inner_corners: tuple[int, int], square_mm: float) -> dict:
    """Scale from a checkerboard with `inner_corners` = (cols, rows) and squares of `square_mm`.

    Uses the mean distance between neighbouring corners, so mild perspective/lens effects average out.
    The target should lie in the same plane as the profile's cut face.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    found, corners = cv2.findChessboardCorners(
        gray, inner_corners, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        raise CalibrationError(f"checkerboard with {inner_corners[0]}x{inner_corners[1]} inner corners not found")
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    grid = corners.reshape(inner_corners[1], inner_corners[0], 2)
    d = np.concatenate([np.hypot(*np.diff(grid, axis=1).reshape(-1, 2).T),
                        np.hypot(*np.diff(grid, axis=0).reshape(-1, 2).T)])
    px_per_mm = float(np.mean(d)) / square_mm
    return dict(px_per_mm=px_per_mm, spread_pct=round(float(np.std(d) / np.mean(d) * 100), 3),
                n_spacings=int(d.size))
