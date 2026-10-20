import numpy as np
import pytest

from profile_check.calibration import (CalibrationError, load_calibration, px_per_mm_from_checkerboard,
                                       save_calibration)


def checkerboard(px_per_mm=23.7, square_mm=5.0, inner=(9, 6), margin=150):
    sq = px_per_mm * square_mm
    w, h = int((inner[0] + 1) * sq) + 2 * margin, int((inner[1] + 1) * sq) + 2 * margin
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    i, j = np.floor((xx - margin) / sq), np.floor((yy - margin) / sq)
    inside = (i >= 0) & (i <= inner[0]) & (j >= 0) & (j <= inner[1])
    img = np.full((h, w), 255, np.uint8)
    img[inside & ((i + j) % 2 == 0)] = 0
    return np.dstack([img] * 3)


def test_checkerboard_scale():
    cal = px_per_mm_from_checkerboard(checkerboard(23.7), (9, 6), 5.0)
    assert cal["px_per_mm"] == pytest.approx(23.7, rel=0.003)


def test_checkerboard_not_found():
    with pytest.raises(CalibrationError):
        px_per_mm_from_checkerboard(np.full((300, 300, 3), 128, np.uint8), (9, 6), 5.0)


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_calibration_file_roundtrip(tmp_path, suffix):
    path = tmp_path / f"cal{suffix}"
    save_calibration(path, 26.5, target="ruler.png")
    assert load_calibration(path) == pytest.approx(26.5)


def test_calibration_file_needs_scale(tmp_path):
    path = tmp_path / "cal.yaml"
    path.write_text("target: ruler.png\n")
    with pytest.raises(CalibrationError):
        load_calibration(path)
