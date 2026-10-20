import cv2
import numpy as np
import pytest

import synthetic as syn
from profile_check.segmentation import SegmentationError, segment


def _area_mm2(poly_px):
    return cv2.contourArea(poly_px.astype(np.float32)) / syn.PX_PER_MM ** 2


def _true_areas():
    outer, cavity = syn.profile_mm()
    return (cv2.contourArea(outer.astype(np.float32)), cv2.contourArea(cavity.astype(np.float32)))


def test_closed_cavity_with_glare(cfg):
    seg = segment(syn.render(angle_deg=20, glare=True), cfg.segmentation, syn.PX_PER_MM)
    true_outer, true_cav = _true_areas()
    assert seg.bridge_mm == 0
    assert _area_mm2(seg.outer) == pytest.approx(true_outer, rel=0.01)
    assert _area_mm2(seg.cavity) == pytest.approx(true_cav, rel=0.01)


def test_glare_does_not_change_cavity(cfg):
    with_glare = segment(syn.render(glare=True), cfg.segmentation, syn.PX_PER_MM)
    without = segment(syn.render(glare=False), cfg.segmentation, syn.PX_PER_MM)
    assert _area_mm2(with_glare.cavity) == pytest.approx(_area_mm2(without.cavity), rel=0.002)


@pytest.mark.parametrize("gap_mm", [1.5, 3.0])
def test_open_legs_still_give_a_cavity(cfg, gap_mm):
    seg = segment(syn.render(angle_deg=70, gap_mm=gap_mm), cfg.segmentation, syn.PX_PER_MM)
    _, true_cav = _true_areas()
    assert seg.bridge_mm > 0
    # Bridged cavity = true cavity + the mouth slot up to the bridged envelope.
    assert true_cav * 0.98 < _area_mm2(seg.cavity) < true_cav * 1.10


def test_no_foam_raises(cfg):
    img = np.full((500, 500, 3), syn.BG_BGR, np.uint8)
    with pytest.raises(SegmentationError):
        segment(img, cfg.segmentation, syn.PX_PER_MM)
