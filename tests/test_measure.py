import pytest

import synthetic as syn
from profile_check.measure import measure
from profile_check.pipeline import part_from_image

SPEC_KEYS = ("length", "inner_width_max", "base_thickness")
INVARIANT_KEYS = SPEC_KEYS + ("outer_width_max", "wall_left", "wall_right")


def _measure(img, cfg):
    p = part_from_image(img, syn.PX_PER_MM, cfg)
    return measure(p.outer, p.cavity, cfg.measurement)


@pytest.fixture(scope="module")
def base_image():
    return syn.render(seed=7)


def test_matches_ground_truth(cfg, base_image):
    m = _measure(base_image, cfg)
    for k, v in syn.TRUE.items():
        assert m[k] == pytest.approx(v, abs=0.15), k
    assert m["wall_asym"] < 0.1
    assert m["tip_height_diff"] < 0.1
    assert m["inner_width_max_height_mm"] == pytest.approx(4.0, abs=0.3)
    heights = [d for d, _ in m["inner_width_profile_from_cavity_bottom"]]
    assert heights[:3] == [5.0, 10.0, 15.0]


def test_rotation_and_mirror_invariant_within_0_3_mm(cfg, base_image):
    ref = _measure(base_image, cfg)
    for angle in range(15, 181, 15):
        for mirror in (False, True):
            m = _measure(syn.warp(base_image, angle, mirror=mirror), cfg)
            for k in INVARIANT_KEYS:
                assert abs(m[k] - ref[k]) <= 0.3, (angle, mirror, k, m[k], ref[k])

