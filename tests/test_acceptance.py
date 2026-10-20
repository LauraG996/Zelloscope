"""Acceptance tests from the task: variants pass, out-of-spec variants fail, runtime."""
import time

import cv2
import pytest

import synthetic as syn
from conftest import SAMPLE_SHAPE
from profile_check.pipeline import inspect_part, load_part, part_from_image

VARIANTS = ([dict(angle_deg=a) for a in (0, 30, 60, 90, 120, 150, 180)]
            + [dict(angle_deg=a, mirror=True) for a in (0, 45, 135, 180)]
            + [dict(angle_deg=a, scale=s) for a, s in ((20, 0.99), (110, 1.01), (165, 0.99))])


def _inspect(img, cfg, reference):
    return inspect_part(part_from_image(img, syn.PX_PER_MM, cfg), *reference, cfg)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(f"{k}{v[k]}" for k in v))
def test_variants_of_sample_pass(cfg, reference, sample_image, variant):
    res = _inspect(syn.warp(sample_image, **variant), cfg, reference)
    assert res.passed, res.reasons
    assert res.data["alignment"]["match_score"] > 1.4          # max is 1 + cavity_weight = 1.5
    for name in ("outer", "cavity"):
        assert res.data["deviation"][name]["p99_abs"] <= cfg.contour_tolerance_mm


@pytest.mark.parametrize("bad, dim", [(dict(sx=0.93), "inner_width_max"), (dict(sy=0.94), "length")],
                         ids=["width93", "length94"])
def test_shrunk_variants_fail_dimension_and_contour(cfg, reference, bad, dim):
    res = _inspect(syn.render(angle_deg=50, seed=5, **bad, **SAMPLE_SHAPE), cfg, reference)
    assert not res.passed
    assert not res.data["spec"][dim]["ok"]
    assert not res.data["contour_ok"]
    assert any("contour" in r for r in res.reasons)


def test_open_legs_part_is_inspected(cfg, reference):
    res = _inspect(syn.render(angle_deg=10, gap_mm=1.5, **SAMPLE_SHAPE), cfg, reference)
    assert res.data["mouth_bridged"]
    for k in ("length", "inner_width_max", "base_thickness"):
        assert res.data["spec"][k]["ok"], k


def test_one_inspection_under_5_s(cfg, reference, sample_image, tmp_path):
    path = tmp_path / "part.png"
    cv2.imwrite(str(path), syn.warp(sample_image, 73))
    t0 = time.perf_counter()
    inspect_part(load_part(path, syn.PX_PER_MM, cfg), *reference, cfg)
    assert time.perf_counter() - t0 < 5.0
