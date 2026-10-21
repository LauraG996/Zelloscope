import numpy as np
import pytest

import synthetic as syn
from profile_check.measure import measure
from profile_check.pipeline import part_from_image
from profile_check.reference import Reference, build_reference


@pytest.fixture(scope="module")
def built(cfg, good_parts):
    return build_reference(good_parts, cfg, px_per_mm=syn.PX_PER_MM)


def test_reference_shape_is_the_mean(cfg, built):
    ref, _ = built
    m = measure(ref.outer, ref.cavity, cfg.measurement)
    mean = lambda key: np.mean([s[key] for s in syn.GOOD_PART_SHAPES])
    assert m["length"] == pytest.approx(mean("length"), abs=0.3)
    assert m["inner_width_max"] == pytest.approx(mean("inner_width"), abs=0.3)
    assert m["base_thickness"] == pytest.approx(mean("base_thickness"), abs=0.3)


def test_good_samples_in_spec_and_leave_one_out_passes(built):
    _, report = built
    assert all(s["spec_ok"] and s["contour_ok"] for s in report["samples"])
    assert len(report["leave_one_out"]) == 5
    assert all(r["PASS"] for r in report["leave_one_out"]), report["leave_one_out"]


def test_out_of_spec_good_sample_is_flagged(cfg, good_parts):
    bad = part_from_image(syn.render(angle_deg=12, sy=0.94), syn.PX_PER_MM, cfg, name="short")
    _, report = build_reference(good_parts[:3] + [bad], cfg, run_loo=False)
    flags = {s["name"]: s["spec_ok"] for s in report["samples"]}
    assert flags["short"] is False
    assert all(v for k, v in flags.items() if k != "short")


def test_save_load_and_extend(cfg, built, good_parts, tmp_path):
    ref, _ = built
    path = tmp_path / "ref.npz"
    ref.save(path)
    loaded = Reference.load(path)
    np.testing.assert_allclose(loaded.outer, ref.outer)
    np.testing.assert_allclose(loaded.cavity, ref.cavity)
    assert [p.name for p in loaded.parts] == [p.name for p in good_parts]
    np.testing.assert_allclose(loaded.parts[2].cavity, good_parts[2].cavity)
    assert loaded.meta["n_parts"] == 5

    # Extending = rebuilding from the stored parts plus new ones.
    extra = part_from_image(syn.render(angle_deg=300, seed=42), syn.PX_PER_MM, cfg, name="new")
    bigger, _ = build_reference(loaded.parts + [extra], cfg, run_loo=False)
    assert bigger.meta["n_parts"] == 6


@pytest.mark.parametrize("mode", ["nominal", "mid"])
def test_ideal_shape_hits_drawing_dimensions(cfg, built, mode):
    from profile_check.reference import make_ideal, target_dimensions
    ref, _ = built
    targets = target_dimensions(cfg, mode)
    ideal = make_ideal(ref, cfg, targets, mode)
    m = measure(ideal.outer, ideal.cavity, cfg.measurement)
    for key, value in targets.items():
        assert m[key] == pytest.approx(value, abs=0.05), key
    # The outline the drawing doesn't specify is kept from the mean shape.
    src = measure(ref.outer, ref.cavity, cfg.measurement)
    assert m["outer_width_max"] == pytest.approx(src["outer_width_max"], abs=0.1)
    assert ideal.meta["kind"] == "ideal"


def test_mid_range_ideal_passes_good_and_fails_bad(cfg, built, sample_image):
    from conftest import SAMPLE_SHAPE
    from profile_check.pipeline import inspect_part
    from profile_check.reference import make_ideal, target_dimensions
    ref, _ = built
    ideal = make_ideal(ref, cfg, target_dimensions(cfg, "mid"), "mid")
    good = part_from_image(syn.warp(sample_image, 65, mirror=True), syn.PX_PER_MM, cfg)
    bad = part_from_image(syn.render(angle_deg=20, sx=0.93, **SAMPLE_SHAPE), syn.PX_PER_MM, cfg)
    assert inspect_part(good, ideal.outer, ideal.cavity, cfg).passed
    assert not inspect_part(bad, ideal.outer, ideal.cavity, cfg).passed
