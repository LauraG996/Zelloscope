"""Shape-agnostic mode (kind: generic) on synthetic parts: a rounded rectangle with a hole, a solid part and a part
with two holes, imaged at any rotation and from either cut end."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from shape_variation.cli import main as sv_main
from shape_variation.config import load_config
from shape_variation.generic import (
    describe_generic,
    extract_generic,
    generalized_procrustes_generic,
    prepare_generic,
)
from shape_variation.io import discover
from shape_variation.pipeline import run
from shape_variation.segment import SegmentationError

PPM, SIZE = 14.0, 1000                       # ~71 mm field of view
CONFIG = Path(__file__).resolve().parent.parent / "config" / "generic_profile.yaml"


def rounded_rect(w, h, r, cx=0.0, cy=0.0, n=400):
    """Closed rounded rectangle (mm), w x h with corner radius r, centred on (cx, cy)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = np.cos(t), np.sin(t)
    cxs = np.where(x >= 0, w / 2 - r, -(w / 2 - r))
    cys = np.where(y >= 0, h / 2 - r, -(h / 2 - r))
    return np.column_stack([cxs + r * x + cx, cys + r * y + cy])


def render(outer, holes=(), angle=0.0, mirror=False, shift=(0.0, 0.0), seed=0):
    """Light foam on a dark background; the part rotated by `angle` degrees, optionally mirrored."""
    rng = np.random.default_rng(seed)
    a = np.radians(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def px(p):
        q = np.asarray(p, float) * ([-1.0, 1.0] if mirror else [1.0, 1.0])
        q = q @ rot.T + shift
        return np.round((q * PPM + SIZE / 2) * 16).astype(np.int32)

    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:] = (20, 25, 30)
    cv2.fillPoly(img, [px(outer)], (205, 170, 120), cv2.LINE_AA, shift=4)
    for h in holes:
        cv2.fillPoly(img, [px(h)], (20, 25, 30), cv2.LINE_AA, shift=4)
    noise = rng.normal(0, 8, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def write_set(folder: Path, images) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scale.yaml").write_text(f"mm_per_px: {1 / PPM}\n")
    for i, img in enumerate(images):
        cv2.imwrite(str(folder / f"2026-10-23_10-{i:02d}-00.png"), img)
    return folder


def generic_cfg(**shape_model):
    cfg = load_config(CONFIG)
    cfg["exclude"] = []
    cfg["shape_model"].update(bootstrap=20, **shape_model)
    return cfg


def shapes_of(folder, cfg):
    return [extract_generic(prepare_generic(r, cfg)) for r in discover([folder])]


def test_same_part_any_rotation_or_mirror_aligns_and_measures_the_same(tmp_path):
    outer, hole = rounded_rect(40, 28, 4), rounded_rect(20, 10, 3, cx=1.5, cy=-1.0)   # hole off-centre
    imgs = [render(outer, [hole], angle=a, mirror=i % 2 == 1, shift=(i - 3, 2 - i), seed=i)
            for i, a in enumerate((0, 33, 90, 150, 211, 300))]
    cfg = generic_cfg()
    shapes = shapes_of(write_set(tmp_path / "rot", imgs), cfg)
    assert all(len(s.holes) == 1 for s in shapes)
    sizes = (200, 200)
    _, aligned = generalized_procrustes_generic([[s.outer] + s.holes for s in shapes], sizes)
    pts = np.array([al.points for al in aligned])
    rms = np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum(-1).mean(-1))
    assert rms.max() < 0.08, rms
    assert [al.mirrored for al in aligned] == [False, True] * 3 or [al.mirrored for al in aligned] == [True, False] * 3
    for al in aligned:
        d = describe_generic(al.points, sizes)
        assert d["outer_width"] == pytest.approx(40, abs=0.2) and d["outer_height"] == pytest.approx(28, abs=0.2)
        assert d["hole_width"] == pytest.approx(20, abs=0.2) and d["hole_height"] == pytest.approx(10, abs=0.2)
        assert abs(d["hole_x"]) == pytest.approx(1.5, abs=0.15) and abs(d["hole_y"]) == pytest.approx(1.0, abs=0.15)
        assert d["hole_wall_left"] + d["hole_wall_right"] == pytest.approx(20, abs=0.3)


def test_aligned_frame_maps_the_part_onto_its_points(tmp_path):
    """Aligned.apply (used to put cells into the common frame) must move the dense outline onto the points."""
    outer, hole = rounded_rect(40, 28, 4), rounded_rect(20, 10, 3, cx=1.5)
    cfg = generic_cfg()
    s1, s2 = shapes_of(write_set(tmp_path / "map", [render(outer, [hole]), render(outer, [hole], 127, True)]), cfg)
    _, (a1, a2) = generalized_procrustes_generic([[s1.outer] + s1.holes, [s2.outer] + s2.holes], (200, 200))
    for s, al in ((s1, a1), (s2, a2)):
        moved = al.apply(s.outer)
        d = np.min(np.hypot(*(moved[:, None] - al.points[None, :200]).transpose(2, 0, 1)), axis=1)
        assert d.max() < 0.5


def test_pipeline_recovers_a_hole_shift(tmp_path):
    """Parts imaged the same way up (orientations 1, no mirror): the hole moves left/right from part to part and
    the first mode, the hole_x feature and the drift chart must show it."""
    rng = np.random.default_rng(3)
    dx = np.linspace(-1.5, 1.5, 9)
    rng.shuffle(dx)
    outer = rounded_rect(40, 28, 4)
    imgs = [render(outer, [rounded_rect(20, 10, 3, cx=d)], angle=rng.uniform(-8, 8), seed=i)
            for i, d in enumerate(dx)]
    cfg = generic_cfg(orientations=1, allow_mirror=False)
    a = run([write_set(tmp_path / "shift", imgs)], cfg, progress=lambda s: None)
    assert a.kind == "generic" and a.model.sizes == (200, 200)
    est = a.table["hole_x"].to_numpy()
    assert np.abs(est - dx).max() < 0.15, (est, dx)
    assert "hole further" in a.modes[0]["name"] and a.model.explained[0] > 0.8
    assert "hole_x" in a.charts and np.isfinite(a.thickness).all()
    # walls beside the hole follow the shift
    assert np.corrcoef(a.table["hole_wall_left"], dx)[0, 1] > 0.99


def test_solid_part_and_two_holes(tmp_path):
    rng = np.random.default_rng(5)
    solid = [render(rounded_rect(30 + w, 18, 5), angle=rng.uniform(0, 360), seed=i)
             for i, w in enumerate((0, 0.5, 1.0, 1.5))]
    a = run([write_set(tmp_path / "solid", solid)], generic_cfg(), progress=lambda s: None)
    assert a.model.sizes == (200,)
    assert not any(c.startswith("hole") for c in a.table.columns)
    assert np.corrcoef(a.table["outer_width"], [30, 30.5, 31, 31.5])[0, 1] > 0.99
    # solid: the largest circle is limited by the far side -> thickness ~ the part's height (18 mm)
    assert np.nanmax(a.thickness) == pytest.approx(18, abs=0.5)

    outer = rounded_rect(50, 24, 4)
    holes = [rounded_rect(12, 10, 2, cx=-11), rounded_rect(14, 8, 2, cx=11)]       # different hole sizes
    two = [render(outer, holes, angle=ang, mirror=m, seed=i) for i, (ang, m) in
           enumerate(((0, False), (180, False), (90, True), (271, True)))]
    a = run([write_set(tmp_path / "two", two)], generic_cfg(), progress=lambda s: None)
    assert a.model.sizes == (200, 200, 200)
    # the larger hole is hole 1 in every part, whatever the orientation
    assert np.allclose(a.table["hole1_width"], 12, atol=0.25) and np.allclose(a.table["hole2_width"], 14, atol=0.25)


def test_part_cut_by_the_frame_and_odd_hole_count_are_skipped(tmp_path):
    outer, hole = rounded_rect(40, 28, 4), rounded_rect(20, 10, 3)
    imgs = [render(outer, [hole], angle=a, seed=i) for i, a in enumerate((0, 20, 40))]
    imgs.append(render(outer, [hole], shift=(25, 0)))                  # runs off the right edge
    imgs.append(render(outer, [], angle=10))                          # no hole
    cfg = generic_cfg()
    folder = write_set(tmp_path / "skip", imgs)
    with pytest.raises(SegmentationError, match="touches the image frame"):
        prepare_generic(discover([folder])[3], cfg)
    a = run([folder], cfg, progress=lambda s: None)
    reasons = dict(a.skipped)
    assert len(a.shapes) == 3
    assert "touches the image frame" in reasons["2026-10-23_10-03-00.png"]
    assert "0 hole(s)" in reasons["2026-10-23_10-04-00.png"]


def test_cli_analyze_and_score_generic(tmp_path):
    outer = rounded_rect(40, 28, 4)
    imgs = [render(outer, [rounded_rect(20 + 0.2 * i, 10, 3)], angle=37 * i, mirror=i % 3 == 0, seed=i)
            for i in range(8)]
    folder = write_set(tmp_path / "cli", imgs)
    cfg_file = tmp_path / "generic.yaml"
    cfg_file.write_text(CONFIG.read_text().replace('exclude: ["2026-06-04_", "capture_sample_"]', "exclude: []")
                        .replace("bootstrap: 200", "bootstrap: 20"))
    out = tmp_path / "res"
    assert sv_main(["analyze", str(folder), "-o", str(out), "--quiet", "--config", str(cfg_file)]) == 0
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Hole width" in html and "Thickness around the part" in html and "U-profiles" not in html
    new = tmp_path / "new"
    write_set(new, [render(outer, [rounded_rect(21, 10, 3)], angle=200, mirror=True, seed=50),
                    render(outer, [rounded_rect(21, 10, 3, cx=4)], angle=10, seed=51)])
    res = tmp_path / "scores.csv"
    sv_main(["score", "--model", str(out / "model.npz"), str(new), "-o", str(res), "--config", str(cfg_file)])
    import pandas as pd
    df = pd.read_csv(res)
    assert not df.loc[0, "shape_abnormal"] or not df.loc[0, "new_deformation"]
    assert bool(df.loc[1, "shape_abnormal"]) or bool(df.loc[1, "new_deformation"])     # hole 4 mm off-centre
    assert df.loc[0, "hole_width"] == pytest.approx(21, abs=0.2)
