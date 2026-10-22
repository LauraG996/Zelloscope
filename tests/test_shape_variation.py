"""Shape-variation analysis: synthetic parts with known, planted variation + one real rig image."""
import base64
import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

import synthetic as syn
from shape_variation.cli import main as sv_main
from shape_variation.config import load_config
from shape_variation.correlate import spearman_table
from shape_variation.io import discover
from shape_variation.monitor import i_chart, t2_leave_one_out
from shape_variation.pipeline import run
from shape_variation.relaxation import early_prediction_check, fit_pooled
from shape_variation.segment import extract, prepare
from shape_variation.shape_model import align_to

PPM, SIZE = 14.0, 1400                 # 100 mm field of view
TIPS_OUT = (0, -300)                   # shift so the tips leave the frame (like the rig images)
N_GOOD = 12
REAL_IMAGE = Path(__file__).resolve().parent.parent / "test_images" / "2026-06-04_09-45-17.png"


def _render(seed, angle=0.0, mirror=False, offset=TIPS_OUT, **shape):
    img = syn.render(angle_deg=angle, seed=seed, size=SIZE, ppm=PPM, offset_px=offset, glare=False, **shape)
    return cv2.flip(img, 1) if mirror else img


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """12 good parts with independent lean / base thickness, every other one mirrored, plus one narrow-cavity
    outlier last. All parts are asymmetric (right leg leans 3 mm more), like the rig parts, so a mirrored image
    is a different picture. Folder has scale.yaml and a parts.csv whose line speed follows the lean."""
    rng = np.random.default_rng(1)
    d = tmp_path_factory.mktemp("parts")
    (d / "scale.yaml").write_text(f"mm_per_px: {1 / PPM}\n")
    truth, rows = [], ["filename,line_speed_m_min"]
    for i in range(N_GOOD + 1):
        lean = float(rng.uniform(2.5, 5.5))
        shape = dict(lean=lean, lean_right=lean + 3.0, base_thickness=float(rng.uniform(10.8, 13.2)),
                     inner_width=25.5 + float(rng.normal(0, 0.05)))
        if i == N_GOOD:
            shape = dict(lean=4.0, lean_right=7.0, base_thickness=12.0, inner_width=21.5)
        name = f"2026-06-04_{9 + i // 6:02d}-{10 * (i % 6):02d}-00.png"
        cv2.imwrite(str(d / name), _render(i, float(rng.uniform(-15, 15)), mirror=bool(i % 2), **shape))
        truth.append(dict(name=Path(name).stem, **shape))
        rows.append(f"{name},{10 + 2 * (shape['lean'] - 4) + rng.normal(0, 0.1):.3f}")
    (d / "parts.csv").write_text("\n".join(rows) + "\n")
    return d, truth


@pytest.fixture(scope="module")
def analysis(dataset, cfg):
    return run([dataset[0]], cfg, progress=lambda s: None)


# ---------------- io / scale -----------------

def _png_with_itxt(path: Path, key: str, text: str):
    ok, buf = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
    data = buf.tobytes()
    body = key.encode() + b"\0" + b"\0\0" + b"\0" + b"\0" + text.encode()
    chunk = struct.pack(">I", len(body)) + b"iTXt" + body + struct.pack(">I", zlib.crc32(b"iTXt" + body) & 0xffffffff)
    iend = data.rfind(b"IEND") - 4
    path.write_bytes(data[:iend] + chunk + data[iend:])


def test_scale_from_png_metadata_beats_folder_file(tmp_path):
    meta = base64.b64encode(json.dumps({"p2mm": 0.02, "sw_ver": "v2.6.1"}).encode()).decode()
    _png_with_itxt(tmp_path / "2026-08-27_09-36.png", "ZelloscopeData", meta)
    (tmp_path / "scale.yaml").write_text("mm_per_px: 0.5\n")
    r = discover([tmp_path])[0]
    assert r.mm_per_px == pytest.approx(0.02) and r.scale_source == "png-metadata"
    assert r.timestamp.isoformat() == "2026-08-27T09:36:00"


def test_per_file_scale_beats_png_metadata(tmp_path):
    meta = base64.b64encode(json.dumps({"p2mm": 0.008063}).encode()).decode()
    _png_with_itxt(tmp_path / "capture_sample_20261022T112322.png", "ZelloscopeData", meta)
    _png_with_itxt(tmp_path / "other.png", "ZelloscopeData", meta)
    (tmp_path / "scale.yaml").write_text("mm_per_px: 0.014\nfiles:\n  capture_sample_20261022T112322.png:\n"
                                         "    mm_per_px: 0.020\n")
    recs = {r.path.name: r for r in discover([tmp_path])}
    assert recs["capture_sample_20261022T112322.png"].mm_per_px == pytest.approx(0.020)
    assert recs["capture_sample_20261022T112322.png"].timestamp.isoformat() == "2026-10-22T11:23:22"
    assert recs["other.png"].mm_per_px == pytest.approx(0.008063)          # no entry: PNG value as before


def test_unknown_scale_is_reported_not_guessed(tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((8, 8, 3), np.uint8))
    r = discover([tmp_path])[0]
    assert r.mm_per_px is None and "scale" in r.scale_source
    assert sv_main(["scales", str(tmp_path)]) == 1


def test_excluded_products(cfg):
    names = [r.path.name for r in discover([REAL_IMAGE.parent], tuple(cfg["exclude"]))]
    assert "squarefoam2.png" not in names and sum(n.startswith("2026-06-04") for n in names) == 11


# ---------------- segmentation -----------------

@pytest.mark.skipif(not REAL_IMAGE.exists(), reason="rig image not available")
def test_real_rig_image(cfg):
    p = prepare(discover([REAL_IMAGE])[0], cfg)
    assert not p.full_view and 45 < p.visible_depth_mm < 55
    s = extract(p, 45.0, 200, 200)
    from profile_check.measure import measure
    m = measure(s.outer, s.cavity, cfg.spec.measurement)
    assert 22 < m["inner_width_max"] < 26 and 12 < m["base_thickness"] < 14.5


def test_partial_view_window(tmp_path, cfg):
    path = tmp_path / "p.png"
    cv2.imwrite(str(path), _render(3, 10.0))
    (tmp_path / "scale.yaml").write_text(f"mm_per_px: {1 / PPM}\n")
    p = prepare(discover([path])[0], cfg)
    assert not p.full_view
    assert 60 < p.visible_depth_mm < 75          # base ~67 mm below the top edge, tilted 10 deg
    s = extract(p, 50.0, 200, 200)
    from profile_check.measure import measure
    m = measure(s.outer, s.cavity, cfg.spec.measurement)
    assert m["inner_width_max"] == pytest.approx(25.5, abs=0.3)
    assert m["base_thickness"] == pytest.approx(12.0, abs=0.3)
    # open curves: left cut -> base -> right cut
    assert s.outer_curve[0, 1] == pytest.approx(-50.0, abs=0.3) and s.outer_curve[-1, 1] == pytest.approx(-50, abs=0.3)
    assert s.outer_curve[0, 0] < 0 < s.outer_curve[-1, 0]


def test_full_view(tmp_path, cfg):
    path = tmp_path / "f.png"
    cv2.imwrite(str(path), _render(4, 30.0, offset=(0, 0)))
    (tmp_path / "scale.yaml").write_text(f"mm_per_px: {1 / PPM}\n")
    p = prepare(discover([path])[0], cfg)
    assert p.full_view
    s = extract(p, None, 200, 200)
    assert s.outer_curve[0, 1] == pytest.approx(s.outer[:, 1].min(), abs=0.5)    # starts at the tip
    assert abs(s.outer_curve[0, 0]) < 2.0


# ---------------- shape model -----------------

def test_mirrored_parts_are_aligned(analysis):
    mirrored = analysis.table["mirrored"].to_numpy()
    assert 0 < mirrored.sum() < len(mirrored)              # both cut ends present and recognised
    # every other part was flipped in the image: the detected flag alternates (up to a global flip)
    pattern = mirrored[:N_GOOD]
    assert np.all(pattern[::2] == pattern[0]) and np.all(pattern[1::2] != pattern[0])


def test_mirror_does_not_change_scores(analysis, dataset, cfg, tmp_path):
    d, truth = dataset
    src = cv2.imread(str(d / f"{truth[2]['name']}.png"))
    cv2.imwrite(str(tmp_path / "m.png"), cv2.flip(src, 1))
    (tmp_path / "scale.yaml").write_text(f"mm_per_px: {1 / PPM}\n")
    s = extract(prepare(discover([tmp_path / "m.png"])[0], cfg), analysis.depth_mm, 200, 200)
    m = analysis.model
    al = align_to(s.outer_curve, s.cavity_curve, m.mean, m.closed)
    np.testing.assert_allclose(m.project(al.points, 2), m.scores[2, :2], atol=0.15 * np.sqrt(m.variance[0]))


def test_planted_variation_is_found(analysis, dataset):
    """Lean and base thickness were varied independently: the main modes together must carry each of them
    (R^2 of a linear fit of the planted value on the first 3 mode scores)."""
    _, truth = dataset
    t = analysis.table.iloc[:N_GOOD]
    k = min(3, analysis.model.n_keep)
    x = np.column_stack([np.ones(N_GOOD)] + [t[f"PC{i + 1}"].to_numpy() for i in range(k)])
    for key in ("lean", "base_thickness"):
        planted = np.array([p[key] for p in truth[:N_GOOD]])
        coef, *_ = np.linalg.lstsq(x, planted, rcond=None)
        r2 = 1 - ((planted - x @ coef) ** 2).sum() / ((planted - planted.mean()) ** 2).sum()
        assert r2 > 0.85, (key, r2)
    # one of the main modes describes a change in leg lean (the planted lean range is 3 mm at the tips)
    assert max(abs(md["deltas"].get("lean_left_deg", 0)) for md in analysis.modes[:3]) > 1.0


def test_rotation_and_mirror_repeatability(analysis, dataset, cfg, tmp_path):
    """The same part imaged at other angles / from the other cut end scores (almost) the same."""
    d, truth = dataset
    shape = {k: v for k, v in truth[2].items() if k != "name"}
    (tmp_path / "scale.yaml").write_text(f"mm_per_px: {1 / PPM}\n")
    m = analysis.model
    scores = []
    for angle, mirror in ((-12, False), (0, True), (15, False), (7, True)):
        cv2.imwrite(str(tmp_path / "x.png"), _render(2, angle, mirror=mirror, **shape))
        s = extract(prepare(discover([tmp_path / "x.png"])[0], cfg), analysis.depth_mm, 200, 200)
        scores.append(m.project(align_to(s.outer_curve, s.cavity_curve, m.mean, m.closed).points, 3))
    spread = np.ptp(np.array(scores), axis=0)
    assert np.all(spread < 0.15 * np.sqrt(m.variance[:3])), spread


def test_outlier_is_flagged(analysis):
    t = analysis.table
    assert bool(t.iloc[-1]["shape_abnormal"] or t.iloc[-1]["new_deformation"])
    assert not t.iloc[-1]["inner_width_max_ok"]                          # 21.5 mm < 24 mm
    assert (t.iloc[:N_GOOD]["shape_abnormal"]).sum() <= 1


def test_thickness_map(analysis, dataset):
    g, th = analysis.thickness_grid, analysis.thickness
    assert len(g) > 50 and th.shape == (len(analysis.table), len(g))
    assert not np.isnan(th).any()
    base = th[:N_GOOD, int(np.argmin(np.abs(g)))]
    planted = np.array([p["base_thickness"] for p in dataset[1][:N_GOOD]])
    assert np.abs(base - planted).max() < 0.6


def test_correlation_with_process_data(analysis):
    c = analysis.correlations
    assert isinstance(c, dict) and "line_speed_m_min" in c["features"]
    top = c["table"][c["table"]["feature"] == "line_speed_m_min"].iloc[0]
    assert abs(top["rho"]) > 0.6


def test_missing_data_is_listed(analysis):
    text = " ".join(analysis.not_done + analysis.collect_next)
    assert "outside the field of view" in text and "after extrusion" in text


# ---------------- monitoring / relaxation units -----------------

def test_i_chart_flags_shift():
    y = np.r_[np.random.default_rng(0).normal(0, 1, 20), 6.0]
    c = i_chart(y)
    assert any(v["index"] == 20 for v in c["violations"])
    run = i_chart(np.r_[np.zeros(5), np.ones(9) * 0.5] + np.random.default_rng(1).normal(0, 0.05, 14))
    assert any("in a row" in v["rule"] for v in run["violations"])


def test_t2_flags_far_point():
    x = np.random.default_rng(0).normal(0, 1, (15, 2))
    x[-1] = [8, 8]
    r = t2_leave_one_out(x, 2, 0.01)
    assert r["abnormal"][-1] and r["abnormal"][:-1].sum() <= 1


def test_relaxation_recovers_tau():
    rng = np.random.default_rng(0)
    series = {}
    for p in range(5):
        t = np.array([0, 10, 30, 60, 240, 1440], float)
        series[f"p{p}"] = (t, 24 + 0.2 * p + 1.5 * np.exp(-t / 45.0) + rng.normal(0, 0.02, len(t)))
    fit = fit_pooled(series)
    assert fit["tau"] == pytest.approx(45, rel=0.15) and fit["amplitude"] == pytest.approx(1.5, rel=0.1)
    check = early_prediction_check(series)
    assert check["rmse"] < 0.2 * check["rmse_without_model"]


def test_spearman_fdr():
    import pandas as pd
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": np.arange(12.0), "noise": rng.normal(size=12)})
    y = pd.DataFrame({"t": np.arange(12.0) + rng.normal(0, 0.5, 12)})
    tab = spearman_table(x, y)
    assert tab.iloc[0]["feature"] == "a" and tab.iloc[0]["q"] < 0.01
    assert (tab["q"] >= tab["p"] - 1e-9).all()


# ---------------- CLI -----------------

def test_cli_analyze_and_score(dataset, tmp_path):
    d, truth = dataset
    out = tmp_path / "out"
    assert sv_main(["analyze", str(d), "-o", str(out), "--quiet"]) == 0
    for f in ("report.html", "parts.csv", "thickness.csv", "modes.json", "model.npz"):
        assert (out / f).exists(), f
    html = (out / "report.html").read_text()
    for section in ("Shape model", "Wall thickness", "Drift", "Anomalies", "What could not be done"):
        assert section in html
    # a good part scores clean, the outlier is flagged
    new = tmp_path / "new"
    new.mkdir()
    (new / "scale.yaml").write_text(f"mm_per_px: {1 / PPM}\n")
    cv2.imwrite(str(new / "good.png"), _render(50, -5.0, mirror=True, lean=4.0, lean_right=7.0, base_thickness=12.0))
    assert sv_main(["score", "--model", str(out / "model.npz"), str(new / "good.png")]) == 0
    cv2.imwrite(str(new / "bad.png"), _render(51, 5.0, lean=4.0, lean_right=7.0, base_thickness=12.0,
                                              inner_width=21.0))
    assert sv_main(["score", "--model", str(out / "model.npz"), str(new / "bad.png")]) == 1


def test_cli_overlay(dataset, tmp_path):
    d, truth = dataset
    out = tmp_path / "ov"
    assert sv_main(["overlay", str(d / f"{truth[0]['name']}.png"), "-o", str(out), "--window", "50"]) == 0
    img = cv2.imread(str(out / f"{truth[0]['name']}_segmentation.jpg"))
    assert img is not None and img.shape[0] > 900


def test_cli_shrink_keeps_scale_and_dimensions(dataset, cfg, tmp_path):
    d, truth = dataset
    src = d / f"{truth[0]['name']}.png"
    out = tmp_path / "small"
    assert sv_main(["shrink", str(src), "-o", str(out), "--px-per-mm", str(PPM / 2)]) == 0
    (r_big,), (r_small,) = discover([src]), discover([out])
    assert r_small.path.name == src.name and r_small.timestamp == r_big.timestamp
    assert r_small.mm_per_px == pytest.approx(2 / PPM, rel=2e-3)
    big = extract(prepare(r_big, cfg), 50.0, 200, 200)
    small = extract(prepare(r_small, cfg), 50.0, 200, 200)
    assert np.abs(big.cavity_curve - small.cavity_curve).max() < 0.1


def test_surface_intercepts_recover_flattened_layer(cfg):
    """Synthetic slab, 40 px/mm: three layers of flattened cells (0.4 mm deep x 1.2 mm wide) under the top
    surface, then round 0.8 mm cells. The intercept method must see the flattening and a round core."""
    from shape_variation.cells import surface_features, surface_intercepts
    from shape_variation.io import ImageRecord
    from shape_variation.segment import Prepared
    res, top = 40, 40
    h, w = top + 9 * res, 20 * res
    img = np.full((h, w), 8, np.uint8)
    img[top:] = 70
    walls = np.zeros((h, w), np.uint8)
    depth_lines = [0.4, 0.8, 1.2] + list(np.arange(2.0, 9.0, 0.8))
    for d in depth_lines:
        walls[top + int(round(d * res))] = 255
    for x in np.arange(0, w, 1.2 * res):
        walls[top:top + int(1.2 * res), int(x)] = 255
    for x in np.arange(0.4 * res, w, 0.8 * res):
        walls[top + int(1.2 * res):, int(x)] = 255
    walls = cv2.dilate(walls, np.ones((2, 2), np.uint8))
    img = np.maximum(img, (walls * 0.6).astype(np.uint8))
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img[:top] = 8
    foam = np.zeros((h, w), bool)
    foam[top:] = True
    prep = Prepared(record=ImageRecord(Path("slab.png"), 1 / res, "test", None), foam=foam,
                    cavity=np.zeros((h, w), bool), valid=np.ones((h, w), bool), res=float(res), base_row=float(h),
                    axis_col=w / 2, visible_depth_mm=9.0, full_view=True,
                    image_to_canvas=np.array([[1.0, 0, 0], [0, 1.0, 0]]))
    p = dict(cfg["cells"], work_px_per_mm=float(res))
    si = surface_intercepts(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), prep, p, None,
                            core_diameter_mm=0.8 / (np.pi / 4))
    flat = si["outer"]["flattening"]
    assert flat[0] > 2.0, flat                                   # 1.2 / 0.4 = 3 in the first 0.5 mm
    assert np.all((flat[3:] > 0.8) & (flat[3:] < 1.25)), flat    # round core from 1.5 mm
    f = surface_features(si)
    assert f["outer_layer_deep_mm"] == pytest.approx(0.4, abs=0.08)
    assert f["outer_flattened_depth_mm"] <= 1.75
