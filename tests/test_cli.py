import json
from pathlib import Path

import cv2
import pytest

import synthetic as syn
from conftest import SAMPLE_SHAPE
from profile_check.calibration import save_calibration
from profile_check.cli import main


def test_scale_is_required(tmp_path):
    with pytest.raises(SystemExit):
        main(["inspect", "--ref", str(tmp_path / "ref.npz"), str(tmp_path)])


def test_build_ref_and_inspect(tmp_path, good_images, capsys):
    good = tmp_path / "good"
    good.mkdir()
    for i, img in enumerate(good_images[:3]):
        cv2.imwrite(str(good / f"good{i}.png"), img)
    test = tmp_path / "test"
    test.mkdir()
    cv2.imwrite(str(test / "ok.png"), syn.render(angle_deg=140, seed=11, **SAMPLE_SHAPE))
    cv2.imwrite(str(test / "narrow.png"), syn.render(angle_deg=80, seed=12, sx=0.93, **SAMPLE_SHAPE))
    cal = tmp_path / "calibration.yaml"
    save_calibration(cal, syn.PX_PER_MM)

    ref = tmp_path / "ref.npz"
    assert main(["build-ref", "--calibration", str(cal), "-o", str(ref), str(good)]) == 0
    out = capsys.readouterr().out
    assert "Leave-one-out" in out and ref.exists() and ref.with_suffix(".report.json").exists()

    # Extend with one more part, without the original images.
    cv2.imwrite(str(tmp_path / "more.png"), good_images[3])
    ref2 = tmp_path / "ref2.npz"
    assert main(["build-ref", "--px-per-mm", str(syn.PX_PER_MM), "--extend", str(ref), "--no-loo",
                 "-o", str(ref2), str(tmp_path / "more.png")]) == 0
    assert json.loads(ref2.with_suffix(".report.json").read_text())["reference"]["n_parts"] == 4

    res_dir = tmp_path / "results"
    assert main(["inspect", "--calibration", str(cal), "--ref", str(ref2), "-o", str(res_dir), str(test)]) == 1
    results = {Path(r["image"]).name: r for r in json.loads((res_dir / "results.json").read_text())["results"]}
    assert results["ok.png"]["PASS"] is True
    assert results["narrow.png"]["PASS"] is False and results["narrow.png"]["reasons"]
    assert (res_dir / "ok_deviation.png").exists() and (res_dir / "narrow_deviation.png").exists()


def test_calibrate_command(tmp_path):
    from test_calibration import checkerboard
    img = tmp_path / "target.png"
    cv2.imwrite(str(img), checkerboard(21.0))
    out = tmp_path / "cal.yaml"
    assert main(["calibrate", str(img), "--inner-corners", "9x6", "--square-mm", "5", "-o", str(out)]) == 0
    from profile_check.calibration import load_calibration
    assert load_calibration(out) == pytest.approx(21.0, rel=0.003)
