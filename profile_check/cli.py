"""Zelloscope profile check - NMC NOMAPACK U 0-24/77 cut-end images.

  1) Scale from a checkerboard target (or write the calibration file by hand: `px_per_mm: 26.5`):
       python -m profile_check calibrate target.png --inner-corners 9x6 --square-mm 5 -o calib.yaml

  2) Golden reference from known-good parts (images and/or directories):
       python -m profile_check build-ref --calibration calib.yaml -o ref.npz good/
     Add more good parts later without the original images:
       python -m profile_check build-ref --calibration calib.yaml --extend ref.npz -o ref_v2.npz new_good/

  3) Inspect parts against the spec + reference contour:
       python -m profile_check inspect --calibration calib.yaml --ref ref.npz -o results/ test/

The image scale is required on every run (--px-per-mm or --calibration); it is never defaulted.
Spec limits and tuning parameters live in the config file (--config).
`inspect` exits with status 1 if any image fails or cannot be processed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .calibration import load_calibration, px_per_mm_from_checkerboard, save_calibration
from .config import DEFAULT_CONFIG, load_config
from .pipeline import expand_images, inspect_part, load_part
from .reference import Reference, build_reference
from .report import draw_overlay, summary_line
from .segmentation import SegmentationError


def _scale(args) -> float:
    px_per_mm = args.px_per_mm if args.px_per_mm is not None else load_calibration(args.calibration)
    if px_per_mm <= 0:
        sys.exit("--px-per-mm must be > 0")
    return px_per_mm


def _contour_tol(args, cfg) -> float:
    return cfg.contour_tolerance_mm if args.contour_tol is None else args.contour_tol


def _images(args) -> list[Path]:
    files = expand_images(args.images)
    if not files and not getattr(args, "extend", None):
        sys.exit("no input images")
    return files


def cmd_build(args) -> int:
    cfg = load_config(args.config)
    px_per_mm, tol = _scale(args), _contour_tol(args, cfg)
    parts = []
    if args.extend:
        old = Reference.load(args.extend)
        parts.extend(old.parts)
        print(f"extending {args.extend}: {len(old.parts)} stored parts")
    for path in _images(args):
        part = load_part(path, px_per_mm, cfg)
        parts.append(part)
        print(f"loaded {part.name}" + (f" (open mouth bridged, {part.bridge_mm} mm)" if part.bridge_mm else ""))
    if not parts:
        sys.exit("no parts to build from")

    ref, report = build_reference(parts, cfg, px_per_mm, tol, run_loo=not args.no_loo)
    print("\nPer-sample measurements (mm) and deviation from the mean reference:")
    for s in report["samples"]:
        flag = "" if s["spec_ok"] else "  <-- OUTSIDE SPEC: " + ", ".join(
            k for k, v in s["spec"].items() if not v["ok"])
        print("  " + summary_line(s["name"], s["measurements"], s["deviation"], flag))
    if report["leave_one_out"]:
        print(f"\nLeave-one-out (reference from the others, contour tol {tol} mm):")
        for r in report["leave_one_out"]:
            print(f"  {r['name']}: {'PASS' if r['PASS'] else 'FAIL'}"
                  f"  p99 outer={r['deviation']['outer']['p99_abs']:.2f} cav={r['deviation']['cavity']['p99_abs']:.2f}"
                  + (f"  [{'; '.join(r['reasons'])}]" if r["reasons"] else ""))
    ref.save(args.output)
    report_path = Path(args.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    rm = report["reference"]["reference_measurements"]
    print(f"\nReference ({len(parts)} parts) saved to {args.output}, report {report_path}")
    print(f"Reference shape: L={rm['length']} W={rm['inner_width_max']} T={rm['base_thickness']} "
          f"outerW={rm['outer_width_max']}")
    return 0


def cmd_inspect(args) -> int:
    cfg = load_config(args.config)
    px_per_mm, tol = _scale(args), _contour_tol(args, cfg)
    ref = Reference.load(args.ref)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results, all_ok = [], True
    for path in _images(args):
        t0 = time.perf_counter()
        try:
            part = load_part(path, px_per_mm, cfg)
        except (SegmentationError, ValueError) as e:
            print(f"{path.stem}: ERROR {e}")
            results.append(dict(image=str(path), PASS=False, error=str(e)))
            all_ok = False
            continue
        res = inspect_part(part, ref.outer, ref.cavity, cfg, tol)
        res.data["runtime_s"] = round(time.perf_counter() - t0, 3)
        results.append(res.data)
        all_ok &= res.passed
        verdict = "PASS" if res.passed else "FAIL"
        draw_overlay(ref.outer, ref.cavity, res, outdir / f"{path.stem}_deviation.png", f"{path.stem}: {verdict}")
        print(summary_line(f"{path.stem}: {verdict}", res.data["measurements"], res.data["deviation"],
                           f"  [{'; '.join(res.reasons)}]" if res.reasons else ""))
    out = dict(reference=str(args.ref), config=cfg.source, px_per_mm=px_per_mm, contour_tol_mm=tol,
               results=results)
    (outdir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults: {outdir / 'results.json'}")
    return 0 if all_ok else 1


def cmd_calibrate(args) -> int:
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read {args.image}")
    cols, rows = (int(v) for v in args.inner_corners.lower().split("x"))
    cal = px_per_mm_from_checkerboard(img, (cols, rows), args.square_mm)
    save_calibration(args.output, cal["px_per_mm"], target=str(args.image),
                     inner_corners=args.inner_corners, square_mm=args.square_mm,
                     corner_spacing_spread_pct=cal["spread_pct"])
    print(f"px_per_mm = {cal['px_per_mm']:.4f} (corner spacing spread {cal['spread_pct']} %) -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m profile_check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build-ref", "inspect"):
        s = sub.add_parser(name)
        s.add_argument("images", nargs="*" if name == "build-ref" else "+", type=Path,
                       help="image files and/or directories")
        scale = s.add_mutually_exclusive_group(required=True)
        scale.add_argument("--px-per-mm", type=float, help="image scale from a calibration")
        scale.add_argument("--calibration", type=Path, help="calibration file (YAML/JSON with px_per_mm)")
        s.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"spec/config file (default {DEFAULT_CONFIG.name})")
        s.add_argument("--contour-tol", type=float, help="allowed p99 contour deviation in mm (default: from config)")
        if name == "build-ref":
            s.add_argument("-o", "--output", type=Path, default=Path("ref.npz"))
            s.add_argument("--extend", type=Path, help="existing reference whose stored parts are included")
            s.add_argument("--no-loo", action="store_true", help="skip the leave-one-out check")
        else:
            s.add_argument("--ref", type=Path, required=True)
            s.add_argument("-o", "--outdir", type=Path, default=Path("results"))
    c = sub.add_parser("calibrate", help="px/mm from a checkerboard target image")
    c.add_argument("image", type=Path)
    c.add_argument("--inner-corners", required=True, help="inner corners as COLSxROWS, e.g. 9x6")
    c.add_argument("--square-mm", type=float, required=True)
    c.add_argument("-o", "--output", type=Path, default=Path("calibration.yaml"))
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"build-ref": cmd_build, "inspect": cmd_inspect, "calibrate": cmd_calibrate}[args.cmd](args)
