"""Zelloscope shape-variation analysis - NMC NOMAPACK U 0-24/77 cut-end images.

  Analyse a set of parts (folders and/or files) and write an HTML report plus tables:
      python -m shape_variation analyze data/nmc_u/2026-06-04 data/nmc_u/2026-08-27 -o results/variation --cells

  Score new parts against the saved model (abnormal shape, new deformation, spec):
      python -m shape_variation score --model results/variation/model.npz new_images/ -o results/new.csv

  Check which images have a known scale (nothing else is done):
      python -m shape_variation scales data/nmc_u

Scale: from the ZelloscopeData PNG metadata or a scale.yaml (mm_per_px) in the image folder - never guessed;
images without one are listed and skipped. Settings: config/shape_variation.yaml (--config).
"""
import argparse
import json
import time
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .io import discover
from .pipeline import run, save_model, score_images
from .report import render_html
from .segment import SegmentationError


def cmd_analyze(args) -> int:
    cfg = load_config(args.config)
    if args.window is not None:
        cfg["window"]["depth_mm"] = args.window if args.window in ("auto", "full") else float(args.window)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    reference = None
    if args.reference:
        reference = [line.strip() for line in Path(args.reference).read_text().splitlines() if line.strip()]
    try:
        a = run(args.inputs, cfg, with_cells=args.cells, reference_names=reference,
                progress=(lambda s: None) if args.quiet else print)
    except SegmentationError as e:
        print(f"error: {e}")
        return 2
    a.table.to_csv(out / "parts.csv", index=False)
    if a.thickness is not None and len(a.thickness_grid):
        import pandas as pd
        pd.DataFrame(a.thickness, columns=[f"{g:+.1f}" for g in a.thickness_grid],
                     index=a.table["name"]).to_csv(out / "thickness.csv")
    (out / "modes.json").write_text(json.dumps(dict(modes=a.modes, stability=a.stability,
                                                    explained=a.model.explained.round(4).tolist(),
                                                    window_depth_mm=a.depth_mm), indent=2))
    save_model(a, out / "model.npz")
    (out / "report.html").write_text(render_html(a), encoding="utf-8")
    print(f"\n{len(a.shapes)} parts analysed, {len(a.skipped)} skipped, "
          f"{'full outline' if a.closed else f'window {a.depth_mm:.1f} mm from the base'}; "
          f"{a.model.n_keep} modes ({100 * a.model.explained[:a.model.n_keep].sum():.0f} % of the variation)")
    for md in a.modes[:3]:
        print(f"  mode {md['mode']} ({100 * md['explained']:.0f} %): {md['name']}")
    flagged = a.table[a.table["shape_abnormal"] | a.table["new_deformation"]]["name"].tolist()
    print(f"  abnormal shapes: {', '.join(flagged) if flagged else 'none'}")
    for name, reason in a.skipped:
        print(f"  skipped {name}: {reason}")
    print(f"Report: {out / 'report.html'}  ({time.perf_counter() - t0:.0f} s)")
    return 0


def cmd_score(args) -> int:
    cfg = load_config(args.config)
    df = score_images(args.model, args.inputs, cfg)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
    bad = 0
    for _, r in df.iterrows():
        if isinstance(r.get("error"), str):
            print(f"{r['name']}: ERROR {r['error']}")
            bad += 1
            continue
        flags = [f for f, on in (("abnormal shape", r["shape_abnormal"]), ("new deformation", r["new_deformation"]))
                 if on]
        flags += [k[:-3] + " out of spec" for k in r.index if k.endswith("_ok") and r[k] is False]
        bad += bool(flags)
        print(f"{r['name']}: {'FLAG ' + ', '.join(flags) if flags else 'ok'}  T2={r['T2']} (limit {r['T2_limit']})  "
              f"SPE={r['SPE']} (limit {r['SPE_limit']})")
    return 1 if bad else 0


def cmd_scales(args) -> int:
    cfg = load_config(args.config)
    missing = 0
    for r in discover(args.inputs, tuple(cfg["exclude"])):
        print(f"{r.path}: " + (f"{r.mm_per_px} mm/px ({r.scale_source})" if r.mm_per_px else f"UNKNOWN - {r.scale_source}"))
        missing += r.mm_per_px is None
    return 1 if missing else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m shape_variation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="shape model, thickness, drift, anomalies, report")
    a.add_argument("inputs", nargs="+", help="image folders and/or files")
    a.add_argument("-o", "--output", default="results/variation", help="output folder")
    a.add_argument("--cells", action="store_true", help="also segment cells (slower, ~5 s per image)")
    a.add_argument("--window", help="window depth from the base in mm, 'auto' or 'full' (default: config)")
    a.add_argument("--reference", help="text file with the names of approved parts (one per line) to use as the "
                                       "reference set for anomaly limits (default: all parts)")
    a.add_argument("--quiet", action="store_true")
    s = sub.add_parser("score", help="score new images against a saved model")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--model", required=True)
    s.add_argument("-o", "--output", help="CSV to write")
    c = sub.add_parser("scales", help="list the scale found for each image")
    c.add_argument("inputs", nargs="+")
    for p in (a, s, c):
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"analyze": cmd_analyze, "score": cmd_score, "scales": cmd_scales}[args.cmd](args)
