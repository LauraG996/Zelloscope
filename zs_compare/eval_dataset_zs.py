"""Run zs-auto-measure (gui.measure_sample, the GUI's own pipeline) on the same dataset_out as
eval_dataset.py and write its errors in the same columns, so the two methods can be compared
with compare_methods.py.

    python eval_dataset_zs.py --code C:/path/to/zs-auto-measure [--preset NAME] [--no-snap]
    python eval_dataset_zs.py --summary-only

Per base image, a ZS reference is built from reference_points.csv: the 6 points of D1..D3 become
landmarks (snapped onto the nearest contour, as a click in the GUI is; --no-snap keeps them
as they are) and (D1_1, D1_2), (D2_1, D2_2), (D3_1, D3_2) the straight pairs.  Every image of
transformed_points.csv is then measured against it and compared with its ground truth.

Writes (resumable - images already in the csv are skipped):
  eval_results/per_image_zs.csv   same columns as per_image.csv, plus
        accepted   1 = ZS accepted the sample (its match decision), 0 = rejected
        reasons    why it was rejected
        ref_snap_px  largest distance a reference point moved when snapped onto its contour
      failed = 1 when ZS gave no landmarks at all (exception, or the solidity / shape early
      reject); a rejected sample that still got landmarks has its errors filled in, with
      accepted = 0, so compare_methods.py can count it either way.
  eval_results/percentiles_zs.csv same layout as percentiles.csv (accepted samples only)
Errors are in pixels of the transformed image, exactly as in eval_dataset.py.

The ZS code is imported from --code (a checkout of zs-auto-measure); its map_settings.json and
presets.json are read from there and never written (presets.json is loaded from a temp copy).
"""
import argparse, csv, json, os, shutil, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # outputs go here
DATA = os.path.dirname(HERE)                                # dataset_out/ and "New folder"/ live in the parent folder
OUT = os.path.join(HERE, "eval_results")
PER = os.path.join(OUT, "per_image_zs.csv")
PCTS = [5, 25, 50, 75, 90, 95, 99]
HEAD = (["image", "shape", "keypoint_source", "mirrored", "seconds", "failed", "n_flagged"]
        + [f"D{d}_e{e}_err" for d in (1, 2, 3) for e in (1, 2)] + [f"D{d}_len_err" for d in (1, 2, 3)]
        + [f"D{d}_e{e}_rel_pct" for d in (1, 2, 3) for e in (1, 2)] + [f"D{d}_len_rel_pct" for d in (1, 2, 3)]
        + ["orientation", "accepted", "reasons", "ref_snap_px"])
PAIRS = [(0, 1), (2, 3), (4, 5)]


def pts(row):
    return np.array([[float(row[f"D{d}_x{e}"]), float(row[f"D{d}_y{e}"])] for d in (1, 2, 3) for e in (1, 2)])


def ln(p):
    return np.linalg.norm(p[0::2] - p[1::2], axis=1)


def load_zs(code, preset_name):
    """Import zs-auto-measure from `code`; returns (sample_core, mapping, settings)."""
    code = os.path.abspath(code)
    sys.path.insert(0, code)
    os.chdir(code)                          # its settings / theme paths are relative to the repo root
    import sample_core, mapping, presets
    settings = sample_core.load_settings(os.path.join(code, "map_settings.json"))
    tmp = os.path.join(tempfile.mkdtemp(), "presets.json")
    shutil.copy(os.path.join(code, "presets.json"), tmp)   # load_presets may write defaults into the file
    plist, selected = presets.load_presets(tmp)
    preset = presets.find_preset(plist, preset_name) if preset_name else plist[selected]
    if preset is None:
        sys.exit(f"preset {preset_name!r} not in {code}/presets.json: {[p['name'] for p in plist]}")
    presets.apply_preset(settings, preset)
    print(f"zs-auto-measure: {code}, preset {preset['name']!r}", flush=True)
    return sample_core, mapping, settings


def build_reference(sc, mapping, settings, image, p0, snap):
    """Segment the reference like the GUI and turn the 6 points into landmarks (x, y, contour_idx)."""
    _, contours, hierarchy = sc.segment_image(image, settings["segmentation_settings"])
    landmarks, moved = [], 0.0
    for p in p0:
        q, dist, idx = mapping.nearest_contour_point(p, contours)
        if snap:
            landmarks.append((q[0], q[1], idx))
            moved = max(moved, dist)
        else:
            landmarks.append((float(p[0]), float(p[1]), idx))
    return contours, hierarchy, landmarks, moved


def run(code, preset_name, snap):
    import cv2
    data = os.path.abspath(DATA)
    sc, mapping, settings = load_zs(code, preset_name)
    refs = {r["base_image"]: r for r in csv.DictReader(open(os.path.join(data, "dataset_out/reference_points.csv")))}
    rows = list(csv.DictReader(open(os.path.join(data, "dataset_out/transformed_points.csv"))))
    os.makedirs(OUT, exist_ok=True)
    done = set(r["image"] for r in csv.DictReader(open(PER))) if os.path.exists(PER) else set()
    new_file = not os.path.exists(PER)
    f = open(PER, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(HEAD)
    cache = {}
    for i, r in enumerate(rows):
        if r["image"] in done:
            continue
        base = r["base_image"]
        if base not in cache:
            img = cv2.imread(os.path.join(data, "New folder", base), cv2.IMREAD_COLOR)
            cache = {base: (img.shape,) + build_reference(sc, mapping, settings, img, pts(refs[base]), snap)}
        ref_shape, ref_contours, ref_hier, ref_lm, moved = cache[base]
        gt = pts(r)
        mir = int("flip" in r["ops"])
        new = cv2.imread(os.path.join(data, "dataset_out/raw", r["image"]), cv2.IMREAD_COLOR)
        t = time.time()
        try:
            s = sc.measure_sample(new, settings, ref_contours, ref_hier, ref_lm, PAIRS,
                                  perp_pair_mode=[False] * 3, pix2mm=None, ref_image_shape=ref_shape)
            res = s["result"]
            if res is None:
                raise RuntimeError("; ".join(s["reasons"]) or "no mapping")
            pred = np.array([(x, y) for x, y, _ in res["refined_landmarks"]], float)
            e = np.linalg.norm(pred - gt, axis=1)
            le = np.abs(np.array([d["pixel_distance"] for d in res["distances"]]) - ln(gt))
            ti = res["transform_info"]
            ang = ti.get("angle")
            orient = (f"rot={ang:.1f}" if isinstance(ang, (int, float)) else "rot=?") + \
                     (" mirror" if ti.get("mirrored") else "")
            w.writerow([r["image"], r["shape"], r["keypoint_source"], mir, f"{time.time() - t:.1f}", 0,
                        len(s["warnings"])] + [f"{v:.2f}" for v in e] + [f"{v:.2f}" for v in le] +
                       [f"{v:.3f}" for v in 100 * e / np.repeat(ln(gt), 2)] + [f"{v:.3f}" for v in 100 * le / ln(gt)] +
                       [orient, int(bool(s["accepted"])), "; ".join(s["reasons"]), f"{moved:.1f}"])
        except Exception as ex:
            print(r["image"], "FAILED", ex, flush=True)
            w.writerow([r["image"], r["shape"], r["keypoint_source"], mir, f"{time.time() - t:.1f}", 1, ""]
                       + [""] * 19 + [0, str(ex), f"{moved:.1f}"])
        f.flush()
        print(f"[{i + 1}/{len(rows)}] {r['image']}", flush=True)
    f.close()


def summarize():
    rows = [r for r in csv.DictReader(open(PER)) if r["failed"] == "0" and r["accepted"] == "1"]
    out = open(os.path.join(OUT, "percentiles_zs.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["subset", "distance", "metric", "n"] + [f"p{p}" for p in PCTS])
    subsets = {"all": rows, "non-mirrored": [r for r in rows if r["mirrored"] == "0"],
               "mirrored": [r for r in rows if r["mirrored"] == "1"]}
    for sname, rs in subsets.items():
        for d in (1, 2, 3, "all"):
            ks = (1, 2, 3) if d == "all" else (d,)
            for metric, cols in (("endpoint_err_px", [f"D{k}_e{e}_err" for k in ks for e in (1, 2)]),
                                 ("length_err_px", [f"D{k}_len_err" for k in ks]),
                                 ("endpoint_err_rel_pct", [f"D{k}_e{e}_rel_pct" for k in ks for e in (1, 2)]),
                                 ("length_err_rel_pct", [f"D{k}_len_rel_pct" for k in ks])):
                v = np.array([float(r[c]) for r in rs for c in cols])
                if len(v):
                    w.writerow([sname, f"D{d}" if d != "all" else "all", metric, len(v)] +
                               [f"{np.percentile(v, p):.2f}" for p in PCTS])
    out.close()
    print(open(os.path.join(OUT, "percentiles_zs.csv")).read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", help="zs-auto-measure checkout")
    ap.add_argument("--preset", help="preset of its presets.json (default: the selected one)")
    ap.add_argument("--no-snap", action="store_true", help="don't snap reference points onto the contour")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        if not a.code:
            ap.error("--code is required (path of the zs-auto-measure checkout)")
        run(a.code, a.preset, not a.no_snap)
    summarize()
