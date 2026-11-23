"""Evaluate the key-point transfer pipeline on the generated dataset.

  python run_evaluation.py                      all shapes, all images
  python run_evaluation.py --shapes 8 3         only these shapes
  python run_evaluation.py --limit 10           first 10 images per run (smoke test)

Images are processed grouped by reference image, so each reference's RoMa encoder features are computed once and
reused for all of its images. The run is resumable: finished images in <out>/per_image.csv are skipped.

Writes   <out>/per_image.csv      per image: time, orientation method, relative errors of D1..D3 endpoints and lengths
         <out>/percentiles.csv    percentiles of the relative errors per distance (D1, D2, D3, all)
Relative errors are in % of the true length of that distance (endpoint error / length, |length error| / length).
Data layout expected under --data:  dataset_out/{reference_points.csv, transformed_points.csv, raw/*.png}
                                    New folder/<base_image>.png   (the full-resolution reference images)
"""
import argparse
import csv
import ctypes
import os
import time

import numpy as np
import torch

import keypoint_transfer as kt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default=os.path.dirname(HERE), help="folder holding dataset_out/ and 'New folder'/ (default: parent folder)")
ap.add_argument("--out", default=os.path.join(HERE, "results"), help="output folder")
ap.add_argument("--shapes", nargs="*", default=None, help="shape numbers to run (default: all)")
ap.add_argument("--limit", type=int, default=None, help="process at most this many images (smoke test)")
a = ap.parse_args()

if os.name == "nt":                                         # no idle sleep while this process runs (a sleeping laptop stalls the run)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
if kt.BACKEND == "onnx":
    import roma_onnx
    roma_onnx.CACHE_MAX = 64                                # keep the features of every reference for the whole run

os.makedirs(a.out, exist_ok=True)
PER = os.path.join(a.out, "per_image.csv")
refs = {r["base_image"]: r for r in csv.DictReader(open(os.path.join(a.data, "dataset_out", "reference_points.csv")))}
rows = list(csv.DictReader(open(os.path.join(a.data, "dataset_out", "transformed_points.csv"))))
if a.shapes: rows = [r for r in rows if r["shape"] in a.shapes]
rows.sort(key=lambda r: (r["base_image"], r["image"]))      # all images of one reference together
if a.limit: rows = rows[:a.limit]
pts = lambda r: np.array([[float(r[f"D{d}_x{e}"]), float(r[f"D{d}_y{e}"])] for d in (1, 2, 3) for e in (1, 2)])
head = (["image", "shape", "base_image", "keypoint_source", "first_of_reference", "seconds", "orientation", "failed"]
        + [f"D{d}_e{e}_rel_pct" for d in (1, 2, 3) for e in (1, 2)] + [f"D{d}_len_rel_pct" for d in (1, 2, 3)] + ["mean_px"])
done = {r["image"] for r in csv.DictReader(open(PER))} if os.path.exists(PER) else set()
f = open(PER, "a", newline="")
w = csv.writer(f)
if not done: w.writerow(head)
print(f"{len(rows)} images, {len({r['base_image'] for r in rows})} references" + (f", {len(done)} already done" if done else ""), flush=True)

# load the models once (and export the ONNX files on the very first run); not part of the timing
r0 = rows[0]
kt.map_points_oriented(kt.load_gray(os.path.join(a.data, "New folder", r0["base_image"])),
                       kt.load_gray(os.path.join(a.data, "dataset_out", "raw", r0["image"])), pts(refs[r0["base_image"]]), verbose=False)
if kt.BACKEND == "onnx":
    roma_onnx._CACHE.clear()
    roma_onnx.STATS.update(cache_hits=0, cache_misses=0)

cur = ref = None
seen = set()
t_start = time.perf_counter()
for i, r in enumerate(rows):
    if r["image"] in done:
        continue
    b = r["base_image"]
    if b != cur:
        ref, cur = kt.load_gray(os.path.join(a.data, "New folder", b)), b
    new = kt.load_gray(os.path.join(a.data, "dataset_out", "raw", r["image"]))
    first = b not in seen
    seen.add(b)
    torch.manual_seed(0)
    try:
        res = kt.map_points_oriented(ref, new, pts(refs[b]), verbose=False)
        gt = pts(r)
        e = np.linalg.norm(res["new_pts"] - gt, axis=1)
        ln = lambda p: np.linalg.norm(p[0::2] - p[1::2], axis=1)
        L = ln(gt)
        w.writerow([r["image"], r["shape"], b, r["keypoint_source"], int(first), f"{res['seconds']:.2f}", res["orientation"], 0]
                   + [f"{v:.3f}" for v in 100 * e / np.repeat(L, 2)] + [f"{v:.3f}" for v in 100 * np.abs(ln(res["new_pts"]) - L) / L] + [f"{e.mean():.2f}"])
    except Exception as ex:
        print(r["image"], "FAILED:", ex, flush=True)
        w.writerow([r["image"], r["shape"], b, r["keypoint_source"], int(first), "", "", 1] + [""] * 10)
    f.flush()
    if (i + 1) % 20 == 0:
        print(f"[{i + 1}/{len(rows)}] {r['image']}", flush=True)
wall = time.perf_counter() - t_start
f.close()

# ---- summary ----
R = [r for r in csv.DictReader(open(PER))]
ok = [r for r in R if r["failed"] == "0"]
sec = np.array([float(r["seconds"]) for r in ok])
print(f"\n{len(ok)} images ok, {len(R) - len(ok)} failed;  wall clock of this run {wall / 60:.1f} min")
print(f"time per image: median {np.median(sec):.2f}s  mean {sec.mean():.2f}s  p90 {np.percentile(sec, 90):.2f}s  (>60 s: {(sec > 60).sum()} images)")
PCTS = [5, 25, 50, 75, 90, 95, 99]
col = lambda names, rr: np.array([float(r[n]) for r in rr for n in names])
sets = {"D1": (["D1_e1_rel_pct", "D1_e2_rel_pct"], ["D1_len_rel_pct"]), "D2": (["D2_e1_rel_pct", "D2_e2_rel_pct"], ["D2_len_rel_pct"]),
        "D3": (["D3_e1_rel_pct", "D3_e2_rel_pct"], ["D3_len_rel_pct"])}
sets["all"] = (sum((v[0] for v in sets.values()), []), sum((v[1] for v in sets.values()), []))
with open(os.path.join(a.out, "percentiles.csv"), "w", newline="") as pf:
    pw = csv.writer(pf)
    pw.writerow(["distance", "measure", "mean"] + [f"p{p}" for p in PCTS])
    print("\nrelative error [%]            mean " + " ".join(f"{'p' + str(p):>7}" for p in PCTS))
    for k, (ep, ln_) in sets.items():
        for measure, nm in (("endpoint", ep), ("length", ln_)):
            v = col(nm, ok)
            pw.writerow([k, measure, f"{v.mean():.3f}"] + [f"{np.percentile(v, p):.3f}" for p in PCTS])
            print(f"  {k:>3} {measure:<9}        {v.mean():7.3f} " + " ".join(f"{np.percentile(v, p):7.3f}" for p in PCTS))
print("results in", a.out)
