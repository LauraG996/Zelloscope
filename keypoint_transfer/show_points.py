"""Run the pipeline on chosen dataset images and save a picture of where the points were placed.

  python show_points.py --data DIR IMAGE [IMAGE ...]          by file name (as in transformed_points.csv)
  python show_points.py --data DIR --shape 7 --limit 5        the first 5 images of shape 7
  python show_points.py --data DIR --worst 10                 the 10 worst images of results/per_image.csv
  ... --open                                                  also open the pictures (Windows)

Each picture shows the reference with its points, the new image with the placed points (red) and the ground truth
(green), and zoomed full-resolution tiles around every point with its error in px. Saved to ./plots (or --out).
DIR is the folder holding dataset_out/ and 'New folder'/ (default: the parent folder, as in run_evaluation.py).
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("images", nargs="*", help="image file names from transformed_points.csv")
ap.add_argument("--data", default=os.path.dirname(HERE), help="folder holding dataset_out/ and 'New folder'/")
ap.add_argument("--out", default=os.path.join(HERE, "plots"), help="where to save the pictures")
ap.add_argument("--shape", help="pick images of this shape")
ap.add_argument("--limit", type=int, default=5, help="with --shape: how many images (default 5)")
ap.add_argument("--worst", type=int, help="the N images with the largest mean error in results/per_image.csv")
ap.add_argument("--results", default=os.path.join(HERE, "results", "per_image.csv"), help="for --worst")
ap.add_argument("--open", action="store_true", help="open each picture after saving it")
a = ap.parse_args()

import cv2
import torch

import keypoint_transfer as kt

refs = {r["base_image"]: r for r in csv.DictReader(open(os.path.join(a.data, "dataset_out", "reference_points.csv")))}
rows = {r["image"]: r for r in csv.DictReader(open(os.path.join(a.data, "dataset_out", "transformed_points.csv")))}
pts = lambda r: np.array([[float(r[f"D{d}_x{e}"]), float(r[f"D{d}_y{e}"])] for d in (1, 2, 3) for e in (1, 2)])
NAMES = [f"D{d}{'ab'[e]}" for d in (1, 2, 3) for e in (0, 1)]
PAIRS = [(0, 1), (2, 3), (4, 5)]

names = list(a.images)
if a.shape:
    names += sorted(n for n, r in rows.items() if r["shape"] == str(a.shape))[:a.limit]
if a.worst:
    res = [r for r in csv.DictReader(open(a.results)) if r["failed"] == "0"]
    names += [r["image"] for r in sorted(res, key=lambda r: -float(r["mean_px"]))[:a.worst]]
if not names:
    sys.exit("no images chosen: give image names, --shape N or --worst N")
missing = [n for n in names if n not in rows]
if missing:
    sys.exit("not in transformed_points.csv: " + ", ".join(missing))

os.makedirs(a.out, exist_ok=True)
for name in dict.fromkeys(names):                            # keeps order, drops duplicates
    r = rows[name]
    ref = kt.load_gray(os.path.join(a.data, "New folder", r["base_image"]))
    new = kt.load_gray(os.path.join(a.data, "dataset_out", "raw", name))
    torch.manual_seed(0)                                     # same sampling as run_evaluation.py
    res = kt.map_points_oriented(ref, new, pts(refs[r["base_image"]]), verbose=False)
    gt = pts(r)
    err = np.linalg.norm(res["new_pts"] - gt, axis=1)
    title = (f"{name}  shape {r['shape']}  ref {r['base_image']}  |  error mean {err.mean():.2f}px  max {err.max():.2f}px"
             f"  |  {res['orientation']}  {res['seconds']:.1f}s")
    img = kt.draw_overlay(ref, new, pts(refs[r["base_image"]]), res, gt=gt, pairs=PAIRS, names=NAMES, title=title)
    path = os.path.join(a.out, os.path.splitext(name)[0] + "_points.png")
    cv2.imwrite(path, img)
    print(f"{name}: mean {err.mean():.2f}px  max {err.max():.2f}px  ->  {path}", flush=True)
    for i, f in enumerate(res["flags"]):
        if f:
            print(f"    {NAMES[i]}: {', '.join(f)}")
    if a.open and os.name == "nt":
        os.startfile(path)
