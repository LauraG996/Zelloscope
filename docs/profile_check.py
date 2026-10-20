#!/usr/bin/env python3
"""
Zelloscope profile check - NMC NOMAPACK U 0-24/77 (cut-end images)

Two steps:

  1) Build a golden reference from known-good parts (5 is fine; more cuts/placements per part is better):
       python profile_check.py build-ref --px-per-mm 26.5 -o ref.npz good_*.jpg

  2) Inspect new images against the spec + the reference contour:
       python profile_check.py inspect --px-per-mm 26.5 --ref ref.npz -o results/ test_*.jpg

What it does
  - Segments the blue foam (HSV threshold) -> outer contour + cavity contour.
    If the legs are open (cavity connects to background) the mouth is bridged automatically.
  - Deskews: principal axis vertical, base at the bottom (cavity is widest towards the base).
  - Aligns each part to the reference (rotation, shift, and mirror - the cut face can be seen from either end).
  - Measures in mm: length, max inner width, base thickness, left/right wall at widest cavity row, outer width,
    tip height difference.
  - Contour deviation: distance from every point on the part's outer and cavity contours to the reference
    contours (+ = part material/cavity bigger than reference, - = smaller).
  - PASS if all spec dimensions are within limits AND the 99th-percentile contour deviation <= --contour-tol.

Requirements: python3, opencv-python, numpy
Scale: --px-per-mm must come from a calibration target imaged with the same camera/setup.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

# ---------------- Spec (mm): (min, max) -----------------
SPEC = {
    "inner_width_max": (24.0, 27.0),   # width 24 -0/+3
    "base_thickness": (11.0, 13.0),    # thickness 12 +-1
    "length": (76.0, 80.0),            # length 77 -1/+3
}
CANVAS_RES = 10.0      # px per mm in the canonical frame (0.1 mm grid)
CANVAS_MM = 130.0      # canonical canvas size (mm)
N = int(CANVAS_MM * CANVAS_RES)


# ---------------- Segmentation -----------------
def segment(img, hue=(90, 130), sat_min=80, val_min=90):
    """Return (outer_contour_px, cavity_contour_px) as Nx2 float arrays in image pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    foam = ((hsv[..., 0] >= hue[0]) & (hsv[..., 0] <= hue[1]) &
            (hsv[..., 1] >= sat_min) & (hsv[..., 2] >= val_min)).astype(np.uint8) * 255
    k = max(3, int(round(min(img.shape[:2]) * 0.004)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    foam = cv2.morphologyEx(foam, cv2.MORPH_CLOSE, ker)
    foam = cv2.morphologyEx(foam, cv2.MORPH_OPEN, ker)

    # keep largest blob
    n, lab, stats, _ = cv2.connectedComponentsWithStats(foam)
    if n < 2:
        raise RuntimeError("no foam found - check lighting / HSV thresholds")
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    foam = (lab == big).astype(np.uint8) * 255

    outer_cs, _ = cv2.findContours(foam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer = max(outer_cs, key=cv2.contourArea)
    outer_area = cv2.contourArea(outer)

    # cavity = largest hole; if legs are open, bridge the mouth with growing closing
    for kk in [0, 15, 31, 61, 101, 151]:
        m = foam if kk == 0 else cv2.morphologyEx(
            foam, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        holes = [cs[i] for i in range(len(cs)) if hier[0][i][3] != -1]
        if holes:
            h = max(holes, key=cv2.contourArea)
            if cv2.contourArea(h) > 0.02 * outer_area:
                hole = np.zeros_like(foam)
                cv2.drawContours(hole, [h], -1, 255, -1)
                hole = cv2.bitwise_and(hole, cv2.bitwise_not(foam))  # only truly empty pixels
                cc, _ = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cav = max(cc, key=cv2.contourArea)
                return outer.reshape(-1, 2).astype(float), cav.reshape(-1, 2).astype(float), kk
    raise RuntimeError("no cavity found")


# ---------------- Canonical frame -----------------
def to_canonical(outer_px, cav_px, px_per_mm):
    """mm coordinates, origin = outer centroid, axis vertical, base at +y (bottom)."""
    o = outer_px / px_per_mm
    c = cav_px / px_per_mm
    M = cv2.moments(o.astype(np.float32))
    cen = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    # PCA on filled area via second moments
    mu20, mu02, mu11 = M["mu20"] / M["m00"], M["mu02"] / M["m00"], M["mu11"] / M["m00"]
    w, v = np.linalg.eigh(np.array([[mu20, mu11], [mu11, mu02]]))
    ax = v[:, 1]                 # major axis
    perp = np.array([-ax[1], ax[0]])
    R = np.stack([perp, ax], 1)  # columns -> new x, new y
    o2 = (o - cen) @ R
    c2 = (c - cen) @ R
    # base at bottom: cavity area centroid lies towards its wide (base) end
    Mc = cv2.moments(c2.astype(np.float32))
    cy = Mc["m01"] / Mc["m00"]
    if cy < (c2[:, 1].min() + c2[:, 1].max()) / 2:
        o2[:, 1] *= -1
        c2[:, 1] *= -1
        o2[:, 0] *= -1  # keep it a rotation (not a mirror)
        c2[:, 0] *= -1
    return o2, c2


def transform(pts, ang_deg=0.0, dx=0.0, dy=0.0, mirror=False):
    p = pts.copy()
    if mirror:
        p[:, 0] *= -1
    a = np.radians(ang_deg)
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return p @ Rm.T + [dx, dy]


def to_canvas(pts_mm):
    return np.round(pts_mm * CANVAS_RES + N / 2).astype(np.int32)


def render(outer_mm, cav_mm):
    om = np.zeros((N, N), np.uint8)
    cm = np.zeros((N, N), np.uint8)
    cv2.fillPoly(om, [to_canvas(outer_mm)], 1)
    cv2.fillPoly(cm, [to_canvas(cav_mm)], 1)
    return om, cm


def iou(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return i / u if u else 0.0


def align(outer, cav, ref_outer_mask, ref_cav_mask):
    """Search mirror / rotation / shift that best matches the reference foam + cavity masks."""
    ref_foam = ref_outer_mask & (1 - ref_cav_mask)

    def score(p):
        om, cm = render(transform(outer, *p[:3], p[3]), transform(cav, *p[:3], p[3]))
        return iou(om & (1 - cm), ref_foam) + 0.5 * iou(cm, ref_cav_mask)

    best = None
    for mir in (False, True):
        p = [0.0, 0.0, 0.0, mir]
        s = score(p)
        for step_a, step_t in [(4.0, 2.0), (1.0, 0.5), (0.25, 0.1)]:
            improved = True
            while improved:
                improved = False
                for i, st in [(0, step_a), (1, step_t), (2, step_t)]:
                    for sgn in (-1, 1):
                        q = list(p)
                        q[i] += sgn * st
                        sq = score(q)
                        if sq > s + 1e-6:
                            p, s, improved = q, sq, True
        if best is None or s > best[1]:
            best = (p, s)
    p, s = best
    return transform(outer, *p[:3], p[3]), transform(cav, *p[:3], p[3]), dict(
        rot_deg=p[0], dx_mm=p[1], dy_mm=p[2], mirrored=bool(p[3]), match_score=float(s))


# ---------------- Measurements -----------------
def measure(outer_mm, cav_mm):
    om, cm = render(outer_mm, cav_mm)
    rows_o = np.where(om.any(1))[0]
    length = (rows_o.max() - rows_o.min() + 1) / CANVAS_RES

    def extent(mask, r):
        xs = np.where(mask[r])[0]
        return (xs.min(), xs.max()) if len(xs) else None

    outer_w = max((e[1] - e[0] + 1) for r in rows_o if (e := extent(om, r))) / CANVAS_RES
    rows_c = np.where(cm.any(1))[0]
    widths = {r: extent(cm, r)[1] - extent(cm, r)[0] + 1 for r in rows_c}
    r_max = max(widths, key=widths.get)
    inner_w = widths[r_max] / CANVAS_RES

    cl, cr = extent(cm, r_max)
    ol, orr = extent(om, r_max)
    wall_l = (cl - ol) / CANVAS_RES
    wall_r = (orr - cr) / CANVAS_RES

    # base thickness: centre column of the widest cavity row, cavity bottom -> outer bottom (band of +-1 mm, median)
    xc = (cl + cr) // 2
    vals = []
    for x in range(xc - int(CANVAS_RES), xc + int(CANVAS_RES) + 1):
        yc = np.where(cm[:, x])[0]
        yo = np.where(om[:, x])[0]
        if len(yc) and len(yo):
            vals.append(yo.max() - yc.max())
    base_t = float(np.median(vals)) / CANVAS_RES if vals else float("nan")

    # tip height difference (left vs right half top)
    xmid = (cl + cr) // 2
    top_l = np.where(om[:, :xmid].any(1))[0].min()
    top_r = np.where(om[:, xmid:].any(1))[0].min()
    tip_diff = abs(top_l - top_r) / CANVAS_RES

    # inner width profile every 5 mm from the base
    prof = []
    ybot = rows_c.max()
    for d in np.arange(0, (rows_c.max() - rows_c.min()) / CANVAS_RES, 5.0):
        r = int(ybot - d * CANVAS_RES)
        if r in widths:
            prof.append((round(float(d), 1), round(widths[r] / CANVAS_RES, 2)))

    return dict(length=round(float(length), 2), inner_width_max=round(float(inner_w), 2), base_thickness=round(base_t, 2),
                outer_width_max=round(float(outer_w), 2), wall_left=round(float(wall_l), 2), wall_right=round(float(wall_r), 2),
                wall_asym=round(float(abs(wall_l - wall_r)), 2), tip_height_diff=round(float(tip_diff), 2),
                inner_width_profile_from_base=prof)


def spec_check(m):
    res = {}
    for k, (lo, hi) in SPEC.items():
        res[k] = dict(value=m[k], min=lo, max=hi, ok=bool(lo <= m[k] <= hi))
    return res


# ---------------- Contour deviation -----------------
def boundary_dist(mask):
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return cv2.distanceTransform((1 - edge).astype(np.uint8), cv2.DIST_L2, 5) / CANVAS_RES


def deviation(outer_mm, cav_mm, ref_om, ref_cm):
    out = {}
    pts_all = {}
    for name, pts, refm in [("outer", outer_mm, ref_om), ("cavity", cav_mm, ref_cm)]:
        d = boundary_dist(refm)
        c = to_canvas(pts)
        c = c[(c[:, 0] >= 0) & (c[:, 0] < N) & (c[:, 1] >= 0) & (c[:, 1] < N)]
        dist = d[c[:, 1], c[:, 0]]
        outside = refm[c[:, 1], c[:, 0]] == 0
        signed = np.where(outside, dist, -dist)
        out[name] = dict(max_abs=round(float(np.abs(signed).max()), 2),
                         p99_abs=round(float(np.percentile(np.abs(signed), 99)), 2),
                         rms=round(float(np.sqrt((signed ** 2).mean())), 2),
                         mean_signed=round(float(signed.mean()), 2))
        pts_all[name] = (c, signed)
    return out, pts_all


def overlay(ref_om, ref_cm, pts_all, tol, path, title):
    img = np.full((N, N, 3), 30, np.uint8)
    foam = (ref_om & (1 - ref_cm)).astype(bool)
    img[foam] = (90, 70, 40)
    for m in (ref_om, ref_cm):
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, (255, 255, 255), 2)
    for c, s in pts_all.values():
        t = np.clip(np.abs(s) / tol, 0, 1)
        for (x, y), tt in zip(c[::2], t[::2]):
            col = (0, int(255 * (1 - tt)), int(255 * tt)) if tt < 1 else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 3, col, -1)
    cv2.putText(img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(img, f"white = reference, dots green->red = 0->{tol} mm deviation",
                (20, N - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    cv2.imwrite(path, img)


# ---------------- Pipeline -----------------
def load_canonical(path, args):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    o, c, bridged = segment(img, (args.hue_min, args.hue_max), args.sat_min, args.val_min)
    o2, c2 = to_canonical(o, c, args.px_per_mm)
    return o2, c2, bridged


def mean_masks(shapes, ref=None, iters=2):
    if ref is None:
        ref = render(*shapes[0])
    for _ in range(iters):
        acc_o = np.zeros((N, N), np.float32)
        acc_c = np.zeros((N, N), np.float32)
        aligned = []
        for o, c in shapes:
            ao, ac, _ = align(o, c, *ref)
            om, cm = render(ao, ac)
            acc_o += om
            acc_c += cm
            aligned.append((ao, ac))
        ref = ((acc_o / len(shapes) >= 0.5).astype(np.uint8), (acc_c / len(shapes) >= 0.5).astype(np.uint8))
    return ref, aligned


def cmd_build(args):
    shapes, names = [], []
    for p in args.images:
        o, c, br = load_canonical(p, args)
        shapes.append((o, c))
        names.append(os.path.basename(p))
        print(f"loaded {names[-1]}" + (" (open mouth bridged)" if br else ""))
    (rom, rcm), aligned = mean_masks(shapes)
    print("\nPer-sample measurements (mm) and deviation from the mean reference:")
    samples = []
    for n, (o, c) in zip(names, aligned):
        m = measure(o, c)
        sc = spec_check(m)
        dev, _ = deviation(o, c, rom, rcm)
        flag = "" if all(v["ok"] for v in sc.values()) else "  <-- outside spec!"
        print(f"  {n}: L={m['length']} W={m['inner_width_max']} T={m['base_thickness']} "
              f"walls={m['wall_left']}/{m['wall_right']}  dev p99 outer/cav="
              f"{dev['outer']['p99_abs']}/{dev['cavity']['p99_abs']}{flag}")
        samples.append(dict(name=n, measurements=m, spec=sc, deviation=dev))

    loo = []
    if len(shapes) >= 3:
        print("\nLeave-one-out (reference from the others, test the left-out part):")
        for i in range(len(shapes)):
            others = [s for j, s in enumerate(shapes) if j != i]
            (lom, lcm), _ = mean_masks(others)
            ao, ac, _ = align(*shapes[i], lom, lcm)
            dev, _ = deviation(ao, ac, lom, lcm)
            ok = max(dev["outer"]["p99_abs"], dev["cavity"]["p99_abs"]) <= args.contour_tol
            print(f"  {names[i]}: p99 outer={dev['outer']['p99_abs']} cav={dev['cavity']['p99_abs']} "
                  f"-> {'OK' if ok else 'would FAIL at tol ' + str(args.contour_tol)}")
            loo.append(dict(name=names[i], deviation=dev, ok=bool(ok)))

    np.savez_compressed(args.output, outer=rom, cavity=rcm, res=CANVAS_RES, canvas=N,
                        px_per_mm=args.px_per_mm, info=json.dumps(dict(samples=samples, loo=loo)))
    ref_meas = measure_masks(rom, rcm)
    print(f"\nReference saved to {args.output}. Reference shape: {ref_meas}")


def measure_masks(om, cm):
    cs_o, _ = cv2.findContours(om, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs_c, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    o = (max(cs_o, key=cv2.contourArea).reshape(-1, 2) - N / 2) / CANVAS_RES
    c = (max(cs_c, key=cv2.contourArea).reshape(-1, 2) - N / 2) / CANVAS_RES
    m = measure(o, c)
    m.pop("inner_width_profile_from_base")
    return m


def cmd_inspect(args):
    ref = np.load(args.ref)
    rom, rcm = ref["outer"], ref["cavity"]
    os.makedirs(args.outdir, exist_ok=True)
    all_res = []
    for p in args.images:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            o, c, br = load_canonical(p, args)
        except RuntimeError as e:
            print(f"{name}: ERROR {e}")
            all_res.append(dict(image=p, error=str(e), pass_=False))
            continue
        ao, ac, al = align(o, c, rom, rcm)
        m = measure(ao, ac)
        sc = spec_check(m)
        dev, pts = deviation(ao, ac, rom, rcm)
        contour_ok = max(dev["outer"]["p99_abs"], dev["cavity"]["p99_abs"]) <= args.contour_tol
        ok = all(v["ok"] for v in sc.values()) and contour_ok
        res = dict(image=p, PASS=bool(ok), spec=sc, contour_ok=bool(contour_ok),
                   contour_tol_mm=args.contour_tol, deviation=dev, measurements=m,
                   alignment=al, mouth_bridged=bool(br))
        all_res.append(res)
        overlay(rom, rcm, pts, args.contour_tol, os.path.join(args.outdir, f"{name}_deviation.png"),
                f"{name}: {'PASS' if ok else 'FAIL'}")
        fails = [k for k, v in sc.items() if not v["ok"]] + ([] if contour_ok else ["contour"])
        print(f"{name}: {'PASS' if ok else 'FAIL'}  L={m['length']} W={m['inner_width_max']} "
              f"T={m['base_thickness']} walls={m['wall_left']}/{m['wall_right']} "
              f"dev p99 outer/cav={dev['outer']['p99_abs']}/{dev['cavity']['p99_abs']}"
              + (f"  [{', '.join(fails)}]" if fails else ""))
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nResults: {os.path.join(args.outdir, 'results.json')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build-ref", "inspect"):
        s = sub.add_parser(name)
        s.add_argument("images", nargs="+")
        s.add_argument("--px-per-mm", type=float, required=True, help="image scale from calibration")
        s.add_argument("--contour-tol", type=float, default=1.0, help="allowed p99 contour deviation, mm")
        s.add_argument("--hue-min", type=int, default=90)
        s.add_argument("--hue-max", type=int, default=130)
        s.add_argument("--sat-min", type=int, default=80)
        s.add_argument("--val-min", type=int, default=90)
        if name == "build-ref":
            s.add_argument("-o", "--output", default="ref.npz")
        else:
            s.add_argument("--ref", required=True)
            s.add_argument("-o", "--outdir", default="results")
    args = ap.parse_args()
    {"build-ref": cmd_build, "inspect": cmd_inspect}[args.cmd](args)


if __name__ == "__main__":
    main()
