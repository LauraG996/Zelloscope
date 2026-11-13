"""Compare keypoint_transfer (eval_dataset.py -> per_image.csv) with zs-auto-measure
(eval_dataset_zs.py -> per_image_zs.csv) on the same dataset images.

    python compare_methods.py [--include-rejected]

Writes eval_results/comparison.csv and prints a short table.
  - failures: per method, share of images with no result (keypoint_transfer: exception;
    ZS: no landmarks, or - unless --include-rejected - a sample ZS rejected)
  - errors: percentiles on the images BOTH methods measured (paired), so neither method is
    judged on easier images than the other; per subset (all, non-mirrored, mirrored,
    shape <k>), distance (D1..D3, all) and metric (as in eval_dataset.py)
  - zs_better_pct: share of paired values where ZS's error is smaller (ties count half)
"""
import argparse, csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_results")
PCTS = [50, 90, 95]
METRICS = {"endpoint_err_px": lambda ks: [f"D{k}_e{e}_err" for k in ks for e in (1, 2)],
           "length_err_px": lambda ks: [f"D{k}_len_err" for k in ks],
           "endpoint_err_rel_pct": lambda ks: [f"D{k}_e{e}_rel_pct" for k in ks for e in (1, 2)],
           "length_err_rel_pct": lambda ks: [f"D{k}_len_rel_pct" for k in ks]}


def load(path):
    return {r["image"]: r for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kt", default=os.path.join(OUT, "per_image.csv"))
    ap.add_argument("--zs", default=os.path.join(OUT, "per_image_zs.csv"))
    ap.add_argument("--include-rejected", action="store_true",
                    help="score ZS samples it rejected but still mapped, instead of counting them as failures")
    a = ap.parse_args()
    kt, zs = load(a.kt), load(a.zs)
    images = sorted(set(kt) & set(zs))
    if not images:
        raise SystemExit("no image is in both csv files")
    kt_ok = lambda r: r["failed"] == "0"
    zs_ok = lambda r: r["failed"] == "0" and (a.include_rejected or r["accepted"] == "1")

    out = open(os.path.join(OUT, "comparison.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["subset", "distance", "metric", "n_images", "kt_failed_pct", "zs_failed_pct", "n_paired"]
               + [f"{m}_p{p}" for m in ("kt", "zs") for p in PCTS] + ["kt_mean", "zs_mean", "zs_better_pct"])
    shapes = sorted({kt[i]["shape"] for i in images}, key=lambda s: (len(s), s))
    subsets = {"all": images,
               "non-mirrored": [i for i in images if kt[i]["mirrored"] == "0"],
               "mirrored": [i for i in images if kt[i]["mirrored"] == "1"]}
    subsets.update({f"shape {s}": [i for i in images if kt[i]["shape"] == s] for s in shapes})
    print(f"{len(images)} images in both files ({len(set(kt) ^ set(zs))} only in one, left out)\n")
    print(f"{'subset':14s} {'n':>5s} {'fail% KT':>9s} {'fail% ZS':>9s}  "
          f"{'len err px p50/p90 KT':>22s} {'ZS':>14s}  {'endpoint px p50/p90 KT':>23s} {'ZS':>14s}  ZS better")
    for sname, ims in subsets.items():
        if not ims:
            continue
        kf = 100 * np.mean([not kt_ok(kt[i]) for i in ims])
        zf = 100 * np.mean([not zs_ok(zs[i]) for i in ims])
        paired = [i for i in ims if kt_ok(kt[i]) and zs_ok(zs[i])]
        summary = {}
        for d in (1, 2, 3, "all"):
            ks = (1, 2, 3) if d == "all" else (d,)
            for metric, cols in METRICS.items():
                cs = cols(ks)
                vk = np.array([float(kt[i][c]) for i in paired for c in cs])
                vz = np.array([float(zs[i][c]) for i in paired for c in cs])
                if len(vk):
                    pk, pz = ([np.percentile(v, p) for p in PCTS] for v in (vk, vz))
                    better = 100 * (np.mean(vz < vk) + 0.5 * np.mean(vz == vk))
                    row = [f"{v:.2f}" for v in pk + pz] + [f"{vk.mean():.2f}", f"{vz.mean():.2f}", f"{better:.1f}"]
                else:
                    pk = pz = [np.nan] * len(PCTS)
                    better = np.nan
                    row = [""] * (2 * len(PCTS) + 3)
                w.writerow([sname, f"D{d}" if d != "all" else "all", metric, len(ims),
                            f"{kf:.1f}", f"{zf:.1f}", len(paired)] + row)
                if d == "all":
                    summary[metric] = (pk, pz, better)
        lk, lz, lb = summary["length_err_px"]
        ek, ez, _ = summary["endpoint_err_px"]
        print(f"{sname:14s} {len(ims):5d} {kf:9.1f} {zf:9.1f}  {lk[0]:10.2f} / {lk[1]:<9.2f} {lz[0]:6.2f} / {lz[1]:<6.2f}"
              f"  {ek[0]:11.2f} / {ek[1]:<9.2f} {ez[0]:6.2f} / {ez[1]:<6.2f}  {lb:5.1f}%")
    out.close()
    print(f"\nfull table: {os.path.join(OUT, 'comparison.csv')}")


if __name__ == "__main__":
    main()
