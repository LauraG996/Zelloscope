# Profile check – NMC NOMAPACK U 0-24/77

Checks the cut end (cross-section) of a foam U-profile against the NMC spec and against a "golden" reference
shape built from known-good parts. Code: `profile_check/` (run as `python -m profile_check`), spec and tuning
parameters: `config/nmc_nomapack_u_0-24-77.yaml`. `docs/profile_check.py` is the original prototype, kept for
reference only.

```
pip install -r requirements-dev.txt
python -m pytest            # ~1 min, synthetic images only
```

## Workflow

1. **Scale.** The image scale is never defaulted. Either pass `--px-per-mm` on every run, or write a
   calibration file once:
   ```
   python -m profile_check calibrate target.png --inner-corners 9x6 --square-mm 5 -o calibration.yaml
   ```
   This needs a printed checkerboard lying in the plane of the cut face. A hand-written file works too: `px_per_mm: 26.5`.
2. **Reference.** Build it from the known-good parts (files and/or directories):
   ```
   python -m profile_check build-ref --calibration calibration.yaml -o refs/nmc_u.npz good/
   ```
   This prints the per-sample measurements and flags any "good" sample that is outside spec. It also runs
   leave-one-out: the reference is rebuilt from N−1 parts and the left-out part is inspected. It writes
   `refs/nmc_u.npz` plus `refs/nmc_u.report.json`.
   To see the reference shape on its own, run `python -m profile_check show-ref refs/nmc_u.npz`. It writes
   `refs/nmc_u.png`: the ideal shape (white) with min/max lines at −/+ the contour tolerance (blue/orange),
   which is the band a part's outline must stay inside, plus the length, inner width and base thickness.
   `--parts` also outlines the parts it was built from; `--no-band` draws the ideal shape only.
3. **Inspect.**
   ```
   python -m profile_check inspect --calibration calibration.yaml --ref refs/nmc_u.npz -o results/ new/
   ```
   For each image it prints PASS/FAIL with reasons and writes an overlay `results/<image>_deviation.png`. The
   reference is drawn in white and the part's contour runs from green to red as |deviation| goes from 0 to the tolerance. All
   numbers go to `results/results.json`. The exit status is 1 if any image fails.

## Adding more reference parts

The reference is the mean of the parts it was built from, so adding parts means rebuilding it. Either:

* rebuild from all good images: `build-ref ... -o refs/nmc_u_v2.npz good/ more_good/`, or
* extend an existing reference without the old images. Each `.npz` stores the aligned shapes of its parts:
  `build-ref ... --extend refs/nmc_u.npz -o refs/nmc_u_v2.npz more_good/`

Check the printed per-sample and leave-one-out results before switching `inspect` to the new file. Several
cuts or placements per part are fine and help the mean.

## Pipeline

| Step | Module | What |
|---|---|---|
| Segmentation | `segmentation.py` | HSV foam mask → largest blob → outer contour. The cavity is the largest hole. If the legs are open, increasingly large closings (`bridge_close_mm`) decide what counts as "outside", but the cavity shape itself comes from the unclosed mask. A glare spot in the cavity cannot split the cavity. |
| Canonical frame | `geometry.py` | Pixels → mm. Origin at the outer centroid, principal axis vertical, base at the bottom (the cavity's area centroid lies towards its wide end). |
| Alignment | `geometry.py` | Mirror × {0°, 180°} start → coarse-to-fine search over rotation and translation, maximising IoU(foam) + 0.5·IoU(cavity) against the reference. No scaling: a size difference is a defect. |
| Measurements | `measure.py` | Exact line/contour intersections in the reference frame. |
| Deviation | `deviation.py` | Signed distance from every part contour point (0.1 mm spacing) to the reference polyline. + means bigger than the reference (more material on the outer contour, a larger cavity on the cavity contour). Reports max, p99, RMS and mean. |
| Reference | `reference.py` | Align all to the first part, average the masks (≥ 0.5), re-align to the mean, average again, then re-centre on the mean's own principal axis. |

**Decision:** PASS when every spec dimension is in range AND the p99 |deviation| of both contours is at most the
contour tolerance (1.0 mm by default; set `contour_tolerance_mm` in the config or pass `--contour-tol`).

### Measurement definitions (mm, base at the bottom)

| Name | Definition | Spec |
|---|---|---|
| `length` | outer extent along the profile axis, base → tips | 76.0 – 80.0 |
| `inner_width_max` | widest cavity row, perpendicular to the axis | 24.0 – 27.0 |
| `base_thickness` | outer base − cavity bottom on the centre line of the widest row; median over ±1 mm | 11.0 – 13.0 |
| `outer_width_max` | outer extent perpendicular to the axis | report only |
| `wall_left/right`, `wall_asym` | outer − cavity edge on the widest cavity row, and their difference | report only |
| `tip_height_diff` | height difference of the topmost outer point left/right of the centre line | report only |
| `inner_width_profile_from_cavity_bottom` | cavity width every 5 mm above the cavity bottom | report only |

## Open questions / TODO

* **NMC:** confirm where the 24 mm width and the 12 mm thickness are measured. The definitions are in
  `measure.py`, a few lines each.
* **NMC:** tip overlap and asymmetry have no spec yet. They are measured and reported, but never cause a FAIL.
* The contour tolerance comes from the spec, not from sample statistics, because 5 parts are too few. Rebuild
  the reference (see above) as more NMC-approved parts arrive.
* An open mouth (legs not touching) is segmented and measured, but its cavity contour differs from a reference
  built from closed parts, so it fails the contour check. Decide with NMC whether an open mouth is acceptable.
* Tune the HSV thresholds on the first images from the NMC rig. The tests use synthetic images, because the
  existing `test_images/` show a different product and are cropped at the frame edge.
* Set up the calibration target (ruler or checkerboard) on the rig. Lens distortion is not corrected; keep the
  part near the image centre, or add a full camera calibration if the error matters.
