# Task: Profile shape check for NMC NOMAPACK U 0-24/77 (Zelloscope)

## Context
Zelloscope is our camera + computer-vision system. For the NMC pilot we image the **cut end (cross-section)** of
foam U-profiles and must decide whether each profile matches a "golden" reference closely enough.

- Product: NMC NOMAPACK **U 0-24/77** — blue PE-LD foam, U-shaped cross-section.
- Relaxed shape: legs lean inward and the tips touch/nearly touch, so the cavity is a **tapered keyhole**
  (widest just above the base, narrowing to the tips). The part may lie at any rotation in the image and may be
  seen mirrored (either cut end).
- Images: ~2000×2000 px, blue foam on a dark/brown background, dark cavity, occasional glare spot inside the cavity.
- We have only **5 known-good parts** to build the reference from.

## Spec (from NMC drawing)
| Dimension | Nominal | Accept range | How to measure |
|---|---|---|---|
| Inner width (max cavity width) | 24 −0/+3 | 24.0 – 27.0 mm | widest cavity row, perpendicular to the profile axis |
| Thickness (base) | 12 ±1 | 11.0 – 13.0 mm | on the centre line, cavity bottom → outer base |
| Length | 77 −1/+3 | 76.0 – 80.0 mm | outer extent along the profile axis, base → tips |

Keep the spec in a config file (YAML/JSON), not hard-coded.

## What to build
A Python module + CLI (OpenCV + NumPy; add to the existing Zelloscope repo structure — look at it first and follow
its conventions, tests and packaging).

1. **Segmentation** – foam mask (HSV threshold, configurable), largest blob → outer contour.
   Cavity = largest hole in the foam mask. If the legs are open (no hole), bridge the mouth with increasing
   morphological closing until a hole appears. Must be robust to the glare spot in the cavity.
2. **Calibration** – px/mm passed in or loaded from a calibration file (we will image a ruler/target).
   Add a small helper to compute px/mm from a calibration target image if it's simple to do.
3. **Canonical frame** – convert to mm, principal axis vertical, **base at the bottom** (the cavity's area
   centroid lies towards its wide/base end).
4. **Alignment to reference** – search mirror (yes/no) + rotation + translation that maximises overlap
   (IoU) of foam and cavity masks with the reference.
5. **Measurements (mm)** – length, max inner width, base thickness, outer width, left/right wall at the widest
   cavity row, wall asymmetry, tip height difference, inner-width profile every 5 mm from the base.
6. **Reference building (`build-ref`)** – from N good images: align all to the first, average the binary masks
   (threshold 0.5), re-align to the mean and average again. Save the reference (masks + metadata).
   Print per-sample measurements and flag any "good" sample that is outside spec.
   Run **leave-one-out**: build from N−1, test the left-out part, report whether it would pass.
7. **Contour deviation** – for every point on the test part's outer and cavity contours, signed distance (mm) to
   the reference contour (+ = bigger than reference). Report max, p99, RMS, mean.
8. **Decision (`inspect`)** – PASS if all spec dimensions are in range AND p99 deviation of both contours
   ≤ contour tolerance (default **1.0 mm**, configurable). Output per image: PASS/FAIL with reasons,
   `results.json` (all numbers), and an overlay PNG (reference contour in white, part contour coloured
   green→red by |deviation| up to the tolerance).

## Acceptance tests (write them)
- Synthetic variants of a sample image (rotated 0–180°, mirrored, ±1 % scale) all align and pass the contour check
  against a reference built from the others.
- A variant scaled to 93 % width and one to 94 % length both FAIL (dimension and contour).
- An image with open legs (mouth not closed) still segments a cavity.
- Measurements are rotation-invariant within 0.3 mm.
- Runtime: one inspection < 5 s on a laptop.

## Notes / open questions — don't guess silently, leave TODOs
- The px/mm scale is not known yet — require it explicitly; never default to a magic number.
- Confirm with NMC where the 24 mm width and 12 mm thickness are measured; keep the measurement definitions
  easy to change.
- Tip overlap/asymmetry has no spec yet — measure and report it, but don't fail on it.
- With only 5 parts the contour tolerance comes from the spec, not from sample statistics; make it easy to
  rebuild the reference later from more NMC-approved parts.

## Reference implementation
`profile_check.py` is a working prototype of all of the above. Use it as a starting point, but restructure
it to fit the repo (separate segmentation / geometry / reference / CLI, config file, tests).
