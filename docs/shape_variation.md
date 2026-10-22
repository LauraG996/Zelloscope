# Shape variation – NMC NOMAPACK U 0-24/77

`profile_check` measures a few dimensions per part. `shape_variation` looks at the **whole cross-section** of many
parts. It describes how the shape varies, tracks drift over time and flags abnormal parts. Where the data
exists, it also links the shape to cell structure and process data.

Code: `shape_variation/`, run as `python -m shape_variation`. Settings: `config/shape_variation.yaml`. The spec
limits are reused from `config/nmc_nomapack_u_0-24-77.yaml`.

```
conda env create -f environment.yml        # or: pip install -r requirements-dev.txt
python -m shape_variation scales data/nmc_u/2026-06-04            # check every image has a scale
python -m shape_variation analyze data/nmc_u/2026-06-04 data/nmc_u/2026-08-27 -o results/variation --cells
python -m shape_variation score --model results/variation/model.npz new_images/ -o results/new.csv
python -m shape_variation overlay data/nmc_u/2026-06-04 -o results/overlays          # check the segmentation
python -m shape_variation shrink D:/raw/2026-06-04 -o data/nmc_u/2026-06-04  # raw frames -> 40 px/mm + scale.yaml
```

`overlay` draws, over each image, the foam and cavity outlines, the window cut and every cell boundary. Measured
cells are yellow; cells that were dropped are violet, either because they touch the foam edge or because they
lie outside the window. It also adds 12 × 12 mm close-ups of the base and a leg.

`analyze` writes the following to `results/variation/`:
* `report.html` – the report: summary, mode plots, thickness heat map, control charts, anomaly ranking, cell maps,
  correlations, and what could not be done / what to collect next.
* `parts.csv` – one row per part: time, process window, dimensions, spec flags, mode scores, T², residual and
  cell features.
* `thickness.csv` – wall thickness per part on a common position grid.
* `modes.json` – mode descriptions, explained variance and bootstrap stability.
* `model.npz` – the shape model, used by `score`.

## Data and scale

Put images under `data/<product>/<session>/` (Git LFS, see `data/README.md`). The scale is never guessed. It
comes from the `ZelloscopeData` PNG metadata (`p2mm`), or from a `scale.yaml` with `mm_per_px` in the folder.
Images with neither are listed in the report and skipped. Raw rig frames (3800×3804 px) are 0.014019 mm/px,
confirmed by NMC on 2026-10-21. Cropped or resized exports need their own scale.

An optional `parts.csv` in the image folder adds `part_id`, `batch`, process settings and
`time_since_extrusion_min`. Numeric columns are used in the correlations automatically. `part_id` together with
`time_since_extrusion_min` enables the relaxation analysis.

## Method

| Step | What | Where |
|---|---|---|
| Segmentation | Grey-level (Otsu) foam mask with small pores filled. The cavity is the background inside the foam's convex hull, which works for a closed cavity, an open mouth and legs cut by the frame. | `segment.py` |
| Canonical frame | mm, cavity axis vertical (principal axis, refined on the cavity midline), base at the bottom | `segment.py` |
| Window | The rig images show only the base and ~50 mm of the legs, so every part is compared within the same depth from the outer base (`window.depth_mm: auto` = deepest depth visible in all images). With the whole profile in view, the full outline is used. | `segment.py` |
| Correspondence | Outer and cavity contours smoothed (0.3 mm) and resampled to 200 + 200 points, from the left window cut through the base to the right cut (full outline: starting at the tip on the axis) | `segment.py` |
| Alignment | Generalised Procrustes: rotation and translation, no scaling (size matters). Each part is mirrored if that fits the mean better, since the cut face can be seen from either end. | `shape_model.py` |
| Shape model | PCA of the aligned points. Keeps the modes covering 95 %. Each mode is described in plain terms (cavity width, base thickness, walls, leg lean) from the change between −2 sd and +2 sd. A bootstrap shows how stable each mode is. | `shape_model.py` |
| Wall thickness | Medial axis (skeleton) of the foam + distance transform: thickness = 2 × distance, along the U, as mm from the base centre (left negative) | `thickness.py` |
| Drift | Individuals (SPC) charts of the main mode scores and key dimensions in time order: beyond 3 sigma, 8 in a row on one side, 6 rising or falling. Process-change windows are shaded. | `monitor.py` |
| Anomalies | Hotelling T² in the first modes, at most one mode per 3 parts. Each part is compared with all the *other* parts, so it cannot hide itself. A leave-one-out residual (SPE) catches deformations the model has not seen. The hard spec check is reported separately. | `monitor.py` |
| Relaxation | Pooled exponential fit for parts imaged repeatedly: y = c_part + A·exp(−t/τ). Gives the time to 95 % settled, and how well the first image predicts the final shape (leave one part out). | `relaxation.py` |
| Cells | Watershed cell segmentation ported from `cell_segmentation.py` (brave-hamilton branch), with parameters in mm. Per cell: size, elongation and orientation, mapped into the aligned frame. Gives maps and per-region features (base / legs / skin / core). | `cells.py` |
| Links | Spearman correlations with a Benjamini-Hochberg q-value, plus a shallow random forest with leave-one-out R² and permutation importance. Exploratory only. | `correlate.py` |

Repeatability on synthetic parts: imaging the same part at other angles or from the other cut end moves the
aligned points by about 0.05 mm RMS. That is well below the part-to-part variation seen so far (about 0.3 mm RMS
per mode sd).

## First results (11 images from 2026-06-04, provisional)

* 4 modes cover 96 % of the variation. The main ones:
  * Mode 1 (53 %): legs closer together at the top of the window, with the whole part narrower.
  * Mode 2 (32 %): left leg leaning further inward.
  * Mode 3 (9 %): right leg more upright.
  Only modes 1–2 are reproducible in the bootstrap.
* Three parts are flagged:
  * 09:16, the first image: a narrow cavity (inner width 22.3 mm).
  * 09:56: a notch in the cavity outline, probably dust or a segmentation defect. Worth a look at the image.
  * 13:16: the only part from process window 3, with a thinner base and a leg leaning further in.
* As measured here, inner width is 22.3–24.9 mm and base thickness 12.7–13.6 mm, so most parts are outside the
  drawing limits. TODO(NMC): confirm where the 24 mm width and 12 mm thickness are measured before calling
  these rejects.
* No cell feature correlates with the shape after the multiple-testing correction. With 11 parts this is
  expected. Collect more parts before reading anything into it.

### Cells at the surfaces

At both surfaces the first cell layer is flattened against the surface. The watershed does not split that
layer into cells, so it is measured separately with the line-intercept method (as in ASTM D3576),
`cells.surface_intercepts()`:

* A wall map comes from a Sato ridge filter, which picks out the bright, line-like cell walls.
* Walls are counted along lines going straight in from each surface (every 0.25 mm along the surface, up to
  3.5 mm deep, stopping halfway to the other surface). The wall-to-wall distance along these lines is the
  cell depth.
* Walls are also counted along lines running parallel to the surface at 0.25, 0.75, 1.25, 1.75 and 2.5 mm
  deep. The wall-to-wall distance there is the cell width. Flattening = width / depth (1 = round).
* The wall-detection sensitivity is set per image: in the core (more than 2 mm in), the wall-to-wall distance
  has to match π/4 × the mean diameter of the watershed cells, which are reliable there. All 11 images landed
  on the same setting.
* A synthetic test with known flattened cells (0.4 × 1.2 mm under the surface, 0.8 mm round below) checks
  that both are recovered.

First result (11 parts): in the first 0.5 mm the cells are about 1.8× wider than deep below the outer
surface. They are nearly round (about 1.1×) from about 1.5–2 mm in. Below the cavity the first 0.5 mm reads
about 3.4×, but that value is less certain: the cut face is blurred along the cavity edge in these images, so
some walls are missed and the cells read too wide. Check with `overlay`, where the detected walls are marked
in cyan.

Flattening map (`cells.flattening_samples()` / `flattening_grid()`, in the report under Cell structure): the
foam is divided into 1 mm squares (`cells.flat_bin_mm`), in the frame where all parts are aligned on each
other. Lines run straight in from each surface (every 0.25 mm) and parallel to it (every 0.25 mm of depth) all
the way to the middle of the wall. In each square, flattening = walls crossed per mm going in / walls crossed
per mm going along the nearest surface. Counting walls per mm instead of measuring whole cells works in small
squares. First result (11 parts): about 1.6× in the outermost millimetre, round (1.0) from about 2.5–3 mm in,
and slightly below 1 (about 0.85) in the middle of the thick base, where cells are a little taller than wide
between the two surfaces.

Other cell limitations: cells touching the foam edge are left out of the per-cell statistics and maps
(the surface layer is covered by the intercepts above). The watershed has only been checked by eye against
the images (`overlay`), not against manual counts or Zelloscope's own cell measurement.

## Open questions / what to collect

* TODO(NMC): where are the 24 mm width and the 12 mm thickness measured? The measurements use the
  definitions in `profile_check/measure.py`.
* TODO(NMC): the tolerance for tip overlap / asymmetry. It is only measurable with the wider field of view.
  It is reported and never fails a part.
* Feed rate on 4 June: 60 → 57 → 55 → 58 → 60 kg/h (`process_windows` in the config; each part gets a
  `feed_rate_kg_h` column used in the correlations). TODO(NMC): exact switch times, and the settings on 27 August.
* The wider field of view (whole cross-section), for length, tips and legs closing at the mouth.
* Repeated images of marked parts at fixed times after extrusion, for the relaxation model.
* Process settings per image (line speed, melt/die temperature, cooling, batch / material lot).
* At least 30 parts over several days, and NMC's list of approved parts to use as the reference set
  (`analyze --reference approved.txt`).
