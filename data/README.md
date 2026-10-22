# Image data

One folder per imaging session, for example:

```
data/nmc_u/2026-06-04/        raw frames, file names YYYY-MM-DD_HH-MM-SS.png
data/nmc_u/2026-08-27/
data/nmc_u/new_fov/           wider field of view
```

PNG/TIFF/JPG files under `data/` are stored with Git LFS (see `.gitattributes`); run `git lfs install` once
per machine before adding images.

## Adding a set of raw frames

Raw frames are ~24 MB each. The analysis never works finer than 40 px/mm (shapes at 20, cells and walls at 40),
so shrink them first; this makes each image ~8 MB and writes the folder's `scale.yaml` with each file's exact
new scale:

```
python -m shape_variation shrink D:/zelloscope/2026-06-04 -o data/nmc_u/2026-06-04
python -m shape_variation scales data/nmc_u/2026-06-04        # every image should show a scale
git add data/nmc_u/2026-06-04 && git commit -m "Add 2026-06-04 frames" && git push
```

If the raw frames carry no scale (no PNG metadata, no `scale.yaml`), add `--mm-per-px 0.014019`. The time must
be in the file name (`YYYY-MM-DD_HH-MM-SS`); `shrink` renames files whose time was only in the PNG metadata.
On a shrunk set the dimensions and the cell results agree with the full-size frames to within 0.01 mm / 1 %;
the cavity surface-layer flattening (already the least certain number) moves by up to ±0.4.

## Scale (required)

The analysis never guesses the scale. For each image it uses, in this order:

1. the `ZelloscopeData` metadata Zelloscope writes into its PNGs (`{"p2mm": ...}` = mm per pixel), if present;
2. a `scale.yaml` in the image's folder:
   ```yaml
   mm_per_px: 0.014019     # raw Zelloscope frames, 3800x3804 px
   source: confirmed by Laura Guerrero 2026-10-21
   ```

Images with neither are reported and skipped. If the value Zelloscope stored in a PNG is wrong (e.g. it was
calibrated for another zoom setting), give that file its own entry in `scale.yaml`; an entry for a file wins
over the PNG value:

```yaml
mm_per_px: 0.014019
files:
  capture_sample_20261022T112322.png: {mm_per_px: 0.020, source: "20 um/px, wider field of view"}
``` Cropped or resized exports have a different scale than the raw
frame: give them their own folder and `scale.yaml`, or don't use them.

## Optional per-folder metadata

A `parts.csv` next to the images adds information per image (all columns optional except `filename`):

```
filename,part_id,batch,line_speed_m_min,melt_temp_c,time_since_extrusion_min,notes
2026-06-04_09-16-15.png,,,,,,
```

Timestamps are taken from the PNG metadata or the file name (`YYYY-MM-DD_HH-MM[-SS]`).
