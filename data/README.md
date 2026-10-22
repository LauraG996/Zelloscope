# Image data

One folder per imaging session, for example:

```
data/nmc_u/2026-06-04/        raw frames, file names YYYY-MM-DD_HH-MM-SS.png
data/nmc_u/2026-08-27/
data/nmc_u/new_fov/           wider field of view
```

PNG/TIFF/JPG files under `data/` are stored with Git LFS (see `.gitattributes`); run `git lfs install` once
per machine before adding images.

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
