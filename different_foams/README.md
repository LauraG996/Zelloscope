# Different Foams

Segmentation work for open-cell foam images, kept separate from the
closed-cell notch/straighten scripts at the repo root.

- `segment_cells.py` — segments individual cells in an open-cell foam
  image (distance-transform + watershed on the flattened, thresholded
  cell-interior mask). Outputs an annotated image with each cell's
  boundary drawn, plus a CSV of per-cell measurements (area, equivalent
  diameter, centroid). Usage:

  ```
  python different_foams/segment_cells.py <input> <output> [--pix2mm MM_PER_PX]
  ```

  `<input>`/`<output>` can be a single image or a directory (in which
  case a `cell_summary.csv` is also written to the output directory).

- `test_images/` — sample open-cell foam images for developing and
  testing the segmentation code.
