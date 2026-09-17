# Different Foams

Segmentation work for open-cell foam images, kept separate from the
closed-cell notch/straighten scripts at the repo root.

- `fuse_exposures.py` — merges several same-scene, pixel-aligned exposures
  (e.g. `opencell1.png`, `opencell1_0.7.png`, `opencell1_1.4.png`,
  `opencell1_2.png`) into one image via Mertens exposure fusion. No single
  exposure captures both the bright cell walls and the darker, translucent
  cell interiors without clipping one or the other; fusing them gives a
  sharper, higher-contrast image than any single shot, which in turn
  improves `segment_cells.py`'s results (in a test crop, cell count went
  from 10,854 on the single best exposure to 11,831 on the fused image,
  with visibly tighter boundaries). Usage:

  ```
  python different_foams/fuse_exposures.py <img1> <img2> [<img3> ...] <output>
  ```

- `segment_cells.py` — segments individual cells in an open-cell foam
  image (distance-transform + watershed on the flattened, thresholded
  cell-interior mask). Outputs an annotated image with each cell's
  boundary drawn, plus a CSV of per-cell measurements (area, equivalent
  diameter, centroid). Works best on a fused image from
  `fuse_exposures.py`, but takes any single image too. Usage:

  ```
  python different_foams/segment_cells.py <input> <output> [--pix2mm MM_PER_PX]
  ```

  `<input>`/`<output>` can be a single image or a directory (in which
  case a `cell_summary.csv` is also written to the output directory).

- `test_images/` — sample open-cell foam images for developing and
  testing the segmentation code.
