#!/usr/bin/env python3
"""Fuse multiple exposures of the same foam sample into one image.

Cell walls (bright, near-specular) and cell interiors (darker,
translucent) span a wider dynamic range than a single exposure can
capture without either blowing out the walls or losing shadow detail in
the cells -- see different_foams/README.md for a walkthrough of what each
exposure level in this dataset shows.

Uses Mertens exposure fusion: each input pixel is weighted by local
contrast, saturation, and "well-exposedness" (distance from clipped
black/white), then blended, so the sharp wall detail from a darker frame
and the well-separated cell interiors from a brighter frame both survive
into one output. Unlike Debevec/Robertson HDR merging this doesn't need
known exposure times/ratios and doesn't produce a physical radiance map --
it just picks the best-exposed detail at each pixel, which is what a
segmentation pipeline actually needs.

Inputs must be the same scene, pixel-aligned (a static rig with only
exposure changed between shots, no camera movement).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def fuse_exposures(images: list[np.ndarray]) -> np.ndarray:
    """Merge same-scene, pixel-aligned exposures into one 8-bit BGR image."""
    merger = cv2.createMergeMertens()
    fused = merger.process(images)
    return np.clip(fused * 255, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs", type=Path, nargs="+",
        help="Two or more same-scene exposure images (any order)",
    )
    parser.add_argument("output", type=Path, help="Output fused image path")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("Need at least two input images to fuse")

    images = []
    for path in args.inputs:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            parser.error(f"Could not read image: {path}")
        images.append(img)

    shapes = {img.shape for img in images}
    if len(shapes) > 1:
        parser.error(f"All input images must be the same size, got: {shapes}")

    fused = fuse_exposures(images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), fused)
    print(f"Fused {len(images)} exposures -> {args.output}")


if __name__ == "__main__":
    main()
