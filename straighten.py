#!/usr/bin/env python3
"""Auto-straighten tilted microscopy images by rotating them level.

Segments the bright foreground object from a dark background, finds its
principal axis with PCA, and rotates the image so that axis is vertical.
Works on single files or a whole directory of images.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def largest_foreground_mask(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove dust/speckle noise and keep only the main connected object.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def principal_axis_angle(mask: np.ndarray) -> float:
    """Angle (degrees) to rotate the image so the object's long axis is vertical."""
    ys, xs = np.where(mask > 0)
    coords = np.column_stack((xs, ys)).astype(np.float64)

    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    principal = eigenvectors[0]  # direction of greatest variance
    vx, vy = principal

    # Angle between the principal axis and the vertical (0, 1) axis.
    angle_deg = np.degrees(np.arctan2(vx, vy))

    # Keep the correction within +/-90 degrees (axis has no inherent direction).
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    return angle_deg


def rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by angle_deg (counter-clockwise positive) expanding the canvas."""
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - cx
    matrix[1, 2] += (new_h / 2) - cy

    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )


def straighten(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = largest_foreground_mask(gray)
    angle = principal_axis_angle(mask)
    # Rotate by -angle to bring the axis to vertical (cv2 rotates CCW for +angle).
    return rotate_bound(image, -angle), angle


def process_file(src: Path, dst: Path) -> float:
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {src}")
    straightened, angle = straighten(image)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), straightened)
    return angle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image file or directory")
    parser.add_argument("output", type=Path, help="Output image file or directory")
    args = parser.parse_args()

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"No images found in {args.input}")
        for src in files:
            angle = process_file(src, args.output / src.name)
            print(f"{src.name}: rotated {angle:+.2f} deg -> {args.output / src.name}")
    else:
        angle = process_file(args.input, args.output)
        print(f"{args.input.name}: rotated {angle:+.2f} deg -> {args.output}")


if __name__ == "__main__":
    main()
