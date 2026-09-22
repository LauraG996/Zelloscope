#!/usr/bin/env python3
"""Plot cell positions from the latest measurement in a celloscope cell_data.csv,
drawn as true-to-relative-scale circles colored by diameter, with an
adjustable diameter filter.

cell_data.csv has one row per measurement run (timestamp, diameters,
x_positions, y_positions, each a Python-literal list of equal length). This
picks the row with the latest timestamp and draws each cell as a circle at
its (x, y) position, restricted to --min-diameter/--max-diameter so you can
re-run with different bounds to see which size range of cells is shown.

diameters and positions turn out not to share a unit in this data set --
positions are raw image pixel coordinates, but the largest recorded diameter
is over 1000x smaller than the frame width, so true 1:1-scale circles would
be invisible. --size-scale (default DEFAULT_SIZE_SCALE) exaggerates every
circle's radius by a constant factor so they're visible on the plot; circles
stay proportional to each other and to each cell's real diameter, only the
overall scale is inflated, and this is labeled on the plot itself.
"""
import argparse
import ast
import csv
import sys
from pathlib import Path

# Some rows have tens of thousands of measurements per column; csv's default
# field size limit (131072 bytes) is too small for that. sys.maxsize itself
# overflows the C long field_size_limit takes on Windows, so cap at INT32_MAX.
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

# Default circle-size exaggeration factor (see module docstring): picked so a
# median-sized cell in this data set renders at a few px radius rather than a
# sub-pixel dot.
DEFAULT_SIZE_SCALE = 100.0


def load_latest_row(csv_path: Path) -> dict:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"No rows found in {csv_path}")
    return max(rows, key=lambda r: r["timestamp"])


def parse_cells(row: dict) -> tuple[list[float], list[float], list[float]]:
    diameters = [float(d) for d in ast.literal_eval(row["diameters"])]
    x = [float(v) for v in ast.literal_eval(row["x_positions"])]
    y = [float(v) for v in ast.literal_eval(row["y_positions"])]
    return diameters, x, y


def plot_cells(
    diameters: list[float],
    x: list[float],
    y: list[float],
    timestamp: str,
    min_diameter: float | None,
    max_diameter: float | None,
    dst: Path,
    size_scale: float = DEFAULT_SIZE_SCALE,
) -> int:
    """Draw each cell as a circle (radius = diameter * size_scale / 2, in x/y's
    units) at its (x, y) position, colored by its real diameter, restricted to
    [min_diameter, max_diameter].

    size_scale exaggerates every circle's radius by a constant factor so they
    show up on the plot -- see module docstring for why. Circles stay
    proportional to each other and to true diameter; only the overall size is
    inflated. Saves to dst and returns how many cells passed the filter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    diameters = np.asarray(diameters)
    x = np.asarray(x)
    y = np.asarray(y)

    keep = np.ones(len(diameters), dtype=bool)
    if min_diameter is not None:
        keep &= diameters >= min_diameter
    if max_diameter is not None:
        keep &= diameters <= max_diameter

    fig, ax = plt.subplots(figsize=(8, 8))

    scaled_radii = diameters[keep] * size_scale / 2
    circles = [Circle((xi, yi), radius=ri) for xi, yi, ri in zip(x[keep], y[keep], scaled_radii)]
    collection = PatchCollection(circles, cmap="viridis", alpha=0.6, linewidths=0)
    collection.set_array(diameters[keep])
    ax.add_collection(collection)

    margin = diameters.max() * size_scale / 2 if len(diameters) else 0
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.set_xlabel("x position (px)")
    ax.set_ylabel("y position (px)")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # x/y are image pixel coordinates: y increases downward

    title = f"{timestamp}: {int(keep.sum())} of {len(diameters)} cells shown"
    if min_diameter is not None or max_diameter is not None:
        lo = f"{min_diameter:.3f}" if min_diameter is not None else "-inf"
        hi = f"{max_diameter:.3f}" if max_diameter is not None else "+inf"
        title += f"\ndiameter in [{lo}, {hi}]"
    title += f"\ncircle size exaggerated {size_scale:g}x for visibility (not 1:1 with position units)"
    ax.set_title(title, fontsize=10)

    cbar = fig.colorbar(collection, ax=ax)
    cbar.set_label("true diameter")

    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)
    return int(keep.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path("cell_data.csv"),
                         help="Input cell_data.csv file (default: cell_data.csv)")
    parser.add_argument("output", type=Path, nargs="?", default=Path("cell_plot.png"),
                         help="Output plot PNG path (default: cell_plot.png)")
    parser.add_argument(
        "--min-diameter", type=float, default=None,
        help="Only show cells with diameter >= this (default: no lower bound)",
    )
    parser.add_argument(
        "--max-diameter", type=float, default=None,
        help="Only show cells with diameter <= this (default: no upper bound)",
    )
    parser.add_argument(
        "--size-scale", type=float, default=DEFAULT_SIZE_SCALE,
        help="Exaggerate circle radius by this factor for visibility, since diameters "
             f"and positions don't share a unit in this data (default: {DEFAULT_SIZE_SCALE:g})",
    )
    args = parser.parse_args()

    row = load_latest_row(args.input)
    diameters, x, y = parse_cells(row)
    shown = plot_cells(
        diameters, x, y, row["timestamp"], args.min_diameter, args.max_diameter,
        args.output, args.size_scale,
    )
    print(f"{row['timestamp']}: {shown} of {len(diameters)} cells shown -> {args.output}")


if __name__ == "__main__":
    main()
