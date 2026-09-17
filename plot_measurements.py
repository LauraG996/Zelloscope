#!/usr/bin/env python3
"""Plot two measurement-log CSVs (same format as straighten.py's --csv output)
against each other on a shared time-of-day axis.

Reproduces the look of plots/4june_vs_27aug_plot.svg: solid line + circle
markers for the first (dense) run, diamond markers for the second (sparse)
run, plus a dashed centered rolling average for each.
"""
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#e1e0d9"
TEXT = "#898781"
BG = "#fcfcfb"

METRICS = [
    ("inner_shoulder_distance_mm", "A — Inner shoulder distance (mm)"),
    ("wall_thickness_mm", "B — Wall thickness (mm)"),
]

# filenames look like "2026-06-04_09-16-15.png" or "2026-08-27_09-36.png"
TIME_FORMATS = ["%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H-%M"]


def parse_time_of_day(filename: str, anchor: date) -> datetime:
    stem = Path(filename).stem
    for fmt in TIME_FORMATS:
        try:
            ts = datetime.strptime(stem, fmt)
            return datetime.combine(anchor, ts.time())
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp from filename: {filename}")


def load_run(csv_path: Path, anchor: date) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["time"] = df["filename"].apply(lambda f: parse_time_of_day(f, anchor))
    return df.sort_values("time").reset_index(drop=True)


def centered_rolling_avg(df: pd.DataFrame, column: str, window: timedelta) -> pd.Series:
    times = df["time"].to_numpy()
    values = df[column].to_numpy()
    out = []
    for t in times:
        mask = (times >= t - window) & (times <= t + window)
        out.append(values[mask].mean())
    return pd.Series(out, index=df.index)


def style_axis(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_a", type=Path, help="Dense/reference run CSV (e.g. 4 June)")
    parser.add_argument("csv_b", type=Path, help="Sparse/comparison run CSV (e.g. 27 August)")
    parser.add_argument("--label-a", default="4 June")
    parser.add_argument("--label-b", default="27 August")
    parser.add_argument("--rolling-minutes", type=float, default=30)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/measurements_comparison.svg"))
    args = parser.parse_args()

    anchor = date(2000, 1, 1)
    run_a = load_run(args.csv_a, anchor)
    run_b = load_run(args.csv_b, anchor)
    window = timedelta(minutes=args.rolling_minutes / 2)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(14, 3.6 * len(METRICS)), sharex=True)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"{args.label_a} vs. {args.label_b} — Inner shoulder distance & Wall thickness",
        fontsize=15, color="#2b2b2b", x=0.01, ha="left", y=0.99,
    )
    fig.text(
        0.01, 0.945,
        f"dashed = {args.rolling_minutes:g}min centered rolling average"
        f" · line+circles = {args.label_a} actual (n={len(run_a)})"
        f" · diamonds = {args.label_b} actual (n={len(run_b)})",
        fontsize=9.5, color=TEXT, ha="left",
    )

    for ax, (column, ylabel) in zip(axes, METRICS):
        style_axis(ax)

        mean_a, std_a = run_a[column].mean(), run_a[column].std()
        mean_b, std_b = run_b[column].mean(), run_b[column].std()

        ax.plot(
            run_a["time"], run_a[column], "-o", color=BLUE, markersize=4.5,
            linewidth=1.4, label=f"{args.label_a} — {mean_a:.2f}±{std_a:.2f} mm, n={len(run_a)}",
        )
        ax.plot(
            run_b["time"], run_b[column], "D", color=ORANGE, markersize=7,
            linestyle="none", label=f"{args.label_b} — {mean_b:.2f}±{std_b:.2f} mm, n={len(run_b)}",
        )
        ax.plot(
            run_a["time"], centered_rolling_avg(run_a, column, window),
            "--", color=BLUE, linewidth=1.6, alpha=0.85,
            label=f"{args.label_a} — {args.rolling_minutes:g}min rolling avg",
        )
        ax.plot(
            run_b["time"], centered_rolling_avg(run_b, column, window),
            "--", color=ORANGE, linewidth=1.6, alpha=0.85,
            label=f"{args.label_b} — {args.rolling_minutes:g}min rolling avg",
        )

        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, labelcolor=TEXT)

    axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Time of day")

    fig.tight_layout(rect=(0, 0, 0.86, 0.90))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=BG)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
