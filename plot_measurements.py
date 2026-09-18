#!/usr/bin/env python3
"""Plot two measurement-log CSVs (same format as straighten.py's --csv output)
against each other on a shared time-of-day axis.

Reproduces the look of plots/4june_vs_27aug_plot.svg: solid line + circle
markers for the first (dense) run, diamond markers for the second (sparse)
run, plus a dashed centered rolling average for each, optional feed-rate
bands, and optional spec min/max lines.
"""
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#e1e0d9"
TEXT = "#898781"
BG = "#fcfcfb"
SPEC_RED = "#c0392b"

FEED_DARK, FEED_MED, FEED_PALE, FEED_BLUE = "#0d366b", "#5a9be8", "#cde2fb", "#2975d1"

METRICS = [
    ("inner_shoulder_distance_mm", "A — Inner shoulder distance (mm)"),
    ("wall_thickness_mm", "B — Wall thickness (mm)"),
]

# filenames look like "2026-06-04_09-16-15.png" or "2026-08-27_09-36.png"
TIME_FORMATS = ["%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H-%M"]

ANCHOR = date(2000, 1, 1)

# Feed-rate schedule for the 4 June run, back-calculated pixel-accurately
# from plots/4june_vs_27aug_plot.svg (its band boundaries land on
# 11:10, 12:57, 13:55 and 14:56, and the gap between the 12:57 and 13:55
# boundaries brackets that plot's "11:51-12:55 removed" annotation exactly).
FEED_RATE_SCHEDULE = [
    (None, "11:10", "60 kg/h", FEED_DARK),
    ("11:10", "12:57", "57 kg/h", FEED_MED),
    ("12:57", "13:55", "55 kg/h", FEED_PALE),
    ("13:55", "14:56", "58 kg/h", FEED_BLUE),
    ("14:56", None, "60 kg/h", FEED_DARK),
]


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


def feed_rate_bounds(xmin: datetime, xmax: datetime) -> list[tuple[datetime, datetime, str, str]]:
    bounds = []
    for start, end, label, color in FEED_RATE_SCHEDULE:
        t0 = xmin if start is None else datetime.combine(ANCHOR, datetime.strptime(start, "%H:%M").time())
        t1 = xmax if end is None else datetime.combine(ANCHOR, datetime.strptime(end, "%H:%M").time())
        bounds.append((t0, t1, label, color))
    return bounds


def draw_feed_rate_strip(ax, bounds):
    ax.set_facecolor(BG)
    for t0, t1, label, color in bounds:
        ax.axvspan(t0, t1, color=color, alpha=0.55, lw=0)
        ax.text(
            t0 + (t1 - t0) / 2, 0.5, label, ha="center", va="center",
            fontsize=9.5, color="white" if color in (FEED_DARK, FEED_BLUE) else "#2b2b2b",
        )
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("feed\nrate", fontsize=9, color=TEXT, rotation=0, ha="right", va="center", labelpad=20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_a", type=Path, help="Dense/reference run CSV (e.g. 4 June)")
    parser.add_argument("csv_b", type=Path, help="Sparse/comparison run CSV (e.g. 27 August)")
    parser.add_argument("--label-a", default="4 June")
    parser.add_argument("--label-b", default="27 August")
    parser.add_argument("--rolling-minutes", type=float, default=30)
    parser.add_argument(
        "--feed-bands", action="store_true",
        help="Shade feed-rate bands using the schedule back-calculated from the reference plot",
    )
    parser.add_argument(
        "--spec-min", type=float, default=None,
        help="Draw a min spec-limit line at this value on the first metric's panel",
    )
    parser.add_argument(
        "--spec-max", type=float, default=None,
        help="Draw a max spec-limit line at this value on the first metric's panel",
    )
    parser.add_argument(
        "--spec-metric", default=METRICS[0][0], choices=[m[0] for m in METRICS],
        help="Which metric panel the spec-limit lines apply to",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/measurements_comparison.svg"))
    args = parser.parse_args()

    run_a = load_run(args.csv_a, ANCHOR)
    run_b = load_run(args.csv_b, ANCHOR)
    window = timedelta(minutes=args.rolling_minutes / 2)

    n_rows = len(METRICS) + (1 if args.feed_bands else 0)
    height_ratios = ([0.5] if args.feed_bands else []) + [4] * len(METRICS)
    fig, all_axes = plt.subplots(
        n_rows, 1, figsize=(16, 0.5 + 3.6 * len(METRICS)), sharex=False,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.12},
    )
    axes = list(all_axes[1:]) if args.feed_bands else list(all_axes)

    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"{args.label_a} vs. {args.label_b} — Inner shoulder distance & Wall thickness",
        fontsize=15, color="#2b2b2b", x=0.01, ha="left", y=0.99,
    )
    fig.text(
        0.01, 0.945,
        f"dashed = {args.rolling_minutes:g}min centered rolling average"
        f" · line+circles = {args.label_a} actual (n={len(run_a)})"
        f" · diamonds = {args.label_b} actual (n={len(run_b)})"
        + (" · shade = material feed rate (kg/h)" if args.feed_bands else ""),
        fontsize=9.5, color=TEXT, ha="left",
    )

    xmin = min(run_a["time"].min(), run_b["time"].min())
    xmax = max(run_a["time"].max(), run_b["time"].max())
    bounds = feed_rate_bounds(xmin, xmax) if args.feed_bands else []

    for ax, (column, ylabel) in zip(axes, METRICS):
        style_axis(ax)

        for t0, t1, _label, color in bounds:
            ax.axvspan(t0, t1, color=color, alpha=0.2, lw=0, zorder=0)

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

        if column == args.spec_metric:
            for value, tag in ((args.spec_min, "min"), (args.spec_max, "max")):
                if value is None:
                    continue
                ax.axhline(value, color=SPEC_RED, linestyle="--", linewidth=1.4, zorder=3)
                ax.text(
                    0.998, value, f"{tag} {value:g}", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom" if tag == "min" else "top",
                    fontsize=8.5, color=SPEC_RED,
                )

        ax.set_xlim(xmin, xmax)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, labelcolor=TEXT)

    if args.feed_bands:
        strip_ax = all_axes[0]
        strip_ax.set_xlim(xmin, xmax)
        draw_feed_rate_strip(strip_ax, bounds)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Time of day")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    fig.tight_layout(rect=(0, 0, 0.82, 0.90))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=BG, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
