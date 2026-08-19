"""Plot power against traffic for a few antennas, separating weekdays and weekend.

Exploratory check for Section 6.2: does the affine relation hold on both
regimes, and is there a vertical spread at near-zero traffic caused by
partial activation inside an averaged hour?
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_processing.data_loader import FULL_CSV_PATH, make_output_path, iter_records


def _collect(csv_path, num_antennas: int):
    """Return {antenna_id: (weekday_points, weekend_points)} for the first ids seen."""
    selected: list[str] = []
    seen: set[str] = set()
    weekday: dict[str, list[tuple[float, float]]] = defaultdict(list)
    weekend: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for timestamp, antenna_id, traffic, power in iter_records(csv_path):
        if antenna_id not in seen:
            if len(seen) >= num_antennas:
                continue
            seen.add(antenna_id)
            selected.append(antenna_id)
        bucket = weekend if timestamp.weekday() >= 5 else weekday
        bucket[antenna_id].append((traffic, power))
    return selected, weekday, weekend


def _fit(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    active = [(d, p) for d, p in points if p > 0.0]
    if len(active) < 3:
        return None
    traffic = np.asarray([d for d, _ in active], dtype=float)
    power = np.asarray([p for _, p in active], dtype=float)
    if float(np.ptp(traffic)) <= 0.0:
        return None
    slope, intercept = np.polyfit(traffic, power, 1)
    return float(intercept), float(slope)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=FULL_CSV_PATH)
    parser.add_argument("--num-antennas", type=int, default=10)
    parser.add_argument("--filename", default="weekday_weekend_scatter.png")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument(
        "--zoom-quantile",
        type=float,
        default=None,
        help="restrict the traffic axis to this quantile of each antenna",
    )
    args = parser.parse_args()

    print(">> Reading the source CSV once", flush=True)
    selected, weekday, weekend = _collect(args.input, args.num_antennas)

    columns = max(1, args.columns)
    rows = (len(selected) + columns - 1) // columns
    width = 4.0 if columns >= 4 else 7.0
    height = 3.4 if columns >= 4 else 4.6
    figure, axes = plt.subplots(
        rows, columns, figsize=(width * columns, height * rows), squeeze=False
    )
    for index, antenna_id in enumerate(selected):
        axis = axes[index // columns][index % columns]
        week_points = weekday[antenna_id]
        end_points = weekend[antenna_id]
        for points, colour, label in (
            (week_points, "#3677B8", "semaine"),
            (end_points, "#C94C4C", "week-end"),
        ):
            if not points:
                continue
            axis.scatter(
                [d for d, _ in points],
                [p for _, p in points],
                s=9,
                alpha=0.55,
                color=colour,
                label=label,
                edgecolors="none",
            )
        combined = week_points + end_points
        fit = _fit(combined)
        if fit is not None:
            intercept, slope = fit
            grid = np.linspace(0.0, max(d for d, _ in combined), 100)
            axis.plot(grid, intercept + slope * grid, color="black", linewidth=1.2)
            axis.axhline(
                intercept,
                color="black",
                linewidth=0.8,
                linestyle=":",
                label="$P^{fixe}$ ajusté",
            )
            axis.set_title(
                f"{antenna_id}\n$P^{{fixe}}$={intercept:.0f} W, $s$={slope:.1f} W/Go",
                fontsize=10,
            )
        else:
            axis.set_title(antenna_id, fontsize=10)
        axis.set_xlabel("Trafic descendant (Go/h)", fontsize=9)
        axis.set_ylabel("Puissance moyenne (W)", fontsize=9)
        axis.tick_params(labelsize=9)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25, linewidth=0.5)
        if args.zoom_quantile is not None:
            limit = float(
                np.quantile([d for d, _ in combined], args.zoom_quantile)
            )
            axis.set_xlim(left=-0.02 * max(limit, 1e-9), right=limit)
        else:
            axis.set_xlim(left=0.0)
        if index == 0:
            axis.legend(fontsize=9, loc="lower right")
    for index in range(len(selected), rows * columns):
        axes[index // columns][index % columns].axis("off")

    figure.tight_layout()
    output = make_output_path(args.filename)
    figure.savefig(output, dpi=140)
    plt.close(figure)

    print(f">> Figure: {output.resolve()}", flush=True)
    for antenna_id in selected:
        week_points = weekday[antenna_id]
        end_points = weekend[antenna_id]
        near_zero = [
            p
            for d, p in week_points + end_points
            if d <= 0.05 * max((x for x, _ in week_points + end_points), default=1.0)
            and p > 0.0
        ]
        spread = (
            f"{min(near_zero):.0f}-{max(near_zero):.0f} W over {len(near_zero)} h"
            if near_zero
            else "no near-zero active hour"
        )
        print(
            f"   {antenna_id}: {len(week_points)} weekday h, "
            f"{len(end_points)} weekend h, near-zero power {spread}",
            flush=True,
        )


if __name__ == "__main__":
    main()
