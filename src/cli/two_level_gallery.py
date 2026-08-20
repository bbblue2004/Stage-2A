"""Gallery of antennas whose power-traffic cloud splits into two levels.

The reduced-power hours are detected per antenna, without assuming a
schedule: one slope is shared across the week, one level is estimated per
hour of the day, and the twenty-four levels are split at their largest gap.
The selected antennas span the range of amplitudes rather than only the most
spectacular cases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data_processing.power_validation import load_calibrated_population  # noqa: E402

CACHE = Path("data") / "processed" / "calibrated_population_7d.npz"
FIGURES = Path("figures") / "diagnostics"
AMPLITUDE_FLOOR = 0.10


def _detect(d: np.ndarray, p: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the set of low-state hours and the amplitude of the split."""
    present = np.unique(h)
    design = np.zeros((p.size, present.size + 1))
    for column, hour in enumerate(present):
        design[h == hour, column] = 1.0
    design[:, -1] = d
    coefficients, *_ = np.linalg.lstsq(design, p, rcond=None)
    levels = np.full(24, np.nan)
    levels[present] = coefficients[:-1]
    finite = np.isfinite(levels)
    ordered = np.sort(levels[finite])
    cut = int(np.argmax(np.diff(ordered)))
    threshold = 0.5 * (ordered[cut] + ordered[cut + 1])
    amplitude = (ordered[-1] - ordered[0]) / p.mean()
    return finite & (levels < threshold), amplitude


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    pop = load_calibrated_population(CACHE)
    n = len(pop.antenna_ids)
    traffic = pop.traffic_gb.reshape(n, -1)
    power = pop.power_w.reshape(n, -1)
    hours = np.tile(np.arange(24), (n, pop.days.size))
    live = power > 0.0

    records = []
    for k in range(n):
        keep = live[k]
        if keep.sum() < 96 or np.ptp(traffic[k][keep]) <= 0:
            continue
        if np.unique(hours[k][keep]).size < 24:
            continue
        low_hours, amplitude = _detect(traffic[k][keep], power[k][keep],
                                       hours[k][keep])
        if amplitude > AMPLITUDE_FLOOR and 3 <= low_hours.sum() <= 10:
            records.append((k, low_hours, amplitude))

    print(f">> Antennes à deux niveaux retenues : {len(records)}")
    amplitudes = np.array([r[2] for r in records])
    order = np.argsort(amplitudes)
    picks = [records[order[int(round(i))]] for i in
             np.linspace(0.45, 0.99, args.count) * (len(records) - 1)]

    rows = 2
    columns = int(np.ceil(args.count / rows))
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 6.6))
    axes = np.atleast_1d(axes).ravel()

    for panel, (k, low_hours, amplitude) in enumerate(picks):
        keep = live[k]
        d, p, h = traffic[k][keep], power[k][keep], hours[k][keep]
        is_low = low_hours[h]
        axis = axes[panel]
        for sel, colour, label in ((~is_low, "#2A6BB0", "état plein"),
                                   (is_low, "#C0504D", "état réduit")):
            axis.scatter(d[sel], p[sel], s=11, alpha=0.7, color=colour,
                         edgecolors="none", label=label)

        design = np.column_stack([np.ones_like(d), is_low.astype(float), d])
        (level, gap, slope), *_ = np.linalg.lstsq(design, p, rcond=None)
        grid = np.linspace(0.0, float(d.max()) * 1.02, 60)
        axis.plot(grid, level + slope * grid, color="#2A6BB0", linewidth=1.2)
        axis.plot(grid, level + gap + slope * grid, color="#C0504D",
                  linewidth=1.2)
        single = np.polyfit(d, p, 1)
        axis.plot(grid, single[1] + single[0] * grid, color="black",
                  linewidth=1.2, linestyle="--")

        span = _describe(low_hours)
        axis.set_title(f"{pop.antenna_ids[k]}\nécart {abs(gap):.0f} W "
                       f"({amplitude:.0%}), état réduit {span}", fontsize=8)
        axis.set_xlabel("trafic (Go/h)", fontsize=8)
        axis.set_ylabel("puissance (W)", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.set_ylim(bottom=0.0)
        axis.set_xlim(left=0.0)
        axis.grid(alpha=0.25, linewidth=0.5)
        if panel == 0:
            axis.legend(fontsize=6.5, loc="lower right")
        print(f"   {pop.antenna_ids[k]} : écart {abs(gap):.0f} W, "
              f"amplitude {amplitude:.1%}, pente {slope:.1f} W/Go, "
              f"état réduit {span}")

    for axis in axes[len(picks):]:
        axis.axis("off")

    figure.suptitle("Antennes dont le nuage puissance-trafic se scinde en deux "
                    "niveaux ; le pointillé noir est la droite unique ajustée "
                    "sur les 168 heures", fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "two_level_gallery.png"
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(f"\n>> Figure : {output.resolve()}")


def _describe(low_hours: np.ndarray) -> str:
    idx = np.flatnonzero(low_hours)
    if idx.size == 0:
        return "aucun"
    wraps = low_hours[0] and low_hours[23]
    if not wraps:
        return f"{idx[0]:02d}h-{idx[-1] + 1:02d}h"
    start = max(h for h in range(24) if low_hours[h] and not low_hours[h - 1])
    end = min(h for h in range(24) if low_hours[h] and not low_hours[(h + 1) % 24])
    return f"{start:02d}h-{end + 1:02d}h"


if __name__ == "__main__":
    main()
