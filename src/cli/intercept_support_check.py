"""Quantify how far P_fixed is extrapolated below the observed traffic range.

F_i is read off the fitted intercept at d = 0. If an antenna never operates
near zero traffic, that intercept lies outside the observed support. This
script measures that gap over the whole population, and whether the weekend
extends the low-traffic range.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from src.data_processing.data_loader import FULL_CSV_PATH, iter_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=FULL_CSV_PATH)
    args = parser.parse_args()

    weekday: dict[str, list[float]] = defaultdict(list)
    weekend: dict[str, list[float]] = defaultdict(list)
    zero_power = 0
    zero_traffic_active = 0
    total_active = 0

    print(">> Reading the source CSV once", flush=True)
    for timestamp, antenna_id, traffic, power in iter_records(args.input):
        if power <= 0.0:
            zero_power += 1
            continue
        total_active += 1
        if traffic <= 0.0:
            zero_traffic_active += 1
        bucket = weekend if timestamp.weekday() >= 5 else weekday
        bucket[antenna_id].append(traffic)

    ratios_week: list[float] = []
    ratios_full: list[float] = []
    improved = 0
    for antenna_id, week_values in weekday.items():
        end_values = weekend.get(antenna_id, [])
        if not week_values:
            continue
        week = np.asarray(week_values, dtype=float)
        peak = float(week.max())
        if peak <= 0.0:
            continue
        min_week = float(week.min())
        ratios_week.append(min_week / peak)
        if end_values:
            full = np.concatenate((week, np.asarray(end_values, dtype=float)))
            min_full = float(full.min())
            ratios_full.append(min_full / float(full.max()))
            if min_full < min_week - 1e-12:
                improved += 1

    def describe(label: str, values: list[float]) -> None:
        array = np.asarray(values, dtype=float)
        q = np.quantile(array, (0.10, 0.25, 0.50, 0.75, 0.90))
        print(
            f"   {label}: n={array.size}  "
            f"q10={q[0]:.3f} q25={q[1]:.3f} median={q[2]:.3f} "
            f"q75={q[3]:.3f} q90={q[4]:.3f}",
            flush=True,
        )
        for threshold in (0.05, 0.10, 0.20, 0.30):
            share = float(np.mean(array > threshold))
            print(f"      share with min/max > {threshold:.2f}: {share:.1%}", flush=True)

    print("\n>> Active-hour accounting")
    print(f"   hours with zero power (excluded): {zero_power}")
    print(f"   active hours: {total_active}")
    print(
        f"   active hours with exactly zero traffic: {zero_traffic_active} "
        f"({zero_traffic_active / max(total_active, 1):.3%})"
    )

    print("\n>> Ratio min(active traffic) / max(traffic) per antenna")
    describe("weekdays only ", ratios_week)
    describe("all seven days", ratios_full)
    print(
        f"\n   antennas whose minimum traffic is lowered by the weekend: "
        f"{improved} / {len(ratios_full)} ({improved / max(len(ratios_full), 1):.1%})",
        flush=True,
    )


if __name__ == "__main__":
    main()
