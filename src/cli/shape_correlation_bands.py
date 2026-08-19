"""Distribution of pairwise correlations between normalised traffic shapes.

Checks whether operator-shape heterogeneity can be controlled by *selecting*
real donor antennas inside a target correlation band, instead of blending two
real profiles into a synthetic one.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from src.data_processing.data_loader import FULL_CSV_PATH, iter_records


def _load_shapes(csv_path) -> tuple[list[str], np.ndarray]:
    series: dict[str, dict[int, float]] = defaultdict(dict)
    stamps: dict[str, int] = {}
    for timestamp, antenna_id, traffic, _power in iter_records(csv_path):
        key = timestamp.isoformat()
        index = stamps.setdefault(key, len(stamps))
        series[antenna_id][index] = traffic

    horizon = len(stamps)
    ids: list[str] = []
    rows: list[np.ndarray] = []
    for antenna_id, values in series.items():
        if len(values) != horizon:
            continue
        row = np.empty(horizon, dtype=float)
        for index, value in values.items():
            row[index] = value
        mean = row.mean()
        if mean <= 0.0 or row.std() <= 0.0:
            continue
        ids.append(antenna_id)
        rows.append(row / mean)
    return ids, np.asarray(rows, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=FULL_CSV_PATH)
    parser.add_argument("--sample", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    print(">> Reading the source CSV once", flush=True)
    ids, shapes = _load_shapes(args.input)
    print(f"   usable antennas: {len(ids)}, hours: {shapes.shape[1]}", flush=True)

    rng = np.random.default_rng(args.seed)
    size = min(args.sample, shapes.shape[0])
    picked = rng.choice(shapes.shape[0], size=size, replace=False)
    block = shapes[picked]

    centred = block - block.mean(axis=1, keepdims=True)
    centred /= np.linalg.norm(centred, axis=1, keepdims=True)
    matrix = centred @ centred.T
    upper = matrix[np.triu_indices(size, k=1)]

    print(f"\n>> Pairwise shape correlation over {size} antennas "
          f"({upper.size} pairs)")
    for level in (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
        print(f"   q{level:.2f} = {np.quantile(upper, level):+.3f}", flush=True)

    print("\n>> Share of pairs inside candidate bands")
    for low, high in ((0.95, 1.00), (0.90, 0.95), (0.80, 0.90),
                      (0.60, 0.80), (0.30, 0.60), (-1.0, 0.30)):
        share = float(np.mean((upper >= low) & (upper < high)))
        print(f"   [{low:+.2f}, {high:+.2f}) : {share:7.2%}", flush=True)

    print("\n>> Per-reference availability of donors, by band")
    for low, high in ((0.95, 1.00), (0.90, 0.95), (0.80, 0.90), (0.30, 0.60)):
        counts = ((matrix >= low) & (matrix < high)).sum(axis=1)
        empty = float(np.mean(counts < 3))
        print(
            f"   [{low:+.2f}, {high:+.2f}) : median {np.median(counts):6.0f} donors, "
            f"references with fewer than 3 donors: {empty:.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
