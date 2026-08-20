"""Which hours of the day fall into the reduced-power state?

For each antenna, one slope is shared across the week and one level is
estimated per hour of the day:

    P(h) = beta_{hour(h)} + gamma * d(h)

The profile of the twenty-four levels is then split into a low group and a
high group at the largest gap, without assuming any schedule. The script
reports which hours end up in the low group.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data_processing.power_validation import load_calibrated_population  # noqa: E402

CACHE = Path("data") / "processed" / "calibrated_population_7d.npz"
FIGURES = Path("figures") / "diagnostics"
AMPLITUDE_FLOOR = 0.10


def main() -> None:
    pop = load_calibrated_population(CACHE)
    n = len(pop.antenna_ids)
    traffic = pop.traffic_gb.reshape(n, -1)
    power = pop.power_w.reshape(n, -1)
    hours = np.tile(np.arange(24), (n, pop.days.size))
    live = power > 0.0

    levels = np.full((n, 24), np.nan)
    amplitude = np.full(n, np.nan)
    for k in range(n):
        keep = live[k]
        d, p, h = traffic[k][keep], power[k][keep], hours[k][keep]
        if p.size < 48 or np.ptp(d) <= 0:
            continue
        present = np.unique(h)
        if present.size < 20:
            continue
        design = np.zeros((p.size, present.size + 1))
        for column, hour in enumerate(present):
            design[h == hour, column] = 1.0
        design[:, -1] = d
        coefficients, *_ = np.linalg.lstsq(design, p, rcond=None)
        levels[k, present] = coefficients[:-1]
        amplitude[k] = (np.nanmax(levels[k]) - np.nanmin(levels[k])) / p.mean()

    usable = np.isfinite(amplitude)
    two_state = usable & (amplitude > AMPLITUDE_FLOOR)
    print(f">> Antennes exploitables : {usable.sum()}")
    print(f"   dont amplitude horaire supérieure à {AMPLITUDE_FLOOR:.0%} "
          f"de la puissance moyenne : {two_state.sum()} "
          f"({two_state.sum() / usable.sum():.1%})")
    print(f"   amplitude médiane : {np.nanmedian(amplitude[usable]):.3f}")

    low_mask = np.zeros((n, 24), dtype=bool)
    for k in np.flatnonzero(two_state):
        row = levels[k]
        finite = np.isfinite(row)
        order = np.argsort(row[finite])
        sorted_values = row[finite][order]
        gaps = np.diff(sorted_values)
        cut = int(np.argmax(gaps))
        threshold = 0.5 * (sorted_values[cut] + sorted_values[cut + 1])
        low_mask[k] = finite & (row < threshold)

    counts = low_mask[two_state].sum(axis=0)
    share = counts / two_state.sum()
    print("\n>> Fréquence d'appartenance à l'état bas, par heure de la journée")
    for start in range(0, 24, 8):
        print("   " + "   ".join(
            f"{hour:02d}h {share[hour]:5.1%}" for hour in range(start, start + 8)
        ))

    sets = low_mask[two_state]
    sizes = sets.sum(axis=1)
    print("\n>> Taille de l'état bas, en heures par jour")
    print(f"   médiane {np.median(sizes):.0f}   q25 {np.quantile(sizes, 0.25):.0f}"
          f"   q75 {np.quantile(sizes, 0.75):.0f}")

    def contiguous_circular(row: np.ndarray) -> bool:
        idx = np.flatnonzero(row)
        if idx.size == 0 or idx.size == 24:
            return True
        doubled = np.concatenate([row, row])
        runs, current = [], 0
        for value in doubled:
            if value:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
        return max(runs) >= idx.size

    print("\n>> Structure de l'état bas")
    contiguous = np.array([contiguous_circular(r) for r in sets])
    print(f"   plage horaire d'un seul tenant, en tenant compte du passage "
          f"par minuit : {contiguous.mean():.1%}")
    exact = np.array([
        np.array_equal(np.flatnonzero(r), np.arange(6)) for r in sets
    ])
    print(f"   exactement les heures 0 h-6 h : {exact.mean():.1%}")
    within = np.array([
        set(np.flatnonzero(r)).issubset({22, 23, 0, 1, 2, 3, 4, 5, 6, 7})
        for r in sets
    ])
    print(f"   entièrement inclus dans 22 h-8 h : {within.mean():.1%}")
    covers = np.array([r[1:5].all() for r in sets])
    print(f"   contient au moins les heures 1 h-5 h : {covers.mean():.1%}")

    starts, ends = [], []
    for r in sets:
        idx = np.flatnonzero(r)
        if idx.size == 0 or idx.size == 24:
            continue
        rolled = [(h, r[h] and not r[(h - 1) % 24]) for h in range(24)]
        begin = [h for h, flag in rolled if flag]
        finish = [h for h in range(24) if r[h] and not r[(h + 1) % 24]]
        if len(begin) == 1 and len(finish) == 1:
            starts.append(begin[0])
            ends.append((finish[0] + 1) % 24)
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    print(f"\n   parmi les plages d'un seul tenant ({starts.size} antennes) :")
    for tag, values in (("début", starts), ("fin  ", ends)):
        uniques, occurrences = np.unique(values, return_counts=True)
        top = np.argsort(occurrences)[::-1][:5]
        detail = "   ".join(
            f"{uniques[i]:02d}h {occurrences[i] / values.size:.0%}" for i in top
        )
        print(f"      {tag} le plus fréquent : {detail}")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axis = axes[0]
    axis.bar(np.arange(24), share * 100.0, color="#C0504D", alpha=0.85)
    axis.set_xlabel("heure de la journée")
    axis.set_ylabel("part des antennes en état bas (%)")
    axis.set_title(f"(a) sur les {two_state.sum()} antennes à deux niveaux,\n"
                   "quelles heures sont en état bas", fontsize=9)
    axis.set_xticks(range(0, 24, 2))
    axis.grid(alpha=0.25, linewidth=0.5, axis="y")

    axis = axes[1]
    axis.hist(sizes, bins=np.arange(0.5, 24.5, 1.0), color="#2A6BB0", alpha=0.85)
    axis.set_xlabel("durée de l'état bas (heures par jour)")
    axis.set_ylabel("antennes")
    axis.set_title(f"(b) durée, médiane {np.median(sizes):.0f} h", fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.5, axis="y")

    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "night_state_hours.png"
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(f"\n>> Figure : {output.resolve()}")


if __name__ == "__main__":
    main()
