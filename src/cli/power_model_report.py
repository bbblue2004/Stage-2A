"""Numbers and figure for Section 6.2, computed on the full seven-day window.

Reports the admissibility counts, the goodness-of-fit indicators, the
five-day versus seven-day stability of the coefficients, and draws the
representative scatter with weekdays and weekend distinguished.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_processing.data_loader import FULL_CSV_PATH
from src.data_processing.power_validation import (
    calibrated_population,
    load_calibrated_population,
    run_campaign,
    save_calibrated_population,
    selection_counts,
)

CACHE = Path("data") / "processed"
FIGURES = Path("figures") / "power_calibration"


def _campaign(num_days: int):
    path = CACHE / f"calibrated_population_{num_days}d.npz"
    campaign = run_campaign(FULL_CSV_PATH, num_days=num_days)
    population = calibrated_population(campaign)
    CACHE.mkdir(parents=True, exist_ok=True)
    save_calibrated_population(population, path)
    return campaign, population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    print(">> Seven-day calibration", flush=True)
    campaign7, pop7 = _campaign(7)
    print(">> Five-day calibration", flush=True)
    campaign5, pop5 = _campaign(5)

    print("\n" + "=" * 70)
    print("ADMISSIBILITÉ — 7 jours")
    print("=" * 70)
    counts = selection_counts(campaign7.antenna_results)
    total = sum(counts.values())
    for reason, value in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {reason:<34} {value:>6}  ({value / total:6.2%})")
    print(f"   {'TOTAL':<34} {total:>6}")

    print("\n" + "=" * 70)
    print("QUALITÉ D'AJUSTEMENT — 7 jours")
    print("=" * 70)
    rmse = pop7.normalized_rmse
    r2 = pop7.r_squared
    mean_traffic = pop7.traffic_gb.mean(axis=(1, 2))
    fixed_share = pop7.p_fixed_w / (pop7.p_fixed_w + pop7.slope_w_per_gb * mean_traffic)
    for name, values, fmt in (
        ("RMSE normalisée", rmse, "{:.4f}"),
        ("R^2", r2, "{:.4f}"),
        ("part fixe au trafic moyen", fixed_share, "{:.4f}"),
    ):
        q = np.quantile(values, (0.10, 0.25, 0.50, 0.75, 0.90))
        print(f"   {name:<28} médiane {fmt.format(q[2])}   "
              f"q10 {fmt.format(q[0])}  q25 {fmt.format(q[1])}  "
              f"q75 {fmt.format(q[3])}  q90 {fmt.format(q[4])}")
    for threshold in (0.10, 0.15, 0.20):
        print(f"   part des antennes avec RMSE normalisée < {threshold:.0%} : "
              f"{np.mean(rmse < threshold):.1%}")
    print(f"   part des antennes avec R^2 > 0,80 : {np.mean(r2 > 0.80):.1%}")
    print(f"   part fixe > 50 % : {np.mean(fixed_share > 0.50):.1%}")

    print("\n" + "=" * 70)
    print("STABILITÉ 5 JOURS CONTRE 7 JOURS")
    print("=" * 70)
    ids7 = {str(a): i for i, a in enumerate(pop7.antenna_ids)}
    ids5 = {str(a): i for i, a in enumerate(pop5.antenna_ids)}
    common = sorted(set(ids7) & set(ids5))
    i7 = np.array([ids7[a] for a in common])
    i5 = np.array([ids5[a] for a in common])
    print(f"   antennes admissibles à 7 jours : {pop7.antenna_ids.size}")
    print(f"   antennes admissibles à 5 jours : {pop5.antenna_ids.size}")
    print(f"   antennes communes              : {len(common)}")
    for name, a7, a5 in (
        ("P_fixe", pop7.p_fixed_w[i7], pop5.p_fixed_w[i5]),
        ("pente", pop7.slope_w_per_gb[i7], pop5.slope_w_per_gb[i5]),
    ):
        rel = np.abs(a7 - a5) / np.maximum(np.abs(a5), 1e-12)
        q = np.quantile(rel, (0.50, 0.90, 0.99))
        print(f"   {name:<8} écart relatif : médiane {q[0]:.3%}, "
              f"q90 {q[1]:.3%}, q99 {q[2]:.3%}")

    print("\n" + "=" * 70)
    print("STRUCTURE DU NUAGE — 7 jours")
    print("=" * 70)
    traffic = pop7.traffic_gb.reshape(pop7.antenna_ids.size, -1)
    power = pop7.power_w.reshape(pop7.antenna_ids.size, -1)
    active = power > 0.0
    zero_traffic_active = np.sum(active & (traffic <= 0.0))
    print(f"   heures actives                       : {active.sum()}")
    print(f"   dont trafic exactement nul           : {zero_traffic_active} "
          f"({zero_traffic_active / active.sum():.2%})")
    ratios = []
    for k in range(pop7.antenna_ids.size):
        values = traffic[k][active[k]]
        if values.size and values.max() > 0.0:
            ratios.append(values.min() / values.max())
    ratios = np.asarray(ratios)
    print(f"   min/max du trafic actif, médiane     : {np.median(ratios):.4f}")
    print(f"   antennes avec min/max > 0,10         : {np.mean(ratios > 0.10):.1%}")

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    letters = "abc"
    for column, (quantile, caption) in enumerate(
        ((0.25, "bon ajustement"), (0.50, "ajustement médian"),
         (0.90, "ajustement médiocre"))
    ):
        pick = int(np.argmin(np.abs(rmse - np.quantile(rmse, quantile))))
        d = pop7.traffic_gb[pick]
        p = pop7.power_w[pick]
        weekday = np.zeros_like(d, dtype=bool)
        weekday[:5] = True
        night = np.tile(np.arange(24) <= 5, (d.shape[0], 1))
        mask = p > 0.0
        axis = axes[column]
        for is_night, colour, tag in ((False, "#2A6BB0", "6 h-24 h"),
                                      (True, "#C0504D", "0 h-6 h")):
            for is_weekday, marker, day_tag in ((True, "o", "semaine"),
                                                (False, "^", "week-end")):
                sel = mask & (night == is_night) & (weekday == is_weekday)
                axis.scatter(d[sel], p[sel], s=15, alpha=0.65, color=colour,
                             marker=marker, edgecolors="none",
                             label=f"{tag}, {day_tag}")
        grid = np.linspace(0.0, float(d[mask].max()), 100)
        axis.plot(grid, pop7.p_fixed_w[pick] + pop7.slope_w_per_gb[pick] * grid,
                  color="black", linewidth=1.3)
        axis.axhline(pop7.p_fixed_w[pick], color="black", linewidth=0.8,
                     linestyle=":")
        axis.set_xlabel("trafic descendant (Go/h)")
        axis.set_ylabel("puissance moyenne (W)")
        axis.set_title(
            f"({letters[column]}) {caption}\n"
            f"$P^{{\\mathrm{{fixe}}}}$={pop7.p_fixed_w[pick]:.0f} W, "
            f"$s$={pop7.slope_w_per_gb[pick]:.1f} W/Go, "
            f"RMSE={pop7.normalized_rmse[pick]:.1%}, "
            f"$R^2$={pop7.r_squared[pick]:.2f}",
            fontsize=9,
        )
        axis.set_ylim(bottom=0.0)
        axis.set_xlim(left=0.0)
        axis.grid(alpha=0.25, linewidth=0.5)
        if column == 0:
            axis.legend(fontsize=7.5, loc="lower right")
        residual = p[mask] - (pop7.p_fixed_w[pick]
                              + pop7.slope_w_per_gb[pick] * d[mask])
        print(f"\n   antenne au quantile {quantile:.0%} de RMSE : "
              f"{pop7.antenna_ids[pick]}")
        print(f"      P_fixe = {pop7.p_fixed_w[pick]:.1f} W, "
              f"pente = {pop7.slope_w_per_gb[pick]:.2f} W/Go, "
              f"R^2 = {pop7.r_squared[pick]:.3f}, "
              f"RMSE norm. = {pop7.normalized_rmse[pick]:.3f}")
        print(f"      résidu moyen {residual.mean():+.2f} W, "
              f"écart-type {residual.std():.1f} W")
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "representative_power_fit.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=140)
    plt.close(figure)
    print(f"\n>> Figure : {output.resolve()}")


if __name__ == "__main__":
    main()
