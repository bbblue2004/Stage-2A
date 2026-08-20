"""Build one virtual site step by step and print every intermediate number.

This is the worked example behind Section 6.1.2. It applies the
selection-based shape rule (each operator inherits the shape of a real
antenna, chosen by correlation rank) and reports, for comparison, what the
convex-mixing rule would have produced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_processing.antenna_metrics import (
    DEFAULT_ELECTRICITY_PRICE_PER_KWH,
    power_coefficients_to_cost,
)
from src.data_processing.data_loader import FULL_CSV_PATH, make_output_path
from src.data_processing.power_validation import (
    calibrated_population,
    load_calibrated_population,
    run_campaign,
    save_calibrated_population,
)

CACHE = Path("data") / "processed" / "calibrated_population_7d.npz"
SHAPE_BANDS = {"close": None, "moderate": 0.99, "distant": 0.50}


def _population(num_days: int):
    if CACHE.is_file():
        try:
            print(f">> Reusing {CACHE}", flush=True)
            return load_calibrated_population(CACHE)
        except Exception:
            print("   cache unusable, recomputing", flush=True)
    print(f">> Calibrating the population over {num_days} days", flush=True)
    campaign = run_campaign(FULL_CSV_PATH, num_days=num_days)
    population = calibrated_population(campaign)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    save_calibrated_population(population, CACHE)
    return population


def _shapes(traffic: np.ndarray) -> np.ndarray:
    flat = traffic.reshape(traffic.shape[0], -1)
    return flat / flat.mean(axis=1, keepdims=True)


def _correlations(shapes: np.ndarray, row: np.ndarray) -> np.ndarray:
    left = shapes - shapes.mean(axis=1, keepdims=True)
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    right = row - row.mean()
    right /= np.linalg.norm(right)
    return left @ right


def _rule(values: np.ndarray, label: str, unit: str) -> str:
    return (
        f"{label}: median {np.median(values):.3f} {unit}, "
        f"q25 {np.quantile(values, 0.25):.3f}, q75 {np.quantile(values, 0.75):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-days", type=int, default=7)
    parser.add_argument("--num-operators", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--rate", type=float, default=0.70)
    parser.add_argument("--shape-level", default="moderate", choices=list(SHAPE_BANDS))
    args = parser.parse_args()

    population = _population(args.num_days)
    n_antennas = int(population.antenna_ids.size)
    n_days = int(population.days.size)
    shapes = _shapes(population.traffic_gb)
    mean_traffic = population.traffic_gb.mean(axis=(1, 2))
    n_op = args.num_operators
    rng = np.random.default_rng(args.seed)

    print("\n" + "=" * 74)
    print("ÉTAPE 0 — population admissible")
    print("=" * 74)
    print(f"   antennes admissibles : {n_antennas}")
    print(f"   jours                : {n_days}  ({n_days * 24} heures par antenne)")
    print("   " + _rule(mean_traffic, "trafic moyen", "Go/h"))
    print("   " + _rule(population.p_fixed_w, "puissance fixe", "W"))
    print("   " + _rule(population.slope_w_per_gb, "pente", "W/(Go/h)"))

    print("\n" + "=" * 74)
    print("ÉTAPE 1 — tirage de l'antenne de référence")
    print("=" * 74)
    reference = int(rng.integers(n_antennas))
    reference_id = str(population.antenna_ids[reference])
    mu = float(mean_traffic[reference])
    quartile = int(population.traffic_group[reference])
    rank = float(np.mean(mean_traffic <= mu))
    print(f"   graine               : {args.seed}")
    print(f"   antenne de référence : {reference_id}  (indice {reference})")
    print(f"   quartile de trafic   : {quartile + 1} sur 4  (rang {rank:.1%})")
    print(f"   mu_a = trafic moyen  : {mu:.4f} Go/h")
    print(f"   pic observé          : {population.traffic_gb[reference].max():.4f} Go/h")
    print(f"   creux observé        : {population.traffic_gb[reference].min():.4f} Go/h")

    print("\n" + "=" * 74)
    print("ÉTAPE 2 — profil normalisé de la référence")
    print("=" * 74)
    s_a = shapes[reference]
    daily = s_a.reshape(n_days, 24).mean(axis=0)
    print(f"   s_a(t) = d_a(t) / mu_a,  moyenne = {s_a.mean():.6f}  (doit valoir 1)")
    print(f"   amplitude : min {s_a.min():.3f}, max {s_a.max():.3f}")
    print("   profil journalier moyen, heure par heure :")
    for start in range(0, 24, 8):
        row = "  ".join(f"{h:02d}h={daily[h]:.2f}" for h in range(start, start + 8))
        print(f"      {row}")

    print("\n" + "=" * 74)
    print("ÉTAPE 3 — facteurs de taille alpha")
    print("=" * 74)
    quantiles = (0.40, 0.50, 0.60, 0.75)[:n_op]
    raw = np.quantile(mean_traffic, quantiles)
    alpha_sorted = raw / raw.mean()
    permutation = rng.permutation(n_op)
    alpha = alpha_sorted[permutation]
    print(f"   niveau « modérés » = quantiles {quantiles} du trafic moyen")
    print("   " + "  ".join(f"q{int(100*q)}={v:.4f} Go/h" for q, v in zip(quantiles, raw)))
    print(f"   division par la moyenne {raw.mean():.4f} -> alpha trié = "
          + ", ".join(f"{v:.3f}" for v in alpha_sorted))
    print(f"   permutation aléatoire {[int(v) for v in permutation]} -> alpha = "
          + ", ".join(f"{v:.3f}" for v in alpha))
    print(f"   contrôle : moyenne(alpha) = {alpha.mean():.6f}  (doit valoir 1)")

    print("\n" + "=" * 74)
    print("ÉTAPE 4 — formes temporelles des opérateurs")
    print("=" * 74)
    corr = _correlations(shapes, population.traffic_gb[reference].reshape(-1))
    corr[reference] = -np.inf
    band = SHAPE_BANDS[args.shape_level]
    if band is None:
        donors = np.full(n_op - 1, reference, dtype=int)
        print("   niveau « proches » : tous les opérateurs héritent de s_a")
    else:
        target = float(np.quantile(corr[np.isfinite(corr)], band))
        pool = np.argsort(np.abs(corr - target))[: max(20, 3 * n_op)]
        donors = rng.choice(pool, size=n_op - 1, replace=False)
        print(f"   niveau « {args.shape_level} » : donneuses tirées autour du "
              f"quantile {band:.0%} de corrélation avec s_a")
        print(f"   corrélation visée : {target:+.3f}")
    operator_shape_index = np.concatenate(([reference], donors))
    for i, index in enumerate(operator_shape_index):
        origin = "référence" if index == reference else "donneuse "
        value = 1.0 if index == reference else float(corr[index])
        print(f"      opérateur {i + 1} : {origin} {population.antenna_ids[index]!s:>12}"
              f"   corr(s_a, s_i) = {value:+.3f}")
    operator_shapes = shapes[operator_shape_index]

    print("\n" + "=" * 74)
    print("ÉTAPE 5 — antennes énergétiques")
    print("=" * 74)
    forbidden = set(int(v) for v in operator_shape_index)
    pool = np.array([i for i in range(n_antennas) if i not in forbidden])
    energy = rng.choice(pool, size=n_op, replace=False)
    p_fixed = population.p_fixed_w[energy]
    slope = population.slope_w_per_gb[energy]
    print("   niveau « moderate » : tirage uniforme, disjoint des sources de trafic")
    print(f"   prix retenu : {DEFAULT_ELECTRICITY_PRICE_PER_KWH} EUR/kWh")
    print(f"\n   {'op':>3} {'antenne':>12} {'P_fixe (W)':>11} {'pente (W/Go)':>13}"
          f" {'F_i (EUR/h)':>12} {'gamma_i':>11}")
    costs = []
    for i, index in enumerate(energy):
        f_cost, g_cost = power_coefficients_to_cost(
            float(p_fixed[i]), float(slope[i])
        )
        costs.append((f_cost, g_cost))
        print(f"   {i + 1:>3} {str(population.antenna_ids[index]):>12}"
              f" {p_fixed[i]:>11.1f} {slope[i]:>13.2f}"
              f" {f_cost:>12.6f} {g_cost:>11.6f}")

    print("\n" + "=" * 74)
    print("ÉTAPE 6 — demandes construites")
    print("=" * 74)
    traffic = mu * alpha[:, None] * operator_shapes
    traffic = traffic.reshape(n_op, n_days, 24)
    peaks = traffic.max(axis=(1, 2))
    print("   d_i(t) = mu_a * alpha_i * s_i(t)")
    print(f"\n   {'op':>3} {'alpha':>7} {'moyenne':>10} {'pic':>10} {'creux':>10}"
          f" {'pic/moyenne':>12}")
    for i in range(n_op):
        series = traffic[i]
        print(f"   {i + 1:>3} {alpha[i]:>7.3f} {series.mean():>10.4f}"
              f" {series.max():>10.4f} {series.min():>10.4f}"
              f" {series.max() / series.mean():>12.2f}")
    print(f"\n   somme des moyennes  : {traffic.mean(axis=(1, 2)).sum():.4f} Go/h")
    print(f"   n * mu_a            : {n_op * mu:.4f} Go/h   (doit coïncider)")

    lam = 0.35
    sample = rng.choice(n_antennas, size=(2, 4000), replace=True)
    keep = sample[0] != sample[1]
    left, right = shapes[sample[0][keep]], shapes[sample[1][keep]]
    mixed = (1.0 - lam) * left + lam * right
    real_ratio = left.max(axis=1)
    mixed_ratio = mixed.max(axis=1)
    print(f"\n   ancienne règle de mélange, lambda = {lam}, donneuse uniforme :")
    print(f"      pic/moyenne réel  : médiane {np.median(real_ratio):.3f}")
    print(f"      pic/moyenne mêlé  : médiane {np.median(mixed_ratio):.3f}")
    print(f"      perte de pic      : {100 * (1 - np.median(mixed_ratio) / np.median(real_ratio)):.1f} %")

    print("\n" + "=" * 74)
    print(f"ÉTAPE 7 — capacités, scénario r = {args.rate:.2f}")
    print("=" * 74)
    capacities = peaks / args.rate
    print("   q_i tel que max_t d_i(t) = r * q_i")
    print(f"\n   {'op':>3} {'pic (Go/h)':>11} {'q_i (Go/h)':>11} {'marge':>8}")
    for i in range(n_op):
        print(f"   {i + 1:>3} {peaks[i]:>11.4f} {capacities[i]:>11.4f}"
              f" {capacities[i] - peaks[i]:>8.4f}")
    print(f"\n   capacité totale du site : {capacities.sum():.4f} Go/h")

    print("\n" + "=" * 74)
    print("ÉTAPE 8 — seuil k_h, heure par heure")
    print("=" * 74)
    order = np.sort(capacities)[::-1]
    cumulative = np.cumsum(order)
    hourly_total = traffic.sum(axis=0).mean(axis=0)
    peak_total = traffic.sum(axis=0).max(axis=0)
    print("   k_h = plus petit nombre d'équipements couvrant la demande totale")
    print(f"\n   {'h':>3} {'demande moy.':>13} {'k_h moy.':>9}"
          f" {'demande max':>12} {'k_h max':>8}")
    counts_mean = []
    for hour in range(24):
        k_mean = int(np.searchsorted(cumulative, hourly_total[hour], "left") + 1)
        k_peak = int(np.searchsorted(cumulative, peak_total[hour], "left") + 1)
        counts_mean.append(k_mean)
        print(f"   {hour:>3} {hourly_total[hour]:>13.4f} {k_mean:>9d}"
              f" {peak_total[hour]:>12.4f} {k_peak:>8d}")
    print(f"\n   nuit 0h-7h : k_h = {sorted(set(counts_mean[:7]))}")
    print(f"   pointe     : k_h = {max(counts_mean)}")

    print("\n" + "=" * 74)
    print("ÉTAPE 8bis — taux de charge d_i / q_i, selon l'agrégation")
    print("=" * 74)
    hourly_mean = traffic.mean(axis=1)
    hourly_max = traffic.max(axis=1)
    print("   Pour chaque heure de la journée on dispose de 7 valeurs, une par jour.")
    print("   Le pic sur 168 h qui fixe q_i n'est atteint qu'un seul de ces 7 jours.")
    print(f"\n   {'op':>3} {'max168':>9} {'max du max':>11} {'max de moy.':>12}"
          f" {'atténuation':>12}")
    for i in range(n_op):
        best = float(traffic[i].max())
        by_max = float(hourly_max[i].max())
        by_mean = float(hourly_mean[i].max())
        print(f"   {i + 1:>3} {best:>9.3f} {by_max:>11.3f} {by_mean:>12.3f}"
              f" {by_mean / best:>12.3f}")
    print("\n   taux de charge maximal atteint, selon r et selon l'agrégation :")
    print(f"\n   {'r':>6} {'sur le max des 7 j':>20} {'sur la moyenne des 7 j':>24}")
    for r in (0.70, 0.80, 0.90, 1.00):
        caps = peaks / r
        by_max = float((hourly_max / caps[:, None]).max())
        by_mean = float((hourly_mean / caps[:, None]).max())
        print(f"   {r:>6.2f} {by_max:>20.3f} {by_mean:>24.3f}")
    print("\n   d_i(t)/q_i = r * d_i(t)/max_t d_i(t) : la forme ne dépend pas de r,")
    print("   seule l'échelle verticale change. Une seule courbe suffit en figure.")

    print("\n" + "=" * 74)
    print("ÉTAPE 9 — la grille balaie-t-elle réellement les seuils ?")
    print("=" * 74)
    print("   max_h k_h sur les 168 heures, pour ce même site de référence")
    volume_levels = {
        "proches": (),
        "modérés": (0.40, 0.50, 0.60, 0.75),
        "distants": (0.25, 0.40, 0.60, 0.90),
    }
    shape_levels = {"proches": None, "modérés": 0.99, "distants": 0.50}
    header = f"   {'volume':>10} {'forme':>10}" + "".join(
        f" {'r=' + format(r, '.2f'):>10}" for r in (0.70, 0.80, 0.90, 1.00)
    )
    print("\n" + header)
    for vname, vq in volume_levels.items():
        base = (np.ones(n_op) if not vq
                else np.quantile(mean_traffic, vq[:n_op]) / np.quantile(mean_traffic, vq[:n_op]).mean())
        for sname, sband in shape_levels.items():
            if sband is None:
                index = np.full(n_op, reference, dtype=int)
            else:
                tgt = float(np.quantile(corr[np.isfinite(corr)], sband))
                cand = np.argsort(np.abs(corr - tgt))[: max(20, 3 * n_op)]
                index = np.concatenate(([reference], cand[: n_op - 1]))
            series = (mu * base[:, None] * shapes[index]).reshape(n_op, -1)
            pk = series.max(axis=1)
            cells = []
            for r in (0.70, 0.80, 0.90, 1.00):
                caps = np.sort(pk / r)[::-1]
                cum = np.cumsum(caps)
                totals = series.sum(axis=0)
                ks = np.searchsorted(cum, totals - 1e-9, "left") + 1
                cells.append(f" {int(ks.max()):>10d}")
            print(f"   {vname:>10} {sname:>10}" + "".join(cells))
    print("\n   lecture : k_h = 4 exige des volumes proches, des formes proches")
    print("   et r proche de 1 ; l'hétérogénéité fait baisser k_h.")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    hours = np.arange(n_days * 24)

    axes[0][0].plot(hours, population.traffic_gb[reference].reshape(-1),
                    color="#1f4e79", linewidth=1.0)
    axes[0][0].set_title(f"(a) entrée réelle : {reference_id}", fontsize=11)
    axes[0][0].set_ylabel("trafic (Go/h)")

    for i, index in enumerate(operator_shape_index):
        label = "référence" if index == reference else f"donneuse {i}"
        axes[0][1].plot(np.arange(24), shapes[index].reshape(n_days, 24).mean(axis=0),
                        linewidth=1.4, label=label)
    axes[0][1].set_title("(b) formes normalisées, moyenne journalière", fontsize=11)
    axes[0][1].set_ylabel("profil (moyenne 1)")
    axes[0][1].set_xlabel("heure")
    axes[0][1].legend(fontsize=8)

    for i in range(n_op):
        axes[1][0].plot(hours, traffic[i].reshape(-1), linewidth=1.0,
                        label=f"op. {i + 1}, alpha={alpha[i]:.2f}")
    axes[1][0].set_title("(c) sortie : les quatre demandes", fontsize=11)
    axes[1][0].set_ylabel("trafic (Go/h)")
    axes[1][0].set_xlabel("heure de la semaine")
    axes[1][0].legend(fontsize=8)

    clock = np.arange(24)
    for i in range(n_op):
        rate_mean = traffic[i].mean(axis=0) / peaks[i]
        rate_low = traffic[i].min(axis=0) / peaks[i]
        rate_high = traffic[i].max(axis=0) / peaks[i]
        line, = axes[1][1].plot(clock, rate_mean, linewidth=1.4,
                                label=f"op. {i + 1}, moyenne des 7 jours")
        axes[1][1].fill_between(clock, rate_low, rate_high, alpha=0.12,
                                color=line.get_color(), linewidth=0)
    for r, style in ((1.00, "-"), (0.90, "--"), (0.80, "-."), (0.70, ":")):
        axes[1][1].axhline(1.0 / r, color="black", linewidth=0.9, linestyle=style)
        axes[1][1].annotate(f"capacité, r={r:.2f}", (23.6, 1.0 / r + 0.012),
                            fontsize=7, va="bottom", ha="right", color="black")
    axes[1][1].set_title("(d) demande rapportée au pic, et capacités selon r",
                         fontsize=11)
    axes[1][1].set_ylabel("$d_i(t)\\,/\\,\\max_t d_i(t)$")
    axes[1][1].set_xlabel("heure")
    axes[1][1].set_ylim(0.0, 1.58)
    axes[1][1].legend(fontsize=7, loc="lower right")

    for row in axes:
        for axis in row:
            axis.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    output = make_output_path("plan_walkthrough.png")
    figure.savefig(output, dpi=140)
    vector = Path("figures") / "protocol" / "plan_walkthrough.pdf"
    vector.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(vector)
    plt.close(figure)
    print(f"\n>> Figure : {output.resolve()}")
    print(f">> Figure : {vector.resolve()}")


if __name__ == "__main__":
    main()
