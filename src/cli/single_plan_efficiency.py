"""Operational efficiency on the single plan of Section 6.1.

Reconstructs the walkthrough site (seed 20260819, central scenario) and
evaluates the hourly optimum, two feasible approximations, and the
persistent policy. All 168 hours are solved separately, then averaged by
hour of day.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.window_optimiser import (
    descending_capacity_hourly_policy,
    optimal_hourly_with_persistent,
    proportional_hourly_policy,
    standalone_energy,
)
from src.data_processing.instance_generator import CAMPAIGN_A_RATES, CENTRAL_RATE
from src.data_processing.power_validation import load_calibrated_population

CACHE = Path("data") / "processed" / "calibrated_population_7d.npz"
FIGURES = Path("figures") / "operational_efficiency"
SEED = 20260819
WINDOW = tuple(range(7))
RATES = CAMPAIGN_A_RATES


def _shapes(traffic: np.ndarray) -> np.ndarray:
    flat = traffic.reshape(traffic.shape[0], -1)
    return flat / flat.mean(axis=1, keepdims=True)


def _correlations(shapes: np.ndarray, row: np.ndarray) -> np.ndarray:
    left = shapes - shapes.mean(axis=1, keepdims=True)
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    right = row - row.mean()
    right /= np.linalg.norm(right)
    return left @ right


def _build_site(population, n_op: int = 4):
    """Same construction as ``plan_walkthrough`` with the same seed."""
    n_antennas = int(population.antenna_ids.size)
    n_days = int(population.days.size)
    shapes = _shapes(population.traffic_gb)
    mean_traffic = population.traffic_gb.mean(axis=(1, 2))
    rng = np.random.default_rng(SEED)

    reference = int(rng.integers(n_antennas))
    mu = float(mean_traffic[reference])
    quantiles = (0.40, 0.50, 0.60, 0.75)[:n_op]
    raw = np.quantile(mean_traffic, quantiles)
    alpha = (raw / raw.mean())[rng.permutation(n_op)]

    corr = _correlations(shapes, population.traffic_gb[reference].reshape(-1))
    corr[reference] = -np.inf
    target = float(np.quantile(corr[np.isfinite(corr)], 0.99))
    pool = np.argsort(np.abs(corr - target))[: max(20, 3 * n_op)]
    donors = rng.choice(pool, size=n_op - 1, replace=False)
    shape_index = np.concatenate(([reference], donors))

    forbidden = set(int(v) for v in shape_index)
    energy_pool = np.array([i for i in range(n_antennas) if i not in forbidden])
    energy = rng.choice(energy_pool, size=n_op, replace=False)

    traffic = (mu * alpha[:, None] * shapes[shape_index]).reshape(n_op, n_days, 24)
    return {
        "reference_id": str(population.antenna_ids[reference]),
        "mu": mu,
        "alpha": alpha,
        "traffic": traffic,
        "peaks": traffic.max(axis=(1, 2)),
        "p_fixed": population.p_fixed_w[energy],
        "slope": population.slope_w_per_gb[energy],
        "n_days": n_days,
        "n_op": n_op,
    }


def _k_h(capacities: np.ndarray, total: float) -> int:
    order = np.sort(capacities)[::-1]
    return int(np.searchsorted(np.cumsum(order), total - 1e-12, side="left") + 1)


def _hour_slice(site: dict, day: int, hour: int, operators=None) -> np.ndarray:
    if operators is None:
        return site["traffic"][:, day, hour : hour + 1]
    return site["traffic"][list(operators)][:, day, hour : hour + 1]


def _evaluate_hours(site: dict, rate: float, operators=None):
    ops = list(range(site["n_op"])) if operators is None else list(operators)
    peaks = site["peaks"][ops]
    capacities = peaks / rate
    p_fixed = site["p_fixed"][ops]
    slope = site["slope"][ops]
    n_days = site["n_days"]

    standalone = np.zeros((n_days, 24))
    optimum = np.zeros((n_days, 24))
    capacity_pol = np.zeros((n_days, 24))
    proportional = np.zeros((n_days, 24))
    n_guardians = np.zeros((n_days, 24), dtype=int)
    k_h = np.zeros((n_days, 24), dtype=int)
    low_traffic = np.zeros((n_days, 24), dtype=bool)
    masks = np.zeros((n_days, 24), dtype=np.int64)

    t0 = time.perf_counter()
    for day in range(n_days):
        for hour in range(24):
            demands = _hour_slice(site, day, hour, ops)
            total = float(demands.sum())
            standalone[day, hour] = standalone_energy(p_fixed, slope, demands)
            hourly, _ = optimal_hourly_with_persistent(
                capacities, p_fixed, slope, demands
            )
            cap = descending_capacity_hourly_policy(
                capacities, p_fixed, slope, demands
            )
            prop = proportional_hourly_policy(
                hourly.guardian_masks, capacities, p_fixed, slope, demands
            )
            optimum[day, hour] = hourly.energy_wh
            capacity_pol[day, hour] = cap.energy_wh
            proportional[day, hour] = prop.energy_wh
            n_guardians[day, hour] = int(hourly.guardian_counts[0])
            k_h[day, hour] = _k_h(capacities, total)
            low_traffic[day, hour] = total <= float(capacities.min()) + 1e-9
            masks[day, hour] = int(hourly.guardian_masks[0])
    elapsed = time.perf_counter() - t0
    return {
        "capacities": capacities,
        "standalone": standalone,
        "optimum": optimum,
        "capacity_pol": capacity_pol,
        "proportional": proportional,
        "n_guardians": n_guardians,
        "k_h": k_h,
        "low_traffic": low_traffic,
        "masks": masks,
        "elapsed_s": elapsed,
        "ops": ops,
    }


def _window_persistent(site: dict, rate: float, operators=None):
    ops = list(range(site["n_op"])) if operators is None else list(operators)
    capacities = site["peaks"][ops] / rate
    p_fixed = site["p_fixed"][ops]
    slope = site["slope"][ops]
    hours = list(WINDOW)
    rows = []
    t0 = time.perf_counter()
    for day in range(site["n_days"]):
        demands = site["traffic"][ops][:, day, hours]
        hourly, persistent = optimal_hourly_with_persistent(
            capacities, p_fixed, slope, demands
        )
        rows.append(
            {
                "standalone": standalone_energy(p_fixed, slope, demands),
                "hourly": hourly.energy_wh,
                "persistent": persistent.energy_wh,
                "changes": hourly.guardian_changes,
                "persistent_g": persistent.num_guardians,
            }
        )
    return rows, time.perf_counter() - t0


def _days_by_value(column: np.ndarray) -> str:
    return ",".join(
        f"{int(k)}:{int(np.sum(column == k))}" for k in sorted(np.unique(column))
    )


def _pct(saved: np.ndarray, standalone: np.ndarray) -> np.ndarray:
    return 100.0 * saved / np.maximum(standalone, 1e-12)


def _mean_hour(values: np.ndarray, days) -> np.ndarray:
    return values[list(days)].mean(axis=0)


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=CENTRAL_RATE)
    args = parser.parse_args()

    population = load_calibrated_population(CACHE)
    site = _build_site(population)
    print(">> Plan central")
    print(f"   référence {site['reference_id']}, mu = {site['mu']:.2f} Go/h")
    print(f"   alpha = {', '.join(f'{a:.3f}' for a in site['alpha'])}")
    print(f"   r = {args.rate:.2f}, n = {site['n_op']}")

    result = _evaluate_hours(site, args.rate)
    q = np.sort(result["capacities"])[::-1]
    total = site["traffic"].sum(axis=0)
    print("\n>> Capacités")
    print("   somme des k plus grandes q_i = couverture max. avec k équipements")
    print("   (cela définit k_h, pas l'ensemble G*)")
    for k in range(1, min(4, q.size + 1)):
        print(f"   k = {k} : {q[:k].sum():.1f} Go/h")
    print(f"   demande totale, max 7 jours : {total.max():.1f} Go/h")
    print(f"   à 20 h, moyenne / max : {total[:, 20].mean():.1f} / {total[:, 20].max():.1f}")
    equal = np.array_equal(result["n_guardians"], result["k_h"])
    print(f"   |G*| = k_h à chaque heure : {'oui' if equal else 'non'}")
    week = range(5)
    weekend = range(5, 7)
    savings = result["standalone"] - result["optimum"]
    rel = _pct(savings, result["standalone"])

    print("\n>> Profil horaire")
    print("   |G*| et k_h : décompte sur 7 jours (valeur:n_jours), pas une moyenne")
    print("   D<=min q : nombre de jours où n'importe quel opérateur suffit seul")
    print(f"   {'h':>3} {'sem. %':>8} {'w.-e. %':>8} {'|G*|':>12} {'k_h':>12} "
          f"{'D<=min q':>9}")
    for hour in range(24):
        print(
            f"   {hour:>3} {rel[list(week), hour].mean():>8.1f}"
            f" {rel[list(weekend), hour].mean():>8.1f}"
            f" {_days_by_value(result['n_guardians'][:, hour]):>12}"
            f" {_days_by_value(result['k_h'][:, hour]):>12}"
            f" {int(result['low_traffic'][:, hour].sum()):>3d}/7"
        )

    print("\n>> Fenêtre H = [0 h, 7 h), non choisie sur le niveau d'économie")
    h_stand = result["standalone"][:, WINDOW]
    h_opt = result["optimum"][:, WINDOW]
    h_rel = _pct(h_stand - h_opt, h_stand)
    print(f"   économie relative, 7 jours : {_fmt(100 * (h_stand - h_opt).sum() / h_stand.sum())} %")
    print(f"   semaine (5 nuits)          : {_fmt(h_rel[list(week)].mean())} %")
    print(f"   week-end (2 nuits)         : {_fmt(h_rel[list(weekend)].mean())} %")
    night_wh = (h_stand - h_opt).sum(axis=1)
    day_wh = savings.sum(axis=1)
    print(f"   énergie évitée, moyenne d'une nuit H : {night_wh.mean() / 1000:.2f} kWh")
    print(f"   énergie évitée, moyenne d'un jour 24 h : {day_wh.mean() / 1000:.2f} kWh")

    print("\n>> Politiques, sur H, moyenne des 7 nuits")
    cap_h = result["capacity_pol"][:, WINDOW]
    prop_h = result["proportional"][:, WINDOW]
    pers_rows, pers_time = _window_persistent(site, args.rate)
    pers = np.array([row["persistent"] for row in pers_rows])
    hourly_w = np.array([row["hourly"] for row in pers_rows])
    stand_w = np.array([row["standalone"] for row in pers_rows])
    changes = np.array([row["changes"] for row in pers_rows], dtype=float)

    def loss_pts(energy: np.ndarray) -> float:
        return float(np.mean(100.0 * (energy - hourly_w) / stand_w))

    print(f"   {'politique':<22} {'économie %':>12} {'perte vs opt.':>14}")
    print(f"   {'optimum horaire':<22} {np.mean(100 * (stand_w - hourly_w) / stand_w):>12.1f} {'—':>14}")
    print(f"   {'sélection par q_i':<22} {np.mean(100 * (stand_w - cap_h.sum(axis=1)) / stand_w):>12.1f} {loss_pts(cap_h.sum(axis=1)):>14.2f}")
    print(f"   {'proportionnelle':<22} {np.mean(100 * (stand_w - prop_h.sum(axis=1)) / stand_w):>12.1f} {loss_pts(prop_h.sum(axis=1)):>14.2f}")
    print(f"   {'persistante':<22} {np.mean(100 * (stand_w - pers) / stand_w):>12.1f} {loss_pts(pers):>14.2f}")
    print(f"   changements de G* entre heures consécutives de H, moyenne : "
          f"{changes.mean():.2f} par nuit")
    same_set = np.mean(
        [np.unique(result["masks"][d, WINDOW]).size == 1 for d in range(site["n_days"])]
    )
    print(f"   nuits où l'optimum horaire est déjà constant sur H : {100 * same_set:.0f} %")

    print("\n>> Taux r, mêmes demandes, q_i = pic / r")
    print(f"   {'r':>6} {'q max':>8} {'H %':>8} {'max k_h':>8} "
          f"{'k 20h':>8} {'|G*| 20h':>10}")
    for rate in RATES:
        other = _evaluate_hours(site, rate) if rate != args.rate else result
        st = other["standalone"][:, WINDOW]
        op = other["optimum"][:, WINDOW]
        q_max = float(other["capacities"].max())
        print(
            f"   {rate:>6.2f} {q_max:>8.1f}"
            f" {100 * (st - op).sum() / st.sum():>8.1f}"
            f" {int(other['k_h'].max()):>8d}"
            f" {int(other['k_h'][:, 20].max()):>8d}"
            f" {int(other['n_guardians'][:, 20].max()):>10d}"
        )

    print(f"\n>> n = 3, opérateurs 1–3 du même plan, r = {args.rate:.2f}")
    three = _evaluate_hours(site, args.rate, operators=(0, 1, 2))
    st3 = three["standalone"][:, WINDOW]
    op3 = three["optimum"][:, WINDOW]
    rel3 = 100 * (st3 - op3).sum() / st3.sum()
    rel4 = 100 * (h_stand - h_opt).sum() / h_stand.sum()
    print(f"   économie relative n = 3 : {rel3:.1f} %")
    print(f"   économie relative n = 4 : {rel4:.1f} %")

    annual_h = (5 * night_wh[:5].mean() + 2 * night_wh[5:].mean()) / 7 * 365 / 1000
    annual_24 = (5 * day_wh[:5].mean() + 2 * day_wh[5:].mean()) / 7 * 365 / 1000
    print("\n>> Ordre de grandeur annuel, pondération 5/7 + 2/7")
    print(f"   une nuit H par jour : {annual_h:.0f} kWh / site / an")
    print(f"   24 h par jour       : {annual_24:.0f} kWh / site / an")
    print("   saisonnalité non observée : une semaine de mars seulement")

    print("\n>> Gardiens sur les 168 heures")
    counts = result["n_guardians"].ravel()
    for k in range(1, site["n_op"] + 1):
        print(f"   |G_h^*(N)| = {k} : {100 * np.mean(counts == k):.1f} %")
    print(f"   d^h(N) <= min q_i sur H : {100 * result['low_traffic'][:, WINDOW].mean():.1f} %")
    print(f"   d^h(N) <= min q_i, 24 h : {100 * result['low_traffic'].mean():.1f} %")

    print("\n>> Temps")
    print(f"   168 optima horaires n = 4 : {result['elapsed_s']:.2f} s")
    print(f"   7 politiques persistantes : {pers_time:.3f} s")
    print(f"   168 optima horaires n = 3 : {three['elapsed_s']:.2f} s")

    _figure_capacity(site, result)
    _figure_savings(rel, result, week, weekend)


def _style(axis) -> None:
    axis.set_xlim(-0.5, 23.5)
    axis.set_xticks(range(0, 24, 3))
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.axvspan(-0.5, 6.5, color="0.85", alpha=0.55, zorder=0)


def _figure_capacity(site: dict, result: dict) -> None:
    hours = np.arange(24)
    traffic = site["traffic"]
    total = traffic.sum(axis=0)
    q = result["capacities"]
    ordered = np.sort(q)[::-1]
    cum = np.cumsum(ordered)
    colours = ("#2A6BB0", "#C0504D", "#6E8B3D", "#E39B22")

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.15))

    axis = axes[0]
    for i in range(site["n_op"]):
        daily = traffic[i]
        axis.fill_between(hours, daily.min(axis=0), daily.max(axis=0),
                          color=colours[i], alpha=0.12, linewidth=0)
        axis.plot(hours, daily.mean(axis=0), color=colours[i], linewidth=1.5,
                  label=f"op. {i + 1}, $q={q[i]:.0f}$")
        axis.axhline(q[i], color=colours[i], linewidth=0.8, linestyle=":")
    axis.set_xlabel("heure")
    axis.set_ylabel("demande (Go/h)")
    axis.set_title("(a) chaque $d_i$ reste sous son $q_i$\n"
                   "(aplat : min--max sur 7 jours)", fontsize=9)
    axis.legend(fontsize=7, loc="upper left", ncol=2)
    axis.set_ylim(bottom=0.0)
    _style(axis)

    axis = axes[1]
    axis.fill_between(hours, total.min(axis=0), total.max(axis=0),
                      color="#2A6BB0", alpha=0.18, linewidth=0,
                      label="demande totale, min--max")
    axis.plot(hours, total.mean(axis=0), color="#2A6BB0", linewidth=1.7,
              label="demande totale, moyenne")
    axis.axhline(float(site["peaks"].sum()), color="0.35", linewidth=1.0,
                 linestyle="-.",
                 label=f"somme des 4 pics = {site['peaks'].sum():.0f}")
    styles = ("--", "-", ":")
    names = (r"$k=1$ : plus grande $q_i$",
             r"$k=2$ : deux plus grandes $q_i$",
             r"$k=3$ : trois plus grandes $q_i$")
    for k, (level, name, style) in enumerate(zip(cum[:3], names, styles)):
        axis.axhline(level, color="#C0504D" if k == 1 else "0.25",
                     linewidth=1.4 if k == 1 else 1.0, linestyle=style,
                     label=f"{name} = {level:.0f}")
    axis.axvline(20, color="0.4", linewidth=0.8, linestyle=":")
    axis.set_xlabel("heure")
    axis.set_ylabel("demande totale (Go/h)")
    axis.set_title("(b) demande totale et couverture max. avec $k$ équipements",
                   fontsize=9)
    axis.legend(fontsize=6.5, loc="upper left")
    axis.set_ylim(0.0, max(float(site["peaks"].sum()) * 1.15, float(cum[2]) * 1.08))
    _style(axis)
    axis.annotate("20 h", (20.2, 0.92 * axis.get_ylim()[1]), fontsize=8,
                  color="0.25")

    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / "single_plan_capacity.pdf"
    figure.savefig(pdf)
    figure.savefig(pdf.with_suffix(".png"), dpi=140)
    plt.close(figure)
    print(f">> Figure : {pdf.resolve()}")


def _figure_savings(rel, result, week, weekend) -> None:
    hours = np.arange(24)
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.15))

    axis = axes[0]
    axis.plot(hours, rel[list(week)].mean(axis=0), color="#2A6BB0",
              marker="o", markersize=4, linewidth=1.5, label="semaine")
    axis.plot(hours, rel[list(weekend)].mean(axis=0), color="#C0504D",
              marker="s", markersize=4, linewidth=1.5, label="week-end")
    axis.set_xlabel("heure")
    axis.set_ylabel(r"$v_h(N)\,/\,\sum_i C_h^*(\{i\})$ (%)")
    axis.set_title("(a) économie relative", fontsize=10)
    axis.set_ylim(0, 100)
    axis.legend(fontsize=8)
    _style(axis)

    axis = axes[1]
    k = result["k_h"]
    bottom = np.zeros(24)
    labels = {1: r"$k_h=1$", 2: r"$k_h=2$", 3: r"$k_h=3$"}
    bar_colours = {1: "#2A6BB0", 2: "#C0504D", 3: "#E39B22"}
    n_days = k.shape[0]
    for value in (1, 2, 3):
        share = 100.0 * (k == value).sum(axis=0) / n_days
        axis.bar(hours, share, bottom=bottom, color=bar_colours[value],
                 width=0.72, label=labels[value])
        bottom += share
    axis.set_xlabel("heure")
    axis.set_ylabel("part des 7 jours (%)")
    axis.set_title("(b) combien de gardiens le seuil exige, jour par jour",
                   fontsize=10)
    axis.set_ylim(0, 100)
    axis.legend(fontsize=8, loc="lower right")
    _style(axis)

    figure.tight_layout()
    pdf = FIGURES / "single_plan_hourly.pdf"
    figure.savefig(pdf)
    figure.savefig(pdf.with_suffix(".png"), dpi=140)
    plt.close(figure)
    print(f">> Figure : {pdf.resolve()}")


if __name__ == "__main__":
    main()
