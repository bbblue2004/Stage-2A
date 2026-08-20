"""Stability and sharing on the single plan of Sections 6.1 and 6.3.

Rebuilds the walkthrough site and, for each night of H, computes v_H, the
Shapley/core/nucleolus diagnostics, and the transfers of Section 5.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cli.single_plan_efficiency import (
    CACHE,
    CENTRAL_RATE,
    RATES,
    WINDOW,
    _build_site,
)
from src.core.game import (
    allocation_check,
    nucleolus_allocation,
    shapley_value,
)
from src.core.window_optimiser import (
    hourly_coalition_costs,
    optimal_hourly_with_persistent,
)
from src.data_processing.power_validation import load_calibrated_population
from src.experiments.coalition_stability import (
    GRAND_MASK,
    _diagnose,
    _savings_from_costs,
)

FIGURES = Path("figures") / "coalition_stability"
DAYS = ("lun", "mar", "mer", "jeu", "ven", "sam", "dim")


def _players(n: int) -> list[int]:
    return list(range(n))


def _as_map(n: int, values: np.ndarray) -> dict[tuple[int, ...], float]:
    mapping: dict[tuple[int, ...], float] = {}
    for mask, value in enumerate(values):
        members = tuple(i for i in range(n) if mask & (1 << i))
        mapping[members] = float(value)
    return mapping


def _physical_costs(
    hourly,
    p_fixed: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    n_op, n_hours = hourly.allocations_gb.shape
    paid = np.zeros(n_op)
    for hour in range(n_hours):
        mask = int(hourly.guardian_masks[hour])
        for player in range(n_op):
            if mask & (1 << player):
                paid[player] += p_fixed[player] + slope[player] * hourly.allocations_gb[
                    player, hour
                ]
    return paid


def _loo_gap(n: int, costs: np.ndarray) -> float:
    grand = (1 << n) - 1
    leave = sum(costs[grand ^ (1 << player)] for player in range(n))
    return float(costs[grand] - leave / (n - 1))


def _evaluate_night(site: dict, rate: float, day: int, operators=None, *,
                    with_nucleolus: bool = True) -> dict:
    ops = list(range(site["n_op"])) if operators is None else list(operators)
    n = len(ops)
    capacities = site["peaks"][ops] / rate
    p_fixed = site["p_fixed"][ops]
    slope = site["slope"][ops]
    demands = site["traffic"][ops][:, day, list(WINDOW)]

    t0 = time.perf_counter()
    costs = hourly_coalition_costs(capacities, p_fixed, slope, demands)
    table_s = time.perf_counter() - t0
    savings_vec = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        members = [i for i in range(n) if mask & (1 << i)]
        standalone = sum(costs[1 << i] for i in members)
        savings_vec[mask] = standalone - costs[mask]

    players = _players(n)
    savings = _as_map(n, savings_vec)
    grand = tuple(players)
    t1 = time.perf_counter()
    shapley = shapley_value(players, savings)
    check = allocation_check(players, savings, shapley)
    shapley_s = time.perf_counter() - t1

    low_traffic = bool(
        np.all(demands.sum(axis=0) <= float(capacities.min()) + 1e-9)
    )
    loo = _loo_gap(n, costs)

    nucleolus = None
    nuc = shapley
    distance = 0.0
    if with_nucleolus:
        result = nucleolus_allocation(players, savings)
        if result.status == "Optimal":
            nucleolus = result.allocation
            nuc = nucleolus
            distance = float(
                np.linalg.norm(
                    np.array([shapley[i] for i in players])
                    - np.array([nuc[i] for i in players])
                )
            )

    hourly, _ = optimal_hourly_with_persistent(capacities, p_fixed, slope, demands)
    physical = _physical_costs(hourly, p_fixed, slope)
    standalone = np.array([costs[1 << i] for i in players])
    z = np.array([shapley[i] for i in players])
    tau = physical - standalone + z

    blocking = check.blocking_coalitions
    blocking_sizes = tuple(len(c) for c in blocking) if blocking else ()

    extra = {}
    if n == 4:
        extra = _diagnose(costs, _savings_from_costs(costs), capacities, demands)

    return {
        "n": n,
        "rate": rate,
        "day": day,
        "v": savings[grand],
        "standalone": float(standalone.sum()),
        "relative": 100.0 * savings[grand] / max(standalone.sum(), 1e-12),
        "shapley": z,
        "nucleolus": np.array([nuc[i] for i in players]),
        "shapley_in_core": check.in_core,
        "nucleolus_in_core": (
            allocation_check(players, savings, nuc).in_core
            if nucleolus is not None else None
        ),
        "max_excess": check.max_excess,
        "blocking_sizes": blocking_sizes,
        "low_traffic": low_traffic,
        "loo": loo,
        "convex": extra.get("convex", None),
        "category": extra.get("category", None),
        "epsilon": extra.get("least_core_epsilon_wh", float("nan")),
        "physical": physical,
        "tau": tau,
        "tau_sum": float(tau.sum()),
        "table_s": table_s,
        "shapley_s": shapley_s,
        "distance": distance,
    }


def main() -> None:
    population = load_calibrated_population(CACHE)
    site = _build_site(population)
    print(">> Plan central, stabilité sur H")
    print(f"   référence {site['reference_id']}, r = {CENTRAL_RATE:.2f}, n = 4")
    print(f"   alpha = {', '.join(f'{a:.3f}' for a in site['alpha'])}")
    print(f"   s (W/Go) = {', '.join(f'{s:.2f}' for s in site['slope'])}")

    nights = [_evaluate_night(site, CENTRAL_RATE, day) for day in range(7)]

    print("\n>> Cascade, 7 nuits, n = 4, r = 0,90")
    print(f"   {'jour':>6} {'v kWh':>8} {'éco %':>7} {'catégorie':>22} "
          f"{'E_Sh/v':>8} {'bloc':>6} {'D<=min q':>9} {'LOO':>6} {'convexe':>8}")
    for row in nights:
        blocking = ",".join(str(s) for s in row["blocking_sizes"]) or "—"
        excess = (
            100.0 * row["max_excess"] / row["v"] if row["v"] > 0 else 0.0
        )
        print(
            f"   {DAYS[row['day']]:>6} {row['v'] / 1000:>8.2f}"
            f" {row['relative']:>7.1f} {str(row['category']):>22}"
            f" {excess:>7.2f}% {blocking:>6}"
            f" {'oui' if row['low_traffic'] else 'non':>9}"
            f" {row['loo']:>6.1f} {'oui' if row['convex'] else 'non':>8}"
        )

    n_shapley = sum(r["shapley_in_core"] for r in nights)
    n_low = sum(r["low_traffic"] for r in nights)
    n_loo = sum(r["loo"] > 1e-6 for r in nights)
    n_convex = sum(bool(r["convex"]) for r in nights)
    print(f"   Shapley dans le cœur : {n_shapley}/7")
    print(f"   D^h(N) <= min q_i toute la nuit : {n_low}/7")
    print(f"   certificat LOO (cœur vide) : {n_loo}/7")
    print(f"   convexe : {n_convex}/7")
    print(f"   table v_H, moyenne : {np.mean([r['table_s'] for r in nights]):.3f} s / nuit")
    print(f"   test Shapley, moyenne : {np.mean([r['shapley_s'] for r in nights]):.4f} s / nuit")
    print(f"   somme des transferts Shapley, max |Στ| : "
          f"{max(abs(r['tau_sum']) for r in nights):.2e} Wh")

    print("\n>> Shapley contre nucléole, parts de v_H (%)")
    print(f"   {'':>6} {'op1':>7} {'op2':>7} {'op3':>7} {'op4':>7} {'||Sh-nu||/v':>12}")
    for label, key in (("Shapley", "shapley"), ("nucléole", "nucleolus")):
        mean = np.mean([100.0 * r[key] / r["v"] for r in nights], axis=0)
        print(f"   {label:>6} " + " ".join(f"{x:7.1f}" for x in mean), end="")
        if label == "nucléole":
            dist = np.mean([r["distance"] / r["v"] for r in nights])
            print(f" {100 * dist:11.2f}%")
        else:
            print()
    mean_tau = np.mean([r["tau"] for r in nights], axis=0)
    print("   transfert net Shapley τ (Wh), moyenne 7 nuits : "
          + ", ".join(f"op{i + 1} {t:+.0f}" for i, t in enumerate(mean_tau)))
    who_pays = mean_tau < -1.0
    who_gets = mean_tau > 1.0
    print("   paie (τ<0) : "
          + ",".join(f"op{i + 1}" for i, p in enumerate(who_pays) if p))
    print("   reçoit (τ>0) : "
          + ",".join(f"op{i + 1}" for i, p in enumerate(who_gets) if p))

    print("\n>> Trois r, n = 4, 7 nuits")
    print(f"   {'r':>6} {'Sh cœur':>10} {'hors Sh':>10} {'vide':>8} "
          f"{'max E_Sh/v':>12}")
    by_rate = {}
    for rate in RATES:
        rows = (
            nights if rate == CENTRAL_RATE
            else [_evaluate_night(site, rate, day, with_nucleolus=False)
                  for day in range(7)]
        )
        by_rate[rate] = rows
        cats = [r["category"] for r in rows]
        excesses = [
            r["max_excess"] / r["v"] for r in rows if not r["shapley_in_core"]
        ]
        print(
            f"   {rate:>6.2f} {cats.count('shapley_in_core'):>10d}"
            f" {cats.count('nonempty_shapley_out'):>10d}"
            f" {cats.count('empty_core'):>8d}"
            f" {100 * max(excesses, default=0.0):>11.2f}%"
        )

    print("\n>> n = 3, mêmes trois premiers opérateurs, r = 0,90")
    three = [_evaluate_night(site, CENTRAL_RATE, day, operators=(0, 1, 2),
                             with_nucleolus=False)
             for day in range(7)]
    print(f"   Shapley dans le cœur : {sum(r['shapley_in_core'] for r in three)}/7")
    print(f"   économie relative moyenne : {np.mean([r['relative'] for r in three]):.1f} %")

    _figure(nights)
    print(f">> Figure : {(FIGURES / 'single_plan_sharing.pdf').resolve()}")


def _figure(nights: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.6))
    x = np.arange(4)
    width = 0.36
    sh = np.mean([100.0 * r["shapley"] / r["v"] for r in nights], axis=0)
    nu = np.mean([100.0 * r["nucleolus"] / r["v"] for r in nights], axis=0)
    axis = axes[0]
    axis.bar(x - width / 2, sh, width, color="#2A6BB0", label="Shapley")
    axis.bar(x + width / 2, nu, width, color="#C0504D", label="nucléole")
    axis.set_xticks(x, [f"op. {i + 1}" for i in range(4)])
    axis.set_ylabel("part de $v_H$ (%)")
    axis.set_title("(a) répartition des économies, moyenne des 7 nuits")
    axis.legend(fontsize=8, loc="upper right")
    axis.set_ylim(0.0, max(sh.max(), nu.max()) * 1.2)
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1]
    tau = np.array([r["tau"] / 1000 for r in nights])
    colours = ["#C0504D" if t < 0 else "#2A6BB0" for t in tau.mean(axis=0)]
    axis.bar(x, tau.mean(axis=0), color=colours)
    for day in range(7):
        axis.scatter(x + (day - 3) * 0.04, tau[day], s=12, color="0.35",
                     zorder=3, alpha=0.7)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xticks(x, [f"op. {i + 1}" for i in range(4)])
    axis.set_ylabel("transfert net (kWh)")
    axis.set_title("(b) $\\tau_i$ Shapley : négatif = paie")
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout(w_pad=1.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "single_plan_sharing.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    main()
