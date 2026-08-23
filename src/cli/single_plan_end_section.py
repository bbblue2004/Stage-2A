"""Threshold, mechanism and sensitivity results for the Section 6 walkthrough plan.

The script keeps the reference draw of Sections 6.1--6.4. Threshold results
combine the central instance and the matched all-close corner needed to
observe all four values of k_h. Sensitivity is one-factor-at-
a-time around the central instance and the central seven-hour window.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import linprog

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cli.single_plan_efficiency import (
    CACHE,
    CENTRAL_RATE,
    RATES,
    _build_site,
    _shapes,
)
from src.core.game import allocation_check, convexity_test, shapley_value
from src.core.window_optimiser import hourly_coalition_costs
from src.data_processing.power_validation import load_calibrated_population
from src.experiments.coalition_stability import _diagnose, _savings_from_costs
from src.experiments.parameter_sensitivity import _evaluate_instance


RESULTS = Path("results") / "single_plan_end_section"
FIGURES = Path("figures") / "coalition_stability"
DAYS = ("lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.")
THETAS = (0.00, 0.25, 0.50, 0.75, 1.00)
CATEGORY_ORDER = (
    "shapley_in_core",
    "nonempty_shapley_out",
    "empty_core",
)
CATEGORY_LABELS = {
    "shapley_in_core": "Shapley dans le cœur",
    "nonempty_shapley_out": "Shapley hors du cœur",
    "empty_core": "cœur vide",
}
FACTOR_LABELS = {
    "traffic": "Niveau de trafic",
    "fixed": "Coût fixe $F_i$",
    "variable": "Coût variable $\\gamma_i$",
    "sleep": "Puissance de veille",
    "position": "Position de $H$",
    "duration": "Durée de $H$",
}


def _close_traffic(population, site: dict) -> np.ndarray:
    """Return the matched close-volume/close-shape corner of the same plan."""
    identifiers = population.antenna_ids.astype(str)
    matches = np.flatnonzero(identifiers == site["reference_id"])
    if matches.size != 1:
        raise RuntimeError("walkthrough reference is absent from the population")
    shape = _shapes(population.traffic_gb)[int(matches[0])]
    traffic = site["mu"] * np.repeat(shape[None, :], site["n_op"], axis=0)
    return traffic.reshape(site["n_op"], site["n_days"], 24)


def _three_player_diagnostics(
    costs: np.ndarray,
    savings: np.ndarray,
    capacities: np.ndarray,
    demands: np.ndarray,
) -> dict[str, object]:
    """Return the Section 6.5 diagnostics for the three-player subgame."""
    players = [0, 1, 2]
    grand_mask = (1 << len(players)) - 1
    coalitions = [
        coalition
        for size in range(len(players) + 1)
        for coalition in combinations(players, size)
    ]
    game = {
        coalition: float(
            savings[sum(1 << player for player in coalition)]
        )
        for coalition in coalitions
    }
    shapley = shapley_value(players, game)
    check = allocation_check(players, game, shapley)
    grand_value = float(savings[grand_mask])
    scale = max(1.0, grand_value)
    tolerance = 1e-8 * scale

    pair_sum = sum(game[coalition] for coalition in combinations(players, 2))
    core_nonempty = pair_sum <= 2.0 * grand_value + tolerance
    if check.in_core and not core_nonempty:
        raise RuntimeError("stable Shapley value with an empty three-player core")

    proper_masks = np.arange(1, grand_mask, dtype=int)
    membership = np.asarray(
        [
            [
                float(mask & (1 << player) != 0)
                for player in players
            ]
            for mask in proper_masks
        ]
    )
    epsilon = float("nan")
    if not core_nonempty:
        result = linprog(
            np.asarray([0.0] * len(players) + [1.0]),
            A_ub=np.column_stack(
                (-membership, -np.ones(proper_masks.size))
            ),
            b_ub=-savings[proper_masks],
            A_eq=np.asarray([[1.0] * len(players) + [0.0]]),
            b_eq=[grand_value],
            bounds=[(0.0, None)] * len(players) + [(None, None)],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(
                f"three-player least-core LP failed: {result.message}"
            )
        epsilon = float(result.x[-1])

    excesses = {
        mask: float(
            savings[mask]
            - sum(
                shapley[player]
                for player in players
                if mask & (1 << player)
            )
        )
        for mask in proper_masks
    }
    blocking_mask = max(excesses, key=excesses.get)
    loo_gap = float(
        costs[grand_mask]
        - sum(
            costs[grand_mask ^ (1 << player)] for player in players
        )
        / (len(players) - 1)
    )
    low_traffic = float(np.max(np.sum(demands, axis=0))) <= float(
        np.min(capacities)
    ) + 1e-9
    convex = convexity_test(players, game).convex
    if check.in_core:
        category = "shapley_in_core"
    elif core_nonempty:
        category = "nonempty_shapley_out"
    else:
        category = "empty_core"
    normalizer = max(grand_value, 1e-12)
    return {
        "core_nonempty": int(core_nonempty),
        "shapley_in_core": int(check.in_core),
        "category": category,
        "convex": int(convex),
        "low_traffic_condition": int(low_traffic),
        "bondareva_gap_wh": max(0.0, pair_sum / 2.0 - grand_value),
        "bondareva_gap_normalized": (
            max(0.0, pair_sum / 2.0 - grand_value) / normalizer
        ),
        "least_core_epsilon_wh": epsilon,
        "least_core_epsilon_normalized": epsilon / normalizer,
        "loo_gap_wh": loo_gap,
        "loo_gap_normalized": loo_gap / normalizer,
        "loo_certificate": int(loo_gap > tolerance),
        "shapley_max_excess_wh": excesses[blocking_mask],
        "shapley_max_excess_normalized": (
            excesses[blocking_mask] / normalizer
        ),
        "blocking_mask": blocking_mask,
        "blocking_size": int(blocking_mask).bit_count(),
    }


def _generic_savings(costs: np.ndarray, num_operators: int) -> np.ndarray:
    savings = np.zeros(1 << num_operators, dtype=float)
    for mask in range(1, 1 << num_operators):
        members = [
            player
            for player in range(num_operators)
            if mask & (1 << player)
        ]
        savings[mask] = (
            sum(costs[1 << player] for player in members) - costs[mask]
        )
    return savings


def _threshold_instances(
    population,
    site: dict,
    num_operators: int = 4,
) -> list[dict[str, object]]:
    if num_operators not in (3, 4):
        raise ValueError("threshold diagnostics support n=3 or n=4")
    variants = {
        "central": (
            site["traffic"][:num_operators],
            site["p_fixed"][:num_operators],
            site["slope"][:num_operators],
        ),
        "proche": (
            _close_traffic(population, site)[:num_operators],
            site["p_fixed_close"][:num_operators],
            site["slope_close"][:num_operators],
        ),
    }
    records: list[dict[str, object]] = []
    for scenario, (traffic, fixed, slopes) in variants.items():
        peaks = np.max(traffic, axis=(1, 2))
        for rate in RATES:
            capacities = peaks / rate
            cumulative = np.cumsum(np.sort(capacities)[::-1])
            for day in range(site["n_days"]):
                for hour in range(24):
                    demands = traffic[:, day, hour : hour + 1]
                    costs = hourly_coalition_costs(
                        capacities, fixed, slopes, demands
                    )
                    savings = (
                        _savings_from_costs(costs)
                        if num_operators == 4
                        else _generic_savings(costs, num_operators)
                    )
                    diagnostics = (
                        _diagnose(costs, savings, capacities, demands)
                        if num_operators == 4
                        else _three_player_diagnostics(
                            costs, savings, capacities, demands
                        )
                    )
                    total = float(np.sum(demands))
                    k_h = int(
                        np.searchsorted(
                            cumulative, total - 1e-12, side="left"
                        )
                        + 1
                    )
                    standalone = float(
                        sum(
                            costs[1 << player]
                            for player in range(num_operators)
                        )
                    )
                    record: dict[str, object] = {
                        "num_operators": num_operators,
                        "scenario": scenario,
                        "rate": float(rate),
                        "day": day,
                        "hour": hour,
                        "k_h": k_h,
                        "standalone_wh": standalone,
                        "savings_wh": float(savings[-1]),
                        "savings_pct": 100.0 * float(savings[-1]) / standalone,
                        **diagnostics,
                        "_capacities": capacities,
                        "_demands": demands,
                        "_fixed": fixed,
                        "_slopes": slopes,
                    }
                    records.append(record)
    return records


def _threshold_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for k_h in range(1, 5):
        selected = [row for row in records if row["k_h"] == k_h]
        unstable = [row for row in selected if not row["shapley_in_core"]]
        empty = [row for row in selected if not row["core_nonempty"]]
        sizes = Counter(int(row["blocking_size"]) for row in unstable)
        modal_size = "—"
        if sizes:
            size, count = sorted(sizes.items(), key=lambda pair: (-pair[1], pair[0]))[0]
            modal_size = f"{size} ({100.0 * count / len(unstable):.1f} %)"
        epsilon = [
            float(row["least_core_epsilon_normalized"])
            for row in empty
        ]
        summary.append(
            {
                "num_operators": int(records[0].get("num_operators", 4)),
                "k_h": k_h,
                "instances": len(selected),
                "empty_core_pct": 100.0 * len(empty) / max(len(selected), 1),
                "shapley_in_core_pct": 100.0
                * sum(int(row["shapley_in_core"]) for row in selected)
                / max(len(selected), 1),
                "epsilon_empty_median_pct": (
                    100.0 * float(np.median(epsilon)) if epsilon else float("nan")
                ),
                "blocking_size_mode": modal_size,
            }
        )
    return summary


def _operator_count_summary(
    records_by_n: dict[int, list[dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for num_operators, records in sorted(records_by_n.items()):
        rows.append(
            {
                "num_operators": num_operators,
                "instances": len(records),
                "low_traffic_condition_pct": 100.0
                * float(
                    np.mean(
                        [row["low_traffic_condition"] for row in records]
                    )
                ),
                "shapley_in_core_pct": 100.0
                * float(np.mean([row["shapley_in_core"] for row in records])),
                "core_nonempty_pct": 100.0
                * float(np.mean([row["core_nonempty"] for row in records])),
                "savings_kwh_mean": float(
                    np.mean([row["savings_wh"] for row in records])
                )
                / 1000.0,
                "savings_pct_mean": float(
                    np.mean([row["savings_pct"] for row in records])
                ),
            }
        )
    return rows


def _heterogeneity_summary(
    records: list[dict[str, object]], site: dict
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for theta in THETAS:
        diagnostics: list[dict[str, object]] = []
        for record in records:
            capacities = np.asarray(record["_capacities"])
            demands = np.asarray(record["_demands"])
            observed_fixed = np.asarray(record["_fixed"])
            observed_slopes = np.asarray(record["_slopes"])
            fixed = np.mean(observed_fixed) + theta * (
                observed_fixed - np.mean(observed_fixed)
            )
            slopes = np.mean(observed_slopes) + theta * (
                observed_slopes - np.mean(observed_slopes)
            )
            costs = hourly_coalition_costs(capacities, fixed, slopes, demands)
            diagnostics.append(
                _diagnose(costs, _savings_from_costs(costs), capacities, demands)
            )
        low = [row for row in diagnostics if row["low_traffic_condition"]]
        rows.append(
            {
                "theta": theta,
                "instances": len(diagnostics),
                "low_traffic_instances": len(low),
                "shapley_in_core_pct": 100.0
                * np.mean([row["shapley_in_core"] for row in diagnostics]),
                "core_nonempty_pct": 100.0
                * np.mean([row["core_nonempty"] for row in diagnostics]),
                "low_traffic_shapley_in_core_pct": 100.0
                * np.mean([row["shapley_in_core"] for row in low]),
            }
        )
    return rows


def _representatives(records: list[dict[str, object]]) -> list[dict[str, object]]:
    representatives: list[dict[str, object]] = []
    for category in CATEGORY_ORDER:
        selected = [row for row in records if row["category"] == category]
        if not selected:
            continue
        features = np.asarray(
            [
                [
                    float(row["savings_pct"]),
                    max(0.0, 100.0 * float(row["shapley_max_excess_normalized"])),
                    (
                        100.0 * float(row["least_core_epsilon_normalized"])
                        if np.isfinite(row["least_core_epsilon_normalized"])
                        else 0.0
                    ),
                ]
                for row in selected
            ]
        )
        median = np.median(features, axis=0)
        scale = np.subtract(*np.percentile(features, [75, 25], axis=0))
        scale[scale <= 1e-12] = 1.0
        distances = np.sum(np.abs(features - median) / scale, axis=1)
        ordering = sorted(
            range(len(selected)),
            key=lambda index: (
                float(distances[index]),
                str(selected[index]["scenario"]),
                int(selected[index]["day"]),
                int(selected[index]["hour"]),
                float(selected[index]["rate"]),
            ),
        )
        chosen = selected[ordering[0]]
        mask = int(chosen["blocking_mask"])
        coalition = "{" + ",".join(
            str(player + 1) for player in range(4) if mask & (1 << player)
        ) + "}"
        representatives.append(
            {
                "category": category,
                "scenario": chosen["scenario"],
                "rate": chosen["rate"],
                "day": chosen["day"],
                "day_label": DAYS[int(chosen["day"])],
                "hour": chosen["hour"],
                "k_h": chosen["k_h"],
                "savings_pct": chosen["savings_pct"],
                "shapley_excess_pct": 100.0
                * float(chosen["shapley_max_excess_normalized"]),
                "epsilon_pct": (
                    100.0 * float(chosen["least_core_epsilon_normalized"])
                    if np.isfinite(chosen["least_core_epsilon_normalized"])
                    else float("nan")
                ),
                "blocking_coalition": coalition,
            }
        )
    return representatives


def _append_sensitivity(
    rows: list[dict[str, object]],
    factor: str,
    setting: str,
    day: int,
    metrics: dict[str, float | int],
) -> None:
    rows.append(
        {"factor": factor, "setting": setting, "day": day, **metrics}
    )


def _sensitivity_instances(site: dict) -> list[dict[str, object]]:
    traffic = site["traffic"]
    capacities = site["peaks"] / CENTRAL_RATE
    fixed = site["p_fixed"]
    slopes = site["slope"]
    rows: list[dict[str, object]] = []

    for day in range(site["n_days"]):
        central_demands = traffic[:, day, 0:7]
        central = _evaluate_instance(capacities, fixed, slopes, central_demands)
        for multiplier in (0.80, 1.00, 1.20):
            metrics = (
                central
                if multiplier == 1.00
                else _evaluate_instance(
                    capacities, fixed, slopes, multiplier * central_demands
                )
            )
            _append_sensitivity(rows, "traffic", f"{multiplier:.2f}", day, metrics)
        for multiplier in (0.80, 1.00, 1.20):
            metrics = (
                central
                if multiplier == 1.00
                else _evaluate_instance(
                    capacities, multiplier * fixed, slopes, central_demands
                )
            )
            _append_sensitivity(rows, "fixed", f"{multiplier:.2f}", day, metrics)
        for multiplier in (0.80, 1.00, 1.20):
            metrics = (
                central
                if multiplier == 1.00
                else _evaluate_instance(
                    capacities, fixed, multiplier * slopes, central_demands
                )
            )
            _append_sensitivity(rows, "variable", f"{multiplier:.2f}", day, metrics)
        for rate in (0.00, 0.05, 0.10):
            metrics = (
                central
                if rate == 0.00
                else _evaluate_instance(
                    capacities, fixed, slopes, central_demands, sleep_rate=rate
                )
            )
            _append_sensitivity(rows, "sleep", f"{rate:.2f}", day, metrics)
        for duration in (5, 7, 9):
            metrics = (
                central
                if duration == 7
                else _evaluate_instance(
                    capacities, fixed, slopes, traffic[:, day, :duration]
                )
            )
            _append_sensitivity(rows, "duration", str(duration), day, metrics)

    # The shifted windows are paired on the last six dates because 22--05
    # starts on the previous calendar day.
    for day in range(1, site["n_days"]):
        windows = {
            "22-05": np.concatenate(
                (traffic[:, day - 1, 22:24], traffic[:, day, 0:5]), axis=1
            ),
            "00-07": traffic[:, day, 0:7],
            "02-09": traffic[:, day, 2:9],
        }
        for setting, demands in windows.items():
            _append_sensitivity(
                rows,
                "position",
                setting,
                day,
                _evaluate_instance(capacities, fixed, slopes, demands),
            )
    return rows


def _sensitivity_summary(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["factor"]), str(row["setting"]))].append(row)
    summaries: list[dict[str, object]] = []
    for (factor, setting), selected in groups.items():
        summaries.append(
            {
                "factor": factor,
                "setting": setting,
                "instances": len(selected),
                "savings_pct_median": float(
                    np.median([row["savings_pct"] for row in selected])
                ),
                "guardians_mean": float(
                    np.mean([row["guardians_mean"] for row in selected])
                ),
                "persistence_gap_pp_median": float(
                    np.median([row["persistence_gap_pp"] for row in selected])
                ),
                "empty_core_pct": 100.0
                * float(np.mean([row["core_empty"] for row in selected])),
                "shapley_in_core_pct": 100.0
                * float(np.mean([row["shapley_stable"] for row in selected])),
            }
        )
    summaries.sort(key=lambda row: (str(row["factor"]), str(row["setting"])))

    effects: list[dict[str, object]] = []
    for factor in FACTOR_LABELS:
        selected = [row for row in summaries if row["factor"] == factor]
        effects.append(
            {
                "factor": factor,
                "label": FACTOR_LABELS[factor],
                "settings": ", ".join(str(row["setting"]) for row in selected),
                "savings_range_pp": float(
                    np.ptp([row["savings_pct_median"] for row in selected])
                ),
                "guardians_range": float(
                    np.ptp([row["guardians_mean"] for row in selected])
                ),
                "persistence_range_pp": float(
                    np.ptp([row["persistence_gap_pp_median"] for row in selected])
                ),
                "empty_core_range_pp": float(
                    np.ptp([row["empty_core_pct"] for row in selected])
                ),
                "shapley_range_pp": float(
                    np.ptp([row["shapley_in_core_pct"] for row in selected])
                ),
            }
        )
    effects.sort(key=lambda row: float(row["savings_range_pp"]), reverse=True)
    return summaries, effects


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    public = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public[0]))
        writer.writeheader()
        writer.writerows(public)


def _figure(
    records: list[dict[str, object]],
    threshold_summary: list[dict[str, object]],
    heterogeneity: list[dict[str, object]],
    records_n3: list[dict[str, object]],
    threshold_summary_n3: list[dict[str, object]],
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.45))

    axis = axes[0]
    for source, colour, label in (
        (records_n3, "#2F8F5B", "trois opérateurs"),
        (records, "#2A6BB0", "quatre opérateurs"),
    ):
        values = []
        for hour in range(24):
            selected = [
                row
                for row in source
                if row["scenario"] == "central" and row["hour"] == hour
            ]
            values.append(
                100.0 * np.mean([row["low_traffic_condition"] for row in selected])
            )
        axis.plot(range(24), values, color=colour, linewidth=1.8, label=label)
    axis.axvspan(0, 6.99, color="0.5", alpha=0.10)
    axis.set_xlim(0, 23)
    axis.set_ylim(0, 104)
    axis.set_xticks((0, 6, 12, 18, 23))
    axis.set_xlabel("heure")
    axis.set_ylabel("combinaisons satisfaisant la condition (%)")
    axis.set_title("(a) Heures favorables")
    axis.legend(fontsize=7, loc="upper right")
    axis.grid(axis="y", alpha=0.22)

    axis = axes[1]
    x = np.arange(1, 5, dtype=float)
    width = 0.34
    for summaries, offset, label, colour, hatch in (
        (threshold_summary_n3, -width / 2, "trois opérateurs", "#D97A7A", "//"),
        (threshold_summary, width / 2, "quatre opérateurs", "#C0504D", None),
    ):
        by_k = {int(row["k_h"]): row for row in summaries}
        percentages = [
            (
                float(by_k[k_h]["empty_core_pct"])
                if int(by_k[k_h]["instances"]) > 0
                else 0.0
            )
            for k_h in range(1, 5)
        ]
        bars = axis.bar(
            x + offset,
            percentages,
            width,
            color=colour,
            hatch=hatch,
            edgecolor="0.25",
            linewidth=0.5,
            label=label,
        )
        for k_h, bar in enumerate(bars, start=1):
            count = int(by_k[k_h]["instances"])
            if count:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(float(bar.get_height()) + 1.5, 2.0),
                    f"{count}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
    axis.set_xticks(range(1, 5))
    axis.set_xlabel("nombre minimal de gardiens")
    axis.set_ylabel("jeux sans partage stable (%)")
    axis.set_ylim(0, 104)
    axis.set_title("(b) Cœur vide observé")
    axis.legend(fontsize=7, loc="upper left")
    axis.grid(axis="y", alpha=0.22)

    axis = axes[2]
    theta = [row["theta"] for row in heterogeneity]
    axis.plot(
        theta,
        [row["shapley_in_core_pct"] for row in heterogeneity],
        marker="o",
        color="#2A6BB0",
        label="toutes les heures observées",
    )
    axis.plot(
        theta,
        [row["low_traffic_shapley_in_core_pct"] for row in heterogeneity],
        marker="s",
        color="#2F8F5B",
        label="heures satisfaisant la condition",
    )
    axis.set_xticks(THETAS, ("0", "25", "50", "75", "100"))
    axis.set_ylim(0, 104)
    axis.set_xlabel("part de l'hétérogénéité observée (%)")
    axis.set_ylabel("heures avec Shapley stable (%)")
    axis.set_title("(c) Effet des coûts")
    axis.legend(fontsize=7, loc="lower left")
    axis.grid(axis="y", alpha=0.22)

    figure.tight_layout(w_pad=1.0)
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "single_plan_thresholds.pdf"
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    population = load_calibrated_population(CACHE)
    site = _build_site(population)
    print(f">> Plan {site['reference_id']}, graine 20260819")

    records = _threshold_instances(population, site, num_operators=4)
    records_n3 = _threshold_instances(
        population, site, num_operators=3
    )
    threshold_summary = _threshold_summary(records)
    threshold_summary_n3 = _threshold_summary(records_n3)
    operator_count_summary = _operator_count_summary(
        {3: records_n3, 4: records}
    )
    heterogeneity = _heterogeneity_summary(records, site)
    representatives = _representatives(records)
    sensitivity_rows = _sensitivity_instances(site)
    sensitivity_summary, effects = _sensitivity_summary(sensitivity_rows)

    low = [row for row in records if row["low_traffic_condition"]]
    low_n3 = [
        row for row in records_n3 if row["low_traffic_condition"]
    ]
    theorem_violations = sum(not row["core_nonempty"] for row in low)
    theorem_violations_n3 = sum(
        not row["core_nonempty"] for row in low_n3
    )
    print("\n>> Nombre minimal de gardiens")
    print("   op.  k  heures  cœur vide  Shapley stable")
    for summaries in (threshold_summary_n3, threshold_summary):
        for row in summaries:
            if not row["instances"]:
                continue
            print(
                f"   {row['num_operators']:>3d} {row['k_h']:>3d}"
                f" {row['instances']:>7d}"
                f" {row['empty_core_pct']:10.1f} %"
                f" {row['shapley_in_core_pct']:14.1f} %"
            )
    print(
        f"   condition suffisante, n=3: {len(low_n3)}/{len(records_n3)}; "
        f"violations: {theorem_violations_n3}"
    )
    print(
        f"   condition suffisante, n=4: {len(low)}/{len(records)}; "
        f"violations: {theorem_violations}"
    )

    print("\n>> Hétérogénéité des coûts")
    for row in heterogeneity:
        print(
            f"   theta={row['theta']:.2f}: Shapley {row['shapley_in_core_pct']:.1f} %, "
            f"sous condition {row['low_traffic_shapley_in_core_pct']:.1f} %"
        )

    print("\n>> Cas représentatifs")
    for row in representatives:
        print(
            f"   {CATEGORY_LABELS[str(row['category'])]}: {row['scenario']}, "
            f"{row['day_label']} {int(row['hour']):02d} h, r={row['rate']:.2f}, "
            f"k={row['k_h']}, économie={row['savings_pct']:.1f} %, "
            f"E_Sh/v={row['shapley_excess_pct']:.2f} %"
        )

    print("\n>> Sensibilité, amplitudes sur les valeurs testées")
    for row in effects:
        print(
            f"   {row['factor']:>9}: dEco={row['savings_range_pp']:.2f} pt, "
            f"dg={row['guardians_range']:.2f}, "
            f"dPers={row['persistence_range_pp']:.2f} pt, "
            f"dSh={row['shapley_range_pp']:.1f} pt, "
            f"dVide={row['empty_core_range_pp']:.1f} pt"
        )

    output = _figure(
        records,
        threshold_summary,
        heterogeneity,
        records_n3,
        threshold_summary_n3,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULTS / "threshold_instances.csv", records)
    _write_csv(RESULTS / "threshold_instances_n3.csv", records_n3)
    _write_csv(RESULTS / "threshold_summary.csv", threshold_summary)
    _write_csv(RESULTS / "threshold_summary_n3.csv", threshold_summary_n3)
    _write_csv(
        RESULTS / "operator_count_summary.csv",
        operator_count_summary,
    )
    _write_csv(RESULTS / "cost_heterogeneity.csv", heterogeneity)
    _write_csv(RESULTS / "representative_cases.csv", representatives)
    _write_csv(RESULTS / "sensitivity_instances.csv", sensitivity_rows)
    _write_csv(RESULTS / "sensitivity_summary.csv", sensitivity_summary)
    _write_csv(RESULTS / "sensitivity_effects.csv", effects)
    analysis = {
        "reference_id": site["reference_id"],
        "seed": 20260819,
        "threshold_instances": {
            "n3": len(records_n3),
            "n4": len(records),
        },
        "low_traffic_instances": {
            "n3": len(low_n3),
            "n4": len(low),
        },
        "low_traffic_theorem_violations": {
            "n3": theorem_violations_n3,
            "n4": theorem_violations,
        },
        "operator_count_summary": operator_count_summary,
        "figure": output.as_posix(),
    }
    (RESULTS / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n>> Figure: {output.resolve()}")
    print(f">> Résultats: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
