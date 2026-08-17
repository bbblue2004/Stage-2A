"""Run the structured parameter-sensitivity study reported in Section 6.5."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import linprog

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.window_optimiser import persistent_coalition_solutions
from src.data_processing.data_loader import ROOT
from src.data_processing.power_validation import (
    CalibratedPopulation,
    load_calibrated_population,
)
from src.data_processing.virtual_sites import load_virtual_sites
from src.experiments.common import (
    file_signature,
    four_player_bondareva_gap,
    inputs_match,
    portable_path,
)


DEFAULT_CALIBRATION_DIR = ROOT / "results" / "power_calibration"
DEFAULT_OPERATIONAL_DIR = ROOT / "results" / "operational_efficiency"
DEFAULT_STABILITY_DIR = ROOT / "results" / "coalition_stability"
DEFAULT_RESULTS_DIR = ROOT / "results" / "parameter_sensitivity"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "parameter_sensitivity"
CAPACITY_RATES = (0.70, 0.80, 0.90, 1.00)
TRAFFIC_MULTIPLIERS = (0.80, 1.00, 1.20)
COEFFICIENT_MULTIPLIERS = (0.80, 1.00, 1.20)
SLEEP_RATES = (0.00, 0.05, 0.10)
WINDOW_DURATIONS = (5, 7, 9)
OPERATOR_COUNTS = (2, 3, 4, 5, 6)
BASELINE_RATE = 0.80
SITE_SEED = 20_260_814
ALGORITHM_VERSION = 1
FACTOR_LABELS = {
    "capacity_margin": "Marge de capacité",
    "traffic_level": "Niveau de trafic",
    "fixed_cost": "Coût fixe $F_i$",
    "variable_cost": "Coût variable $\\gamma_i$",
    "sleep_power": "Puissance de veille",
    "window_position": "Position de $H$",
    "window_duration": "Durée de $H$",
    "operator_count": "Nombre d'opérateurs",
}
CENTRAL_SETTINGS = {
    "capacity_margin": "0.80",
    "traffic_level": "1.00",
    "fixed_cost": "1.00",
    "variable_cost": "1.00",
    "sleep_power": "0.00",
    "window_position": "00-07",
    "window_duration": "7",
    "operator_count": "4",
}


def _largest_remainder(counts: np.ndarray, total: int) -> np.ndarray:
    quotas = total * counts.astype(float) / np.sum(counts)
    allocation = np.floor(quotas).astype(int)
    remainder = total - int(np.sum(allocation))
    if remainder:
        order = np.argsort(-(quotas - allocation), kind="mergesort")
        allocation[order[:remainder]] += 1
    return allocation


def _generate_stratified_indices(
    population: CalibratedPopulation,
    num_operators: int,
    num_sites: int,
    seed: int = SITE_SEED,
) -> np.ndarray:
    """Generate a reproducible stratified site sample of a given size."""
    if num_operators < 2 or num_sites <= 0:
        raise ValueError("at least two operators and one site are required")
    groups = np.column_stack(
        (population.traffic_group, population.fixed_power_group)
    )
    unique_groups, inverse, counts = np.unique(
        groups, axis=0, return_inverse=True, return_counts=True
    )
    allocation = _largest_remainder(counts, num_sites)
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for group_index, target in enumerate(allocation):
        if target == 0:
            continue
        candidates = np.flatnonzero(inverse == group_index)
        if candidates.size < num_operators:
            raise ValueError(
                f"group {tuple(unique_groups[group_index])} has fewer than "
                f"{num_operators} antennas"
            )
        if math.comb(int(candidates.size), num_operators) < int(target):
            raise ValueError("not enough distinct coalitions in a stratum")
        selected: set[tuple[int, ...]] = set()
        while len(selected) < target:
            selected.add(
                tuple(
                    sorted(
                        int(index)
                        for index in rng.choice(
                            candidates, size=num_operators, replace=False
                        )
                    )
                )
            )
        rows.extend(np.asarray(row, dtype=np.int32) for row in sorted(selected))
    sample = np.stack(rows)
    return sample[rng.permutation(num_sites)]


def _mask_memberships(num_operators: int) -> np.ndarray:
    return np.asarray(
        [
            [float(mask & (1 << player) != 0) for player in range(num_operators)]
            for mask in range(1 << num_operators)
        ]
    )


def _savings_from_costs(costs: np.ndarray, num_operators: int) -> np.ndarray:
    num_masks = 1 << num_operators
    if costs.shape != (num_masks,):
        raise ValueError("coalition-cost array has an inconsistent size")
    savings = np.zeros(num_masks, dtype=float)
    for mask in range(1, num_masks):
        standalone = sum(
            costs[1 << player]
            for player in range(num_operators)
            if mask & (1 << player)
        )
        value = standalone - costs[mask]
        tolerance = 1e-10 * max(1.0, standalone)
        if value < -tolerance:
            raise RuntimeError("negative coalition saving beyond numerical tolerance")
        savings[mask] = 0.0 if abs(value) <= tolerance else value
    return savings


def _shapley_value(savings: np.ndarray, num_operators: int) -> np.ndarray:
    allocation = np.zeros(num_operators, dtype=float)
    denominator = math.factorial(num_operators)
    for player in range(num_operators):
        player_bit = 1 << player
        for mask in range(1 << num_operators):
            if mask & player_bit:
                continue
            size = mask.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(num_operators - size - 1)
                / denominator
            )
            allocation[player] += weight * (
                savings[mask | player_bit] - savings[mask]
            )
    return allocation


def _core_status(savings: np.ndarray, num_operators: int) -> tuple[bool, bool]:
    grand_mask = (1 << num_operators) - 1
    scale = max(1.0, float(savings[grand_mask]))
    tolerance = 1e-8 * scale
    membership = _mask_memberships(num_operators)
    shapley = _shapley_value(savings, num_operators)
    excesses = savings[1:grand_mask] - membership[1:grand_mask] @ shapley
    shapley_in_core = float(np.max(excesses, initial=-np.inf)) <= tolerance
    if shapley_in_core:
        return True, True

    if num_operators == 4:
        core_nonempty = four_player_bondareva_gap(savings) <= tolerance
        return core_nonempty, False

    result = linprog(
        np.zeros(num_operators),
        A_ub=-membership[1:grand_mask],
        b_ub=-savings[1:grand_mask] + tolerance,
        A_eq=np.ones((1, num_operators)),
        b_eq=[savings[grand_mask]],
        bounds=(0.0, None),
        method="highs",
    )
    if result.success:
        return True, False
    if result.status == 2:
        return False, False
    raise RuntimeError(f"core-feasibility LP failed: {result.message}")


def _subset_sums(values: np.ndarray) -> np.ndarray:
    result = np.zeros(1 << values.size, dtype=float)
    for mask in range(1, result.size):
        bit = mask & -mask
        player = bit.bit_length() - 1
        result[mask] = result[mask ^ bit] + values[player]
    return result


def _evaluate_instance(
    capacities: np.ndarray,
    fixed: np.ndarray,
    slopes: np.ndarray,
    demands: np.ndarray,
    *,
    sleep_rate: float = 0.0,
) -> dict[str, float | int]:
    """Compute operational and stability metrics for one parameter setting."""
    if not 0.0 <= sleep_rate < 1.0:
        raise ValueError("sleep_rate must lie in [0, 1)")
    num_operators = capacities.size
    grand_mask = (1 << num_operators) - 1
    effective_fixed = (1.0 - sleep_rate) * fixed
    solutions = persistent_coalition_solutions(
        capacities, effective_fixed, slopes, demands
    )
    sleep_constants = (
        demands.shape[1] * sleep_rate * _subset_sums(fixed)
    )
    costs = solutions.costs_wh + sleep_constants
    savings = _savings_from_costs(costs, num_operators)
    standalone = float(
        sum(costs[1 << player] for player in range(num_operators))
    )
    grand_cost = float(costs[grand_mask])
    hourly_oracle = (
        solutions.grand_hourly_oracle_wh + float(sleep_constants[grand_mask])
    )
    core_nonempty, shapley_in_core = _core_status(savings, num_operators)
    return {
        "savings_pct": 100.0 * (standalone - grand_cost) / standalone,
        "guardians": int(solutions.guardian_masks[grand_mask]).bit_count(),
        "oracle_gap_pp": max(
            0.0, 100.0 * (grand_cost - hourly_oracle) / standalone
        ),
        "core_empty": int(not core_nonempty),
        "shapley_stable": int(shapley_in_core),
    }


def _load_main_metrics(
    operational_path: Path,
    stability_path: Path,
) -> dict[tuple[str, str, float], dict[str, float | int]]:
    stability: dict[tuple[str, str, float], tuple[int, int]] = {}
    with stability_path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            key = (raw["site_id"], raw["day"], float(raw["capacity_rate"]))
            stability[key] = (
                int(raw["category"] == "empty_core"),
                int(raw["shapley_in_core"]),
            )
    metrics: dict[tuple[str, str, float], dict[str, float | int]] = {}
    with operational_path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            key = (raw["site_id"], raw["day"], float(raw["capacity_rate"]))
            standalone = float(raw["standalone_energy_wh"])
            optimum = float(raw["optimal_energy_wh"])
            oracle = float(raw["hourly_oracle_energy_wh"])
            core_empty, shapley_stable = stability[key]
            metrics[key] = {
                "savings_pct": 100.0 * (standalone - optimum) / standalone,
                "guardians": int(raw["optimal_guardians"]),
                "oracle_gap_pp": max(0.0, 100.0 * (optimum - oracle) / standalone),
                "core_empty": core_empty,
                "shapley_stable": shapley_stable,
            }
    return metrics


def _result_row(
    factor: str,
    setting: str,
    setting_value: float,
    site_id: str,
    day: str,
    metrics: dict[str, float | int],
    *,
    capacity_rate: float = BASELINE_RATE,
    traffic_multiplier: float = 1.0,
    num_operators: int = 4,
) -> dict[str, object]:
    return {
        "factor": factor,
        "setting": setting,
        "setting_value": setting_value,
        "site_id": site_id,
        "day": day,
        "capacity_rate": capacity_rate,
        "traffic_multiplier": traffic_multiplier,
        "num_operators": num_operators,
        **metrics,
    }


def _main_site_rows(
    site_index: int,
    indices: np.ndarray,
    site_id: str,
    population: CalibratedPopulation,
    main_metrics: dict[tuple[str, str, float], dict[str, float | int]],
) -> list[dict[str, object]]:
    fixed = population.p_fixed_w[indices]
    slopes = population.slope_w_per_gb[indices]
    peaks = population.peak_traffic_gb[indices]
    traffic = population.traffic_gb[indices]
    rows: list[dict[str, object]] = []

    for day_index, day_value in enumerate(population.days):
        day = str(day_value)
        demands = traffic[:, day_index, 0:7]
        base_by_rate = {
            rate: main_metrics[(site_id, day, rate)] for rate in CAPACITY_RATES
        }
        central = base_by_rate[BASELINE_RATE]

        for rate in CAPACITY_RATES:
            rows.append(
                _result_row(
                    "capacity_margin", f"{rate:.2f}", rate, site_id, day,
                    base_by_rate[rate], capacity_rate=rate,
                )
            )
            rows.append(
                _result_row(
                    "traffic_capacity", f"1.00|{rate:.2f}", rate, site_id,
                    day, base_by_rate[rate], capacity_rate=rate,
                )
            )

        for factor, setting in (
            ("traffic_level", "1.00"),
            ("fixed_cost", "1.00"),
            ("variable_cost", "1.00"),
            ("sleep_power", "0.00"),
            ("window_duration", "7"),
            ("operator_count", "4"),
        ):
            value = 4.0 if factor == "operator_count" else float(setting)
            rows.append(
                _result_row(
                    factor, setting, value, site_id, day, central,
                    num_operators=4,
                )
            )

        for traffic_multiplier in (0.80, 1.20):
            admissible_rates = (
                CAPACITY_RATES
                if traffic_multiplier == 0.80
                else (0.70, 0.80)
            )
            for rate in admissible_rates:
                metrics = _evaluate_instance(
                    peaks / rate,
                    fixed,
                    slopes,
                    traffic_multiplier * demands,
                )
                rows.append(
                    _result_row(
                        "traffic_capacity",
                        f"{traffic_multiplier:.2f}|{rate:.2f}",
                        rate,
                        site_id,
                        day,
                        metrics,
                        capacity_rate=rate,
                        traffic_multiplier=traffic_multiplier,
                    )
                )
                if rate == BASELINE_RATE:
                    rows.append(
                        _result_row(
                            "traffic_level",
                            f"{traffic_multiplier:.2f}",
                            traffic_multiplier,
                            site_id,
                            day,
                            metrics,
                            traffic_multiplier=traffic_multiplier,
                        )
                    )

        for multiplier in (0.80, 1.20):
            fixed_metrics = _evaluate_instance(
                peaks / BASELINE_RATE, multiplier * fixed, slopes, demands
            )
            rows.append(
                _result_row(
                    "fixed_cost", f"{multiplier:.2f}", multiplier,
                    site_id, day, fixed_metrics,
                )
            )
            variable_metrics = _evaluate_instance(
                peaks / BASELINE_RATE, fixed, multiplier * slopes, demands
            )
            rows.append(
                _result_row(
                    "variable_cost", f"{multiplier:.2f}", multiplier,
                    site_id, day, variable_metrics,
                )
            )

        for sleep_rate in (0.05, 0.10):
            metrics = _evaluate_instance(
                peaks / BASELINE_RATE,
                fixed,
                slopes,
                demands,
                sleep_rate=sleep_rate,
            )
            rows.append(
                _result_row(
                    "sleep_power", f"{sleep_rate:.2f}", sleep_rate,
                    site_id, day, metrics,
                )
            )

        for duration in (5, 9):
            metrics = _evaluate_instance(
                peaks / BASELINE_RATE,
                fixed,
                slopes,
                traffic[:, day_index, :duration],
            )
            rows.append(
                _result_row(
                    "window_duration", str(duration), float(duration),
                    site_id, day, metrics,
                )
            )

    for start_day in range(len(population.days) - 1):
        comparison_day = str(population.days[start_day + 1])
        central = main_metrics[(site_id, comparison_day, BASELINE_RATE)]
        rows.append(
            _result_row(
                "window_position", "00-07", 1.0, site_id,
                comparison_day, central,
            )
        )
        overnight = np.concatenate(
            (traffic[:, start_day, 22:24], traffic[:, start_day + 1, 0:5]),
            axis=1,
        )
        late_metrics = _evaluate_instance(
            peaks / BASELINE_RATE, fixed, slopes, overnight
        )
        rows.append(
            _result_row(
                "window_position", "22-05", 0.0, site_id,
                comparison_day, late_metrics,
            )
        )
        morning_metrics = _evaluate_instance(
            peaks / BASELINE_RATE,
            fixed,
            slopes,
            traffic[:, start_day + 1, 2:9],
        )
        rows.append(
            _result_row(
                "window_position", "02-09", 2.0, site_id,
                comparison_day, morning_metrics,
            )
        )
    return rows


def _operator_count_rows(
    population: CalibratedPopulation,
    samples: dict[int, np.ndarray],
    num_sites: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for num_operators in (2, 3, 5, 6):
        sample = samples[num_operators]
        for site_index in range(num_sites):
            indices = sample[site_index]
            fixed = population.p_fixed_w[indices]
            slopes = population.slope_w_per_gb[indices]
            capacities = population.peak_traffic_gb[indices] / BASELINE_RATE
            for day_index, day_value in enumerate(population.days):
                metrics = _evaluate_instance(
                    capacities,
                    fixed,
                    slopes,
                    population.traffic_gb[indices, day_index, 0:7],
                )
                rows.append(
                    _result_row(
                        "operator_count",
                        str(num_operators),
                        float(num_operators),
                        f"n{num_operators}_site_{site_index + 1:04d}",
                        str(day_value),
                        metrics,
                        num_operators=num_operators,
                    )
                )
            if (site_index + 1) % 200 == 0:
                print(
                    f">> n={num_operators}: {site_index + 1}/{num_sites} sites",
                    flush=True,
                )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, object]]:
    string_columns = {"factor", "setting", "site_id", "day"}
    integer_columns = {
        "num_operators", "guardians", "core_empty", "shapley_stable"
    }
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {
                key: (
                    value
                    if key in string_columns
                    else int(value)
                    if key in integer_columns
                    else float(value)
                )
                for key, value in raw.items()
            }
            for raw in csv.DictReader(file)
        ]


def _save_operator_samples(
    path: Path,
    samples: dict[int, np.ndarray],
    population: CalibratedPopulation,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["num_operators", "site_id", *[f"antenna_{i}" for i in range(1, 7)]]
        )
        for num_operators, sample in sorted(samples.items()):
            for site_index, indices in enumerate(sample):
                identifiers = [str(value) for value in population.antenna_ids[indices]]
                writer.writerow(
                    [
                        num_operators,
                        f"n{num_operators}_site_{site_index + 1:04d}",
                        *identifiers,
                        *([""] * (6 - num_operators)),
                    ]
                )


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["factor"]), str(row["setting"])), []).append(row)
    summaries: list[dict[str, object]] = []
    for (factor, setting), selected in sorted(
        groups.items(), key=lambda item: (item[0][0], float(item[1][0]["setting_value"]))
    ):
        savings = np.asarray([row["savings_pct"] for row in selected], dtype=float)
        guardians = np.asarray([row["guardians"] for row in selected], dtype=float)
        oracle = np.asarray([row["oracle_gap_pp"] for row in selected], dtype=float)
        summaries.append(
            {
                "factor": factor,
                "setting": setting,
                "setting_value": float(selected[0]["setting_value"]),
                "instances": len(selected),
                "savings_pct_median": float(np.median(savings)),
                "guardians_mean": float(np.mean(guardians)),
                "oracle_gap_pp_median": float(np.median(oracle)),
                "empty_core_pct": 100.0
                * float(np.mean([row["core_empty"] for row in selected])),
                "shapley_stable_pct": 100.0
                * float(np.mean([row["shapley_stable"] for row in selected])),
            }
        )
    return summaries


def _factor_effects(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    effects: list[dict[str, object]] = []
    for factor in FACTOR_LABELS:
        selected = [row for row in summaries if row["factor"] == factor]
        central = next(
            row for row in selected if row["setting"] == CENTRAL_SETTINGS[factor]
        )
        effect: dict[str, object] = {
            "factor": factor,
            "label": FACTOR_LABELS[factor],
            "central_setting": central["setting"],
        }
        for metric in (
            "savings_pct_median",
            "guardians_mean",
            "oracle_gap_pp_median",
            "empty_core_pct",
            "shapley_stable_pct",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            baseline = float(central[metric])
            effect[f"{metric}_range"] = float(np.max(values) - np.min(values))
            effect[f"{metric}_min_delta"] = float(np.min(values - baseline))
            effect[f"{metric}_max_delta"] = float(np.max(values - baseline))
        effect["best_savings_setting"] = max(
            selected, key=lambda row: float(row["savings_pct_median"])
        )["setting"]
        effect["best_stability_setting"] = max(
            selected, key=lambda row: float(row["shapley_stable_pct"])
        )["setting"]
        effects.append(effect)
    return effects


def _analysis(
    summaries: list[dict[str, object]], effects: list[dict[str, object]]
) -> dict[str, object]:
    interaction = [row for row in summaries if row["factor"] == "traffic_capacity"]
    return {
        "factor_summaries": summaries,
        "factor_effects": effects,
        "energy_ranking": [
            row["factor"]
            for row in sorted(
                effects,
                key=lambda row: float(row["savings_pct_median_range"]),
                reverse=True,
            )
        ],
        "stability_ranking": [
            row["factor"]
            for row in sorted(
                effects,
                key=lambda row: float(row["shapley_stable_pct_range"]),
                reverse=True,
            )
        ],
        "traffic_capacity_interaction": interaction,
        "masked_interaction_cells": ["1.20|0.90", "1.20|1.00"],
    }


def _tornado_figure(
    effects: list[dict[str, object]], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        effects,
        key=lambda row: float(row["shapley_stable_pct_range"]),
    )
    labels = [str(row["label"]) for row in ordered]
    y = np.arange(len(ordered))
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), sharey=True)
    specifications = (
        ("savings_pct_median", "Économie médiane (points)", "(a) Efficacité"),
        ("shapley_stable_pct", "Fréquence stable (points)", "(b) Stabilité de Shapley"),
    )
    for axis, (metric, xlabel, title) in zip(axes, specifications, strict=True):
        lower = np.asarray([row[f"{metric}_min_delta"] for row in ordered])
        upper = np.asarray([row[f"{metric}_max_delta"] for row in ordered])
        axis.hlines(y, lower, upper, color="#4C78A8", linewidth=5, alpha=0.78)
        axis.scatter(lower, y, color="#D95F4A", s=25, zorder=3)
        axis.scatter(upper, y, color="#2F8F5B", s=25, zorder=3)
        axis.axvline(0.0, color="0.25", linewidth=0.9, linestyle="--")
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(y, labels)
    figure.suptitle("Variation par rapport au scénario central", y=0.99)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    paths = [
        output_dir / "parameter_sensitivity_tornado.pdf",
        output_dir / "parameter_sensitivity_tornado.png",
    ]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=220, bbox_inches="tight")
    plt.close(figure)
    return paths


def _interaction_figure(
    summaries: list[dict[str, object]], output_dir: Path
) -> list[Path]:
    selected = [row for row in summaries if row["factor"] == "traffic_capacity"]
    lookup = {str(row["setting"]): row for row in selected}
    savings = np.full((len(TRAFFIC_MULTIPLIERS), len(CAPACITY_RATES)), np.nan)
    stability = np.full_like(savings, np.nan)
    for row_index, multiplier in enumerate(TRAFFIC_MULTIPLIERS):
        for column_index, rate in enumerate(CAPACITY_RATES):
            key = f"{multiplier:.2f}|{rate:.2f}"
            if key in lookup:
                savings[row_index, column_index] = lookup[key]["savings_pct_median"]
                stability[row_index, column_index] = lookup[key]["shapley_stable_pct"]

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    for axis, matrix, title, color_map in (
        (axes[0], savings, "(a) Économie médiane (%)", "YlGn"),
        (axes[1], stability, "(b) Shapley dans le cœur (%)", "Blues"),
    ):
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.get_cmap(color_map).copy()
        cmap.set_bad("#D9D9D9")
        image = axis.imshow(masked, aspect="auto", cmap=cmap)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                text = "hors\ndomaine" if np.isnan(value) else f"{value:.1f}".replace(".", ",")
                axis.text(column_index, row_index, text, ha="center", va="center", fontsize=8)
        axis.set_xticks(
            np.arange(len(CAPACITY_RATES)),
            [f"{rate:.2f}".replace(".", ",") for rate in CAPACITY_RATES],
        )
        axis.set_yticks(
            np.arange(len(TRAFFIC_MULTIPLIERS)),
            [f"{value:.1f}".replace(".", ",") for value in TRAFFIC_MULTIPLIERS],
        )
        axis.set_xlabel("Taux de charge maximal $r$")
        axis.set_ylabel("Multiplicateur de trafic")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    paths = [
        output_dir / "traffic_capacity_interaction.pdf",
        output_dir / "traffic_capacity_interaction.png",
    ]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=220, bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--operational-dir", type=Path, default=DEFAULT_OPERATIONAL_DIR)
    parser.add_argument("--stability-dir", type=Path, default=DEFAULT_STABILITY_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--num-sites", type=int, default=1_000)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cache_path = args.calibration_dir / "calibrated_population.npz"
    sites_path = args.calibration_dir / "virtual_sites.csv"
    operational_path = args.operational_dir / "operational_instances.csv"
    stability_path = args.stability_dir / "stability_instances.csv"
    for path in (cache_path, sites_path, operational_path, stability_path):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if not 1 <= args.num_sites <= 1_000:
        parser.error("num-sites must lie between 1 and 1000")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.results_dir / "sensitivity_instances.csv"
    summary_path = args.results_dir / "sensitivity_summary.csv"
    analysis_path = args.results_dir / "analysis.json"
    samples_path = args.results_dir / "operator_count_sites.csv"
    manifest_path = args.results_dir / "manifest.json"
    expected = {
        "algorithm_version": ALGORITHM_VERSION,
        "calibrated_population": file_signature(cache_path),
        "virtual_sites": file_signature(sites_path),
        "operational_instances": file_signature(operational_path),
        "stability_instances": file_signature(stability_path),
        "num_sites": args.num_sites,
        "capacity_rates": list(CAPACITY_RATES),
        "traffic_multipliers": list(TRAFFIC_MULTIPLIERS),
        "coefficient_multipliers": list(COEFFICIENT_MULTIPLIERS),
        "sleep_rates": list(SLEEP_RATES),
        "window_durations": list(WINDOW_DURATIONS),
        "operator_counts": list(OPERATOR_COUNTS),
        "site_seed": SITE_SEED,
    }
    current = False
    if not args.rebuild and manifest_path.is_file() and rows_path.is_file():
        try:
            recorded = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )["inputs"]
            current = inputs_match(
                recorded,
                expected,
                {
                    "calibrated_population": cache_path,
                    "virtual_sites": sites_path,
                    "operational_instances": operational_path,
                    "stability_instances": stability_path,
                },
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            current = False

    if current:
        print(f">> Reusing sensitivity results: {rows_path.resolve()}", flush=True)
        rows = _read_rows(rows_path)
    else:
        population = load_calibrated_population(cache_path)
        main_sites = load_virtual_sites(sites_path, population)
        main_metrics = _load_main_metrics(operational_path, stability_path)
        rows: list[dict[str, object]] = []
        for site_index in range(args.num_sites):
            rows.extend(
                _main_site_rows(
                    site_index,
                    main_sites.antenna_indices[site_index],
                    str(main_sites.site_ids[site_index]),
                    population,
                    main_metrics,
                )
            )
            if (site_index + 1) % 100 == 0:
                print(
                    f">> n=4 sensitivity: {site_index + 1}/{args.num_sites} sites",
                    flush=True,
                )

        samples = {
            count: _generate_stratified_indices(
                population, count, args.num_sites, SITE_SEED
            )
            for count in (2, 3, 5, 6)
        }
        _save_operator_samples(samples_path, samples, population)
        rows.extend(_operator_count_rows(population, samples, args.num_sites))
        _write_rows(rows_path, rows)

    summaries = _summaries(rows)
    effects = _factor_effects(summaries)
    analysis = _analysis(summaries, effects)
    figures = [
        *_tornado_figure(effects, args.figures_dir),
        *_interaction_figure(summaries, args.figures_dir),
    ]
    _write_rows(summary_path, summaries)
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "inputs": expected,
        "outputs": {
            "instances": portable_path(rows_path),
            "summary": portable_path(summary_path),
            "analysis": portable_path(analysis_path),
            "operator_samples": portable_path(samples_path),
            "figures": [portable_path(path) for path in figures],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f">> {len(rows)} sensitivity rows completed", flush=True)
    if not args.quiet:
        print(json.dumps(analysis, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
