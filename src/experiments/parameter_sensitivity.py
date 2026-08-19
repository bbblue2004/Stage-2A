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

from src.core.window_optimiser import coalition_window_solutions
from src.data_processing.data_loader import ROOT
from src.data_processing.instance_generator import (
    CAMPAIGN_A_RATES,
    CENTRAL_RATE,
    DEFAULT_NUM_SITES,
    DEFAULT_SITE_SEED,
    ScenarioSpec,
    calibrate_protocol,
    capacities_for_site,
    generate_site_blueprints,
    iter_materialized_sites,
    materialize_site,
)
from src.data_processing.power_validation import CalibratedPopulation
from src.experiments.common import (
    file_signature,
    four_player_bondareva_gap,
    inputs_match,
    portable_path,
)
from src.experiments.protocol_io import is_central_campaign_a, load_protocol_inputs


DEFAULT_CALIBRATION_DIR = ROOT / "results" / "power_calibration"
DEFAULT_OPERATIONAL_DIR = ROOT / "results" / "operational_efficiency"
DEFAULT_STABILITY_DIR = ROOT / "results" / "coalition_stability"
DEFAULT_RESULTS_DIR = ROOT / "results" / "parameter_sensitivity"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "parameter_sensitivity"
CAPACITY_RATES = CAMPAIGN_A_RATES
TRAFFIC_MULTIPLIERS = (0.80, 1.00, 1.20)
COEFFICIENT_MULTIPLIERS = (0.80, 1.00, 1.20)
SLEEP_RATES = (0.00, 0.05, 0.10)
WINDOW_DURATIONS = (5, 7, 9)
OPERATOR_COUNTS = (2, 3, 4, 5, 6)
BASELINE_RATE = CENTRAL_RATE
SITE_SEED = DEFAULT_SITE_SEED
ALGORITHM_VERSION = 3
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
    "capacity_margin": "0.70",
    "traffic_level": "1.00",
    "fixed_cost": "1.00",
    "variable_cost": "1.00",
    "sleep_power": "0.00",
    "window_position": "00-07",
    "window_duration": "7",
    "operator_count": "4",
}


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
    solutions = coalition_window_solutions(
        capacities, effective_fixed, slopes, demands
    )
    sleep_constants = (
        demands.shape[1] * sleep_rate * _subset_sums(fixed)
    )
    costs = solutions.hourly_costs_wh + sleep_constants
    savings = _savings_from_costs(costs, num_operators)
    standalone = float(
        sum(costs[1 << player] for player in range(num_operators))
    )
    grand_cost = float(costs[grand_mask])
    persistent_cost = (
        solutions.persistent_costs_wh[grand_mask]
        + float(sleep_constants[grand_mask])
    )
    hourly_masks = solutions.hourly_guardian_masks[grand_mask]
    core_nonempty, shapley_in_core = _core_status(savings, num_operators)
    return {
        "savings_pct": 100.0 * (standalone - grand_cost) / standalone,
        "guardians_mean": float(
            np.mean([int(mask).bit_count() for mask in hourly_masks])
        ),
        "persistence_gap_pp": max(
            0.0, 100.0 * (persistent_cost - grand_cost) / standalone
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
            if "campaign" in raw and not is_central_campaign_a(raw):
                continue
            key = (raw["site_id"], raw["day"], float(raw["capacity_rate"]))
            stability[key] = (
                int(raw["category"] == "empty_core"),
                int(raw["shapley_in_core"]),
            )
    metrics: dict[tuple[str, str, float], dict[str, float | int]] = {}
    with operational_path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            if "campaign" in raw and not is_central_campaign_a(raw):
                continue
            key = (raw["site_id"], raw["day"], float(raw["capacity_rate"]))
            standalone = float(raw["standalone_energy_wh"])
            optimum = float(raw["hourly_optimal_energy_wh"])
            persistent = float(raw["persistent_energy_wh"])
            core_empty, shapley_stable = stability[key]
            metrics[key] = {
                "savings_pct": 100.0 * (standalone - optimum) / standalone,
                "guardians_mean": float(
                    raw["hourly_optimal_guardians_mean"]
                ),
                "persistence_gap_pp": max(
                    0.0,
                    100.0 * (persistent - optimum) / standalone,
                ),
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
    site,
    population: CalibratedPopulation,
    main_metrics: dict[tuple[str, str, float], dict[str, float | int]],
) -> list[dict[str, object]]:
    fixed = site.p_fixed_w
    slopes = site.slope_w_per_gb
    peaks = site.peak_traffic_gb
    traffic = site.traffic_gb
    site_id = site.site_id
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
    num_sites: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for num_operators in (2, 3, 5, 6):
        protocol = calibrate_protocol(
            population,
            num_sites=num_sites,
            seed=SITE_SEED,
            num_operators=num_operators,
        )
        blueprints = generate_site_blueprints(
            population,
            num_sites=num_sites,
            seed=SITE_SEED,
            num_operators=num_operators,
        )
        spec = ScenarioSpec(
            "A",
            "moderate",
            "moderate",
            "moderate",
            capacity_rate=BASELINE_RATE,
        )
        for site_index, site in enumerate(
            iter_materialized_sites(
                blueprints, population, [spec], protocol, num_sites=num_sites
            )
        ):
            for day_index, day_value in enumerate(population.days):
                demands = site.traffic_gb[:, day_index, 0:7]
                capacities = capacities_for_site(site, demands)
                metrics = _evaluate_instance(
                    capacities,
                    site.p_fixed_w,
                    site.slope_w_per_gb,
                    demands,
                )
                rows.append(
                    _result_row(
                        "operator_count",
                        str(num_operators),
                        float(num_operators),
                        site.site_id.replace("site_", f"n{num_operators}_site_"),
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
        "num_operators", "core_empty", "shapley_stable"
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


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["factor"]), str(row["setting"])), []).append(row)
    summaries: list[dict[str, object]] = []
    for (factor, setting), selected in sorted(
        groups.items(), key=lambda item: (item[0][0], float(item[1][0]["setting_value"]))
    ):
        savings = np.asarray([row["savings_pct"] for row in selected], dtype=float)
        guardians = np.asarray(
            [row["guardians_mean"] for row in selected], dtype=float
        )
        persistence = np.asarray(
            [row["persistence_gap_pp"] for row in selected], dtype=float
        )
        summaries.append(
            {
                "factor": factor,
                "setting": setting,
                "setting_value": float(selected[0]["setting_value"]),
                "instances": len(selected),
                "savings_pct_median": float(np.median(savings)),
                "guardians_mean": float(np.mean(guardians)),
                "persistence_gap_pp_median": float(
                    np.median(persistence)
                ),
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
            "persistence_gap_pp_median",
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
    parser.add_argument("--num-sites", type=int, default=DEFAULT_NUM_SITES)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cache_path = args.calibration_dir / "calibrated_population.npz"
    sites_path = args.calibration_dir / "site_blueprints.csv"
    protocol_path = args.calibration_dir / "protocol_parameters.json"
    operational_path = args.operational_dir / "operational_instances.csv"
    stability_path = args.stability_dir / "stability_instances.csv"
    for path in (cache_path, sites_path, protocol_path, operational_path, stability_path):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.num_sites <= 0:
        parser.error("num-sites must be positive")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.results_dir / "sensitivity_instances.csv"
    summary_path = args.results_dir / "sensitivity_summary.csv"
    analysis_path = args.results_dir / "analysis.json"
    samples_path = args.results_dir / "operator_count_sites.csv"
    manifest_path = args.results_dir / "manifest.json"
    expected = {
        "algorithm_version": ALGORITHM_VERSION,
        "calibrated_population": file_signature(cache_path),
        "site_blueprints": file_signature(sites_path),
        "protocol_parameters": file_signature(protocol_path),
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
                    "site_blueprints": sites_path,
                    "protocol_parameters": protocol_path,
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
        population, blueprints, protocol = load_protocol_inputs(args.calibration_dir)
        if args.num_sites > blueprints.num_sites:
            parser.error(
                f"num-sites cannot exceed the frozen list ({blueprints.num_sites})"
            )
        main_metrics = _load_main_metrics(operational_path, stability_path)
        central_spec = ScenarioSpec(
            "A", "moderate", "moderate", "moderate", capacity_rate=BASELINE_RATE
        )
        rows: list[dict[str, object]] = []
        for site_index in range(args.num_sites):
            site = materialize_site(
                blueprints, site_index, population, central_spec, protocol
            )
            rows.extend(_main_site_rows(site, population, main_metrics))
            if (site_index + 1) % 100 == 0:
                print(
                    f">> n=4 sensitivity: {site_index + 1}/{args.num_sites} sites",
                    flush=True,
                )
        rows.extend(_operator_count_rows(population, args.num_sites))
        samples_path.write_text(
            "num_operators,construction\n"
            "2-6,same generator as Section 6.1 with n operators\n",
            encoding="utf-8",
        )
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
