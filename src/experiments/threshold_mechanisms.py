"""Run the twenty-plan threshold and mechanism study reported in Section 6.5."""

from __future__ import annotations

import argparse
import csv
import json
from math import factorial
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.window_optimiser import coalition_window_solutions
from src.data_processing.data_loader import ROOT
from src.data_processing.instance_generator import (
    CAMPAIGN_A_RATES,
    DEFAULT_NUM_SITES,
    ScenarioSpec,
    materialize_site,
)
from src.experiments.coalition_stability import (
    GRAND_MASK,
    _bondareva_gap,
    _max_excess,
    _savings_from_costs,
    _shapley_value,
)
from src.experiments.common import file_signature, inputs_match, portable_path
from src.experiments.protocol_io import load_protocol_inputs


DEFAULT_CALIBRATION_DIR = ROOT / "results" / "power_calibration"
DEFAULT_RESULTS_DIR = ROOT / "results" / "threshold_mechanisms"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "coalition_stability"
ALGORITHM_VERSION = 4
NUM_OPERATORS = (3, 4)
THETAS = (0.00, 0.25, 0.50, 0.75, 1.00)
CONFIGURATIONS = {
    "reference": ("moderate", "moderate", "close"),
    "profils_proches": ("close", "close", "close"),
}


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, object]]:
    string_columns = {"site_id", "configuration", "day", "category"}
    integer_columns = {
        "num_operators",
        "hour",
        "k_h",
        "core_nonempty",
        "shapley_in_core",
        "low_traffic_condition",
        "loo_certificate",
        "blocking_size",
        "instances",
        "low_traffic_instances",
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


def _public_diagnostics(diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        key: diagnostics[key]
        for key in (
            "core_nonempty",
            "shapley_in_core",
            "category",
            "low_traffic_condition",
            "loo_certificate",
            "bondareva_gap_normalized",
            "shapley_max_excess_normalized",
            "blocking_size",
        )
    }


def _generic_savings(costs: np.ndarray, num_operators: int) -> np.ndarray:
    savings = np.zeros(1 << num_operators, dtype=float)
    for mask in range(1, 1 << num_operators):
        standalone = sum(
            costs[1 << player]
            for player in range(num_operators)
            if mask & (1 << player)
        )
        savings[mask] = standalone - costs[mask]
    return savings


def _generic_shapley(savings: np.ndarray, num_operators: int) -> np.ndarray:
    allocation = np.zeros(num_operators, dtype=float)
    denominator = factorial(num_operators)
    for player in range(num_operators):
        player_bit = 1 << player
        for mask in range(1 << num_operators):
            if mask & player_bit:
                continue
            size = int(mask).bit_count()
            weight = (
                factorial(size)
                * factorial(num_operators - size - 1)
                / denominator
            )
            allocation[player] += weight * (
                savings[mask | player_bit] - savings[mask]
            )
    return allocation


def _diagnostics_for_game(
    costs: np.ndarray,
    capacities: np.ndarray,
    demands: np.ndarray,
    num_operators: int,
) -> tuple[np.ndarray, dict[str, object]]:
    savings = (
        _savings_from_costs(costs)
        if num_operators == 4
        else _generic_savings(costs, num_operators)
    )
    grand_mask = (1 << num_operators) - 1
    grand_value = float(savings[grand_mask])
    scale = max(1.0, grand_value)
    tolerance = 1e-8 * scale
    shapley = (
        _shapley_value(savings)
        if num_operators == 4
        else _generic_shapley(savings, num_operators)
    )
    proper_masks = np.arange(1, grand_mask, dtype=int)
    excesses = np.asarray(
        [
            savings[mask]
            - sum(
                shapley[player]
                for player in range(num_operators)
                if mask & (1 << player)
            )
            for mask in proper_masks
        ],
        dtype=float,
    )
    blocking_index = int(np.argmax(excesses))
    blocking_mask = int(proper_masks[blocking_index])
    maximum_excess = float(excesses[blocking_index])
    shapley_in_core = maximum_excess <= tolerance
    if shapley_in_core:
        bondareva_gap = 0.0
        core_nonempty = True
    elif num_operators == 4:
        bondareva_gap = _bondareva_gap(savings)
        core_nonempty = bondareva_gap <= tolerance
    else:
        pair_sum = sum(
            savings[(1 << first) | (1 << second)]
            for first in range(num_operators)
            for second in range(first + 1, num_operators)
        )
        bondareva_gap = max(0.0, pair_sum / 2.0 - grand_value)
        core_nonempty = bondareva_gap <= tolerance

    loo_gap = float(
        costs[grand_mask]
        - sum(
            costs[grand_mask ^ (1 << player)]
            for player in range(num_operators)
        )
        / (num_operators - 1)
    )
    low_traffic = float(np.max(np.sum(demands, axis=0))) <= float(
        np.min(capacities)
    ) + 1e-9
    category = (
        "shapley_in_core"
        if shapley_in_core
        else "nonempty_shapley_out"
        if core_nonempty
        else "empty_core"
    )
    diagnostics = {
        "core_nonempty": int(core_nonempty),
        "shapley_in_core": int(shapley_in_core),
        "category": category,
        "low_traffic_condition": int(low_traffic),
        "loo_certificate": int(loo_gap > tolerance),
        "bondareva_gap_normalized": bondareva_gap
        / max(grand_value, 1e-12),
        "shapley_max_excess_normalized": maximum_excess
        / max(grand_value, 1e-12),
        "blocking_size": blocking_mask.bit_count(),
    }
    return savings, diagnostics


def _evaluate(
    calibration_dir: Path,
    num_sites: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    population, blueprints, protocol = load_protocol_inputs(calibration_dir)
    if not 1 <= num_sites <= blueprints.num_sites:
        raise ValueError(
            f"num_sites must lie between 1 and {blueprints.num_sites}"
        )

    records: list[dict[str, object]] = []
    heterogeneity: list[dict[str, object]] = []
    for site_index in range(num_sites):
        for configuration, levels in CONFIGURATIONS.items():
            volume, shape, equipment = levels
            for rate in CAMPAIGN_A_RATES:
                spec = ScenarioSpec(
                    "A", volume, shape, equipment, capacity_rate=rate
                )
                site = materialize_site(
                    blueprints, site_index, population, spec, protocol
                )
                for num_operators in NUM_OPERATORS:
                    traffic = site.traffic_gb[:num_operators]
                    fixed = site.p_fixed_w[:num_operators]
                    slopes = site.slope_w_per_gb[:num_operators]
                    capacities = site.peak_traffic_gb[:num_operators] / rate
                    cumulative = np.cumsum(np.sort(capacities)[::-1])
                    for day_index, day in enumerate(population.days):
                        day_demands = traffic[:, day_index, :]
                        base_costs = coalition_window_solutions(
                            capacities, fixed, slopes, day_demands
                        ).hourly_costs_by_hour_wh
                        theta_costs_by_hour: dict[float, np.ndarray] = {}
                        if num_operators == 4:
                            for theta in THETAS:
                                if theta == 1.0:
                                    theta_costs_by_hour[theta] = base_costs
                                    continue
                                theta_fixed = np.mean(fixed) + theta * (
                                    fixed - np.mean(fixed)
                                )
                                theta_slopes = np.mean(slopes) + theta * (
                                    slopes - np.mean(slopes)
                                )
                                theta_costs_by_hour[theta] = (
                                    coalition_window_solutions(
                                        capacities,
                                        theta_fixed,
                                        theta_slopes,
                                        day_demands,
                                    ).hourly_costs_by_hour_wh
                                )
                        for hour in range(24):
                            demands = day_demands[:, hour : hour + 1]
                            costs = base_costs[:, hour]
                            savings, diagnostics = _diagnostics_for_game(
                                costs, capacities, demands, num_operators
                            )
                            total_demand = float(np.sum(demands))
                            k_h = int(
                                np.searchsorted(
                                    cumulative,
                                    total_demand - 1e-12,
                                    side="left",
                                )
                                + 1
                            )
                            standalone = float(
                                sum(
                                    costs[1 << player]
                                    for player in range(num_operators)
                                )
                            )
                            base = {
                                "site_id": site.site_id,
                                "configuration": configuration,
                                "capacity_rate": float(rate),
                                "num_operators": num_operators,
                                "day": str(day),
                                "hour": hour,
                                "k_h": k_h,
                                "standalone_wh": standalone,
                                "savings_wh": float(savings[-1]),
                                "savings_pct": (
                                    100.0 * float(savings[-1]) / standalone
                                ),
                                **_public_diagnostics(diagnostics),
                            }
                            records.append(base)

                            if num_operators != 4:
                                continue
                            for theta in THETAS:
                                if theta == 1.0:
                                    shapley_in_core = int(
                                        diagnostics["shapley_in_core"]
                                    )
                                else:
                                    theta_costs = theta_costs_by_hour[theta][
                                        :, hour
                                    ]
                                    theta_savings = _savings_from_costs(
                                        theta_costs
                                    )
                                    shapley = _shapley_value(theta_savings)
                                    maximum_excess, _ = _max_excess(
                                        theta_savings, shapley
                                    )
                                    tolerance = 1e-8 * max(
                                        1.0,
                                        float(theta_savings[GRAND_MASK]),
                                    )
                                    shapley_in_core = int(
                                        maximum_excess <= tolerance
                                    )
                                heterogeneity.append(
                                    {
                                        "site_id": site.site_id,
                                        "configuration": configuration,
                                        "capacity_rate": float(rate),
                                        "day": str(day),
                                        "hour": hour,
                                        "theta": theta,
                                        "shapley_in_core": shapley_in_core,
                                        "low_traffic_condition": int(
                                            diagnostics["low_traffic_condition"]
                                        ),
                                    }
                                )
        print(
            f">> threshold mechanisms: {site_index + 1}/{num_sites} plans",
            flush=True,
        )
    return records, heterogeneity


def _threshold_summary(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for num_operators in NUM_OPERATORS:
        for k_h in range(1, num_operators + 1):
            selected = [
                row
                for row in records
                if int(row["num_operators"]) == num_operators
                and int(row["k_h"]) == k_h
            ]
            empty = [row for row in selected if not row["core_nonempty"]]
            rows.append(
                {
                    "num_operators": num_operators,
                    "k_h": k_h,
                    "instances": len(selected),
                    "empty_core_pct": 100.0 * len(empty) / len(selected),
                    "shapley_in_core_pct": 100.0
                    * sum(int(row["shapley_in_core"]) for row in selected)
                    / len(selected),
                }
            )
    return rows


def _operator_summary(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for num_operators in NUM_OPERATORS:
        selected = [
            row
            for row in records
            if int(row["num_operators"]) == num_operators
        ]
        low = [row for row in selected if row["low_traffic_condition"]]
        rows.append(
            {
                "num_operators": num_operators,
                "instances": len(selected),
                "low_traffic_instances": len(low),
                "low_traffic_condition_pct": 100.0 * len(low) / len(selected),
                "core_nonempty_pct": 100.0
                * np.mean([row["core_nonempty"] for row in selected]),
                "shapley_in_core_pct": 100.0
                * np.mean([row["shapley_in_core"] for row in selected]),
                "savings_kwh_mean": float(
                    np.mean([row["savings_wh"] for row in selected])
                )
                / 1_000.0,
            }
        )
    return rows


def _hourly_low_traffic(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for num_operators in NUM_OPERATORS:
        for configuration in CONFIGURATIONS:
            for hour in range(24):
                selected = [
                    row
                    for row in records
                    if int(row["num_operators"]) == num_operators
                    and str(row["configuration"]) == configuration
                    and int(row["hour"]) == hour
                ]
                rows.append(
                    {
                        "num_operators": num_operators,
                        "configuration": configuration,
                        "hour": hour,
                        "instances": len(selected),
                        "low_traffic_condition_pct": 100.0
                        * np.mean(
                            [row["low_traffic_condition"] for row in selected]
                        ),
                    }
                )
    return rows


def _heterogeneity_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for theta in THETAS:
        selected = [row for row in rows if float(row["theta"]) == theta]
        low = [row for row in selected if row["low_traffic_condition"]]
        summary.append(
            {
                "theta": theta,
                "instances": len(selected),
                "low_traffic_instances": len(low),
                "shapley_in_core_pct": 100.0
                * np.mean([row["shapley_in_core"] for row in selected]),
                "low_traffic_shapley_in_core_pct": 100.0
                * np.mean([row["shapley_in_core"] for row in low]),
            }
        )
    return summary


def _analysis(
    records: list[dict[str, object]],
    operator_summary: list[dict[str, object]],
    hourly: list[dict[str, object]],
    heterogeneity: list[dict[str, object]],
) -> dict[str, object]:
    low = [row for row in records if row["low_traffic_condition"]]
    empty_n4 = [
        row
        for row in records
        if int(row["num_operators"]) == 4 and not row["core_nonempty"]
    ]
    ideal_hours: dict[str, object] = {}
    for num_operators in NUM_OPERATORS:
        selected = [
            row
            for row in hourly
            if int(row["num_operators"]) == num_operators
            and str(row["configuration"]) == "reference"
        ]
        ordered = sorted(
            selected,
            key=lambda row: (
                -float(row["low_traffic_condition_pct"]),
                int(row["hour"]),
            ),
        )
        ideal_hours[str(num_operators)] = ordered[:5]

    first_half = [
        row for row in records if int(str(row["site_id"]).split("_")[1]) <= 10
    ]
    second_half = [
        row for row in records if int(str(row["site_id"]).split("_")[1]) > 10
    ]

    def category_fractions(selected: list[dict[str, object]]) -> dict[str, float]:
        return {
            category: float(np.mean([row["category"] == category for row in selected]))
            for category in (
                "shapley_in_core",
                "nonempty_shapley_out",
                "empty_core",
            )
        }

    half_one = category_fractions(first_half)
    half_two = category_fractions(second_half)
    homogeneous_low = next(
        row for row in heterogeneity if float(row["theta"]) == 0.0
    )
    if any(not row["core_nonempty"] for row in low):
        raise RuntimeError("the low-traffic sufficient condition was violated")
    if float(homogeneous_low["low_traffic_shapley_in_core_pct"]) < 100.0:
        raise RuntimeError("the homogeneous low-traffic corollary was violated")
    return {
        "sites": len({str(row["site_id"]) for row in records}),
        "instances_by_operator_count": {
            str(row["num_operators"]): int(row["instances"])
            for row in operator_summary
        },
        "operator_summary": operator_summary,
        "low_traffic_theorem_violations": sum(
            not row["core_nonempty"] for row in low
        ),
        "ideal_low_traffic_hours_reference": ideal_hours,
        "leave_one_out_detection_fraction_among_empty_n4": (
            float(np.mean([row["loo_certificate"] for row in empty_n4]))
            if empty_n4
            else float("nan")
        ),
        "blocking_size_distribution_when_shapley_out_n4": {
            str(size): float(
                np.mean(
                    [
                        int(row["blocking_size"]) == size
                        for row in records
                        if int(row["num_operators"]) == 4
                        and not row["shapley_in_core"]
                    ]
                )
            )
            for size in (2, 3)
        },
        "heterogeneity": heterogeneity,
        "half_sample_maximum_category_difference_pp": 100.0
        * max(abs(half_one[key] - half_two[key]) for key in half_one),
    }


def _figure(
    thresholds: list[dict[str, object]],
    hourly: list[dict[str, object]],
    heterogeneity: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.55))

    axis = axes[0]
    for num_operators, colour, label in (
        (3, "#2F8F5B", "trois opérateurs"),
        (4, "#2A6BB0", "quatre opérateurs"),
    ):
        selected = [
            row
            for row in hourly
            if int(row["num_operators"]) == num_operators
            and str(row["configuration"]) == "reference"
        ]
        axis.plot(
            [int(row["hour"]) for row in selected],
            [float(row["low_traffic_condition_pct"]) for row in selected],
            color=colour,
            linewidth=1.8,
            label=label,
        )
    axis.axvspan(0, 6.99, color="0.5", alpha=0.10)
    axis.set_xlim(0, 23)
    axis.set_ylim(0, 104)
    axis.set_xticks((0, 6, 12, 18, 23))
    axis.set_xlabel("Heure")
    axis.set_ylabel("Jeux vérifiant la condition (%)")
    axis.set_title("(a) Heures favorables")
    axis.legend(fontsize=7, loc="upper right")
    axis.grid(axis="y", alpha=0.22)

    axis = axes[1]
    x = np.arange(1, 5, dtype=float)
    width = 0.34
    for num_operators, offset, label, colour, hatch in (
        (3, -width / 2, "trois opérateurs", "#D97A7A", "//"),
        (4, width / 2, "quatre opérateurs", "#C0504D", None),
    ):
        lookup = {
            int(row["k_h"]): row
            for row in thresholds
            if int(row["num_operators"]) == num_operators
        }
        percentages = [
            float(lookup[k]["empty_core_pct"]) if k in lookup else 0.0
            for k in range(1, 5)
        ]
        axis.bar(
            x + offset,
            percentages,
            width,
            color=colour,
            hatch=hatch,
            edgecolor="0.25",
            linewidth=0.5,
            label=label,
        )
    axis.set_xticks(range(1, 5))
    axis.set_xlabel("Nombre minimal d'équipements nécessaires")
    axis.set_ylabel("Jeux à cœur vide (%)")
    axis.set_ylim(0, 104)
    axis.set_title("(b) Stabilité selon la charge")
    axis.legend(fontsize=7, loc="upper left")
    axis.grid(axis="y", alpha=0.22)

    axis = axes[2]
    theta = [100.0 * float(row["theta"]) for row in heterogeneity]
    axis.plot(
        theta,
        [float(row["shapley_in_core_pct"]) for row in heterogeneity],
        marker="o",
        color="#2A6BB0",
        label="tous les jeux",
    )
    axis.plot(
        theta,
        [
            float(row["low_traffic_shapley_in_core_pct"])
            for row in heterogeneity
        ],
        marker="s",
        color="#2F8F5B",
        label="jeux vérifiant la condition",
    )
    axis.set_xticks((0, 25, 50, 75, 100))
    axis.set_ylim(0, 104)
    axis.set_xlabel("Dispersion des coûts réintroduite (%)")
    axis.set_ylabel("Jeux avec Shapley stable (%)")
    axis.set_title("(c) Effet de la dispersion des coûts")
    axis.legend(fontsize=7, loc="lower left")
    axis.grid(axis="y", alpha=0.22)

    figure.tight_layout(w_pad=0.8)
    path = output_dir / "threshold_mechanisms.pdf"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--num-sites", type=int, default=DEFAULT_NUM_SITES)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cache_path = args.calibration_dir / "calibrated_population.npz"
    sites_path = args.calibration_dir / "site_blueprints.csv"
    protocol_path = args.calibration_dir / "protocol_parameters.json"
    for path in (cache_path, sites_path, protocol_path):
        if not path.is_file():
            parser.error(f"missing input: {path}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.results_dir / "threshold_instances.csv"
    heterogeneity_path = args.results_dir / "cost_heterogeneity_instances.csv"
    threshold_path = args.results_dir / "threshold_summary.csv"
    operator_path = args.results_dir / "operator_count_summary.csv"
    hourly_path = args.results_dir / "hourly_low_traffic.csv"
    heterogeneity_summary_path = args.results_dir / "cost_heterogeneity_summary.csv"
    analysis_path = args.results_dir / "analysis.json"
    manifest_path = args.results_dir / "manifest.json"
    expected = {
        "algorithm_version": ALGORITHM_VERSION,
        "calibrated_population": file_signature(cache_path),
        "site_blueprints": file_signature(sites_path),
        "protocol_parameters": file_signature(protocol_path),
        "num_sites": args.num_sites,
        "operator_counts": list(NUM_OPERATORS),
        "capacity_rates": list(CAMPAIGN_A_RATES),
        "configurations": {
            name: list(levels) for name, levels in CONFIGURATIONS.items()
        },
        "heterogeneity_levels": list(THETAS),
    }
    current = False
    if not args.rebuild and manifest_path.is_file():
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
                },
            ) and all(
                path.is_file()
                for path in (
                    records_path,
                    heterogeneity_path,
                    threshold_path,
                    operator_path,
                    hourly_path,
                    heterogeneity_summary_path,
                )
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            current = False

    if current:
        print(
            f">> Reusing threshold results: {records_path.resolve()}",
            flush=True,
        )
        records = _read_rows(records_path)
        thresholds = _read_rows(threshold_path)
        operator_summary = _read_rows(operator_path)
        hourly = _read_rows(hourly_path)
        heterogeneity_summary = _read_rows(heterogeneity_summary_path)
    else:
        records, heterogeneity_records = _evaluate(
            args.calibration_dir, args.num_sites
        )
        thresholds = _threshold_summary(records)
        operator_summary = _operator_summary(records)
        hourly = _hourly_low_traffic(records)
        heterogeneity_summary = _heterogeneity_summary(
            heterogeneity_records
        )
        _write_rows(records_path, records)
        _write_rows(heterogeneity_path, heterogeneity_records)
        _write_rows(threshold_path, thresholds)
        _write_rows(operator_path, operator_summary)
        _write_rows(hourly_path, hourly)
        _write_rows(heterogeneity_summary_path, heterogeneity_summary)

    analysis = _analysis(
        records, operator_summary, hourly, heterogeneity_summary
    )
    figures = _figure(
        thresholds, hourly, heterogeneity_summary, args.figures_dir
    )
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "inputs": expected,
        "outputs": {
            "instances": portable_path(records_path),
            "heterogeneity_instances": portable_path(heterogeneity_path),
            "threshold_summary": portable_path(threshold_path),
            "operator_summary": portable_path(operator_path),
            "hourly_low_traffic": portable_path(hourly_path),
            "heterogeneity_summary": portable_path(
                heterogeneity_summary_path
            ),
            "analysis": portable_path(analysis_path),
            "figures": [portable_path(path) for path in figures],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f">> {len(records)} threshold games completed", flush=True)
    if not args.quiet:
        print(json.dumps(analysis, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
