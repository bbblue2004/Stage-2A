"""Run the parameter-sensitivity study reported in Section 6.6."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from src.core.window_optimiser import coalition_window_solutions
from src.data_processing.data_loader import ROOT
from src.data_processing.instance_generator import (
    CENTRAL_EQUIPMENT,
    CENTRAL_RATE,
    DEFAULT_NUM_SITES,
    DEFAULT_SITE_SEED,
    ScenarioSpec,
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
DEFAULT_RESULTS_DIR = ROOT / "results" / "parameter_sensitivity"
TRAFFIC_MULTIPLIERS = (0.80, 1.00, 1.20)
COEFFICIENT_MULTIPLIERS = (0.80, 1.00, 1.20)
SLEEP_RATES = (0.00, 0.05, 0.10)
WINDOW_DURATIONS = (5, 7, 9)
BASELINE_RATE = CENTRAL_RATE
SITE_SEED = DEFAULT_SITE_SEED
ALGORITHM_VERSION = 6
FACTOR_LABELS = {
    "traffic_level": "Niveau de trafic",
    "fixed_cost": "Coût fixe $F_i$",
    "variable_cost": "Coût variable $\\gamma_i$",
    "sleep_power": "Puissance de veille",
    "window_position": "Position de $H$",
    "window_duration": "Durée de $H$",
}
CENTRAL_SETTINGS = {
    "traffic_level": "1.00",
    "fixed_cost": "1.00",
    "variable_cost": "1.00",
    "sleep_power": "0.00",
    "window_position": "00-07",
    "window_duration": "7",
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
    for mask in range(1, grand_mask + 1):
        members = np.asarray(
            [bool(mask & (1 << player)) for player in range(num_operators)]
        )
        coalition_capacity = float(np.sum(capacities[members]))
        coalition_demand = np.sum(demands[members], axis=0)
        tolerance = 1e-10 * max(1.0, coalition_capacity)
        if np.any(coalition_demand > coalition_capacity + tolerance):
            return {
                "savings_pct": float("nan"),
                "guardians_mean": float("nan"),
                "persistence_gap_pp": float("nan"),
                "core_empty": -1,
                "shapley_stable": -1,
                "feasible": 0,
            }
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
        "feasible": 1,
    }


def _load_main_metrics(
    operational_path: Path,
) -> dict[tuple[str, str, float], dict[str, float]]:
    """Load the three operational metrics used to cross-check the baseline.

    Stability is deliberately recomputed for the seven-hour game.  Section 6.4
    stores hourly games, whose stability status cannot be attached to a nightly
    game by retaining one arbitrary hour.
    """
    metrics: dict[tuple[str, str, float], dict[str, float | int]] = {}
    with operational_path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            if "campaign" in raw and not is_central_campaign_a(raw):
                continue
            key = (raw["site_id"], raw["day"], float(raw["capacity_rate"]))
            standalone = float(raw["standalone_energy_wh"])
            optimum = float(raw["hourly_optimal_energy_wh"])
            persistent = float(raw["persistent_energy_wh"])
            metrics[key] = {
                "savings_pct": 100.0 * (standalone - optimum) / standalone,
                "guardians_mean": float(
                    raw["hourly_optimal_guardians_mean"]
                ),
                "persistence_gap_pp": max(
                    0.0,
                    100.0 * (persistent - optimum) / standalone,
                ),
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
    main_metrics: dict[tuple[str, str, float], dict[str, float]],
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
        capacities = peaks / BASELINE_RATE
        central = _evaluate_instance(capacities, fixed, slopes, demands)
        expected = main_metrics[(site_id, day, BASELINE_RATE)]
        for metric in ("savings_pct", "guardians_mean", "persistence_gap_pp"):
            if not np.isclose(
                float(central[metric]), float(expected[metric]), rtol=1e-9, atol=1e-9
            ):
                raise RuntimeError(
                    f"baseline mismatch for {site_id}, {day}, {metric}"
                )

        for factor, setting in (
            ("traffic_level", "1.00"),
            ("fixed_cost", "1.00"),
            ("variable_cost", "1.00"),
            ("sleep_power", "0.00"),
            ("window_duration", "7"),
        ):
            rows.append(
                _result_row(
                    factor, setting, float(setting), site_id, day, central,
                )
            )

        for traffic_multiplier in (0.80, 1.20):
            metrics = _evaluate_instance(
                capacities, fixed, slopes, traffic_multiplier * demands
            )
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
                capacities, multiplier * fixed, slopes, demands
            )
            rows.append(
                _result_row(
                    "fixed_cost", f"{multiplier:.2f}", multiplier,
                    site_id, day, fixed_metrics,
                )
            )
            variable_metrics = _evaluate_instance(
                capacities, fixed, multiplier * slopes, demands
            )
            rows.append(
                _result_row(
                    "variable_cost", f"{multiplier:.2f}", multiplier,
                    site_id, day, variable_metrics,
                )
            )

        for sleep_rate in (0.05, 0.10):
            metrics = _evaluate_instance(
                capacities,
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
                capacities,
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
        capacities = peaks / BASELINE_RATE
        central = _evaluate_instance(
            capacities, fixed, slopes, traffic[:, start_day + 1, 0:7]
        )
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
        late_metrics = _evaluate_instance(capacities, fixed, slopes, overnight)
        rows.append(
            _result_row(
                "window_position", "22-05", 0.0, site_id,
                comparison_day, late_metrics,
            )
        )
        morning_metrics = _evaluate_instance(
            capacities,
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
        "num_operators", "core_empty", "shapley_stable", "feasible"
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


def _plan_setting_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Average repeated days before comparing plans or parameter settings."""
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["site_id"]), str(row["factor"]), str(row["setting"]))
        groups.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for (site_id, factor, setting), all_selected in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0], item[0][1], float(item[1][0]["setting_value"])
        ),
    ):
        selected = [
            row for row in all_selected if int(row.get("feasible", 1)) == 1
        ]
        if not selected:
            raise RuntimeError(
                f"no feasible day for {site_id}, {factor}, {setting}"
            )
        result.append(
            {
                "site_id": site_id,
                "factor": factor,
                "setting": setting,
                "setting_value": float(selected[0]["setting_value"]),
                "instances": len(all_selected),
                "feasible_instances": len(selected),
                "out_of_domain_pct": 100.0
                * (len(all_selected) - len(selected))
                / len(all_selected),
                "savings_pct": float(
                    np.mean([row["savings_pct"] for row in selected])
                ),
                "guardians_mean": float(
                    np.mean([row["guardians_mean"] for row in selected])
                ),
                "persistence_gap_pp": float(
                    np.mean([row["persistence_gap_pp"] for row in selected])
                ),
                "empty_core_pct": 100.0
                * float(np.mean([row["core_empty"] for row in selected])),
                "shapley_stable_pct": 100.0
                * float(np.mean([row["shapley_stable"] for row in selected])),
            }
        )
    return result


def _summaries(
    rows: list[dict[str, object]],
    plan_summaries: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if plan_summaries is None:
        plan_summaries = _plan_setting_summaries(rows)
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in plan_summaries:
        groups.setdefault((str(row["factor"]), str(row["setting"])), []).append(row)
    summaries: list[dict[str, object]] = []
    raw_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["factor"]), str(row["setting"]))
        raw_groups.setdefault(key, []).append(row)
    metrics = (
        "savings_pct",
        "guardians_mean",
        "persistence_gap_pp",
        "empty_core_pct",
        "shapley_stable_pct",
    )
    for (factor, setting), selected in sorted(
        groups.items(),
        key=lambda item: (item[0][0], float(item[1][0]["setting_value"])),
    ):
        summary: dict[str, object] = {
            "factor": factor,
            "setting": setting,
            "setting_value": float(selected[0]["setting_value"]),
            "plans": len(selected),
            "instances": len(raw_groups[(factor, setting)]),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            summary[f"{metric}_q75"] = float(np.quantile(values, 0.75))
        raw_selected = [
            row
            for row in raw_groups[(factor, setting)]
            if int(row.get("feasible", 1)) == 1
        ]
        all_raw_selected = raw_groups[(factor, setting)]
        summary["feasible_instances"] = len(raw_selected)
        summary["out_of_domain_pct_pooled"] = 100.0 * (
            len(all_raw_selected) - len(raw_selected)
        ) / len(all_raw_selected)
        summary["empty_core_pct_pooled"] = 100.0 * float(
            np.mean([row["core_empty"] for row in raw_selected])
        )
        summary["shapley_stable_pct_pooled"] = 100.0 * float(
            np.mean([row["shapley_stable"] for row in raw_selected])
        )
        summaries.append(summary)
    return summaries


def _factor_effects(
    summaries: list[dict[str, object]],
    plan_summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    effects: list[dict[str, object]] = []
    for factor in FACTOR_LABELS:
        selected = [row for row in summaries if row["factor"] == factor]
        effect: dict[str, object] = {
            "factor": factor,
            "label": FACTOR_LABELS[factor],
            "central_setting": CENTRAL_SETTINGS[factor],
        }
        for metric in (
            "savings_pct",
            "guardians_mean",
            "persistence_gap_pp",
            "empty_core_pct",
            "shapley_stable_pct",
        ):
            ranges = []
            for site_id in sorted(
                {str(row["site_id"]) for row in plan_summaries}
            ):
                values = np.asarray(
                    [
                        row[metric]
                        for row in plan_summaries
                        if row["site_id"] == site_id and row["factor"] == factor
                    ],
                    dtype=float,
                )
                if values.size != 3:
                    raise RuntimeError(
                        f"expected three settings for {site_id}, {factor}"
                    )
                ranges.append(float(np.max(values) - np.min(values)))
            range_values = np.asarray(ranges, dtype=float)
            effect[f"{metric}_complete_plans"] = len(ranges)
            effect[f"{metric}_range_median"] = float(np.median(range_values))
            effect[f"{metric}_range_q25"] = float(np.quantile(range_values, 0.25))
            effect[f"{metric}_range_q75"] = float(np.quantile(range_values, 0.75))
        effect["best_savings_setting"] = max(
            selected, key=lambda row: float(row["savings_pct_median"])
        )["setting"]
        effect["best_stability_setting"] = max(
            selected, key=lambda row: float(row["shapley_stable_pct_median"])
        )["setting"]
        effect["shapley_stable_pct_pooled_range"] = float(
            max(float(row["shapley_stable_pct_pooled"]) for row in selected)
            - min(float(row["shapley_stable_pct_pooled"]) for row in selected)
        )
        effects.append(effect)
    return effects


def _analysis(
    summaries: list[dict[str, object]], effects: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "factor_summaries": summaries,
        "factor_effects": effects,
        "energy_ranking": [
            row["factor"]
            for row in sorted(
                effects,
                key=lambda row: float(row["savings_pct_range_median"]),
                reverse=True,
            )
        ],
        "stability_ranking": [
            row["factor"]
            for row in sorted(
                effects,
                key=lambda row: float(row["shapley_stable_pct_range_median"]),
                reverse=True,
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--operational-dir", type=Path, default=DEFAULT_OPERATIONAL_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--num-sites", type=int, default=DEFAULT_NUM_SITES)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cache_path = args.calibration_dir / "calibrated_population.npz"
    sites_path = args.calibration_dir / "site_blueprints.csv"
    protocol_path = args.calibration_dir / "protocol_parameters.json"
    operational_path = args.operational_dir / "operational_instances.csv"
    for path in (cache_path, sites_path, protocol_path, operational_path):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.num_sites <= 0:
        parser.error("num-sites must be positive")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.results_dir / "sensitivity_instances.csv"
    plan_summary_path = args.results_dir / "sensitivity_plan_summary.csv"
    summary_path = args.results_dir / "sensitivity_summary.csv"
    analysis_path = args.results_dir / "analysis.json"
    manifest_path = args.results_dir / "manifest.json"
    expected = {
        "algorithm_version": ALGORITHM_VERSION,
        "calibrated_population": file_signature(cache_path),
        "site_blueprints": file_signature(sites_path),
        "protocol_parameters": file_signature(protocol_path),
        "operational_instances": file_signature(operational_path),
        "num_sites": args.num_sites,
        "traffic_multipliers": list(TRAFFIC_MULTIPLIERS),
        "coefficient_multipliers": list(COEFFICIENT_MULTIPLIERS),
        "sleep_rates": list(SLEEP_RATES),
        "window_durations": list(WINDOW_DURATIONS),
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
        main_metrics = _load_main_metrics(operational_path)
        central_spec = ScenarioSpec(
            "A",
            "moderate",
            "moderate",
            CENTRAL_EQUIPMENT,
            capacity_rate=BASELINE_RATE,
        )
        rows: list[dict[str, object]] = []
        for site_index in range(args.num_sites):
            site = materialize_site(
                blueprints, site_index, population, central_spec, protocol
            )
            rows.extend(_main_site_rows(site, population, main_metrics))
            if (site_index + 1) % 5 == 0:
                print(
                    f">> n=4 sensitivity: {site_index + 1}/{args.num_sites} sites",
                    flush=True,
                )
        _write_rows(rows_path, rows)

    plan_summaries = _plan_setting_summaries(rows)
    summaries = _summaries(rows, plan_summaries)
    effects = _factor_effects(summaries, plan_summaries)
    analysis = _analysis(summaries, effects)
    _write_rows(plan_summary_path, plan_summaries)
    _write_rows(summary_path, summaries)
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "inputs": expected,
        "outputs": {
            "instances": portable_path(rows_path),
            "plan_summary": portable_path(plan_summary_path),
            "summary": portable_path(summary_path),
            "analysis": portable_path(analysis_path),
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
