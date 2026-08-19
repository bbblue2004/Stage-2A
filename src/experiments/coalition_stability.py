"""Run the coalition-stability experiment reported in Section 6.4."""

from __future__ import annotations

import argparse
import csv
import json
from math import factorial
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import linprog

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.game import (
    bondareva_shapley_test,
    least_core_allocation,
    nucleolus_allocation,
)
from src.core.time_window import inclusive_hour_window
from src.core.window_optimiser import (
    hourly_coalition_costs,
    optimal_hourly_policy,
)
from src.data_processing.data_loader import ROOT
from src.data_processing.instance_generator import (
    CAMPAIGN_A_RATES,
    DEFAULT_NUM_SITES,
    ScenarioSpec,
    capacities_for_site,
    iter_materialized_sites,
    materialize_site,
)
from src.experiments.common import (
    file_signature,
    four_player_bondareva_gap,
    inputs_match,
    portable_path,
)
from src.experiments.protocol_io import (
    is_central_campaign_a,
    load_protocol_inputs,
    scenarios_for_grid,
)


DEFAULT_CALIBRATION_DIR = ROOT / "results" / "power_calibration"
DEFAULT_OPERATIONAL_DIR = ROOT / "results" / "operational_efficiency"
DEFAULT_RESULTS_DIR = ROOT / "results" / "coalition_stability"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "coalition_stability"
CAPACITY_RATES = CAMPAIGN_A_RATES
BOOTSTRAP_SEED = 20_260_818
ALGORITHM_VERSION = 6
NUM_PLAYERS = 4
NUM_MASKS = 1 << NUM_PLAYERS
GRAND_MASK = NUM_MASKS - 1
PROPER_MASKS = np.arange(1, GRAND_MASK, dtype=int)
CATEGORIES = (
    "empty_core",
    "nonempty_shapley_out",
    "shapley_in_core",
)
CATEGORY_LABELS = {
    "empty_core": "Cœur vide",
    "nonempty_shapley_out": "Cœur non vide, Shapley hors du cœur",
    "shapley_in_core": "Shapley dans le cœur",
}
CATEGORY_COLORS = {
    "empty_core": "#C94C4C",
    "nonempty_shapley_out": "#E2A23A",
    "shapley_in_core": "#3677B8",
}


def _members(mask: int) -> tuple[int, ...]:
    return tuple(player for player in range(NUM_PLAYERS) if mask & (1 << player))


MASK_MEMBERS = tuple(_members(mask) for mask in range(NUM_MASKS))
MEMBERSHIP = np.asarray(
    [
        [float(player in MASK_MEMBERS[mask]) for player in range(NUM_PLAYERS)]
        for mask in range(NUM_MASKS)
    ]
)
LEAST_CORE_A_UB = np.column_stack(
    (-MEMBERSHIP[PROPER_MASKS], -np.ones(PROPER_MASKS.size))
)
LEAST_CORE_A_EQ = np.asarray([[1.0] * NUM_PLAYERS + [0.0]])


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, object]]:
    string_columns = {
        "site_id",
        "day",
        "scenario",
        "campaign",
        "volume_level",
        "shape_level",
        "equipment_level",
        "category",
        "guardian_target",
    }
    integer_columns = {
        "core_nonempty",
        "shapley_in_core",
        "convex",
        "low_traffic_condition",
        "loo_certificate",
        "blocking_mask",
        "blocking_size",
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


def _load_operational_rows(path: Path) -> dict[tuple[str, str, str], tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as file:
        return {
            (raw["site_id"], raw["scenario"], raw["day"]): (
                float(raw["standalone_energy_wh"]),
                float(raw["hourly_optimal_energy_wh"]),
            )
            for raw in csv.DictReader(file)
        }


def _savings_from_costs(costs: np.ndarray) -> np.ndarray:
    if costs.shape != (NUM_MASKS,):
        raise ValueError(f"expected {NUM_MASKS} coalition costs")
    savings = np.zeros(NUM_MASKS, dtype=float)
    for mask in range(1, NUM_MASKS):
        standalone = sum(costs[1 << player] for player in MASK_MEMBERS[mask])
        value = standalone - costs[mask]
        tolerance = 1e-10 * max(1.0, standalone)
        if value < -tolerance:
            raise RuntimeError("coalition savings violate superadditivity")
        savings[mask] = 0.0 if abs(value) <= tolerance else value
    return savings


def _shapley_value(savings: np.ndarray) -> np.ndarray:
    allocation = np.zeros(NUM_PLAYERS, dtype=float)
    denominator = factorial(NUM_PLAYERS)
    for player in range(NUM_PLAYERS):
        player_bit = 1 << player
        for mask in range(NUM_MASKS):
            if mask & player_bit:
                continue
            size = len(MASK_MEMBERS[mask])
            weight = factorial(size) * factorial(NUM_PLAYERS - size - 1) / denominator
            allocation[player] += weight * (
                savings[mask | player_bit] - savings[mask]
            )
    return allocation


def _max_excess(savings: np.ndarray, allocation: np.ndarray) -> tuple[float, int]:
    allocated = MEMBERSHIP[PROPER_MASKS] @ allocation
    excesses = savings[PROPER_MASKS] - allocated
    index = int(np.argmax(excesses))
    return float(excesses[index]), int(PROPER_MASKS[index])


def _convexity_violation(savings: np.ndarray) -> float:
    maximum = 0.0
    for player in range(NUM_PLAYERS):
        player_bit = 1 << player
        for smaller in range(NUM_MASKS):
            if smaller & player_bit:
                continue
            smaller_marginal = savings[smaller | player_bit] - savings[smaller]
            for larger in range(NUM_MASKS):
                if larger & player_bit or smaller & ~larger:
                    continue
                larger_marginal = savings[larger | player_bit] - savings[larger]
                maximum = max(maximum, smaller_marginal - larger_marginal)
    return maximum


def _bondareva_gap(savings: np.ndarray) -> float:
    return four_player_bondareva_gap(savings)


def _least_core(savings: np.ndarray) -> tuple[float, np.ndarray]:
    result = linprog(
        np.asarray([0.0] * NUM_PLAYERS + [1.0]),
        A_ub=LEAST_CORE_A_UB,
        b_ub=-savings[PROPER_MASKS],
        A_eq=LEAST_CORE_A_EQ,
        b_eq=[savings[GRAND_MASK]],
        bounds=[(0.0, None)] * NUM_PLAYERS + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"least-core LP failed: {result.message}")
    return float(result.x[-1]), np.asarray(result.x[:-1], dtype=float)


def _as_game_map(values: np.ndarray) -> dict[tuple[int, ...], float]:
    return {MASK_MEMBERS[mask]: float(values[mask]) for mask in range(NUM_MASKS)}


def _diagnose(
    costs: np.ndarray,
    savings: np.ndarray,
    capacities: np.ndarray,
    demands: np.ndarray,
) -> dict[str, object]:
    grand_value = float(savings[GRAND_MASK])
    scale = max(1.0, grand_value)
    tolerance = 1e-8 * scale
    shapley = _shapley_value(savings)
    shapley_excess, blocking_mask = _max_excess(savings, shapley)
    shapley_in_core = shapley_excess <= tolerance

    if shapley_in_core:
        core_nonempty = True
        bondareva_gap = 0.0
    else:
        bondareva_gap = _bondareva_gap(savings)
        core_nonempty = bondareva_gap <= tolerance

    least_core_epsilon = float("nan")
    if not core_nonempty:
        least_core_epsilon, _ = _least_core(savings)
        if least_core_epsilon <= tolerance:
            raise RuntimeError("least-core and Bondareva diagnostics disagree")

    if shapley_in_core:
        category = "shapley_in_core"
    elif core_nonempty:
        category = "nonempty_shapley_out"
    else:
        category = "empty_core"

    loo_gap = float(
        costs[GRAND_MASK]
        - sum(costs[GRAND_MASK ^ (1 << player)] for player in range(NUM_PLAYERS))
        / (NUM_PLAYERS - 1)
    )
    loo_certificate = loo_gap > tolerance
    if loo_certificate and core_nonempty:
        raise RuntimeError("leave-one-out certificate contradicts core feasibility")

    convexity_violation = _convexity_violation(savings)
    low_traffic = float(np.max(np.sum(demands, axis=0))) <= float(
        np.min(capacities)
    ) + 1e-9
    if low_traffic and not core_nonempty:
        raise RuntimeError("sufficient low-traffic condition contradicts core feasibility")

    normalizer = max(grand_value, 1e-12)
    return {
        "core_nonempty": int(core_nonempty),
        "shapley_in_core": int(shapley_in_core),
        "category": category,
        "convex": int(convexity_violation <= tolerance),
        "convexity_violation_wh": convexity_violation,
        "low_traffic_condition": int(low_traffic),
        "bondareva_gap_wh": bondareva_gap,
        "bondareva_gap_normalized": bondareva_gap / normalizer,
        "least_core_epsilon_wh": least_core_epsilon,
        "least_core_epsilon_normalized": least_core_epsilon / normalizer,
        "loo_gap_wh": loo_gap,
        "loo_gap_normalized": loo_gap / normalizer,
        "loo_certificate": int(loo_certificate),
        "shapley_max_excess_wh": shapley_excess,
        "shapley_max_excess_normalized": shapley_excess / normalizer,
        "blocking_mask": blocking_mask,
        "blocking_size": len(MASK_MEMBERS[blocking_mask]),
    }


def _validate_with_independent_solver(
    savings: np.ndarray,
    diagnostics: dict[str, object],
) -> tuple[float, float, bool]:
    game = _as_game_map(savings)
    players = list(range(NUM_PLAYERS))
    bondareva = bondareva_shapley_test(players, game)
    if bondareva.status != "Optimal":
        raise RuntimeError(f"independent balancedness LP failed: {bondareva.status}")
    bondareva_difference = abs(
        max(0.0, bondareva.gap) - float(diagnostics["bondareva_gap_wh"])
    )
    least_core_difference = 0.0
    if not diagnostics["core_nonempty"]:
        independent = least_core_allocation(players, game)
        if independent.status != "Optimal":
            raise RuntimeError(f"independent least-core LP failed: {independent.status}")
        least_core_difference = abs(
            independent.epsilon - float(diagnostics["least_core_epsilon_wh"])
        )
    agrees = bondareva.core_nonempty == bool(diagnostics["core_nonempty"])
    return bondareva_difference, least_core_difference, agrees


def _spec_from_row(row: dict[str, object]) -> ScenarioSpec:
    campaign = str(row["campaign"])
    volume = str(row["volume_level"])
    shape = str(row["shape_level"])
    equipment = str(row["equipment_level"])
    if campaign == "A":
        return ScenarioSpec(
            "A",
            volume,
            shape,
            equipment,
            capacity_rate=float(row["capacity_rate"]),
        )
    return ScenarioSpec(
        "B",
        volume,
        shape,
        equipment,
        guardian_target=int(row["guardian_target"]),
        window_peak_rate=float(row["capacity_rate"]),
    )


def _site_index(site_id: str) -> int:
    return int(str(site_id).split("_")[1]) - 1


def _run(
    calibration_dir: Path,
    operational_path: Path,
    num_sites: int,
    hours: tuple[int, ...],
    validation_instances: int,
    grid: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    population, blueprints, protocol = load_protocol_inputs(calibration_dir)
    operational = _load_operational_rows(operational_path)
    if not 1 <= num_sites <= blueprints.num_sites:
        raise ValueError(f"num_sites must be between 1 and {blueprints.num_sites}")
    scenarios = scenarios_for_grid(grid)

    rows: list[dict[str, object]] = []
    max_operational_difference = 0.0
    max_bondareva_difference = 0.0
    max_least_core_difference = 0.0
    independent_disagreements = 0
    validated = 0
    evaluated = 0

    for site in iter_materialized_sites(
        blueprints, population, scenarios, protocol, num_sites=num_sites
    ):
        spec = site.scenario
        for day_index, day in enumerate(population.days):
            demands = site.traffic_gb[:, day_index][:, hours]
            capacities = capacities_for_site(site, demands)
            costs = hourly_coalition_costs(
                capacities, site.p_fixed_w, site.slope_w_per_gb, demands
            )
            savings = _savings_from_costs(costs)
            diagnostics = _diagnose(costs, savings, capacities, demands)
            standalone = float(
                sum(costs[1 << player] for player in range(NUM_PLAYERS))
            )
            grand_cost = float(costs[GRAND_MASK])
            key = (site.site_id, spec.key, str(day))
            expected_standalone, expected_grand = operational[key]
            difference = max(
                abs(standalone - expected_standalone),
                abs(grand_cost - expected_grand),
            )
            max_operational_difference = max(max_operational_difference, difference)
            if difference > 1e-7 * max(1.0, expected_standalone):
                raise RuntimeError(f"operational cross-check failed for {key}")

            if validated < validation_instances:
                bond_diff, least_diff, agrees = _validate_with_independent_solver(
                    savings, diagnostics
                )
                max_bondareva_difference = max(max_bondareva_difference, bond_diff)
                max_least_core_difference = max(max_least_core_difference, least_diff)
                independent_disagreements += int(not agrees)
                validated += 1

            rows.append(
                {
                    "site_id": site.site_id,
                    "scenario": spec.key,
                    "campaign": spec.campaign,
                    "volume_level": spec.volume_level,
                    "shape_level": spec.shape_level,
                    "equipment_level": spec.equipment_level,
                    "day": str(day),
                    "capacity_rate": (
                        spec.capacity_rate
                        if spec.campaign == "A"
                        else spec.window_peak_rate
                    ),
                    "guardian_target": (
                        spec.guardian_target if spec.campaign == "B" else ""
                    ),
                    "standalone_energy_wh": standalone,
                    "grand_cost_wh": grand_cost,
                    "grand_savings_wh": float(savings[GRAND_MASK]),
                    "savings_pct": 100.0 * float(savings[GRAND_MASK]) / standalone,
                    **diagnostics,
                }
            )
        evaluated += 1
        if evaluated % 100 == 0:
            print(
                f">> {evaluated}/{num_sites * len(scenarios)} site-scenarios evaluated",
                flush=True,
            )

    validation = {
        "instances_cross_checked_with_pulp": validated,
        "independent_classification_disagreements": independent_disagreements,
        "max_operational_cost_difference_wh": max_operational_difference,
        "max_bondareva_gap_difference_wh": max_bondareva_difference,
        "max_least_core_difference_wh": max_least_core_difference,
    }
    return rows, validation


def _cluster_bootstrap_intervals(
    rows: list[dict[str, object]],
    rng: np.random.Generator,
    replications: int,
) -> dict[str, tuple[float, float]]:
    sites = sorted({str(row["site_id"]) for row in rows})
    site_index = {site: index for index, site in enumerate(sites)}
    counts = np.zeros((len(sites), len(CATEGORIES)), dtype=int)
    totals = np.zeros(len(sites), dtype=int)
    category_index = {category: index for index, category in enumerate(CATEGORIES)}
    for row in rows:
        index = site_index[str(row["site_id"])]
        counts[index, category_index[str(row["category"])]] += 1
        totals[index] += 1

    samples = np.empty((replications, len(CATEGORIES)), dtype=float)
    for start in range(0, replications, 200):
        stop = min(replications, start + 200)
        draws = rng.integers(0, len(sites), size=(stop - start, len(sites)))
        sampled_counts = np.sum(counts[draws], axis=1)
        sampled_totals = np.sum(totals[draws], axis=1)
        samples[start:stop] = sampled_counts / sampled_totals[:, None]
    lower, upper = np.quantile(samples, (0.025, 0.975), axis=0)
    return {
        category: (float(lower[index]), float(upper[index]))
        for index, category in enumerate(CATEGORIES)
    }


def _summary_rows(
    rows: list[dict[str, object]], replications: int
) -> list[dict[str, object]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries: list[dict[str, object]] = []
    for rate in (*CAPACITY_RATES, None):
        selected = [
            row for row in rows if rate is None or row["capacity_rate"] == rate
        ]
        intervals = _cluster_bootstrap_intervals(selected, rng, replications)
        summary: dict[str, object] = {
            "capacity_rate": "all" if rate is None else f"{rate:.2f}",
            "instances": len(selected),
        }
        for category in CATEGORIES:
            count = sum(row["category"] == category for row in selected)
            lower, upper = intervals[category]
            summary[f"{category}_count"] = count
            summary[f"{category}_pct"] = 100.0 * count / len(selected)
            summary[f"{category}_ci_low_pct"] = 100.0 * lower
            summary[f"{category}_ci_high_pct"] = 100.0 * upper
        summaries.append(summary)
    return summaries


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    q05, median, q95 = np.quantile(values, (0.05, 0.50, 0.95))
    return {
        "mean": float(np.mean(values)),
        "q05": float(q05),
        "median": float(median),
        "q95": float(q95),
    }


def _finite_quantiles(rows: list[dict[str, object]], column: str) -> dict[str, float]:
    values = np.asarray([float(row[column]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return _quantiles(values) if values.size else {}


def _analysis(
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    validation: dict[str, object],
) -> dict[str, object]:
    empty = [row for row in rows if row["category"] == "empty_core"]
    shapley_out_nonempty = [
        row for row in rows if row["category"] == "nonempty_shapley_out"
    ]
    shapley_out = [row for row in rows if not row["shapley_in_core"]]
    low_traffic = [row for row in rows if row["low_traffic_condition"]]
    n_sites = len({str(row["site_id"]) for row in rows})
    split = max(1, n_sites // 2)
    first_half = [
        row for row in rows if _site_index(str(row["site_id"])) < split
    ]
    savings = np.asarray([float(row["savings_pct"]) for row in rows])
    top_decile_threshold = float(np.quantile(savings, 0.90))
    top_decile = [
        row for row in rows if float(row["savings_pct"]) >= top_decile_threshold
    ]

    def fractions(selected: list[dict[str, object]]) -> dict[str, float]:
        if not selected:
            return {category: float("nan") for category in CATEGORIES}
        return {
            category: float(
                np.mean([row["category"] == category for row in selected])
            )
            for category in CATEGORIES
        }

    full_fractions = fractions(rows)
    half_fractions = fractions(first_half)
    return {
        "instances": len(rows),
        "sites": len({row["site_id"] for row in rows}),
        "category_summary": {
            str(summary["capacity_rate"]): summary for summary in summaries
        },
        "energy_savings_pct_by_category": {
            category: _quantiles(
                np.asarray(
                    [
                        float(row["savings_pct"])
                        for row in rows
                        if row["category"] == category
                    ]
                )
            )
            for category in CATEGORIES
        },
        "diagnostics": {
            "convex_fraction": float(np.mean([row["convex"] for row in rows])),
            "low_traffic_condition_fraction": len(low_traffic) / len(rows),
            "low_traffic_condition_all_core_nonempty": all(
                row["core_nonempty"] for row in low_traffic
            ),
            "low_traffic_category_fractions": fractions(low_traffic),
            "top_savings_decile_threshold_pct": top_decile_threshold,
            "top_savings_decile_category_fractions": fractions(top_decile),
            "loo_detection_fraction_among_empty_cores": (
                float(np.mean([row["loo_certificate"] for row in empty]))
                if empty
                else float("nan")
            ),
            "loo_false_positives": sum(
                row["loo_certificate"] and row["core_nonempty"] for row in rows
            ),
            "empty_core_bondareva_gap_normalized": _finite_quantiles(
                empty, "bondareva_gap_normalized"
            ),
            "empty_core_least_core_epsilon_normalized": _finite_quantiles(
                empty, "least_core_epsilon_normalized"
            ),
            "nonempty_shapley_out_excess_normalized": _finite_quantiles(
                shapley_out_nonempty, "shapley_max_excess_normalized"
            ),
            "blocking_size_distribution_when_shapley_out": {
                str(size): float(
                    np.mean([row["blocking_size"] == size for row in shapley_out])
                )
                for size in range(1, NUM_PLAYERS)
            },
        },
        "convergence": {
            "category_fractions_first_half_sites": half_fractions,
            "category_fractions_all_sites": full_fractions,
            "maximum_absolute_difference_pp": 100.0
            * max(
                abs(half_fractions[category] - full_fractions[category])
                for category in CATEGORIES
            ),
        },
        "validation": validation,
    }


def _representative_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Select one deterministic multivariate medoid in each category."""
    selected_rows: list[dict[str, object]] = []
    for category in CATEGORIES:
        candidates = [
            row for row in rows if str(row["category"]) == category
        ]
        if not candidates:
            continue
        features = np.asarray(
            [
                [
                    float(row["savings_pct"]),
                    (
                        float(row["least_core_epsilon_normalized"])
                        if np.isfinite(
                            float(row["least_core_epsilon_normalized"])
                        )
                        else 0.0
                    ),
                    max(
                        0.0,
                        float(row["shapley_max_excess_normalized"]),
                    ),
                ]
                for row in candidates
            ],
            dtype=float,
        )
        center = np.median(features, axis=0)
        q25, q75 = np.quantile(features, (0.25, 0.75), axis=0)
        scale = q75 - q25
        scale[scale <= 1e-12] = 1.0
        distances = np.sum(np.abs((features - center) / scale), axis=1)
        order = sorted(
            range(len(candidates)),
            key=lambda index: (
                float(distances[index]),
                str(candidates[index]["site_id"]),
                str(candidates[index]["day"]),
                float(candidates[index]["capacity_rate"]),
            ),
        )
        representative = dict(candidates[order[0]])
        representative["medoid_distance"] = float(distances[order[0]])
        selected_rows.append(representative)
    return selected_rows


def _case_outputs(
    rows: list[dict[str, object]],
    calibration_dir: Path,
    hours: tuple[int, ...],
    results_dir: Path,
    figures_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Recompute Shapley, nucleolus and transfers for representative cases."""
    representatives = _representative_rows(rows)
    if not representatives:
        return ([], [])
    population, blueprints, protocol = load_protocol_inputs(calibration_dir)
    day_lookup = {
        str(day): index for index, day in enumerate(population.days)
    }
    players = list(range(NUM_PLAYERS))
    case_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    blocking_rows: list[dict[str, object]] = []

    for case_index, representative in enumerate(representatives, start=1):
        site_id = str(representative["site_id"])
        day = str(representative["day"])
        spec = _spec_from_row(representative)
        site = materialize_site(
            blueprints, _site_index(site_id), population, spec, protocol
        )
        demands = site.traffic_gb[:, day_lookup[day]][:, hours]
        capacities = capacities_for_site(site, demands)
        fixed = site.p_fixed_w
        slopes = site.slope_w_per_gb
        rate = float(representative["capacity_rate"])
        costs = hourly_coalition_costs(
            capacities, fixed, slopes, demands
        )
        savings = _savings_from_costs(costs)
        diagnostics = _diagnose(costs, savings, capacities, demands)
        if diagnostics["category"] != representative["category"]:
            raise RuntimeError("representative case did not reproduce its category")
        hourly = optimal_hourly_policy(
            capacities, fixed, slopes, demands
        )
        physical = np.asarray(
            [
                float(
                    np.sum(
                        (
                            hourly.guardian_masks & (1 << player)
                            != 0
                        )
                        * fixed[player]
                    )
                    + slopes[player]
                    * np.sum(hourly.allocations_gb[player])
                )
                for player in players
            ],
            dtype=float,
        )
        standalone = np.asarray(
            [costs[1 << player] for player in players], dtype=float
        )
        shapley = _shapley_value(savings)
        game_scale = max(1.0, float(savings[GRAND_MASK]))
        nucleolus_result = nucleolus_allocation(
            players, _as_game_map(savings / game_scale)
        )
        if nucleolus_result.status != "Optimal":
            raise RuntimeError(
                "representative nucleolus failed: "
                f"{nucleolus_result.status}"
            )
        nucleolus = np.asarray(
            [nucleolus_result.allocation[player] for player in players],
            dtype=float,
        ) * game_scale
        nucleolus *= float(savings[GRAND_MASK]) / float(np.sum(nucleolus))
        tolerance = 1e-8 * max(1.0, float(savings[GRAND_MASK]))
        if abs(float(np.sum(physical)) - float(costs[GRAND_MASK])) > tolerance:
            raise RuntimeError("hourly physical costs do not recover grand cost")
        case_id = f"case_{case_index}_{representative['category']}"
        case_rows.append(
            {
                "case_id": case_id,
                "category": representative["category"],
                "site_id": site_id,
                "day": day,
                "capacity_rate": rate,
                "scenario": spec.key,
                "savings_pct": representative["savings_pct"],
                "least_core_epsilon_normalized": (
                    representative["least_core_epsilon_normalized"]
                ),
                "shapley_max_excess_normalized": (
                    representative["shapley_max_excess_normalized"]
                ),
                "medoid_distance": representative["medoid_distance"],
                "hourly_guardian_masks": "|".join(
                    str(int(mask)) for mask in hourly.guardian_masks
                ),
            }
        )
        for rule, allocation in (
            ("shapley", shapley),
            ("nucleolus", nucleolus),
        ):
            net = standalone - allocation
            transfers = physical - net
            if (
                abs(float(np.sum(allocation)) - float(savings[GRAND_MASK]))
                > tolerance
                or abs(float(np.sum(transfers))) > tolerance
            ):
                raise RuntimeError(
                    f"representative {rule} allocation violates efficiency"
                )
            excesses = savings[PROPER_MASKS] - (
                MEMBERSHIP[PROPER_MASKS] @ allocation
            )
            for player in players:
                allocation_rows.append(
                    {
                        "case_id": case_id,
                        "category": representative["category"],
                        "rule": rule,
                        "operator": player + 1,
                        "standalone_cost_wh": standalone[player],
                        "physical_cost_wh": physical[player],
                        "allocated_savings_wh": allocation[player],
                        "net_cost_wh": net[player],
                        "transfer_received_wh": transfers[player],
                    }
                )
            for mask, excess in zip(
                PROPER_MASKS, excesses, strict=True
            ):
                if excess > tolerance:
                    blocking_rows.append(
                        {
                            "case_id": case_id,
                            "category": representative["category"],
                            "rule": rule,
                            "coalition_mask": int(mask),
                            "coalition": "|".join(
                                str(player + 1)
                                for player in MASK_MEMBERS[int(mask)]
                            ),
                            "excess_wh": float(excess),
                        }
                    )

    results_dir.mkdir(parents=True, exist_ok=True)
    case_path = results_dir / "representative_cases.csv"
    allocation_path = results_dir / "representative_allocations.csv"
    blocking_path = results_dir / "representative_blocking_coalitions.csv"
    _write_rows(case_path, case_rows)
    _write_rows(allocation_path, allocation_rows)
    if blocking_rows:
        _write_rows(blocking_path, blocking_rows)
    else:
        with blocking_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "case_id",
                    "category",
                    "rule",
                    "coalition_mask",
                    "coalition",
                    "excess_wh",
                ],
            )
            writer.writeheader()

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2,
        len(case_rows),
        figsize=(10.0, 5.6),
        squeeze=False,
    )
    x = np.arange(NUM_PLAYERS)
    width = 0.36
    for column, case in enumerate(case_rows):
        case_allocations = [
            row
            for row in allocation_rows
            if row["case_id"] == case["case_id"]
        ]
        for rule_index, (rule, label, color) in enumerate(
            (
                ("shapley", "Shapley", "#3677B8"),
                ("nucleolus", "Nucléole", "#D98E04"),
            )
        ):
            selected = [
                row for row in case_allocations if row["rule"] == rule
            ]
            positions = x + (rule_index - 0.5) * width
            axes[0, column].bar(
                positions,
                [row["allocated_savings_wh"] for row in selected],
                width,
                color=color,
                label=label,
            )
            axes[1, column].bar(
                positions,
                [row["transfer_received_wh"] for row in selected],
                width,
                color=color,
                label=label,
            )
        axes[0, column].set_title(
            CATEGORY_LABELS[str(case["category"])], fontsize=9
        )
        axes[0, column].set_xticks(x, [f"O{i + 1}" for i in x])
        axes[1, column].set_xticks(x, [f"O{i + 1}" for i in x])
        axes[0, column].grid(axis="y", alpha=0.22)
        axes[1, column].grid(axis="y", alpha=0.22)
        axes[1, column].axhline(0.0, color="0.25", linewidth=0.8)
    axes[0, 0].set_ylabel("Économie attribuée (Wh)")
    axes[1, 0].set_ylabel("Transfert reçu (Wh)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure_paths = [
        figures_dir / "representative_allocations.pdf",
        figures_dir / "representative_allocations.png",
    ]
    figure.savefig(figure_paths[0], bbox_inches="tight")
    figure.savefig(figure_paths[1], dpi=220, bbox_inches="tight")
    plt.close(figure)
    return (
        [case_path, allocation_path, blocking_path],
        figure_paths,
    )


def _figure(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))

    x = np.arange(len(CAPACITY_RATES))
    bottom = np.zeros(len(CAPACITY_RATES))
    for category in CATEGORIES:
        fractions = np.asarray(
            [
                np.mean(
                    [
                        row["category"] == category
                        for row in rows
                        if row["capacity_rate"] == rate
                    ]
                )
                for rate in CAPACITY_RATES
            ]
        )
        axes[0].bar(
            x,
            100.0 * fractions,
            bottom=100.0 * bottom,
            color=CATEGORY_COLORS[category],
            label=CATEGORY_LABELS[category],
            width=0.68,
        )
        bottom += fractions
    axes[0].set_xticks(x, [f"{rate:.2f}".replace(".", ",") for rate in CAPACITY_RATES])
    axes[0].set_xlabel("Taux de charge maximal $r$")
    axes[0].set_ylabel("Part des instances (%)")
    axes[0].set_ylim(0.0, 100.0)
    axes[0].set_title("(a) Catégories de stabilité")
    axes[0].grid(axis="y", alpha=0.22)

    values = [
        np.asarray(
            [float(row["savings_pct"]) for row in rows if row["category"] == category]
        )
        for category in CATEGORIES
    ]
    box = axes[1].boxplot(
        values,
        tick_labels=("Cœur\nvide", "Shapley\nhors", "Shapley\ndans"),
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
    )
    for patch, category in zip(box["boxes"], CATEGORIES, strict=True):
        patch.set_facecolor(CATEGORY_COLORS[category])
        patch.set_alpha(0.82)
    axes[1].set_ylabel("Énergie évitée (%)")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_title("(b) Efficacité et stabilité")
    axes[1].grid(axis="y", alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    paths = [
        output_dir / "coalition_stability.pdf",
        output_dir / "coalition_stability.png",
    ]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=220, bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--operational-dir", type=Path, default=DEFAULT_OPERATIONAL_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--num-sites", type=int, default=DEFAULT_NUM_SITES)
    parser.add_argument(
        "--grid",
        choices=("central", "full", "thresholds"),
        default="full",
    )
    parser.add_argument("--hours", nargs=2, type=int, default=(0, 6), metavar=("START", "END"))
    parser.add_argument("--bootstrap-replications", type=int, default=2_000)
    parser.add_argument("--validation-instances", type=int, default=20)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    hours = inclusive_hour_window(*args.hours)
    cache_path = args.calibration_dir / "calibrated_population.npz"
    sites_path = args.calibration_dir / "site_blueprints.csv"
    protocol_path = args.calibration_dir / "protocol_parameters.json"
    operational_path = args.operational_dir / "operational_instances.csv"
    for path in (cache_path, sites_path, protocol_path, operational_path):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.bootstrap_replications <= 0 or args.validation_instances < 0:
        parser.error("replication count must be positive and validation count non-negative")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.results_dir / "stability_instances.csv"
    summary_path = args.results_dir / "stability_summary.csv"
    analysis_path = args.results_dir / "analysis.json"
    manifest_path = args.results_dir / "manifest.json"
    expected = {
        "algorithm_version": ALGORITHM_VERSION,
        "calibrated_population": file_signature(cache_path),
        "site_blueprints": file_signature(sites_path),
        "protocol_parameters": file_signature(protocol_path),
        "operational_instances": file_signature(operational_path),
        "num_sites": args.num_sites,
        "hours": list(hours),
        "grid": args.grid,
        "capacity_rates": list(CAPACITY_RATES),
        "bootstrap_replications": args.bootstrap_replications,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "validation_instances": args.validation_instances,
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
        print(f">> Reusing stability results: {rows_path.resolve()}", flush=True)
        rows = _read_rows(rows_path)
        previous = json.loads(analysis_path.read_text(encoding="utf-8"))
        validation = previous["validation"]
    else:
        rows, validation = _run(
            args.calibration_dir,
            operational_path,
            args.num_sites,
            hours,
            args.validation_instances,
            args.grid,
        )
        _write_rows(rows_path, rows)

    figure_rows = [row for row in rows if is_central_campaign_a(row)] or rows
    summaries = _summary_rows(figure_rows, args.bootstrap_replications)
    analysis = _analysis(figure_rows, summaries, validation)
    analysis["grid"] = args.grid
    analysis["instances_all_scenarios"] = len(rows)
    figures = _figure(figure_rows, args.figures_dir)
    case_tables, case_figures = _case_outputs(
        figure_rows,
        args.calibration_dir,
        hours,
        args.results_dir,
        args.figures_dir,
    )
    figures.extend(case_figures)
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
            "figures": [portable_path(path) for path in figures],
            "representative_tables": [
                portable_path(path) for path in case_tables
            ],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f">> {len(rows)} stability instances completed", flush=True)
    if not args.quiet:
        print(json.dumps(analysis, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
