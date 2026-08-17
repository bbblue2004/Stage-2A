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

from src.core.game import bondareva_shapley_test, least_core_allocation
from src.core.time_window import inclusive_hour_window
from src.core.window_optimiser import persistent_coalition_costs
from src.data_processing.data_loader import ROOT
from src.data_processing.power_validation import load_calibrated_population
from src.data_processing.virtual_sites import load_virtual_sites
from src.experiments.common import (
    file_signature,
    four_player_bondareva_gap,
    inputs_match,
    portable_path,
)


DEFAULT_CALIBRATION_DIR = ROOT / "results" / "power_calibration"
DEFAULT_OPERATIONAL_DIR = ROOT / "results" / "operational_efficiency"
DEFAULT_RESULTS_DIR = ROOT / "results" / "coalition_stability"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "coalition_stability"
CAPACITY_RATES = (0.70, 0.80, 0.90, 1.00)
BOOTSTRAP_SEED = 20_260_814
ALGORITHM_VERSION = 2
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
    string_columns = {"site_id", "day", "category"}
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


def _load_operational_rows(path: Path) -> dict[tuple[str, str, float], tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as file:
        return {
            (raw["site_id"], raw["day"], float(raw["capacity_rate"])): (
                float(raw["standalone_energy_wh"]),
                float(raw["optimal_energy_wh"]),
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


def _run(
    cache_path: Path,
    sites_path: Path,
    operational_path: Path,
    num_sites: int,
    hours: tuple[int, ...],
    validation_instances: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    population = load_calibrated_population(cache_path)
    sites = load_virtual_sites(sites_path, population)
    operational = _load_operational_rows(operational_path)
    if not 1 <= num_sites <= sites.num_sites:
        raise ValueError(f"num_sites must be between 1 and {sites.num_sites}")

    rows: list[dict[str, object]] = []
    max_operational_difference = 0.0
    max_bondareva_difference = 0.0
    max_least_core_difference = 0.0
    independent_disagreements = 0
    validated = 0

    for site_index in range(num_sites):
        indices = sites.antenna_indices[site_index]
        fixed = population.p_fixed_w[indices]
        slopes = population.slope_w_per_gb[indices]
        peaks = population.peak_traffic_gb[indices]
        for day_index, day in enumerate(population.days):
            demands = population.traffic_gb[indices, day_index][:, hours]
            for rate in CAPACITY_RATES:
                capacities = peaks / rate
                costs = persistent_coalition_costs(capacities, fixed, slopes, demands)
                savings = _savings_from_costs(costs)
                diagnostics = _diagnose(costs, savings, capacities, demands)
                standalone = float(
                    sum(costs[1 << player] for player in range(NUM_PLAYERS))
                )
                grand_cost = float(costs[GRAND_MASK])
                key = (str(sites.site_ids[site_index]), str(day), rate)
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
                        "site_id": key[0],
                        "day": key[1],
                        "capacity_rate": rate,
                        "standalone_energy_wh": standalone,
                        "grand_cost_wh": grand_cost,
                        "grand_savings_wh": float(savings[GRAND_MASK]),
                        "savings_pct": 100.0 * float(savings[GRAND_MASK]) / standalone,
                        **diagnostics,
                    }
                )
        if (site_index + 1) % 100 == 0:
            print(f">> {site_index + 1}/{num_sites} sites evaluated", flush=True)

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
    first_half = [row for row in rows if str(row["site_id"]) <= "site_0500"]
    savings = np.asarray([float(row["savings_pct"]) for row in rows])
    top_decile_threshold = float(np.quantile(savings, 0.90))
    top_decile = [
        row for row in rows if float(row["savings_pct"]) >= top_decile_threshold
    ]

    def fractions(selected: list[dict[str, object]]) -> dict[str, float]:
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
            "category_fractions_first_500_sites": half_fractions,
            "category_fractions_1000_sites": full_fractions,
            "maximum_absolute_difference_pp": 100.0
            * max(
                abs(half_fractions[category] - full_fractions[category])
                for category in CATEGORIES
            ),
        },
        "validation": validation,
    }


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
    parser.add_argument("--num-sites", type=int, default=1_000)
    parser.add_argument("--hours", nargs=2, type=int, default=(0, 6), metavar=("START", "END"))
    parser.add_argument("--bootstrap-replications", type=int, default=2_000)
    parser.add_argument("--validation-instances", type=int, default=20)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    hours = inclusive_hour_window(*args.hours)
    cache_path = args.calibration_dir / "calibrated_population.npz"
    sites_path = args.calibration_dir / "virtual_sites.csv"
    operational_path = args.operational_dir / "operational_instances.csv"
    for path in (cache_path, sites_path, operational_path):
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
        "virtual_sites": file_signature(sites_path),
        "operational_instances": file_signature(operational_path),
        "num_sites": args.num_sites,
        "hours": list(hours),
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
                    "virtual_sites": sites_path,
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
            cache_path,
            sites_path,
            operational_path,
            args.num_sites,
            hours,
            args.validation_instances,
        )
        _write_rows(rows_path, rows)

    summaries = _summary_rows(rows, args.bootstrap_replications)
    analysis = _analysis(rows, summaries, validation)
    figures = _figure(rows, args.figures_dir)
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
