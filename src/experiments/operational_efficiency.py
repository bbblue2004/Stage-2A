"""Run the multi-plan operational experiment reported in Section 6.3."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.time_window import HOURS_PER_DAY, inclusive_hour_window
from src.core.window_optimiser import (
    HourlySolution,
    descending_capacity_hourly_policy,
    optimal_hourly_with_persistent,
    proportional_hourly_policy,
)
from src.data_processing.data_loader import ROOT
from src.data_processing.instance_generator import (
    CAMPAIGN_A_RATES,
    CENTRAL_RATE,
    ScenarioSpec,
    capacities_for_site,
    iter_materialized_sites,
    minimal_guardian_counts,
)
from src.experiments.common import file_signature, inputs_match, portable_path
from src.experiments.protocol_io import (
    is_central_campaign_a,
    load_protocol_inputs,
    scenarios_for_grid,
)


DEFAULT_CALIBRATION_DIR = ROOT / "results" / "power_calibration"
DEFAULT_RESULTS_DIR = ROOT / "results" / "operational_efficiency"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "operational_efficiency"
CAPACITY_RATES = CAMPAIGN_A_RATES
PROFILE_HOURS = tuple(range(HOURS_PER_DAY))
ALGORITHM_VERSION = 6
POLICIES = (
    ("hourly_optimal", "Optimum horaire", "hourly_optimal_savings_pct", None),
    (
        "hourly_descending_capacity",
        "Sélection par capacité",
        "hourly_capacity_savings_pct",
        "capacity_loss_vs_hourly_optimum_pp",
    ),
    (
        "hourly_proportional",
        "Mêmes gardiens, répartition proportionnelle",
        "hourly_proportional_savings_pct",
        "proportional_loss_vs_hourly_optimum_pp",
    ),
    (
        "persistent",
        "Mêmes gardiens pendant toute la fenêtre",
        "persistent_savings_pct",
        "persistent_loss_vs_hourly_optimum_pp",
    ),
)


def _is_efficiency_reference(row: dict[str, object]) -> bool:
    """Return whether a row belongs to the Section 6.3 reference setting."""
    return is_central_campaign_a(row)


def _efficiency_scenarios(grid: str) -> tuple[ScenarioSpec, ...]:
    """Ensure that the reference setting is evaluated at every capacity rate."""
    return scenarios_for_grid(grid)


STRING_COLUMNS = {
    "site_id",
    "day",
    "day_type",
    "scenario",
    "campaign",
    "volume_level",
    "shape_level",
    "equipment_level",
    "guardian_target",
    "hourly_optimal_guardian_masks",
    "hourly_capacity_guardian_masks",
    "persistent_guardian_set",
    "policy",
    "label",
}
INTEGER_COLUMNS = {
    "hour",
    "num_operators",
    "persistent_guardians",
    "hourly_guardian_changes",
    "hourly_optimal_guardians",
    "minimum_guardians",
    "low_traffic_condition",
    "instances",
    "plans",
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
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {
                key: (
                    value
                    if key in STRING_COLUMNS
                    else int(value)
                    if key in INTEGER_COLUMNS
                    else float(value)
                )
                for key, value in raw.items()
            }
            for raw in csv.DictReader(file)
        ]


def _day_type(day_index: int) -> str:
    return "weekday" if day_index < 5 else "weekend"


def _hourly_energy(
    solution: HourlySolution,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
) -> np.ndarray:
    fixed = np.asarray(fixed_power_w, dtype=float)
    slopes = np.asarray(slopes_w_per_gb, dtype=float)
    fixed_by_hour = np.asarray(
        [
            sum(
                fixed[index]
                for index in range(fixed.size)
                if int(mask) & (1 << index)
            )
            for mask in solution.guardian_masks
        ],
        dtype=float,
    )
    return fixed_by_hour + np.sum(
        slopes[:, None] * solution.allocations_gb, axis=0
    )


def _standalone_hourly_energy(
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> np.ndarray:
    return float(np.sum(fixed_power_w)) + np.sum(
        np.asarray(slopes_w_per_gb)[:, None] * demands_gb, axis=0
    )


def _window_positions(
    profile_hours: tuple[int, ...], window_hours: tuple[int, ...]
) -> np.ndarray:
    positions = {hour: index for index, hour in enumerate(profile_hours)}
    try:
        return np.asarray([positions[hour] for hour in window_hours], dtype=int)
    except KeyError as error:
        raise ValueError("the profile does not cover the complete window") from error


def _evaluate_demands(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
    profile_hours: tuple[int, ...],
    window_hours: tuple[int, ...],
) -> dict[str, object]:
    """Evaluate all hourly policies and the persistent policy on ``H``."""
    started = time.perf_counter()
    positions = _window_positions(profile_hours, window_hours)
    optimum, _ = optimal_hourly_with_persistent(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    capacity = descending_capacity_hourly_policy(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    proportional = proportional_hourly_policy(
        optimum.guardian_masks,
        capacities_gb,
        fixed_power_w,
        slopes_w_per_gb,
        demands_gb,
    )
    window_demands = demands_gb[:, positions]
    window_optimum, persistent = optimal_hourly_with_persistent(
        capacities_gb, fixed_power_w, slopes_w_per_gb, window_demands
    )
    if not np.array_equal(
        optimum.guardian_masks[positions], window_optimum.guardian_masks
    ):
        raise RuntimeError("hourly optimum changed when the study window was sliced")

    standalone = _standalone_hourly_energy(
        fixed_power_w, slopes_w_per_gb, demands_gb
    )
    optimum_energy = _hourly_energy(optimum, fixed_power_w, slopes_w_per_gb)
    capacity_energy = _hourly_energy(capacity, fixed_power_w, slopes_w_per_gb)
    proportional_energy = _hourly_energy(
        proportional, fixed_power_w, slopes_w_per_gb
    )
    tolerance = 1e-7 * max(1.0, float(np.sum(standalone)))
    if not (
        np.all(optimum_energy <= capacity_energy + tolerance)
        and np.all(optimum_energy <= proportional_energy + tolerance)
        and np.all(optimum_energy <= standalone + tolerance)
        and window_optimum.energy_wh <= persistent.energy_wh + tolerance
    ):
        raise RuntimeError("operational cost ordering failed")

    totals = np.sum(demands_gb, axis=0)
    minimum_guardians = np.maximum(
        1, minimal_guardian_counts(capacities_gb, demands_gb)
    )
    return {
        "standalone_hourly_wh": standalone,
        "optimum_hourly_wh": optimum_energy,
        "capacity_hourly_wh": capacity_energy,
        "proportional_hourly_wh": proportional_energy,
        "optimum_masks": optimum.guardian_masks.astype(int),
        "capacity_masks": capacity.guardian_masks.astype(int),
        # Guardian sets are non-empty in Definition 3; zero traffic still has k_h=1.
        "minimum_guardians": minimum_guardians,
        "low_traffic": totals <= float(np.min(capacities_gb)) + 1e-9,
        "window_positions": positions,
        "window_optimum": window_optimum,
        "persistent": persistent,
        "elapsed_s": time.perf_counter() - started,
    }


def _relative_savings(energy: float, standalone: float) -> float:
    return 100.0 * (1.0 - energy / standalone)


def _window_row(
    site,
    day: str,
    day_index: int,
    capacities: np.ndarray,
    result: dict[str, object],
    num_operators: int,
) -> dict[str, object]:
    spec = site.scenario
    positions = np.asarray(result["window_positions"], dtype=int)
    standalone_hourly = np.asarray(result["standalone_hourly_wh"], dtype=float)
    optimum_hourly = np.asarray(result["optimum_hourly_wh"], dtype=float)
    capacity_hourly = np.asarray(result["capacity_hourly_wh"], dtype=float)
    proportional_hourly = np.asarray(
        result["proportional_hourly_wh"], dtype=float
    )
    optimum_masks = np.asarray(result["optimum_masks"], dtype=int)[positions]
    capacity_masks = np.asarray(result["capacity_masks"], dtype=int)[positions]
    optimum_counts = np.asarray(
        [int(mask).bit_count() for mask in optimum_masks], dtype=int
    )
    capacity_counts = np.asarray(
        [int(mask).bit_count() for mask in capacity_masks], dtype=int
    )
    minimum = np.asarray(result["minimum_guardians"], dtype=int)[positions]
    low_traffic = np.asarray(result["low_traffic"], dtype=bool)[positions]
    window_optimum = result["window_optimum"]
    persistent = result["persistent"]
    standalone = float(np.sum(standalone_hourly[positions]))
    standalone_fixed = float(
        positions.size * np.sum(site.p_fixed_w[:num_operators])
    )
    optimum = float(np.sum(optimum_hourly[positions]))
    capacity_energy = float(np.sum(capacity_hourly[positions]))
    proportional_energy = float(np.sum(proportional_hourly[positions]))
    if abs(optimum - window_optimum.energy_wh) > 1e-7 * max(1.0, optimum):
        raise RuntimeError("window energy is inconsistent with hourly additivity")

    return {
        "site_id": site.site_id,
        "scenario": spec.key,
        "campaign": spec.campaign,
        "volume_level": spec.volume_level,
        "shape_level": spec.shape_level,
        "equipment_level": spec.equipment_level,
        "day": str(day),
        "day_type": _day_type(day_index),
        "num_operators": num_operators,
        "capacity_rate": (
            spec.capacity_rate if spec.campaign == "A" else spec.window_peak_rate
        ),
        "guardian_target": spec.guardian_target if spec.campaign == "B" else "",
        "standalone_energy_wh": standalone,
        "standalone_fixed_energy_wh": standalone_fixed,
        "standalone_fixed_fraction": standalone_fixed / standalone,
        "hourly_optimal_energy_wh": optimum,
        "hourly_capacity_energy_wh": capacity_energy,
        "hourly_proportional_energy_wh": proportional_energy,
        "persistent_energy_wh": persistent.energy_wh,
        "hourly_optimal_savings_wh": standalone - optimum,
        "hourly_optimal_savings_pct": _relative_savings(optimum, standalone),
        "hourly_capacity_savings_pct": _relative_savings(capacity_energy, standalone),
        "hourly_proportional_savings_pct": _relative_savings(
            proportional_energy, standalone
        ),
        "persistent_savings_pct": _relative_savings(
            persistent.energy_wh, standalone
        ),
        "capacity_loss_vs_hourly_optimum_pp": max(
            0.0, 100.0 * (capacity_energy - optimum) / standalone
        ),
        "proportional_loss_vs_hourly_optimum_pp": max(
            0.0, 100.0 * (proportional_energy - optimum) / standalone
        ),
        "persistent_loss_vs_hourly_optimum_pp": max(
            0.0, 100.0 * (persistent.energy_wh - optimum) / standalone
        ),
        "hourly_optimal_guardians_mean": float(np.mean(optimum_counts)),
        "hourly_capacity_guardians_mean": float(np.mean(capacity_counts)),
        "minimum_guardians_mean": float(np.mean(minimum)),
        "low_traffic_condition_fraction": float(np.mean(low_traffic)),
        "persistent_guardians": persistent.num_guardians,
        "hourly_optimal_guardian_masks": "|".join(str(mask) for mask in optimum_masks),
        "hourly_capacity_guardian_masks": "|".join(str(mask) for mask in capacity_masks),
        "persistent_guardian_set": "|".join(
            str(index + 1) for index in persistent.guardians
        ),
        "capacity_same_guardian_count_fraction": float(
            np.mean(optimum_counts == capacity_counts)
        ),
        "capacity_same_guardian_set_fraction": float(
            np.mean(optimum_masks == capacity_masks)
        ),
        "hourly_guardian_changes": window_optimum.guardian_changes,
        "full_day_standalone_energy_wh": float(np.sum(standalone_hourly)),
        "full_day_optimal_energy_wh": float(np.sum(optimum_hourly)),
        "full_day_optimal_savings_wh": float(
            np.sum(standalone_hourly - optimum_hourly)
        ),
        "evaluation_time_ms": 1_000.0 * float(result["elapsed_s"]),
        "maximum_capacity_gb": float(np.max(capacities)),
        "minimum_fixed_share": float(
            np.min(site.p_fixed_w) / np.sum(site.p_fixed_w)
        ),
    }


def _hourly_rows(
    site,
    day: str,
    day_index: int,
    profile_hours: tuple[int, ...],
    result: dict[str, object],
) -> list[dict[str, object]]:
    spec = site.scenario
    standalone = np.asarray(result["standalone_hourly_wh"], dtype=float)
    optimum = np.asarray(result["optimum_hourly_wh"], dtype=float)
    capacity = np.asarray(result["capacity_hourly_wh"], dtype=float)
    proportional = np.asarray(result["proportional_hourly_wh"], dtype=float)
    optimum_masks = np.asarray(result["optimum_masks"], dtype=int)
    minimum = np.asarray(result["minimum_guardians"], dtype=int)
    low_traffic = np.asarray(result["low_traffic"], dtype=bool)
    return [
        {
            "site_id": site.site_id,
            "scenario": spec.key,
            "campaign": spec.campaign,
            "volume_level": spec.volume_level,
            "shape_level": spec.shape_level,
            "equipment_level": spec.equipment_level,
            "day": str(day),
            "day_type": _day_type(day_index),
            "capacity_rate": spec.capacity_rate,
            "hour": hour,
            "standalone_energy_wh": standalone[index],
            "hourly_optimal_energy_wh": optimum[index],
            "hourly_capacity_energy_wh": capacity[index],
            "hourly_proportional_energy_wh": proportional[index],
            "hourly_optimal_savings_wh": standalone[index] - optimum[index],
            "hourly_optimal_savings_pct": _relative_savings(
                optimum[index], standalone[index]
            ),
            "hourly_proportional_savings_pct": _relative_savings(
                proportional[index], standalone[index]
            ),
            "proportional_loss_vs_hourly_optimum_pp": 100.0
            * (proportional[index] - optimum[index])
            / standalone[index],
            "hourly_optimal_guardians": int(optimum_masks[index]).bit_count(),
            "minimum_guardians": int(minimum[index]),
            "low_traffic_condition": int(low_traffic[index]),
        }
        for index, hour in enumerate(profile_hours)
    ]


def _run(
    calibration_dir: Path,
    num_sites: int,
    window_hours: tuple[int, ...],
    grid: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, float],
]:
    population, blueprints, protocol = load_protocol_inputs(calibration_dir)
    if not 1 <= num_sites <= blueprints.num_sites:
        raise ValueError(f"num_sites must be between 1 and {blueprints.num_sites}")
    scenarios = _efficiency_scenarios(grid)
    window_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    operator_rows: list[dict[str, object]] = []
    timing = {"campaign_wall_time_s": 0.0, "n3_time_s": 0.0, "n4_time_s": 0.0}
    started = time.perf_counter()
    evaluated = 0
    for site in iter_materialized_sites(
        blueprints, population, scenarios, protocol, num_sites=num_sites
    ):
        spec = site.scenario
        profile_hours = PROFILE_HOURS if spec.campaign == "A" else window_hours
        for day_index, day in enumerate(population.days):
            window_demands = site.traffic_gb[:, day_index][:, window_hours]
            capacities = capacities_for_site(site, window_demands)
            demands = site.traffic_gb[:, day_index][:, profile_hours]
            result = _evaluate_demands(
                capacities,
                site.p_fixed_w,
                site.slope_w_per_gb,
                demands,
                profile_hours,
                window_hours,
            )
            timing["n4_time_s"] += float(result["elapsed_s"])
            row = _window_row(
                site, str(day), day_index, capacities, result, num_operators=4
            )
            window_rows.append(row)
            if spec.campaign == "A":
                hourly_rows.extend(
                    _hourly_rows(site, str(day), day_index, profile_hours, result)
                )
            if _is_efficiency_reference(row) and float(spec.capacity_rate) == CENTRAL_RATE:
                operator_rows.append(dict(row))
                n3_capacities = site.peak_traffic_gb[:3] / float(spec.capacity_rate)
                n3_result = _evaluate_demands(
                    n3_capacities,
                    site.p_fixed_w[:3],
                    site.slope_w_per_gb[:3],
                    demands[:3],
                    profile_hours,
                    window_hours,
                )
                timing["n3_time_s"] += float(n3_result["elapsed_s"])
                operator_rows.append(
                    _window_row(
                        site,
                        str(day),
                        day_index,
                        n3_capacities,
                        n3_result,
                        num_operators=3,
                    )
                )
        evaluated += 1
        if evaluated % 20 == 0:
            print(
                f">> {evaluated}/{num_sites * len(scenarios)} site-scenarios evaluated",
                flush=True,
            )
    timing["campaign_wall_time_s"] = time.perf_counter() - started
    return window_rows, hourly_rows, operator_rows, timing


def _site_means(
    rows: Iterable[dict[str, object]], column: str
) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["site_id"]), []).append(float(row[column]))
    sites = sorted(grouped)
    return sites, np.asarray([np.mean(grouped[site]) for site in sites], dtype=float)


def _descriptive_interval(
    values: np.ndarray,
    statistic: Callable[..., np.ndarray | float] = np.median,
) -> dict[str, float]:
    """Return the centre and middle half of the plan-level observations."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("summary values must be a non-empty vector")
    lower, upper = np.quantile(values, (0.25, 0.75))
    return {
        "estimate": float(statistic(values)),
        "q25": float(lower),
        "q75": float(upper),
    }


def _summarise_column(
    rows: list[dict[str, object]],
    column: str,
    statistic: Callable[..., np.ndarray | float] = np.median,
) -> dict[str, float]:
    _, values = _site_means(rows, column)
    return _descriptive_interval(values, statistic)


def _add_interval(
    target: dict[str, object], prefix: str, interval: dict[str, float]
) -> None:
    for suffix, value in interval.items():
        target[f"{prefix}_{suffix}"] = value


def _capacity_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for rate in CAPACITY_RATES:
        selected = [
            row for row in rows if float(row["capacity_rate"]) == rate
        ]
        summary: dict[str, object] = {
            "capacity_rate": rate,
            "plans": len({str(row["site_id"]) for row in selected}),
            "instances": len(selected),
        }
        _add_interval(
            summary,
            "savings_pct",
            _summarise_column(selected, "hourly_optimal_savings_pct"),
        )
        saved_kwh_rows = [
            {
                **row,
                "saved_kwh": float(row["hourly_optimal_savings_wh"]) / 1_000.0,
            }
            for row in selected
        ]
        _add_interval(
            summary,
            "saved_kwh",
            _summarise_column(saved_kwh_rows, "saved_kwh"),
        )
        summaries.append(summary)
    return summaries


def _policy_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected = [row for row in rows if float(row["capacity_rate"]) == CENTRAL_RATE]
    summaries: list[dict[str, object]] = []
    for key, label, savings_column, loss_column in POLICIES:
        summary: dict[str, object] = {
            "capacity_rate": CENTRAL_RATE,
            "policy": key,
            "label": label,
            "plans": len({str(row["site_id"]) for row in selected}),
            "instances": len(selected),
        }
        _add_interval(
            summary,
            "savings_pct",
            _summarise_column(selected, savings_column),
        )
        interval = (
            {"estimate": 0.0, "q25": 0.0, "q75": 0.0}
            if loss_column is None
            else _summarise_column(selected, loss_column)
        )
        _add_interval(summary, "loss_pp", interval)
        summaries.append(summary)
    return summaries


def _operator_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for count in (3, 4):
        selected = [row for row in rows if row["num_operators"] == count]
        summary: dict[str, object] = {
            "num_operators": count,
            "capacity_rate": CENTRAL_RATE,
            "plans": len({str(row["site_id"]) for row in selected}),
            "instances": len(selected),
            "structural_bound_pct": 100.0 * (count - 1) / count,
        }
        _add_interval(
            summary,
            "savings_pct",
            _summarise_column(selected, "hourly_optimal_savings_pct"),
        )
        _add_interval(
            summary,
            "evaluation_time_ms",
            _summarise_column(selected, "evaluation_time_ms"),
        )
        summaries.append(summary)
    return summaries


def _hourly_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected_rate = [
        row for row in rows if float(row["capacity_rate"]) == CENTRAL_RATE
    ]
    summaries: list[dict[str, object]] = []
    metric_specs = (
        ("savings_pct", "hourly_optimal_savings_pct", np.median),
        ("saved_kwh", "saved_kwh", np.median),
        ("optimal_guardians", "hourly_optimal_guardians", np.mean),
        ("minimum_guardians", "minimum_guardians", np.mean),
        ("low_traffic_pct", "low_traffic_pct", np.mean),
    )
    for hour in PROFILE_HOURS:
        selected = [row for row in selected_rate if row["hour"] == hour]
        enriched = [
            {
                **row,
                "saved_kwh": float(row["hourly_optimal_savings_wh"]) / 1_000.0,
                "low_traffic_pct": 100.0 * float(row["low_traffic_condition"]),
            }
            for row in selected
        ]
        summary: dict[str, object] = {
            "capacity_rate": CENTRAL_RATE,
            "hour": hour,
            "plans": len({str(row["site_id"]) for row in selected}),
            "instances": len(selected),
        }
        for prefix, column, statistic in metric_specs:
            _add_interval(
                summary,
                prefix,
                _summarise_column(enriched, column, statistic),
            )
        summaries.append(summary)
    return summaries


def _paired_operator_effect(
    rows: list[dict[str, object]],
) -> dict[str, float]:
    by_count: dict[int, dict[str, float]] = {}
    for count in (3, 4):
        sites, values = _site_means(
            [row for row in rows if row["num_operators"] == count],
            "hourly_optimal_savings_pct",
        )
        by_count[count] = dict(zip(sites, values, strict=True))
    sites = sorted(set(by_count[3]) & set(by_count[4]))
    differences = np.asarray(
        [by_count[4][site] - by_count[3][site] for site in sites], dtype=float
    )
    return _descriptive_interval(differences)


def _annual_order(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    selected = [row for row in rows if float(row["capacity_rate"]) == CENTRAL_RATE]
    by_site: dict[str, dict[str, list[float]]] = {}
    for row in selected:
        bucket = by_site.setdefault(
            str(row["site_id"]),
            {
                "night_saved": [],
                "night_standalone": [],
                "night_optimal": [],
                "day_saved": [],
            },
        )
        bucket["night_saved"].append(float(row["hourly_optimal_savings_wh"]))
        bucket["night_standalone"].append(float(row["standalone_energy_wh"]))
        bucket["night_optimal"].append(float(row["hourly_optimal_energy_wh"]))
        bucket["day_saved"].append(float(row["full_day_optimal_savings_wh"]))

    annual_night: list[float] = []
    annual_day: list[float] = []
    typical_night: list[float] = []
    typical_night_standalone: list[float] = []
    typical_night_optimal: list[float] = []
    typical_day: list[float] = []
    for values in by_site.values():
        night = float(np.mean(values["night_saved"]))
        standalone = float(np.mean(values["night_standalone"]))
        optimum = float(np.mean(values["night_optimal"]))
        day = float(np.mean(values["day_saved"]))
        typical_night.append(night / 1_000.0)
        typical_night_standalone.append(standalone / 1_000.0)
        typical_night_optimal.append(optimum / 1_000.0)
        typical_day.append(day / 1_000.0)
        annual_night.append(365.0 * night / 1_000.0)
        annual_day.append(365.0 * day / 1_000.0)
    return {
        "typical_night_kwh": _descriptive_interval(np.asarray(typical_night)),
        "typical_night_standalone_kwh": _descriptive_interval(
            np.asarray(typical_night_standalone)
        ),
        "typical_night_optimal_kwh": _descriptive_interval(
            np.asarray(typical_night_optimal)
        ),
        "typical_day_kwh": _descriptive_interval(np.asarray(typical_day)),
        "annual_night_kwh": _descriptive_interval(np.asarray(annual_night)),
        "annual_day_kwh": _descriptive_interval(np.asarray(annual_day)),
    }


def _analysis(
    window_rows: list[dict[str, object]],
    hourly_rows: list[dict[str, object]],
    operator_rows: list[dict[str, object]],
    capacity_summaries: list[dict[str, object]],
    policy_summaries: list[dict[str, object]],
    operator_summaries: list[dict[str, object]],
    timing: dict[str, float],
    window_hours: tuple[int, ...],
) -> dict[str, object]:
    central = [row for row in window_rows if float(row["capacity_rate"]) == CENTRAL_RATE]
    central_hourly = [
        row for row in hourly_rows if float(row["capacity_rate"]) == CENTRAL_RATE
    ]
    changes = _summarise_column(central, "hourly_guardian_changes")
    k_window = [
        row for row in central_hourly if int(row["hour"]) in window_hours
    ]
    all_k = np.asarray([int(row["minimum_guardians"]) for row in central_hourly])
    window_k = np.asarray([int(row["minimum_guardians"]) for row in k_window])
    all_guardians = np.asarray(
        [int(row["hourly_optimal_guardians"]) for row in central_hourly]
    )
    sites, savings = _site_means(central, "hourly_optimal_savings_pct")
    half = max(1, len(sites) // 2)
    return {
        "plans": len(sites),
        "window_instances": len(window_rows),
        "central_window_instances": len(central),
        "hourly_instances": len(hourly_rows),
        "capacity": capacity_summaries,
        "policies": policy_summaries,
        "operator_count": operator_summaries,
        "paired_n4_minus_n3_savings_pp": _paired_operator_effect(operator_rows),
        "annual_order_of_magnitude": _annual_order(window_rows),
        "hourly_guardian_changes": changes,
        "constant_guardian_set_fraction": float(
            np.mean([float(row["hourly_guardian_changes"]) == 0.0 for row in central])
        ),
        "minimum_guardian_distribution_24h": {
            str(count): float(np.mean(all_k == count)) for count in range(1, 5)
        },
        "minimum_guardian_distribution_window": {
            str(count): float(np.mean(window_k == count)) for count in range(1, 5)
        },
        "optimal_guardian_distribution_24h": {
            str(count): float(np.mean(all_guardians == count))
            for count in range(1, 5)
        },
        "low_traffic_condition_fraction_24h": float(
            np.mean([int(row["low_traffic_condition"]) for row in central_hourly])
        ),
        "low_traffic_condition_fraction_window": float(
            np.mean([int(row["low_traffic_condition"]) for row in k_window])
        ),
        "one_optimal_guardian_fraction_window": float(
            np.mean(
                [int(row["hourly_optimal_guardians"]) == 1 for row in k_window]
            )
        ),
        "proportional_loss_multi_guardian_pp": _summarise_column(
            [
                row
                for row in k_window
                if int(row["hourly_optimal_guardians"]) > 1
            ],
            "proportional_loss_vs_hourly_optimum_pp",
        ),
        "minimum_fixed_share_median": float(
            _summarise_column(central, "minimum_fixed_share")["estimate"]
        ),
        "standalone_fixed_fraction_median": float(
            _summarise_column(central, "standalone_fixed_fraction")["estimate"]
        ),
        "maximum_observed_k_h": int(np.max(all_k)),
        "convergence_pilot": {
            "median_first_half_plans_pct": float(np.median(savings[:half])),
            "median_all_plans_pct": float(np.median(savings)),
            "absolute_difference_pp": float(
                abs(np.median(savings[:half]) - np.median(savings))
            ),
        },
        "timing": timing,
    }


def _hourly_figure(
    summaries: list[dict[str, object]], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hours = np.asarray(PROFILE_HOURS)
    colour = "#2A6FBB"
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 3.5), sharex=True)

    def values(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = sorted(summaries, key=lambda row: int(row["hour"]))
        return tuple(
            np.asarray([float(row[f"{prefix}_{suffix}"]) for row in rows])
            for suffix in ("estimate", "q25", "q75")
        )

    estimate, lower, upper = values("savings_pct")
    axes[0].plot(hours, estimate, color=colour)
    axes[0].fill_between(hours, lower, upper, color=colour, alpha=0.16)
    estimate, lower, upper = values("saved_kwh")
    axes[1].plot(hours, estimate, color=colour)
    axes[1].fill_between(hours, lower, upper, color=colour, alpha=0.16)
    guardians, guardians_low, guardians_high = values("optimal_guardians")
    axes[2].plot(hours, guardians, color=colour)
    axes[2].fill_between(
        hours, guardians_low, guardians_high, color=colour, alpha=0.12
    )

    axes[0].set_title("(a) Part économisée")
    axes[0].set_ylabel("part économisée (%)")
    axes[0].set_ylim(0.0, 100.0)
    axes[1].set_title("(b) Énergie économisée")
    axes[1].set_ylabel("énergie économisée (kWh par heure)")
    axes[2].set_title("(c) Gardiens retenus par l'optimum")
    axes[2].set_ylabel("nombre moyen de gardiens")
    axes[2].set_ylim(0.8, 4.1)
    for axis in axes:
        axis.axvspan(-0.5, 6.5, color="0.88", alpha=0.65, zorder=0)
        axis.set_xlim(-0.5, 23.5)
        axis.set_xticks(range(0, 24, 3))
        axis.grid(alpha=0.24, linewidth=0.5)
        axis.set_xlabel("heure")
    figure.tight_layout()
    paths = [
        output_dir / "hourly_profiles.pdf",
        output_dir / "hourly_profiles.png",
    ]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=220, bbox_inches="tight")
    plt.close(figure)
    return paths


def _policy_figure(
    rows: list[dict[str, object]], output_dir: Path
) -> list[Path]:
    selected = [row for row in rows if float(row["capacity_rate"]) == CENTRAL_RATE]
    savings: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    for _, _, savings_column, loss_column in POLICIES:
        _, values = _site_means(selected, savings_column)
        savings.append(values)
        if loss_column is not None:
            _, values = _site_means(selected, loss_column)
            losses.append(values)

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    box = axes[0].boxplot(
        savings,
        tick_labels=(
            "Optimum",
            "Capacité\ndécroissante",
            "Proportionnelle\n(mêmes gardiens)",
            "Gardiens fixes\nsur la nuit",
        ),
        whis=(0, 100),
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(
        box["boxes"],
        ("#2A6FBB", "#D98E04", "#5B9A55", "#7557A5"),
        strict=True,
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set_ylabel("part économisée (%)")
    axes[0].set_title("(a) Part économisée sur la fenêtre nocturne")
    axes[0].tick_params(axis="x", labelsize=7.5)
    axes[0].grid(axis="y", alpha=0.25)

    loss_box = axes[1].boxplot(
        losses,
        tick_labels=(
            "Capacité\ndécroissante",
            "Proportionnelle\n(mêmes gardiens)",
            "Gardiens fixes\nsur la nuit",
        ),
        whis=(0, 100),
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(
        loss_box["boxes"],
        ("#D98E04", "#5B9A55", "#7557A5"),
        strict=True,
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].set_ylabel("perte d'économie (points)")
    axes[1].set_title("(b) Perte par rapport à l'optimum")
    axes[1].tick_params(axis="x", labelsize=7.5)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    paths = [
        output_dir / "operational_efficiency.pdf",
        output_dir / "operational_efficiency.png",
    ]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=220, bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--num-sites", type=int, default=DEFAULT_NUM_SITES)
    parser.add_argument(
        "--grid",
        choices=("central", "full", "thresholds"),
        default="central",
        help="central evaluates the three capacity rates used in Section 6.3",
    )
    parser.add_argument(
        "--hours", nargs=2, type=int, default=(0, 6), metavar=("START", "END")
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    window_hours = inclusive_hour_window(*args.hours)
    cache_path = args.calibration_dir / "calibrated_population.npz"
    sites_path = args.calibration_dir / "site_blueprints.csv"
    protocol_path = args.calibration_dir / "protocol_parameters.json"
    for path in (cache_path, sites_path, protocol_path):
        if not path.is_file():
            parser.error(f"missing calibrated input: {path}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.results_dir / "operational_instances.csv"
    hourly_path = args.results_dir / "hourly_profiles.csv"
    operator_path = args.results_dir / "operator_count_instances.csv"
    capacity_summary_path = args.results_dir / "capacity_summary.csv"
    policy_summary_path = args.results_dir / "policy_summary.csv"
    operator_summary_path = args.results_dir / "operator_count_summary.csv"
    hourly_summary_path = args.results_dir / "hourly_profile_summary.csv"
    analysis_path = args.results_dir / "analysis.json"
    manifest_path = args.results_dir / "manifest.json"
    expected = {
        "algorithm_version": ALGORITHM_VERSION,
        "calibrated_population": file_signature(cache_path),
        "site_blueprints": file_signature(sites_path),
        "protocol_parameters": file_signature(protocol_path),
        "num_sites": args.num_sites,
        "window_hours": list(window_hours),
        "profile_hours": list(PROFILE_HOURS),
        "grid": args.grid,
        "capacity_rates": list(CAPACITY_RATES),
    }
    cached_paths = (rows_path, hourly_path, operator_path, analysis_path)
    current = False
    if (
        not args.rebuild
        and manifest_path.is_file()
        and all(path.is_file() for path in cached_paths)
    ):
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
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            current = False

    if current:
        print(f">> Reusing operational results: {rows_path.resolve()}", flush=True)
        window_rows = _read_rows(rows_path)
        hourly_rows = _read_rows(hourly_path)
        operator_rows = _read_rows(operator_path)
        timing = json.loads(analysis_path.read_text(encoding="utf-8"))["timing"]
    else:
        window_rows, hourly_rows, operator_rows, timing = _run(
            args.calibration_dir, args.num_sites, window_hours, args.grid
        )
        _write_rows(rows_path, window_rows)
        _write_rows(hourly_path, hourly_rows)
        _write_rows(operator_path, operator_rows)

    figure_rows = [row for row in window_rows if _is_efficiency_reference(row)] or window_rows
    figure_hourly_rows = [
        row
        for row in hourly_rows
        if _is_efficiency_reference(row)
    ] or hourly_rows
    capacity_summaries = _capacity_summaries(figure_rows)
    policy_summaries = _policy_summaries(figure_rows)
    operator_summaries = _operator_summaries(operator_rows)
    hourly_summaries = _hourly_summaries(figure_hourly_rows)
    analysis = _analysis(
        figure_rows,
        figure_hourly_rows,
        operator_rows,
        capacity_summaries,
        policy_summaries,
        operator_summaries,
        timing,
        window_hours,
    )
    analysis["grid"] = args.grid
    analysis["window_hours"] = list(window_hours)
    analysis["pilot"] = args.num_sites < DEFAULT_NUM_SITES
    figures = _hourly_figure(hourly_summaries, args.figures_dir)
    figures.extend(_policy_figure(figure_rows, args.figures_dir))

    _write_rows(capacity_summary_path, capacity_summaries)
    _write_rows(policy_summary_path, policy_summaries)
    _write_rows(operator_summary_path, operator_summaries)
    _write_rows(hourly_summary_path, hourly_summaries)
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "inputs": expected,
        "outputs": {
            "instances": portable_path(rows_path),
            "hourly_profiles": portable_path(hourly_path),
            "operator_count_instances": portable_path(operator_path),
            "capacity_summary": portable_path(capacity_summary_path),
            "policy_summary": portable_path(policy_summary_path),
            "operator_count_summary": portable_path(operator_summary_path),
            "hourly_profile_summary": portable_path(hourly_summary_path),
            "analysis": portable_path(analysis_path),
            "figures": [portable_path(path) for path in figures],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f">> {len(window_rows)} operational instances completed on {args.num_sites} plans",
        flush=True,
    )
    if not args.quiet:
        print(json.dumps(analysis, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
