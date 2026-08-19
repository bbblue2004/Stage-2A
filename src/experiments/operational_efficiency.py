"""Run the hourly operational experiment reported in Section 6.3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.time_window import inclusive_hour_window
from src.core.window_optimiser import (
    descending_capacity_hourly_policy,
    optimal_hourly_with_persistent,
    proportional_hourly_policy,
    standalone_energy,
)
from src.data_processing.data_loader import ROOT
from src.data_processing.instance_generator import (
    CAMPAIGN_A_RATES,
    DEFAULT_NUM_SITES,
    capacities_for_site,
    iter_materialized_sites,
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
ALGORITHM_VERSION = 3
POLICIES = (
    (
        "hourly_optimal",
        "Optimum horaire",
        "hourly_optimal_energy_wh",
        "hourly_optimal_guardians_mean",
    ),
    (
        "hourly_descending_capacity",
        "Capacités décroissantes horaires",
        "hourly_capacity_energy_wh",
        "hourly_capacity_guardians_mean",
    ),
    (
        "hourly_proportional",
        "Allocation proportionnelle horaire",
        "hourly_proportional_energy_wh",
        "hourly_optimal_guardians_mean",
    ),
    (
        "persistent",
        "Gardiens persistants",
        "persistent_energy_wh",
        "persistent_guardians",
    ),
)
STRING_COLUMNS = {
    "site_id",
    "day",
    "scenario",
    "campaign",
    "volume_level",
    "shape_level",
    "equipment_level",
    "guardian_target",
    "hourly_optimal_guardian_masks",
    "hourly_capacity_guardian_masks",
    "persistent_guardian_set",
}
INTEGER_COLUMNS = {"persistent_guardians", "hourly_guardian_changes"}


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


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q05, median, q95 = np.quantile(values, (0.05, 0.50, 0.95))
    return {
        "mean": float(np.mean(values)),
        "q05": float(q05),
        "median": float(median),
        "q95": float(q95),
    }


def _policy_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for rate in (*CAPACITY_RATES, None):
        selected = [
            row
            for row in rows
            if rate is None or row["capacity_rate"] == rate
        ]
        baseline = np.asarray(
            [row["standalone_energy_wh"] for row in selected], dtype=float
        )
        optimum = np.asarray(
            [row["hourly_optimal_energy_wh"] for row in selected],
            dtype=float,
        )
        for key, label, energy_column, guardian_column in POLICIES:
            energy = np.asarray(
                [row[energy_column] for row in selected], dtype=float
            )
            savings = 100.0 * (1.0 - energy / baseline)
            loss = np.maximum(
                0.0, 100.0 * (energy - optimum) / baseline
            )
            guardians = np.asarray(
                [row[guardian_column] for row in selected], dtype=float
            )
            summaries.append(
                {
                    "capacity_rate": (
                        "all" if rate is None else f"{rate:.2f}"
                    ),
                    "policy": key,
                    "label": label,
                    "instances": len(selected),
                    **{
                        f"savings_pct_{name}": value
                        for name, value in _quantiles(savings).items()
                    },
                    **{
                        f"loss_vs_hourly_optimum_pp_{name}": value
                        for name, value in _quantiles(loss).items()
                    },
                    "guardians_mean": float(np.mean(guardians)),
                    "guardians_median": float(np.median(guardians)),
                }
            )
    return summaries


def _parse_masks(value: object) -> np.ndarray:
    return np.asarray([int(item) for item in str(value).split("|")], dtype=int)


def _analysis(rows: list[dict[str, object]]) -> dict[str, object]:
    baseline = np.asarray(
        [row["standalone_energy_wh"] for row in rows], dtype=float
    )
    optimum = np.asarray(
        [row["hourly_optimal_energy_wh"] for row in rows], dtype=float
    )
    capacity = np.asarray(
        [row["hourly_capacity_energy_wh"] for row in rows], dtype=float
    )
    proportional = np.asarray(
        [row["hourly_proportional_energy_wh"] for row in rows], dtype=float
    )
    persistent = np.asarray(
        [row["persistent_energy_wh"] for row in rows], dtype=float
    )
    savings = 100.0 * (1.0 - optimum / baseline)
    persistence_gap = np.maximum(
        0.0, 100.0 * (persistent - optimum) / baseline
    )
    first_half = np.asarray(
        [str(row["site_id"]) <= "site_0200" for row in rows]
    )
    optimal_counts = np.concatenate(
        [
            np.asarray(
                [int(mask).bit_count() for mask in _parse_masks(
                    row["hourly_optimal_guardian_masks"]
                )],
                dtype=int,
            )
            for row in rows
        ]
    )
    capacity_counts = np.concatenate(
        [
            np.asarray(
                [int(mask).bit_count() for mask in _parse_masks(
                    row["hourly_capacity_guardian_masks"]
                )],
                dtype=int,
            )
            for row in rows
        ]
    )
    return {
        "instances": len(rows),
        "hourly_decisions": int(optimal_counts.size),
        "all_policies_feasible": True,
        "hourly_optimal_savings_pct": _quantiles(savings),
        "capacity_loss_vs_hourly_optimum_pp": _quantiles(
            np.maximum(
                0.0, 100.0 * (capacity - optimum) / baseline
            )
        ),
        "proportional_loss_vs_hourly_optimum_pp": _quantiles(
            np.maximum(
                0.0, 100.0 * (proportional - optimum) / baseline
            )
        ),
        "persistent_loss_vs_hourly_optimum_pp": _quantiles(persistence_gap),
        "capacity_matches_hourly_optimum_fraction": float(
            np.mean(np.isclose(capacity, optimum, rtol=1e-10, atol=1e-7))
        ),
        "proportional_matches_hourly_optimum_fraction": float(
            np.mean(
                np.isclose(proportional, optimum, rtol=1e-10, atol=1e-7)
            )
        ),
        "same_hourly_guardian_count_fraction": float(
            np.mean(
                [
                    row["capacity_same_guardian_count_fraction"]
                    for row in rows
                ]
            )
        ),
        "same_hourly_guardian_set_fraction": float(
            np.mean(
                [row["capacity_same_guardian_set_fraction"] for row in rows]
            )
        ),
        "persistent_matches_hourly_optimum_fraction": float(
            np.mean(
                np.isclose(persistent, optimum, rtol=1e-10, atol=1e-7)
            )
        ),
        "hourly_optimal_guardian_distribution": {
            str(count): float(np.mean(optimal_counts == count))
            for count in range(1, 5)
        },
        "hourly_capacity_guardian_distribution": {
            str(count): float(np.mean(capacity_counts == count))
            for count in range(1, 5)
        },
        "hourly_guardian_changes": _quantiles(
            np.asarray(
                [row["hourly_guardian_changes"] for row in rows],
                dtype=float,
            )
        ),
        "convergence": {
            "median_savings_first_200_sites_pct": float(
                np.median(savings[first_half])
            ),
            "median_savings_all_sites_pct": float(np.median(savings)),
            "absolute_difference_pp": float(
                abs(
                    np.median(savings[first_half])
                    - np.median(savings)
                )
            ),
        },
    }


def _figure(
    rows: list[dict[str, object]], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = np.asarray(
        [row["standalone_energy_wh"] for row in rows], dtype=float
    )
    optimum = np.asarray(
        [row["hourly_optimal_energy_wh"] for row in rows], dtype=float
    )
    capacity = np.asarray(
        [row["hourly_capacity_energy_wh"] for row in rows], dtype=float
    )
    proportional = np.asarray(
        [row["hourly_proportional_energy_wh"] for row in rows], dtype=float
    )
    persistent = np.asarray(
        [row["persistent_energy_wh"] for row in rows], dtype=float
    )
    savings = [
        100.0 * (1.0 - capacity / baseline),
        100.0 * (1.0 - proportional / baseline),
        100.0 * (1.0 - persistent / baseline),
        100.0 * (1.0 - optimum / baseline),
    ]
    losses = [
        np.maximum(0.0, 100.0 * (capacity - optimum) / baseline),
        np.maximum(
            0.0, 100.0 * (proportional - optimum) / baseline
        ),
        np.maximum(0.0, 100.0 * (persistent - optimum) / baseline),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    box = axes[0].boxplot(
        savings,
        tick_labels=(
            "Capacité\nhoraire",
            "Proportionnelle\nhoraire",
            "Persistante",
            "Optimum\nhoraire",
        ),
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(
        box["boxes"],
        ("#D98E04", "#5B9A55", "#7557A5", "#2A6FBB"),
        strict=True,
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Énergie évitée (%)")
    axes[0].set_title("(a) Énergie évitée")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    loss_box = axes[1].boxplot(
        losses,
        tick_labels=(
            "Sélection\npar capacité",
            "Allocation\nproportionnelle",
            "Persistance\ndes gardiens",
        ),
        whis=(5, 95),
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
    axes[1].axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Perte d'économie (points)")
    axes[1].set_title("(b) Coût des restrictions")
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


def _run(
    calibration_dir: Path,
    num_sites: int,
    hours: tuple[int, ...],
    grid: str,
) -> list[dict[str, object]]:
    population, blueprints, protocol = load_protocol_inputs(calibration_dir)
    if not 1 <= num_sites <= blueprints.num_sites:
        raise ValueError(f"num_sites must be between 1 and {blueprints.num_sites}")
    scenarios = scenarios_for_grid(grid)
    rows: list[dict[str, object]] = []
    evaluated = 0
    for site in iter_materialized_sites(
        blueprints, population, scenarios, protocol, num_sites=num_sites
    ):
        spec = site.scenario
        for day_index, day in enumerate(population.days):
            demands = site.traffic_gb[:, day_index][:, hours]
            capacities = capacities_for_site(site, demands)
            autonomous = standalone_energy(
                site.p_fixed_w, site.slope_w_per_gb, demands
            )
            optimum, persistent = optimal_hourly_with_persistent(
                capacities, site.p_fixed_w, site.slope_w_per_gb, demands
            )
            capacity = descending_capacity_hourly_policy(
                capacities, site.p_fixed_w, site.slope_w_per_gb, demands
            )
            proportional = proportional_hourly_policy(
                optimum.guardian_masks,
                capacities,
                site.p_fixed_w,
                site.slope_w_per_gb,
                demands,
            )
            tolerance = 1e-7 * max(1.0, autonomous)
            if not (
                optimum.energy_wh <= persistent.energy_wh + tolerance
                and optimum.energy_wh <= capacity.energy_wh + tolerance
                and optimum.energy_wh <= proportional.energy_wh + tolerance
                and optimum.energy_wh <= autonomous + tolerance
            ):
                raise RuntimeError(
                    "operational cost ordering failed for "
                    f"{site.site_id}, {spec.key}, {day}"
                )
            optimal_masks = optimum.guardian_masks.astype(int)
            capacity_masks = capacity.guardian_masks.astype(int)
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
                    "standalone_energy_wh": autonomous,
                    "hourly_optimal_energy_wh": optimum.energy_wh,
                    "hourly_capacity_energy_wh": capacity.energy_wh,
                    "hourly_proportional_energy_wh": proportional.energy_wh,
                    "persistent_energy_wh": persistent.energy_wh,
                    "hourly_optimal_guardians_mean": optimum.mean_guardians,
                    "hourly_capacity_guardians_mean": capacity.mean_guardians,
                    "persistent_guardians": persistent.num_guardians,
                    "hourly_optimal_guardian_masks": "|".join(
                        str(mask) for mask in optimal_masks
                    ),
                    "hourly_capacity_guardian_masks": "|".join(
                        str(mask) for mask in capacity_masks
                    ),
                    "persistent_guardian_set": "|".join(
                        str(index + 1) for index in persistent.guardians
                    ),
                    "capacity_same_guardian_count_fraction": float(
                        np.mean(
                            optimum.guardian_counts
                            == capacity.guardian_counts
                        )
                    ),
                    "capacity_same_guardian_set_fraction": float(
                        np.mean(optimal_masks == capacity_masks)
                    ),
                    "hourly_guardian_changes": optimum.guardian_changes,
                }
            )
        evaluated += 1
        if evaluated % 100 == 0:
            print(
                f">> {evaluated}/{num_sites * len(scenarios)} site-scenarios evaluated",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR
    )
    parser.add_argument("--num-sites", type=int, default=DEFAULT_NUM_SITES)
    parser.add_argument(
        "--grid",
        choices=("central", "full", "thresholds"),
        default="full",
        help="central: campaign A rates only; full: protocol grid; thresholds: campaign B",
    )
    parser.add_argument(
        "--hours",
        nargs=2,
        type=int,
        default=(0, 6),
        metavar=("START", "END"),
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    hours = inclusive_hour_window(*args.hours)
    cache_path = args.calibration_dir / "calibrated_population.npz"
    sites_path = args.calibration_dir / "site_blueprints.csv"
    protocol_path = args.calibration_dir / "protocol_parameters.json"
    for path in (cache_path, sites_path, protocol_path):
        if not path.is_file():
            parser.error(f"missing calibrated input: {path}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.results_dir / "operational_instances.csv"
    summary_path = args.results_dir / "policy_summary.csv"
    analysis_path = args.results_dir / "analysis.json"
    manifest_path = args.results_dir / "manifest.json"
    expected = {
        "algorithm_version": ALGORITHM_VERSION,
        "calibrated_population": file_signature(cache_path),
        "site_blueprints": file_signature(sites_path),
        "protocol_parameters": file_signature(protocol_path),
        "num_sites": args.num_sites,
        "hours": list(hours),
        "grid": args.grid,
        "capacity_rates": list(CAPACITY_RATES),
    }
    current = False
    if (
        not args.rebuild
        and manifest_path.is_file()
        and rows_path.is_file()
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
        print(
            f">> Reusing operational results: {rows_path.resolve()}",
            flush=True,
        )
        rows = _read_rows(rows_path)
    else:
        rows = _run(args.calibration_dir, args.num_sites, hours, args.grid)
        _write_rows(rows_path, rows)

    figure_rows = [row for row in rows if is_central_campaign_a(row)] or rows
    summaries = _policy_summaries(figure_rows)
    analysis = _analysis(figure_rows)
    analysis["grid"] = args.grid
    analysis["instances_all_scenarios"] = len(rows)
    figures = _figure(figure_rows, args.figures_dir)
    _write_rows(summary_path, summaries)
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
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
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f">> {len(rows)} operational instances completed", flush=True)
    if not args.quiet:
        print(json.dumps(analysis, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
