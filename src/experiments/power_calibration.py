"""Reproduce the direct affine-power calibration reported in Section 6.2."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np

from src.data_processing.data_loader import FULL_CSV_PATH, ROOT
from src.data_processing.power_validation import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_NUM_DAYS,
    calibrated_population,
    load_calibrated_population,
    population_summaries,
    run_campaign,
    save_calibrated_population,
    selection_counts,
)
from src.data_processing.virtual_sites import (
    DEFAULT_NUM_SITES,
    DEFAULT_SITE_SEED,
    generate_virtual_sites,
    save_virtual_sites,
)
from src.data_processing.power_validation_figures import (
    generate_calibration_figure,
    generate_representative_fit_figure,
)
from src.experiments.common import (
    file_signature,
    portable_outputs,
    portable_path,
    recorded_path_matches,
    signature_matches,
)


DEFAULT_RESULTS_DIR = ROOT / "results" / "power_calibration"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "power_calibration"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cache_is_current(
    manifest_path: Path,
    cache_path: Path,
    sites_path: Path,
    input_path: Path,
    num_days: int,
) -> bool:
    if (
        not manifest_path.is_file()
        or not cache_path.is_file()
        or not sites_path.is_file()
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            manifest["cache_schema_version"] == CACHE_SCHEMA_VERSION
            and recorded_path_matches(manifest["input_path"], input_path)
            and signature_matches(manifest["input_signature"], input_path)
            and manifest["parameters"]["num_days"] == num_days
            and manifest["parameters"]["num_sites"] == DEFAULT_NUM_SITES
            and manifest["parameters"]["site_seed"] == DEFAULT_SITE_SEED
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one descriptive affine traffic--power model per antenna "
            "on the first observed days."
        )
    )
    parser.add_argument("--input", type=Path, default=FULL_CSV_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--num-days", type=int, default=DEFAULT_NUM_DAYS)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the final numerical summary",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="re-read the raw CSV and rebuild all calibrated artefacts",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input CSV does not exist: {args.input}")
    if args.num_days <= 0:
        parser.error("--num-days must be positive")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    antenna_path = args.results_dir / "antenna_calibration.csv"
    population_path = args.results_dir / "population_summary.csv"
    selection_path = args.results_dir / "selection_summary.csv"
    cache_path = args.results_dir / "calibrated_population.npz"
    sites_path = args.results_dir / "virtual_sites.csv"
    manifest_path = args.results_dir / "manifest.json"
    if not args.rebuild_cache and _cache_is_current(
        manifest_path, cache_path, sites_path, args.input, args.num_days
    ):
        population = load_calibrated_population(cache_path)
        representative_paths, representative_id = generate_representative_fit_figure(
            population, args.figures_dir
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["input_path"] = portable_path(args.input)
        manifest["input_signature"] = file_signature(args.input)
        manifest["outputs"] = portable_outputs(manifest["outputs"])
        manifest["representative_antenna_id"] = representative_id
        manifest["outputs"]["representative_figure"] = [
            portable_path(path) for path in representative_paths
        ]
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f">> Reusing calibrated cache: {cache_path.resolve()}", flush=True)
        print(">> Use --rebuild-cache to re-read the raw CSV.", flush=True)
        return

    print(f">> Calibrating affine power models from {args.input.resolve()}", flush=True)
    campaign = run_campaign(args.input, num_days=args.num_days)
    summaries = population_summaries(campaign.antenna_results)
    counts = selection_counts(campaign.antenna_results)
    population = calibrated_population(campaign)
    save_calibrated_population(population, cache_path)
    virtual_sites = generate_virtual_sites(population)
    save_virtual_sites(virtual_sites, population, sites_path)
    _write_rows(
        antenna_path,
        [result.to_dict() for result in campaign.antenna_results],
    )
    _write_rows(population_path, summaries)
    _write_rows(
        selection_path,
        [
            {"category": category, "count": count}
            for category, count in sorted(counts.items())
        ],
    )

    print(">> Generating the Section 6.2 figure", flush=True)
    figure_paths = generate_calibration_figure(campaign, args.figures_dir)
    representative_paths, representative_id = generate_representative_fit_figure(
        population, args.figures_dir
    )
    manifest = {
        "input_path": portable_path(args.input),
        "input_signature": file_signature(args.input),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "parameters": {
            "num_days": args.num_days,
            "num_sites": DEFAULT_NUM_SITES,
            "site_seed": DEFAULT_SITE_SEED,
            "fit": "ordinary least squares with intercept",
            "active_state_rule": "strictly positive observed power",
            "admissibility_rule": "strictly positive intercept and slope",
            "prediction_or_cross_validation": False,
            "alternative_models": False,
        },
        "software": {
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "audit": asdict(campaign.audit),
        "selection": counts,
        "representative_antenna_id": representative_id,
        "outputs": {
            "antenna_calibration": portable_path(antenna_path),
            "population_summary": portable_path(population_path),
            "selection_summary": portable_path(selection_path),
            "calibrated_population": portable_path(cache_path),
            "virtual_sites": portable_path(sites_path),
            "figure": [portable_path(path) for path in figure_paths],
            "representative_figure": [
                portable_path(path) for path in representative_paths
            ],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_by_metric = {row["metric"]: row for row in summaries}
    included = counts["included"]
    if not args.quiet:
        print(
            f">> Antennas: {counts['total']} total, {included} included "
            f"({included / counts['total']:.2%}).",
            flush=True,
        )
        print(
            f">> Median R^2: {summary_by_metric['r_squared']['median']:.6f}; "
            "median normalized RMSE: "
            f"{summary_by_metric['normalized_rmse']['median']:.6f}; "
            "median fixed-power share: "
            f"{summary_by_metric['fixed_power_share']['median']:.6f}.",
            flush=True,
        )
    print(f">> Results: {args.results_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
