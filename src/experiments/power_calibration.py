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
from src.data_processing.instance_generator import (
    DEFAULT_NUM_SITES,
    DEFAULT_SITE_SEED,
    GENERATOR_VERSION,
    calibrate_protocol,
    generate_site_blueprints,
    save_protocol_spec,
    save_site_blueprints,
)
from src.data_processing.power_validation_figures import (
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


def _population_cache_is_current(
    manifest_path: Path,
    cache_path: Path,
    input_path: Path,
    num_days: int,
) -> bool:
    if not manifest_path.is_file() or not cache_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            manifest["cache_schema_version"] == CACHE_SCHEMA_VERSION
            and recorded_path_matches(manifest["input_path"], input_path)
            and signature_matches(manifest["input_signature"], input_path)
            and manifest["parameters"]["num_days"] == num_days
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _blueprints_are_current(manifest: dict[str, object], sites_path: Path, protocol_path: Path) -> bool:
    if not sites_path.is_file() or not protocol_path.is_file():
        return False
    try:
        parameters = manifest["parameters"]
        return (
            parameters["num_sites"] == DEFAULT_NUM_SITES
            and parameters["site_seed"] == DEFAULT_SITE_SEED
            and parameters.get("generator_version") == GENERATOR_VERSION
        )
    except (KeyError, TypeError):
        return False


def _write_site_artefacts(population, results_dir: Path) -> tuple[Path, Path]:
    sites_path = results_dir / "site_blueprints.csv"
    protocol_path = results_dir / "protocol_parameters.json"
    protocol = calibrate_protocol(population)
    blueprints = generate_site_blueprints(population)
    save_site_blueprints(blueprints, population, sites_path)
    save_protocol_spec(protocol, protocol_path)
    return sites_path, protocol_path


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
    sites_path = args.results_dir / "site_blueprints.csv"
    protocol_path = args.results_dir / "protocol_parameters.json"
    manifest_path = args.results_dir / "manifest.json"
    population_ready = (
        not args.rebuild_cache
        and _population_cache_is_current(
            manifest_path, cache_path, args.input, args.num_days
        )
    )
    if population_ready:
        population = load_calibrated_population(cache_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _blueprints_are_current(manifest, sites_path, protocol_path):
            representative_paths, representative_id = generate_representative_fit_figure(
                population, args.figures_dir
            )
            manifest["input_path"] = portable_path(args.input)
            manifest["input_signature"] = file_signature(args.input)
            manifest["outputs"] = portable_outputs(manifest["outputs"])
            manifest["outputs"].pop("figure", None)
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
        print(">> Reusing power fits; rebuilding site blueprints", flush=True)
        sites_path, protocol_path = _write_site_artefacts(population, args.results_dir)
        representative_paths, representative_id = generate_representative_fit_figure(
            population, args.figures_dir
        )
        manifest["parameters"]["num_sites"] = DEFAULT_NUM_SITES
        manifest["parameters"]["site_seed"] = DEFAULT_SITE_SEED
        manifest["parameters"]["generator_version"] = GENERATOR_VERSION
        manifest["representative_antenna_id"] = representative_id
        manifest["outputs"]["site_blueprints"] = portable_path(sites_path)
        manifest["outputs"]["protocol_parameters"] = portable_path(protocol_path)
        manifest["outputs"]["representative_figure"] = [
            portable_path(path) for path in representative_paths
        ]
        manifest["outputs"].pop("virtual_sites", None)
        manifest["outputs"].pop("figure", None)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f">> Results: {args.results_dir.resolve()}", flush=True)
        return

    print(f">> Calibrating affine power models from {args.input.resolve()}", flush=True)
    campaign = run_campaign(args.input, num_days=args.num_days)
    summaries = population_summaries(campaign.antenna_results)
    counts = selection_counts(campaign.antenna_results)
    population = calibrated_population(campaign)
    save_calibrated_population(population, cache_path)
    sites_path, protocol_path = _write_site_artefacts(population, args.results_dir)
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
            "generator_version": GENERATOR_VERSION,
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
            "site_blueprints": portable_path(sites_path),
            "protocol_parameters": portable_path(protocol_path),
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
