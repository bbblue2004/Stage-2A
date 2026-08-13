"""Reproduce the population-level validation of the affine power model."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.data_processing.data_loader import FULL_CSV_PATH, ROOT
from src.data_processing.power_validation import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_RANDOM_SEED,
    MIN_COMPLETE_DAYS,
    population_summaries,
    representative_antenna_id,
    run_campaign,
    selection_counts,
)
from src.data_processing.power_validation_figures import (
    generate_validation_figures,
)


DEFAULT_RESULTS_DIR = ROOT / "results" / "power_validation"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "power_validation"


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(path: Path, campaign) -> None:
    diagnostics = campaign.diagnostics
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as file:
        fieldnames = (
            "antenna_id",
            "held_out_day",
            "hour",
            "traffic_gb_per_hour",
            "observed_power_w",
            "predicted_constant_w",
            "predicted_affine_w",
            "predicted_nonnegative_w",
            "predicted_quadratic_w",
            "affine_residual_w",
            "relative_traffic_rank",
            "is_extrapolation",
        )
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        for index in range(diagnostics.observed.size):
            writer.writerow(
                (
                    diagnostics.antenna_id[index],
                    diagnostics.day[index],
                    int(diagnostics.hour[index]),
                    float(diagnostics.traffic[index]),
                    float(diagnostics.observed[index]),
                    float(diagnostics.predicted_constant[index]),
                    float(diagnostics.predicted_affine[index]),
                    float(diagnostics.predicted_nonnegative[index]),
                    float(diagnostics.predicted_quadratic[index]),
                    float(
                        diagnostics.observed[index]
                        - diagnostics.predicted_affine[index]
                    ),
                    float(diagnostics.relative_traffic_rank[index]),
                    int(diagnostics.is_extrapolation[index]),
                )
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate P = F_tilde + gamma_tilde d on raw hourly observations "
            "with leave-one-complete-day-out folds."
        )
    )
    parser.add_argument("--input", type=Path, default=FULL_CSV_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--min-complete-days",
        type=int,
        default=MIN_COMPLETE_DAYS,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    if args.min_complete_days < 3:
        parser.error("--min-complete-days must be at least 3")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    if not args.input.is_file():
        parser.error(f"input CSV does not exist: {args.input}")

    print(f">> Loading and validating {args.input.resolve()}", flush=True)
    campaign = run_campaign(args.input, args.min_complete_days)
    summaries = population_summaries(
        campaign.antenna_results,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    counts = selection_counts(campaign.antenna_results)
    representative = representative_antenna_id(campaign.antenna_results)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    antenna_path = args.results_dir / "antenna_validation.csv"
    folds_path = args.results_dir / "fold_validation.csv"
    predictions_path = args.results_dir / "cross_validated_predictions.csv.gz"
    population_path = args.results_dir / "population_summary.csv"
    selection_path = args.results_dir / "selection_summary.csv"
    manifest_path = args.results_dir / "manifest.json"

    _write_dict_rows(
        antenna_path,
        [result.to_dict() for result in campaign.antenna_results],
    )
    _write_dict_rows(
        folds_path,
        [result.to_dict() for result in campaign.fold_results],
    )
    _write_predictions(predictions_path, campaign)
    _write_dict_rows(population_path, summaries)
    _write_dict_rows(
        selection_path,
        [
            {"category": category, "count": count}
            for category, count in sorted(counts.items())
        ],
    )

    print(">> Generating publication figures", flush=True)
    figures = generate_validation_figures(
        campaign,
        args.figures_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    manifest = {
        "input_path": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "parameters": {
            "min_complete_days": args.min_complete_days,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_unit": "antenna",
            "seed": args.seed,
            "cross_validation_unit": "complete_day",
            "global_outage_rule": (
                "exclude timestamps with joint zero traffic and power for "
                "every antenna"
            ),
            "active_state_rule": (
                "fit only observations with strictly positive power; retain "
                "zero-power rows in the audit"
            ),
        },
        "software": {"numpy": np.__version__},
        "audit": asdict(campaign.audit),
        "selection": counts,
        "representative_antenna_id": representative,
        "outputs": {
            "antenna_validation": str(antenna_path.resolve()),
            "fold_validation": str(folds_path.resolve()),
            "cross_validated_predictions": str(predictions_path.resolve()),
            "population_summary": str(population_path.resolve()),
            "selection_summary": str(selection_path.resolve()),
            "figures": {
                name: [str(path.resolve()) for path in paths]
                for name, paths in figures.items()
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    analyzable = [
        result
        for result in campaign.antenna_results
        if np.isfinite(result.cv_r_squared)
    ]
    operational = [
        result for result in analyzable if result.status == "included"
    ]
    print(
        f">> Antennas: {len(campaign.antenna_results)} total, "
        f"{len(analyzable)} analyzable, {len(operational)} operational.",
        flush=True,
    )
    print(
        f">> Median CV R^2: "
        f"{np.median([r.cv_r_squared for r in analyzable]):.6f}; "
        f"affine beats constant for "
        f"{np.mean([r.cv_rmse_gain_affine > 0 for r in analyzable]):.2%}.",
        flush=True,
    )
    print(f">> Representative antenna: {representative}", flush=True)
    print(f">> Results: {args.results_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
