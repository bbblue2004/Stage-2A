"""Plausibility diagnostics for the semi-empirical instance generator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.time_window import inclusive_hour_window
from src.data_processing.data_loader import ROOT
from src.data_processing.instance_generator import (
    CENTRAL_RATE,
    DEFAULT_NUM_SITES,
    ScenarioSpec,
    antenna_usage_counts,
    capacities_for_site,
    iter_materialized_sites,
    load_protocol_spec,
    load_site_blueprints,
    minimal_guardian_counts,
    normalised_shapes,
    protocol_scenarios,
)
from src.data_processing.power_validation import load_calibrated_population
from src.experiments.common import file_signature, inputs_match, portable_path


DEFAULT_CALIBRATION_DIR = ROOT / "results" / "power_calibration"
DEFAULT_RESULTS_DIR = ROOT / "results" / "instance_diagnostics"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "instance_diagnostics"
DIAGNOSTIC_VERSION = 1
CENTRAL_SPEC = ScenarioSpec(
    "A", "moderate", "moderate", "moderate", capacity_rate=CENTRAL_RATE
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q10, q25, median, q75, q90 = np.quantile(values, (0.10, 0.25, 0.50, 0.75, 0.90))
    return {
        "mean": float(np.mean(values)),
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
    }


def _row_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
    return numerator / np.maximum(denominator, 1e-12)


def _pairwise_operator_corr(traffic: np.ndarray) -> np.ndarray:
    flat = np.asarray(traffic, dtype=float).reshape(traffic.shape[0], traffic.shape[1], -1)
    n_sites, n_op, _ = flat.shape
    values: list[float] = []
    for site_index in range(n_sites):
        profiles = flat[site_index]
        centred = profiles - profiles.mean(axis=1, keepdims=True)
        grams = centred @ centred.T
        norms = np.sqrt(np.diag(grams))
        denom = np.outer(norms, norms)
        corr = grams / np.maximum(denom, 1e-12)
        values.extend(
            float(corr[i, j]) for i in range(n_op) for j in range(i + 1, n_op)
        )
    return np.asarray(values, dtype=float)


def _daily_profile(traffic: np.ndarray) -> np.ndarray:
    daily = traffic.reshape(-1, traffic.shape[-3], traffic.shape[-1]).mean(axis=-2)
    means = np.maximum(daily.mean(axis=1, keepdims=True), 1e-12)
    return daily / means


def _collect_sites(sites, hours: tuple[int, ...]) -> dict[str, np.ndarray]:
    traffic = np.stack([site.traffic_gb for site in sites])
    p_fixed = np.stack([site.p_fixed_w for site in sites])
    slope = np.stack([site.slope_w_per_gb for site in sites])
    alpha = np.stack([site.alpha for site in sites])
    means = traffic.mean(axis=(2, 3))
    peaks = traffic.max(axis=(2, 3))
    cv = traffic.reshape(traffic.shape[0], traffic.shape[1], -1).std(axis=2) / np.maximum(
        means, 1e-12
    )
    window = traffic[:, :, :, hours]
    load_rows = []
    guardian_rows = []
    for site, window_traffic in zip(sites, window, strict=True):
        for day_index in range(window_traffic.shape[1]):
            demands = window_traffic[:, day_index]
            capacities = capacities_for_site(site, demands)
            load_rows.append((demands / capacities[:, None]).reshape(-1))
            guardian_rows.append(minimal_guardian_counts(capacities, demands))
    return {
        "traffic": traffic,
        "p_fixed": p_fixed,
        "slope": slope,
        "alpha": alpha,
        "mean": means,
        "peak": peaks,
        "cv": cv,
        "load": np.concatenate(load_rows),
        "guardians": np.concatenate(guardian_rows),
    }


def _summarise_population(population, hours: tuple[int, ...]) -> dict[str, np.ndarray]:
    traffic = population.traffic_gb
    means = traffic.mean(axis=(1, 2))
    peaks = traffic.max(axis=(1, 2))
    cv = traffic.reshape(traffic.shape[0], -1).std(axis=1) / np.maximum(means, 1e-12)
    return {
        "traffic": traffic,
        "p_fixed": population.p_fixed_w,
        "slope": population.slope_w_per_gb,
        "mean": means,
        "peak": peaks,
        "cv": cv,
        "night_load_proxy": traffic[:, :, hours].max(axis=(1, 2)) / np.maximum(peaks, 1e-12),
    }


def run_diagnostics(
    population,
    blueprints,
    protocol,
    hours: tuple[int, ...],
    num_sites: int | None = None,
) -> dict[str, object]:
    real = _summarise_population(population, hours)
    real_shapes = normalised_shapes(population.traffic_gb)
    rng = np.random.default_rng(protocol.seed)
    pair_i = rng.integers(0, real_shapes.shape[0], size=12_000)
    pair_j = rng.integers(0, real_shapes.shape[0], size=12_000)
    mask = pair_i != pair_j
    real_pair_corr = _row_corr(real_shapes[pair_i[mask]], real_shapes[pair_j[mask]])
    lag1_real = _row_corr(real_shapes[:, :-1], real_shapes[:, 1:])

    collected: dict[str, dict[str, np.ndarray]] = {}
    scenario_rows: list[dict[str, object]] = []
    for spec in protocol_scenarios():
        sites = list(
            iter_materialized_sites(
                blueprints, population, [spec], protocol, num_sites=num_sites
            )
        )
        arrays = _collect_sites(sites, hours)
        collected[spec.key] = arrays
        shapes = arrays["traffic"].reshape(arrays["traffic"].shape[0] * 4, -1)
        shapes = shapes / np.maximum(shapes.mean(axis=1, keepdims=True), 1e-12)
        lag1 = _row_corr(shapes[:, :-1], shapes[:, 1:])
        operator_corr = _pairwise_operator_corr(arrays["traffic"])
        guardian_counts = arrays["guardians"]
        feasible = guardian_counts <= 4
        scenario_rows.append(
            {
                "scenario": spec.key,
                "campaign": spec.campaign,
                "volume_level": spec.volume_level,
                "shape_level": spec.shape_level,
                "equipment_level": spec.equipment_level,
                "instances_hours": int(guardian_counts.size),
                "mean_traffic_gb_median": float(np.median(arrays["mean"])),
                "peak_traffic_gb_median": float(np.median(arrays["peak"])),
                "p_fixed_w_median": float(np.median(arrays["p_fixed"])),
                "slope_w_per_gb_median": float(np.median(arrays["slope"])),
                "corr_mean_traffic_p_fixed": float(
                    np.corrcoef(arrays["mean"].reshape(-1), arrays["p_fixed"].reshape(-1))[0, 1]
                ),
                "operator_shape_corr_median": float(np.median(operator_corr)),
                "lag1_corr_median": float(np.median(lag1)),
                "load_factor_median": float(np.median(arrays["load"])),
                "guardian_min_share_1": float(np.mean(guardian_counts == 1)),
                "guardian_min_share_2": float(np.mean(guardian_counts == 2)),
                "guardian_min_share_3": float(np.mean(guardian_counts == 3)),
                "guardian_min_share_4": float(np.mean(guardian_counts == 4)),
                "infeasible_hour_share": float(np.mean(~feasible)),
            }
        )

    usage = antenna_usage_counts(blueprints)
    central = collected[CENTRAL_SPEC.key]
    comparison_rows = []
    for metric, real_values, synth_values in (
        ("mean_traffic_gb", real["mean"], central["mean"].reshape(-1)),
        ("peak_traffic_gb", real["peak"], central["peak"].reshape(-1)),
        ("cv_120h", real["cv"], central["cv"].reshape(-1)),
        ("p_fixed_w", real["p_fixed"], central["p_fixed"].reshape(-1)),
        ("slope_w_per_gb", real["slope"], central["slope"].reshape(-1)),
        ("pairwise_shape_corr", real_pair_corr, _pairwise_operator_corr(central["traffic"])),
        ("lag1_shape_corr", lag1_real, _row_corr(
            (central["traffic"].reshape(-1, 120) / np.maximum(
                central["traffic"].reshape(-1, 120).mean(axis=1, keepdims=True), 1e-12
            ))[:, :-1],
            (central["traffic"].reshape(-1, 120) / np.maximum(
                central["traffic"].reshape(-1, 120).mean(axis=1, keepdims=True), 1e-12
            ))[:, 1:],
        )),
    ):
        comparison_rows.append(
            {
                "metric": metric,
                "source": "empirical",
                **{f"{name}": value for name, value in _quantiles(real_values).items()},
            }
        )
        comparison_rows.append(
            {
                "metric": metric,
                "source": "synthetic_central",
                **{f"{name}": value for name, value in _quantiles(synth_values).items()},
            }
        )

    analysis = {
        "num_sites": blueprints.num_sites if num_sites is None else num_sites,
        "num_operators": blueprints.num_operators,
        "antenna_usage": {
            "distinct_antennas": int(usage.size),
            "max_uses": int(np.max(usage)),
            "median_uses": float(np.median(usage)),
            "share_used_at_most_twice": float(np.mean(usage <= 2)),
        },
        "central_corr_mean_traffic_p_fixed": float(
            np.corrcoef(central["mean"].reshape(-1), central["p_fixed"].reshape(-1))[0, 1]
        ),
        "empirical_corr_mean_traffic_p_fixed": float(
            np.corrcoef(real["mean"], real["p_fixed"])[0, 1]
        ),
        "convergence": {
            "median_mean_traffic_first_half": float(
                np.median(central["mean"][: max(1, central["mean"].shape[0] // 2)])
            ),
            "median_mean_traffic_all": float(np.median(central["mean"])),
        },
    }
    return {
        "scenario_rows": scenario_rows,
        "comparison_rows": comparison_rows,
        "analysis": analysis,
        "collected": collected,
        "real": real,
        "real_pair_corr": real_pair_corr,
    }


def _figures(payload: dict[str, object], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    real = payload["real"]
    central = payload["collected"][CENTRAL_SPEC.key]
    real_daily = _daily_profile(real["traffic"])
    synth_daily = _daily_profile(central["traffic"])
    hours = np.arange(24)

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    axes[0, 0].plot(hours, np.median(real_daily, axis=0), color="#2A6FBB", label="Antennes réelles")
    axes[0, 0].fill_between(
        hours,
        np.quantile(real_daily, 0.25, axis=0),
        np.quantile(real_daily, 0.75, axis=0),
        color="#2A6FBB",
        alpha=0.18,
    )
    axes[0, 0].plot(
        hours,
        np.median(synth_daily, axis=0),
        color="#D95F4A",
        linestyle="--",
        label="Opérateurs synthétiques",
    )
    axes[0, 0].set_xlabel("Heure")
    axes[0, 0].set_ylabel("Profil journalier normalisé")
    axes[0, 0].set_title("(a) Forme journalière")
    axes[0, 0].set_xticks([0, 6, 12, 18, 23])
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].hist(
        payload["real_pair_corr"],
        bins=30,
        density=True,
        color="#2A6FBB",
        alpha=0.45,
        label="Paires d'antennes",
    )
    axes[0, 1].hist(
        _pairwise_operator_corr(central["traffic"]),
        bins=30,
        density=True,
        color="#D95F4A",
        alpha=0.45,
        label="Opérateurs d'un même site",
    )
    axes[0, 1].set_xlabel("Corrélation des profils")
    axes[0, 1].set_ylabel("Densité")
    axes[0, 1].set_title("(b) Corrélation temporelle")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].scatter(
        real["mean"],
        real["p_fixed"],
        s=8,
        alpha=0.25,
        color="#2A6FBB",
        label="Antennes réelles",
    )
    axes[1, 0].scatter(
        central["mean"].reshape(-1),
        central["p_fixed"].reshape(-1),
        s=12,
        alpha=0.45,
        color="#D95F4A",
        label="Opérateurs synthétiques",
    )
    axes[1, 0].set_xlabel("Trafic moyen (Go/h)")
    axes[1, 0].set_ylabel(r"$P^{\mathrm{fixe}}$ (W)")
    axes[1, 0].set_title("(c) Volume et puissance fixe")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    regime_keys = [
        spec.key
        for spec in protocol_scenarios()
        if spec.campaign == "B"
    ]
    labels = ["1 gardien", "Frontière 1–2", "Frontière 2–3", "Contraint"]
    shares = np.zeros((len(regime_keys), 4))
    for row_index, key in enumerate(regime_keys):
        counts = payload["collected"][key]["guardians"]
        for k in range(1, 5):
            shares[row_index, k - 1] = float(np.mean(counts == k))
    left = np.zeros(len(regime_keys))
    colors = ("#3677B8", "#5B9A55", "#E2A23A", "#C94C4C")
    for k in range(4):
        axes[1, 1].barh(
            labels,
            shares[:, k],
            left=left,
            color=colors[k],
            label=f"{k + 1}",
        )
        left += shares[:, k]
    axes[1, 1].set_xlim(0.0, 1.0)
    axes[1, 1].set_xlabel("Part des heures de $H$")
    axes[1, 1].set_title("(d) Gardiens minimaux, campagne B")
    axes[1, 1].legend(title="Nombre", frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    paths = [
        output_dir / "protocol_diagnostics.pdf",
        output_dir / "protocol_diagnostics.png",
    ]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=220, bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--num-sites", type=int, default=DEFAULT_NUM_SITES)
    parser.add_argument("--hours", nargs=2, type=int, default=(0, 6), metavar=("START", "END"))
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
    comparison_path = args.results_dir / "distribution_comparison.csv"
    scenario_path = args.results_dir / "scenario_diagnostics.csv"
    analysis_path = args.results_dir / "analysis.json"
    manifest_path = args.results_dir / "manifest.json"
    expected = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "calibrated_population": file_signature(cache_path),
        "site_blueprints": file_signature(sites_path),
        "protocol_parameters": file_signature(protocol_path),
        "num_sites": args.num_sites,
        "hours": list(hours),
    }
    current = False
    if not args.rebuild and manifest_path.is_file() and comparison_path.is_file():
        try:
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["inputs"]
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

    population = load_calibrated_population(cache_path)
    blueprints = load_site_blueprints(sites_path, population)
    protocol = load_protocol_spec(protocol_path)
    if current:
        print(f">> Reusing instance diagnostics: {args.results_dir.resolve()}", flush=True)
        payload = run_diagnostics(
            population, blueprints, protocol, hours, num_sites=args.num_sites
        )
    else:
        payload = run_diagnostics(
            population, blueprints, protocol, hours, num_sites=args.num_sites
        )
        _write_rows(comparison_path, payload["comparison_rows"])
        _write_rows(scenario_path, payload["scenario_rows"])
        analysis_path.write_text(
            json.dumps(payload["analysis"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    figures = _figures(payload, args.figures_dir)
    if not current:
        comparison_path = args.results_dir / "distribution_comparison.csv"
        scenario_path = args.results_dir / "scenario_diagnostics.csv"
    manifest = {
        "inputs": expected,
        "outputs": {
            "comparison": portable_path(args.results_dir / "distribution_comparison.csv"),
            "scenarios": portable_path(args.results_dir / "scenario_diagnostics.csv"),
            "analysis": portable_path(args.results_dir / "analysis.json"),
            "figures": [portable_path(path) for path in figures],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f">> Instance diagnostics written to {args.results_dir.resolve()}", flush=True)
    if not args.quiet:
        print(json.dumps(payload["analysis"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
