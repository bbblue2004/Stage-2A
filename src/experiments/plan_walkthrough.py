"""Generate the worked protocol example and the three traffic-shape cases.

The figures use the first frozen plan and the same materialisation code as the
Section 6 experiments. They therefore remain aligned with the numerical
pipeline instead of reimplementing the construction independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_processing.instance_generator import (
    ScenarioSpec,
    campaign_a_capacities,
    materialize_site,
    normalised_shapes,
)
from src.experiments.protocol_io import load_protocol_inputs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_DIR = ROOT / "results" / "power_calibration"
DEFAULT_FIGURE_DIR = ROOT / "figures" / "protocol"
SHAPE_LEVELS = (
    ("close", "profils proches"),
    ("moderate", "profils intermédiaires"),
    ("distant", "profils différents"),
)
COLOURS = ("#1F4E79", "#C0504D", "#6E8B3D", "#8064A2")


def _scenario(shape_level: str, rate: float) -> ScenarioSpec:
    return ScenarioSpec(
        "A",
        "moderate",
        shape_level,
        "close",
        capacity_rate=rate,
    )


def _operator_shapes(site) -> np.ndarray:
    return site.traffic_gb / (site.mu_gb * site.alpha[:, None, None])


def _daily_mean(series: np.ndarray) -> np.ndarray:
    return np.mean(series, axis=0)


def _correlations_with_reference(population, blueprints, site_index, shapes) -> np.ndarray:
    empirical = normalised_shapes(population.traffic_gb)
    reference = empirical[int(blueprints.reference_index[site_index])]
    return np.asarray(
        [np.corrcoef(reference, row.reshape(-1))[0, 1] for row in shapes],
        dtype=float,
    )


def _plot_walkthrough(
    site,
    population,
    blueprints,
    protocol,
    site_index: int,
    rate: float,
    figure_dir: Path,
) -> None:
    _, n_days, _ = site.traffic_gb.shape
    slots = np.arange(n_days * 24)
    clock = np.arange(24)
    reference = int(blueprints.reference_index[site_index])
    peaks = site.peak_traffic_gb

    day_ticks = 12 + 24 * np.arange(n_days)
    day_labels = [f"jour {day + 1}" for day in range(n_days)]

    figure, axes = plt.subplots(2, 3, figsize=(14.5, 7.0))
    axes[0, 0].plot(
        slots,
        population.traffic_gb[reference].reshape(-1),
        color=COLOURS[0],
        linewidth=1.0,
    )
    axes[0, 0].set_title("(a) trafic mesuré de la référence", fontsize=10.0)
    axes[0, 0].set_ylabel("trafic (Go)")
    axes[0, 0].set_xlabel("jour observé")
    axes[0, 0].set_xticks(day_ticks, day_labels, rotation=45, ha="right", fontsize=7)

    shape_axes = (axes[0, 1], axes[0, 2], axes[1, 0])
    for panel_index, ((level, title), axis) in enumerate(
        zip(SHAPE_LEVELS, shape_axes, strict=True), start=1
    ):
        level_site = materialize_site(
            blueprints,
            site_index,
            population,
            _scenario(level, rate),
            protocol,
        )
        shapes = _operator_shapes(level_site)
        for operator, colour in enumerate(COLOURS):
            axis.plot(
                clock,
                _daily_mean(shapes[operator]),
                color=colour,
                linewidth=1.35,
                label=f"opérateur {operator + 1}",
            )
        panel = "bcd"[panel_index - 1]
        axis.set_title(f"({panel}) {title}", fontsize=10.0)
        axis.set_xlabel("heure de la journée")
        axis.set_xticks((0, 6, 12, 18, 23))
        axis.set_ylabel("trafic normalisé (moyenne 1)")
    axes[0, 1].legend(fontsize=7.0, loc="upper left")

    for operator, colour in enumerate(COLOURS):
        axes[1, 1].plot(
            slots,
            site.traffic_gb[operator].reshape(-1),
            color=colour,
            linewidth=1.0,
            label=f"opérateur {operator + 1}",
        )
    axes[1, 1].set_title("(e) demandes construites", fontsize=10.0)
    axes[1, 1].set_ylabel("trafic (Go)")
    axes[1, 1].set_xlabel("jour observé")
    axes[1, 1].set_xticks(day_ticks, day_labels, rotation=45, ha="right", fontsize=7)

    for operator, colour in enumerate(COLOURS):
        relative = site.traffic_gb[operator] / peaks[operator]
        axes[1, 2].plot(
            clock,
            np.mean(relative, axis=0),
            color=colour,
            linewidth=1.5,
            label=f"opérateur {operator + 1}",
        )
        axes[1, 2].fill_between(
            clock,
            np.min(relative, axis=0),
            np.max(relative, axis=0),
            color=colour,
            alpha=0.10,
            linewidth=0,
        )
    for rate, style in ((1.00, "-"), (0.90, "--"), (0.80, ":")):
        axes[1, 2].axhline(1.0 / rate, color="black", linewidth=0.9, linestyle=style)
        axes[1, 2].annotate(
            f"capacité pour r = {rate:.2f}",
            (23.5, 1.0 / rate + 0.012),
            fontsize=7,
            ha="right",
            va="bottom",
        )
    axes[1, 2].set_title("(f) demande rapportée au pic", fontsize=10.0)
    axes[1, 2].set_ylabel("part du pic de l’opérateur")
    axes[1, 2].set_xlabel("heure de la journée")
    axes[1, 2].set_ylim(0.0, 1.40)

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.24, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(
        figure_dir / "plan_walkthrough.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def _print_values(site, population, blueprints, site_index: int, rate: float) -> None:
    shapes = _operator_shapes(site)
    correlations = _correlations_with_reference(
        population, blueprints, site_index, shapes
    )
    capacities = campaign_a_capacities(site.peak_traffic_gb, rate)
    print(f"plan: {site.site_id}")
    print(f"référence: {site.reference_id}")
    print(f"trafic moyen de référence: {site.mu_gb:.4f} Go")
    print("donneuses: " + ", ".join(site.donor_ids))
    print("équipements: " + ", ".join(site.energy_ids))
    for operator in range(site.alpha.size):
        print(
            f"opérateur {operator + 1}: alpha={site.alpha[operator]:.3f}, "
            f"corrélation={correlations[operator]:.3f}, "
            f"P_fixe={site.p_fixed_w[operator]:.1f} W, "
            f"pente={site.slope_w_per_gb[operator]:.2f} W/Go, "
            f"moyenne={site.traffic_gb[operator].mean():.2f} Go, "
            f"pic={site.peak_traffic_gb[operator]:.2f} Go, "
            f"capacité={capacities[operator]:.2f} Go"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--site-index", type=int, default=0)
    parser.add_argument("--rate", type=float, default=0.90)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    population, blueprints, protocol = load_protocol_inputs(args.calibration_dir)
    if not 0 <= args.site_index < blueprints.num_sites:
        raise ValueError("site-index is outside the frozen plan list")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    site = materialize_site(
        blueprints,
        args.site_index,
        population,
        _scenario("moderate", args.rate),
        protocol,
    )
    _plot_walkthrough(
        site,
        population,
        blueprints,
        protocol,
        args.site_index,
        args.rate,
        args.figure_dir,
    )
    if not args.quiet:
        _print_values(site, population, blueprints, args.site_index, args.rate)
    print(f">> Protocol figure: {args.figure_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
