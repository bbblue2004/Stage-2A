"""Figure d'ajustement affine utilisée dans la partie 6.2."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .power_validation import CalibratedPopulation


def _representative_indices(population: CalibratedPopulation) -> tuple[int, ...]:
    targets = np.quantile(population.normalized_rmse, (0.25, 0.50, 0.90))
    available = set(range(population.antenna_ids.size))
    selected: list[int] = []
    for target in targets:
        index = min(
            available,
            key=lambda candidate: abs(
                float(population.normalized_rmse[candidate]) - float(target)
            ),
        )
        selected.append(index)
        available.remove(index)
    return tuple(selected)


def generate_representative_fit_figure(
    population: CalibratedPopulation,
    output_dir: Path,
) -> tuple[tuple[Path, ...], str]:
    """Tracer trois antennes aux quantiles 25, 50 et 90 de l'erreur."""
    indices = _representative_indices(population)
    labels = ("(a) Bon ajustement", "(b) Ajustement médian", "(c) Ajustement médiocre")
    figure, axes = plt.subplots(1, 3, figsize=(9.4, 3.25), sharey=False)

    for axis, index, label in zip(axes, indices, labels, strict=True):
        traffic = population.traffic_gb[index]
        # Chaque observation couvre exactement une heure : la puissance
        # moyenne divisée par 1000 est donc l'énergie du créneau en kWh.
        energy_kwh = population.power_w[index] / 1000.0
        active = (
            np.isfinite(traffic)
            & np.isfinite(energy_kwh)
            & (energy_kwh > 0.0)
        )
        night = np.broadcast_to(
            np.arange(traffic.shape[1]) < 7,
            traffic.shape,
        )
        for mask, color, name in (
            (active & night, "#C94C4C", "[0 h, 7 h["),
            (active & ~night, "#3677B8", "[7 h, 24 h["),
        ):
            axis.scatter(
                traffic[mask],
                energy_kwh[mask],
                s=10,
                alpha=0.42,
                color=color,
                edgecolors="none",
                label=name,
            )

        active_traffic = traffic[active]
        grid = np.linspace(
            float(np.min(active_traffic)),
            float(np.max(active_traffic)),
            200,
        )
        prediction = (
            population.p_fixed_w[index]
            + population.slope_w_per_gb[index] * grid
        ) / 1000.0
        axis.plot(grid, prediction, color="black", linewidth=1.5)
        axis.axhline(
            population.p_fixed_w[index] / 1000.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            label=r"$P^{\mathrm{fixe}}$",
        )
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("Trafic (Go)")
        axis.grid(alpha=0.22)
        axis.text(
            0.04,
            0.96,
            f"Erreur : {100.0 * population.normalized_rmse[index]:.1f} %",
            transform=axis.transAxes,
            va="top",
            fontsize=7.5,
        )

    axes[0].set_ylabel("Énergie sur le créneau horaire (kWh)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "representative_power_fit.pdf"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    identifiers = ",".join(str(population.antenna_ids[index]) for index in indices)
    return (path,), identifiers
