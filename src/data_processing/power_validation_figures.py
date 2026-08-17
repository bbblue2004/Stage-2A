"""Publication figure for the direct affine-power calibration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from .power_validation import CalibratedPopulation, CalibrationCampaign


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    probability = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    return ordered, probability


def _draw_distribution(
    axis: plt.Axes,
    values: np.ndarray,
    xlabel: str,
    title: str,
    color: str,
    percentage: bool = False,
) -> None:
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    x, probability = _ecdf(values)
    axis.plot(x, probability, color=color, linewidth=2.0)
    median_label = f"{100.0 * median:.1f} %" if percentage else f"{median:.3f}"
    axis.axvspan(q25, q75, color=color, alpha=0.13, label="50 % central")
    axis.axvline(
        median,
        color=color,
        linestyle="--",
        linewidth=1.5,
        label=f"Médiane : {median_label}",
    )
    axis.set(
        xlabel=xlabel,
        ylabel="Proportion cumulée d'antennes",
        title=title,
        ylim=(0.0, 1.01),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    if percentage:
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def generate_calibration_figure(
    campaign: CalibrationCampaign,
    output_dir: Path,
) -> tuple[Path, Path]:
    included = [
        result for result in campaign.antenna_results if result.status == "included"
    ]
    if not included:
        raise ValueError("No included antenna is available for plotting")
    r_squared = np.asarray([result.r_squared for result in included], dtype=float)
    normalized_rmse = np.asarray(
        [result.normalized_rmse for result in included], dtype=float
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    _draw_distribution(
        axes[0],
        r_squared,
        r"Coefficient de détermination $R_i^2$",
        "(a) Variance de puissance décrite par le trafic",
        "#3568A8",
    )
    axes[0].set_xlim(0.0, 1.0)
    _draw_distribution(
        axes[1],
        normalized_rmse,
        "RMSE / puissance active moyenne",
        "(b) Erreur relative d'ajustement",
        "#C44E52",
        percentage=True,
    )
    axes[1].set_xlim(left=0.0)
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "power_model_calibration.pdf"
    png_path = output_dir / "power_model_calibration.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def generate_representative_fit_figure(
    population: CalibratedPopulation,
    output_dir: Path,
) -> tuple[tuple[Path, Path], str]:
    """Illustrate the fit nearest to the population-median normalized RMSE."""
    median_error = float(np.median(population.normalized_rmse))
    index = int(np.argmin(np.abs(population.normalized_rmse - median_error)))
    traffic = population.traffic_gb[index].reshape(-1)
    power = population.power_w[index].reshape(-1)
    active = np.isfinite(traffic) & np.isfinite(power) & (power > 0.0)
    traffic, power = traffic[active], power[active]
    grid = np.linspace(float(np.min(traffic)), float(np.max(traffic)), 200)
    prediction = (
        population.p_fixed_w[index]
        + population.slope_w_per_gb[index] * grid
    )

    figure, axis = plt.subplots(figsize=(4.8, 3.25))
    axis.scatter(
        traffic,
        power,
        s=14,
        alpha=0.45,
        color="#5B7FA3",
        edgecolors="none",
        label="Observations actives",
    )
    axis.plot(grid, prediction, color="#B6423C", linewidth=2.0, label="Ajustement affine")
    axis.set_xlabel("Trafic descendant horaire (Go)")
    axis.set_ylabel("Puissance moyenne (W)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    axis.text(
        0.03,
        0.96,
        f"RMSE normalisée : {100.0 * population.normalized_rmse[index]:.1f} %"
        + "\n"
        + rf"$R^2$ : {population.r_squared[index]:.2f}",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "representative_power_fit.pdf",
        output_dir / "representative_power_fit.png",
    )
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths, str(population.antenna_ids[index])
