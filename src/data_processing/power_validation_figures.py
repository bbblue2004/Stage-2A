"""Publication figures for the affine power-model validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .power_validation import (
    DEFAULT_RANDOM_SEED,
    NUMERICAL_TOLERANCE,
    ValidationCampaign,
    fit_affine,
    representative_antenna_id,
)


FIGURE_DPI = 300


def _save_figure(fig: plt.Figure, path_without_suffix: Path) -> tuple[Path, Path]:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path_without_suffix.with_suffix(".pdf")
    png_path = path_without_suffix.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return finite, finite
    return finite, np.arange(1, finite.size + 1) / finite.size


def _central_xlim(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lower, upper = np.quantile(finite, [0.01, 0.99])
    if abs(upper - lower) <= NUMERICAL_TOLERANCE:
        return float(lower - 1.0), float(upper + 1.0)
    margin = 0.05 * (upper - lower)
    return float(lower - margin), float(upper + margin)


def _representative_figure(
    campaign: ValidationCampaign,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[Path, Path]:
    antenna_id = representative_antenna_id(campaign.antenna_results)
    series = next(item for item in campaign.series if item.antenna_id == antenna_id)
    result = next(
        item for item in campaign.antenna_results if item.antenna_id == antenna_id
    )
    x_raw = series.traffic.reshape(-1)
    y_raw = series.power.reshape(-1)
    valid = np.isfinite(x_raw) & np.isfinite(y_raw) & (y_raw > 0.0)
    x = x_raw[valid]
    y = y_raw[valid]
    fit = fit_affine(x, y)
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)

    rng = np.random.default_rng(seed)
    bootstrap_predictions: list[np.ndarray] = []
    active_days = np.flatnonzero(
        np.any(
            np.isfinite(series.traffic)
            & np.isfinite(series.power)
            & (series.power > 0.0),
            axis=1,
        )
    )
    for _ in range(bootstrap_replicates):
        sampled_days = rng.choice(
            active_days,
            size=active_days.size,
            replace=True,
        )
        x_sample_raw = series.traffic[sampled_days].reshape(-1)
        y_sample_raw = series.power[sampled_days].reshape(-1)
        valid_sample = (
            np.isfinite(x_sample_raw)
            & np.isfinite(y_sample_raw)
            & (y_sample_raw > 0.0)
        )
        x_sample = x_sample_raw[valid_sample]
        y_sample = y_sample_raw[valid_sample]
        if np.ptp(x_sample) <= NUMERICAL_TOLERANCE:
            continue
        sample_fit = fit_affine(x_sample, y_sample)
        bootstrap_predictions.append(
            sample_fit.f_tilde + sample_fit.gamma_tilde * grid
        )
    prediction_matrix = np.asarray(bootstrap_predictions, dtype=float)
    lower, upper = np.quantile(prediction_matrix, [0.025, 0.975], axis=0)

    mask = campaign.diagnostics.antenna_id == antenna_id
    cv_traffic = campaign.diagnostics.traffic[mask]
    cv_residual = (
        campaign.diagnostics.observed[mask]
        - campaign.diagnostics.predicted_affine[mask]
    )
    cv_rank = campaign.diagnostics.relative_traffic_rank[mask]
    cv_extrapolation = campaign.diagnostics.is_extrapolation[mask]
    decile = np.minimum(9, np.floor(10.0 * cv_rank).astype(int))
    bin_x = np.asarray(
        [np.median(cv_traffic[decile == index]) for index in range(10)]
    )
    bin_median = np.asarray(
        [np.median(cv_residual[decile == index]) for index in range(10)]
    )
    bin_q25 = np.asarray(
        [np.quantile(cv_residual[decile == index], 0.25) for index in range(10)]
    )
    bin_q75 = np.asarray(
        [np.quantile(cv_residual[decile == index], 0.75) for index in range(10)]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    colors = plt.cm.tab10(np.linspace(0.0, 0.8, series.num_days))
    for day_index, day in enumerate(series.days):
        valid_day = (
            np.isfinite(series.traffic[day_index])
            & np.isfinite(series.power[day_index])
            & (series.power[day_index] > 0.0)
        )
        axes[0].scatter(
            series.traffic[day_index, valid_day],
            series.power[day_index, valid_day],
            s=17,
            alpha=0.72,
            color=colors[day_index],
            label=day.strftime("%d/%m"),
        )
    axes[0].plot(
        grid,
        fit.f_tilde + fit.gamma_tilde * grid,
        color="black",
        linewidth=1.8,
        label=f"OLS sur {active_days.size} jours",
    )
    axes[0].fill_between(
        grid,
        lower,
        upper,
        color="black",
        alpha=0.14,
        label="IC bootstrap 95 %",
    )
    axes[0].set(
        xlabel="Trafic descendant (Go/h)",
        ylabel="Puissance moyenne (W)",
        title=f"(a) Antenne représentative {antenna_id}",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].scatter(
        cv_traffic,
        cv_residual,
        s=15,
        alpha=0.42,
        color="#4C72B0",
        label="Résidus hors échantillon",
    )
    axes[1].scatter(
        cv_traffic[cv_extrapolation],
        cv_residual[cv_extrapolation],
        s=55,
        marker="x",
        linewidths=1.5,
        color="#DD8452",
        label="Extrapolation hors support d'apprentissage",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].plot(
        bin_x,
        bin_median,
        color="#C44E52",
        marker="o",
        linewidth=1.8,
        label="Médiane par décile",
    )
    axes[1].fill_between(
        bin_x,
        bin_q25,
        bin_q75,
        color="#C44E52",
        alpha=0.18,
        label="Intervalle interquartile",
    )
    axes[1].set(
        xlabel="Trafic descendant (Go/h)",
        ylabel="Observation - prédiction (W)",
        title=(
            "(b) Résidus leave-one-day-out\n"
            rf"$R^2_{{CV}}={result.cv_r_squared:.3f}$, "
            rf"NRMSE={100.0 * result.cv_nrmse_affine:.1f}\%"
        ),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, output_dir / "power_model_representative")


def _cross_validation_figure(
    campaign: ValidationCampaign,
    output_dir: Path,
) -> tuple[Path, Path]:
    diagnostics = campaign.diagnostics
    observed = diagnostics.observed / diagnostics.antenna_mean_power
    predicted = diagnostics.predicted_affine / diagnostics.antenna_mean_power
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    axis_lower, axis_upper = np.quantile(
        np.concatenate((observed, predicted)), [0.005, 0.995]
    )

    results = [
        result
        for result in campaign.antenna_results
        if np.isfinite(result.cv_rmse_gain_affine)
    ]
    gains = {
        "Affine libre": np.asarray(
            [result.cv_rmse_gain_affine for result in results]
        ),
        "Affine contrainte": np.asarray(
            [result.cv_rmse_gain_nonnegative for result in results]
        ),
        "Quadratique": np.asarray(
            [result.cv_rmse_gain_quadratic for result in results]
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    density = axes[0].hexbin(
        observed,
        predicted,
        gridsize=75,
        bins="log",
        mincnt=1,
        cmap="viridis",
        extent=(axis_lower, axis_upper, axis_lower, axis_upper),
    )
    axes[0].plot(
        [axis_lower, axis_upper],
        [axis_lower, axis_upper],
        color="white",
        linewidth=1.4,
        linestyle="--",
    )
    axes[0].set(
        xlim=(axis_lower, axis_upper),
        ylim=(axis_lower, axis_upper),
        xlabel="Puissance observée / moyenne de l'antenne",
        ylabel="Puissance prédite / moyenne de l'antenne",
        title="(a) Prédictions leave-one-day-out",
    )
    fig.colorbar(density, ax=axes[0], label="Nombre d'observations (échelle log)")

    combined = np.concatenate(tuple(gains.values()))
    xlim = _central_xlim(combined)
    for label, values in gains.items():
        x, probability = _ecdf(values)
        axes[1].plot(x, probability, linewidth=1.8, label=label)
    axes[1].axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set(
        xlim=xlim,
        xlabel=(
            r"Gain de RMSE : $1-\mathrm{RMSE}_{modèle}/"
            r"\mathrm{RMSE}_{constant}$"
        ),
        ylabel="Fonction de répartition empirique",
        title="(b) Comparaison appariée par antenne (axe q1--q99)",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir / "power_model_cross_validation")


def _population_figure(
    campaign: ValidationCampaign,
    output_dir: Path,
) -> tuple[Path, Path]:
    analyzable = [
        result
        for result in campaign.antenna_results
        if np.isfinite(result.cv_r_squared)
    ]
    operational = [
        result for result in analyzable if result.status == "included"
    ]
    train_r2 = np.asarray([result.r_squared_train for result in analyzable])
    cv_r2 = np.asarray([result.cv_r_squared for result in analyzable])
    fixed_share = np.asarray(
        [result.fixed_power_share for result in operational], dtype=float
    )

    diagnostics = campaign.diagnostics
    normalized_residual = (
        diagnostics.observed - diagnostics.predicted_affine
    ) / diagnostics.antenna_mean_power
    traffic_decile = np.minimum(
        9, np.floor(10.0 * diagnostics.relative_traffic_rank).astype(int)
    )
    residual_matrix = np.full((10, 24), np.nan)
    for decile in range(10):
        for hour in range(24):
            mask = (traffic_decile == decile) & (diagnostics.hour == hour)
            if np.any(mask):
                residual_matrix[decile, hour] = np.median(
                    normalized_residual[mask]
                )

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))
    x, probability = _ecdf(train_r2)
    axes[0, 0].plot(x, probability, linewidth=1.8, color="#4C72B0")
    axes[0, 0].set(
        xlim=(0.0, 1.0),
        xlabel=r"$R^2$ d'ajustement",
        ylabel="Fonction de répartition empirique",
        title="(a) Ajustement sur les sept jours",
    )
    axes[0, 0].grid(alpha=0.25)

    x, probability = _ecdf(cv_r2)
    axes[0, 1].plot(x, probability, linewidth=1.8, color="#55A868")
    axes[0, 1].axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[0, 1].set(
        xlim=_central_xlim(cv_r2),
        xlabel=r"$R^2_{\mathrm{CV}}$ relatif au modèle constant",
        ylabel="Fonction de répartition empirique",
        title="(b) Généralisation interjournalière (axe q1--q99)",
    )
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].hist(
        fixed_share[np.isfinite(fixed_share)],
        bins=np.linspace(0.0, 1.0, 31),
        color="#C44E52",
        alpha=0.82,
        edgecolor="white",
    )
    axes[1, 0].set(
        xlim=(0.0, 1.0),
        xlabel=r"Part fixe $\widetilde F/(\widetilde F+\widetilde\gamma\bar d)$",
        ylabel="Nombre d'antennes",
        title="(c) Part fixe de la puissance active",
    )
    axes[1, 0].grid(axis="y", alpha=0.25)

    scale = float(np.nanmax(np.abs(residual_matrix)))
    image = axes[1, 1].imshow(
        residual_matrix,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
        extent=(-0.5, 23.5, 0.0, 10.0),
    )
    axes[1, 1].set(
        xlabel="Heure",
        ylabel="Décile de trafic intra-antenne",
        title="(d) Résidu CV médian / puissance moyenne",
        xticks=range(0, 24, 3),
    )
    fig.colorbar(image, ax=axes[1, 1], label="Résidu normalisé")
    fig.tight_layout()
    return _save_figure(fig, output_dir / "power_model_population")


def generate_validation_figures(
    campaign: ValidationCampaign,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, tuple[Path, Path]]:
    return {
        "representative": _representative_figure(
            campaign, output_dir, bootstrap_replicates, seed
        ),
        "cross_validation": _cross_validation_figure(campaign, output_dir),
        "population": _population_figure(campaign, output_dir),
    }
