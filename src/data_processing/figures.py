"""Power-vs-traffic regression figure."""

from collections import defaultdict
from datetime import date
from math import nan
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from . import data_loader
from .antenna_metrics import load_antenna_profiles, power_regression_from_profiles


def plot_weekly_traffic(antenna_id: str, output: Path | None = None) -> Path:
    """Overlay the first seven daily traffic profiles of one antenna."""
    rows = data_loader.extract_antenna_time_series(antenna_id)
    traffic_by_day: dict[date, dict[int, float]] = defaultdict(dict)
    for timestamp, traffic, _ in rows:
        traffic_by_day[timestamp.date()][timestamp.hour] = traffic

    days = sorted(traffic_by_day)[:7]
    if not days:
        raise ValueError(f"No traffic data for antenna {antenna_id}")

    fig, ax = plt.subplots(figsize=(10, 6))
    day_names = ("lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.")
    for day in days:
        values = [traffic_by_day[day].get(hour, nan) for hour in range(24)]
        ax.plot(
            range(24),
            values,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=f"{day_names[day.weekday()]} {day:%d/%m}",
        )

    ax.set(
        xlabel="Heure de la journée",
        ylabel="Trafic descendant (Go par heure)",
        title=f"Profils journaliers de trafic — antenne {antenna_id}",
        xticks=range(0, 24, 2),
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Jour", ncol=2)
    fig.tight_layout()

    path = output or data_loader.make_output_path(
        f"weekly_traffic_{antenna_id}.png"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_power_vs_traffic(antenna_id: str, output: Path | None = None) -> Path:
    traffic, power = load_antenna_profiles(antenna_id)
    fit = power_regression_from_profiles(traffic, power)
    x = sorted(traffic)
    y = [fit.f_tilde + fit.gamma_tilde * demand for demand in x]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(traffic, power, alpha=0.65, s=24, label=f"{antenna_id} data")
    ax.plot(
        x,
        y,
        linewidth=2,
        label=(
            rf"$P_{{\mathrm{{conso}}}}={fit.f_tilde:.2f}"
            rf"+{fit.gamma_tilde:.2f}d$  ($R^2={fit.r_squared:.3f}$)"
        ),
    )
    ax.set(
        xlabel="Traffic d (GB)",
        ylabel="Power (W)",
        title=(
            r"Power regression: "
            r"$P_{\mathrm{conso}}=\widetilde{F}+\widetilde{\gamma}d$"
        ),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = output or data_loader.make_output_path(f"power_regression_{antenna_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
