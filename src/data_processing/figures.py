"""Power-vs-traffic regression figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from . import data_loader
from .antenna_metrics import load_antenna_profiles, power_regression_from_profiles


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
