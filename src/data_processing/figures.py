"""
Exploratory figures from radio_sites.csv.

Run: python -m src.data_processing.figures [--antenna-id ID] [--num N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from . import data_loader
from .antenna_metrics import compute_hourly_traffic_profile


def plot_hourly_traffic_profile(antenna_id: str, output: Path | None = None) -> Path:
    profile = compute_hourly_traffic_profile(antenna_id)
    hours = list(range(24))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, profile, marker="o", linewidth=2)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean traffic (GB)")
    ax.set_title(f"Weekday-type hourly profile — {antenna_id}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output or data_loader.make_output_path(f"hourly_traffic_{antenna_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_traffic_series(antenna_ids: list[str], output: Path | None = None) -> Path:
    ids, data = data_loader.extract_ids_data(ids=antenna_ids)
    fig, ax = plt.subplots(figsize=(12, 6))
    for nid in ids:
        ax.plot(data[nid]["datetime"], data[nid]["traffic"], marker=".", ms=3, label=nid)
    ax.set_title("Traffic time series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic (GB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:00"))
    fig.autofmt_xdate()
    fig.tight_layout()

    name = "traffic_single.png" if len(ids) == 1 else "traffic_multi.png"
    path = output or data_loader.make_output_path(name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_power_series(antenna_ids: list[str], output: Path | None = None) -> Path:
    ids, data = data_loader.extract_ids_data(ids=antenna_ids, include_power=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for nid in ids:
        ax.plot(data[nid]["datetime"], data[nid]["power"], marker=".", ms=3, label=nid)
    ax.set_title("Power consumption time series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:00"))
    fig.autofmt_xdate()
    fig.tight_layout()

    name = "power_single.png" if len(ids) == 1 else "power_multi.png"
    path = output or data_loader.make_output_path(name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_power_vs_rho(antenna_ids: list[str], output: Path | None = None) -> Path:
    ids, data = data_loader.extract_ids_data(ids=antenna_ids, include_power=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for nid in ids:
        rho = data_loader.compute_rho_from_traffic(data[nid]["traffic"])
        ax.scatter(rho, data[nid]["power"], alpha=0.5, s=12, label=nid)
    ax.set_xlabel("ρ (normalized traffic)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Power vs normalized traffic")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output or data_loader.make_output_path("power_vs_rho.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exploratory CSV figures.")
    parser.add_argument("--antenna-id", default=None)
    parser.add_argument("--num", type=int, default=1, help="Number of antennas when --antenna-id omitted")
    args = parser.parse_args()

    if not data_loader.CSV_PATH.is_file():
        raise SystemExit(f"CSV not found: {data_loader.CSV_PATH}")

    antenna_id = args.antenna_id or data_loader.DEFAULT_ANTENNA_ID
    ids = [antenna_id] if args.antenna_id else None
    if ids is None:
        ids, _ = data_loader.extract_ids_data(n=args.num)

    for path in (
        plot_hourly_traffic_profile(antenna_id),
        plot_traffic_series(ids),
        plot_power_series(ids),
        plot_power_vs_rho(ids),
    ):
        print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
