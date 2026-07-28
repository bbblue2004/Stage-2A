"""Generate the power-vs-traffic regression graph for one antenna."""

import argparse

from src.data_processing import data_loader
from src.data_processing.figures import plot_power_vs_traffic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot P_conso = F_tilde + gamma_tilde * d.",
    )
    parser.add_argument("--antenna-id", default=None)
    args = parser.parse_args()

    antenna_id = args.antenna_id or data_loader.first_antenna_id()
    output = data_loader.make_output_path(f"power_regression_{antenna_id}.png")
    path = plot_power_vs_traffic(antenna_id, output=output)
    print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
