"""Run the RAN-sharing simulation over a configurable hour window."""

import argparse

from src.core.generate_data import DEFAULT_ELECTRICITY_PRICE_PER_KWH, load_scenario
from src.core.reporting import run_report
from src.core.simulation import ALLOCATION_PRIORITIES
from src.core.time_window import (
    DEFAULT_END_HOUR,
    DEFAULT_START_HOUR,
    inclusive_hour_window,
)
from src.data_processing.data_loader import first_antenna_ids
from src.data_processing.figures import plot_weekly_traffic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAN-sharing savings over selected hours.",
    )
    parser.add_argument(
        "antenna",
        nargs="?",
        type=int,
        default=1,
        help="1-based index among the first 10 antennas (default: 1)",
    )
    parser.add_argument(
        "--traffic-mode",
        choices=("average", "daily"),
        default="average",
        help="five-day hourly average, or one of the first four days per operator",
    )
    parser.add_argument(
        "--price-per-kwh",
        type=float,
        default=None,
        help="Electricity price (currency/kWh) used to convert F and gamma",
    )
    parser.add_argument(
        "--hours",
        nargs=2,
        type=int,
        default=(DEFAULT_START_HOUR, DEFAULT_END_HOUR),
        metavar=("START", "END"),
        help=(
            "inclusive study window, with overnight wrap allowed "
            f"(default: {DEFAULT_START_HOUR} {DEFAULT_END_HOUR})"
        ),
    )
    parser.add_argument(
        "--allocation-priority",
        choices=ALLOCATION_PRIORITIES,
        default="contribution",
        help=(
            "choose the Shapley-nearest stable allocation, or the nucleolus "
            "when Shapley is outside the core (default: contribution)"
        ),
    )
    parser.add_argument(
        "--max-instability-ratio",
        type=float,
        default=0.01,
        help=(
            "maximum acceptable least-core epsilon / v(N) "
            "(default: 0.01)"
        ),
    )
    parser.add_argument(
        "--plot-weekly-traffic",
        action="store_true",
        help="save a graph overlaying the antenna's first seven daily profiles",
    )
    args = parser.parse_args()

    try:
        hours = inclusive_hour_window(*args.hours)
    except ValueError as error:
        parser.error(str(error))
    if args.max_instability_ratio < 0.0:
        parser.error("--max-instability-ratio must be non-negative")

    antenna_ids = first_antenna_ids(10)
    if not 1 <= args.antenna <= len(antenna_ids):
        parser.error(f"antenna must be between 1 and {len(antenna_ids)}")
    antenna_id = antenna_ids[args.antenna - 1]

    if args.plot_weekly_traffic:
        figure_path = plot_weekly_traffic(antenna_id)
        print(f">> Weekly traffic figure: {figure_path}", flush=True)

    print(">> Loading scenario...", flush=True)
    scenario = load_scenario(
        antenna_id=antenna_id,
        traffic_mode=args.traffic_mode,
        price_per_kwh=args.price_per_kwh or DEFAULT_ELECTRICITY_PRICE_PER_KWH,
    )
    print(
        f">> Antenna {args.antenna}: {antenna_id}; mode: {args.traffic_mode}.",
        flush=True,
    )
    run_report(
        scenario,
        hours,
        allocation_priority=args.allocation_priority,
        max_instability_ratio=args.max_instability_ratio,
    )


if __name__ == "__main__":
    main()
