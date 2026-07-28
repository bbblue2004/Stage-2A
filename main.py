"""Run the 24-hour RAN-sharing simulation."""

import argparse

from src.core.generate_data import DEFAULT_ELECTRICITY_PRICE_PER_KWH, load_scenario
from src.core.reporting import run_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAN-sharing savings over one day.",
    )
    parser.add_argument(
        "--antenna-id",
        default=None,
        help="SYS.NIDT antenna used for operator 1",
    )
    parser.add_argument(
        "--price-per-kwh",
        type=float,
        default=None,
        help="Electricity price (currency/kWh) used to convert F and gamma",
    )
    args = parser.parse_args()

    print(">> Loading scenario...", flush=True)
    scenario = load_scenario(
        antenna_id=args.antenna_id,
        price_per_kwh=args.price_per_kwh or DEFAULT_ELECTRICITY_PRICE_PER_KWH,
    )
    print(f">> Scenario ready ({scenario.data_source}).", flush=True)
    run_report(scenario)


if __name__ == "__main__":
    main()
