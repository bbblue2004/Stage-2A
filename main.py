"""RAN sharing cooperative game — entry point."""

import argparse

from src.core.generate_data import DEFAULT_ELECTRICITY_PRICE_PER_KWH, load_scenario
from src.core.reporting import run_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAN-sharing cooperative gains over one day.",
    )
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--antenna-id", default=None, help="SYS.NIDT antenna for operator 1")
    parser.add_argument(
        "--price-per-kwh",
        type=float,
        default=None,
        help="Electricity price (currency/kWh) for beta/K conversion from CSV power data",
    )
    args = parser.parse_args()

    print(">> Loading scenario...", flush=True)
    scenario = load_scenario(
        step_minutes=args.step_minutes,
        antenna_id=args.antenna_id,
        price_per_kwh=args.price_per_kwh or DEFAULT_ELECTRICITY_PRICE_PER_KWH,
    )
    print(f">> Scenario ready ({scenario.data_source}).", flush=True)

    run_report(scenario)


if __name__ == "__main__":
    main()
