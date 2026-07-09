import argparse

from src.core.diagnostics import verify_super_additivity
from src.core.generate_data import load_scenario
from src.core.reporting import run_simulation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="RAN sharing simulation entrypoint.")
    parser.add_argument("--scenario", default="realistic", choices=["example", "realistic"])
    parser.add_argument(
        "--mode",
        default="simulate",
        choices=["simulate", "verify", "all"],
        help="simulate: run comparison report, verify: only super-additivity, all: both",
    )
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--safety-margin", type=float, default=0.15)
    parser.add_argument("--debug-verify", action="store_true")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario, step_minutes=args.step_minutes)
    num_operators = len(scenario.operators)

    if args.mode in {"simulate", "all"}:
        run_simulation_report(
            scenario=scenario,
            safety_margin=args.safety_margin,
            window_size=args.window_size,
        )

    if args.mode in {"verify", "all"}:
        print("\n" + "=" * 60)
        print("Verifying Super-Additivity Property...")
        print("=" * 60)
        verify_super_additivity(
            scenario.operators,
            scenario.traffic,
            max_coalition_size=num_operators,
            test_times=scenario.sample_steps(7),
            debug=args.debug_verify,
        )


if __name__ == "__main__":
    main()