"""Console report for the daily simulation."""

from typing import Any

from src.core.generate_data import Scenario
from src.core.simulation import evaluate_day

WIDTH = 94


def _header(scenario: Scenario) -> None:
    sources = {
        "compact_csv": "radio_sites_10x7.csv",
        "csv": "radio_sites.csv",
        "fallback": "hard-coded fallback",
    }
    print("=" * WIDTH)
    print("RAN sharing - operational costs and cooperative savings")
    print("=" * WIDTH)
    print(f"Site              : {scenario.antenna_id}")
    print(f"Traffic source    : {sources[scenario.data_source]}")
    print("Traffic profile   : five-day hourly means")
    print(f"Horizon           : {scenario.num_hours} hourly periods")
    print("Game              : v(S) = sum_i C_i^0 - C*(S)")

    fit = scenario.power_regression
    if fit:
        print(
            f"\nRegression        : P_conso = {fit.f_tilde:.4f} "
            f"+ {fit.gamma_tilde:.4f} d    R^2 = {fit.r_squared:.6f}"
        )

    print(f"\n{'Op':<4} {'q_i (GB)':>11} {'F_i':>14} {'gamma_i':>14}")
    print("-" * 48)
    for i, operator in enumerate(scenario.operators, 1):
        print(
            f"{i:<4} {operator.q:>11.3f} "
            f"{operator.F:>14.6f} {operator.gamma:>14.6f}"
        )


def _hourly_table(result: dict[str, Any]) -> None:
    print("\nHourly optimal cost, cooperative savings and guardians (^ = change)")
    print("-" * WIDTH)
    print(
        f"{'Hour':>5} | {'sum C_i^0':>12} | {'C*(N)':>12} | "
        f"{'v(N)':>12} | {'Chg':>3} | Guardians"
    )
    print("-" * WIDTH)
    for row in result["hourly"]:
        changed = "^" if row["guardians_changed"] else ""
        print(
            f"{row['hour']:02d}:00 | {row['standalone_cost']:>12.6f} | "
            f"{row['coalition_cost']:>12.6f} | {row['savings']:>+12.6f} | "
            f"{changed:>3} | {row['guardians_label']}"
        )

    peak = max(result["hourly"], key=lambda row: row["savings"])
    print("-" * WIDTH)
    print(
        f"Peak saving at {peak['hour']:02d}:00 ({peak['savings']:+.6f}); "
        f"guardians {peak['guardians_label']}"
    )
    print(f"Guardian changes: {result['guardian_changes']}")
    print(f"Total daily saving v(N): {result['total_savings']:+.6f}")


def _daily_summary(scenario: Scenario, result: dict[str, Any]) -> None:
    core, test = result["core_summary"], result["bondareva_summary"]
    print(
        f"\nCore LP: {core['status']}.  "
        f"Bondareva--Shapley: B(N)={test['balanced_value']:.8f}, "
        f"v(N)={test['grand_value']:.8f}, gap={test['gap']:+.3e}."
    )
    conclusion = {
        True: "The core is non-empty.",
        False: "The core is empty.",
        None: "The balancedness test is inconclusive.",
    }
    print(conclusion[test["core_nonempty"]])
    if not core["feasible"]:
        return

    standalone = result["standalone_costs"]
    physical = result["physical_costs"]
    savings = result["savings_allocation"]
    net = result["net_costs"]
    transfers = result["transfers"]
    print("\nDaily accounting per operator")
    print("-" * WIDTH)
    print(
        f"{'Op':<4} {'C_i^0':>14} {'C_i^*(N)':>14} {'z_i':>14} "
        f"{'y_i':>14} {'tau_i':>14}"
    )
    print("-" * WIDTH)
    for i in scenario.coalition:
        print(
            f"{i + 1:<4} {standalone[i]:>14.6f} {physical[i]:>14.6f} "
            f"{savings[i]:>14.6f} {net[i]:>14.6f} {transfers[i]:>+14.6f}"
        )
    print("-" * WIDTH)
    print(
        f"{'Tot':<4} {sum(standalone.values()):>14.6f} "
        f"{sum(physical.values()):>14.6f} {sum(savings.values()):>14.6f} "
        f"{sum(net.values()):>14.6f} {sum(transfers.values()):>+14.3e}"
    )
    print(f"Budget residual: {result['budget_residual']:+.3e}")


def run_report(scenario: Scenario) -> dict[str, Any]:
    _header(scenario)
    print("\n>> Computing the 24 hourly games...")
    result = evaluate_day(scenario)
    _hourly_table(result)
    _daily_summary(scenario, result)
    return result
