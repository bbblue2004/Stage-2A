"""Console report for a selected simulation period."""

from collections.abc import Iterable
from typing import Any

from src.core.generate_data import Scenario
from src.core.simulation import evaluate_period
from src.core.time_window import (
    describe_hours,
    inclusive_hour_window,
    validate_hours,
)

WIDTH = 108


def _header(scenario: Scenario, hours: tuple[int, ...]) -> None:
    sources = {
        "compact_csv": "radio_sites_10x7.csv",
        "csv": "radio_sites.csv",
        "full_csv": "radio_sites.csv",
        "fallback": "hard-coded fallback",
    }
    print("=" * WIDTH)
    print("RAN sharing - operational costs and cooperative savings")
    print("=" * WIDTH)
    print(f"Site              : {scenario.antenna_id}")
    print(f"Traffic source    : {sources[scenario.data_source]}")
    profile = {
        "average": "five-day hourly mean + three nearby profiles",
        "daily": "first four days, one distinct day per operator",
    }[scenario.traffic_mode]
    print(f"Traffic profile   : {profile}")
    print(f"Study window      : {describe_hours(hours)}")
    print(f"Capacity basis    : maximum traffic over all {scenario.num_hours} hours")
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
    print(f"Total period saving v(N): {result['total_savings']:+.6f}")


def _period_summary(
    scenario: Scenario,
    result: dict[str, Any],
    max_instability_ratio: float,
) -> None:
    convexity = result["convexity_summary"]
    convex_label = "yes" if convexity["convex"] else "no"
    print(
        f"\nConvex game: {convex_label}; "
        f"maximum marginal-contribution violation "
        f"={convexity['max_violation']:.6g}."
    )
    if convexity["witness"] is not None:
        smaller, larger, player = convexity["witness"]

        def format_coalition(coalition: tuple[int, ...]) -> str:
            return "{" + ",".join(str(i + 1) for i in coalition) + "}"

        print(
            "Convexity witness: "
            f"S={format_coalition(smaller)}, "
            f"Q={format_coalition(larger)}, i={player + 1}."
        )

    core, test = result["core_summary"], result["bondareva_summary"]
    print(
        f"Core LP: {core['status']}.  "
        f"Bondareva--Shapley: B(N)={test['balanced_value']:.8f}, "
        f"v(N)={test['grand_value']:.8f}, gap={test['gap']:+.3e}."
    )
    conclusion = {
        True: "The core is non-empty.",
        False: "The core is empty.",
        None: "The balancedness test is inconclusive.",
    }
    print(conclusion[test["core_nonempty"]])

    least_core = result["least_core_summary"]
    print(
        f"Least-core epsilon: {least_core['epsilon']:+.8f}; "
        f"relative instability: {least_core['instability_ratio']:.4%}."
    )
    print(
        "Shapley belongs to the core: "
        + ("yes." if result["shapley_in_core"] else "no.")
    )

    print("\nSystematic allocation comparison")
    print("-" * WIDTH)
    print(
        f"{'Rule':<38} {'Core':>7} {'Max excess':>16} "
        f"{'Blocking':>10} {'Distance to Shapley':>22}"
    )
    print("-" * WIDTH)
    for candidate in result["allocation_candidates"].values():
        print(
            f"{candidate['label']:<38} "
            f"{('yes' if candidate['in_core'] else 'no'):>7} "
            f"{candidate['max_excess']:>+16.8f} "
            f"{candidate['blocking_count']:>10d} "
            f"{candidate['distance_to_shapley']:>22.8f}"
        )
    print("-" * WIDTH)
    print(
        f"Selected rule: {result['selected_allocation_label']} "
        f"[priority: {result['allocation_priority']}]."
    )
    print(result["selection_reason"])

    if not core["feasible"]:
        if least_core["instability_ratio"] <= max_instability_ratio:
            print(
                "Operational assessment: approximate cooperation is below "
                f"the accepted instability threshold "
                f"({max_instability_ratio:.2%})."
            )
        else:
            print(
                "Operational assessment: do not implement the grand "
                "coalition without changing the window or coalition; "
                f"the instability exceeds {max_instability_ratio:.2%}."
            )

    standalone = result["standalone_costs"]
    physical = result["physical_costs"]
    savings = result["savings_allocation"]
    net = result["net_costs"]
    transfers = result["transfers"]
    print(
        f"\nPeriod accounting with {result['selected_allocation_label']}"
    )
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


def run_report(
    scenario: Scenario,
    hours: Iterable[int] | None = None,
    allocation_priority: str = "contribution",
    max_instability_ratio: float = 0.01,
) -> dict[str, Any]:
    if max_instability_ratio < 0.0:
        raise ValueError("max_instability_ratio must be non-negative")
    selected_hours = validate_hours(
        hours
        if hours is not None
        else inclusive_hour_window(total_hours=scenario.num_hours),
        scenario.num_hours,
    )
    _header(scenario, selected_hours)
    print(f"\n>> Computing {len(selected_hours)} hourly games...")
    result = evaluate_period(
        scenario,
        selected_hours,
        allocation_priority=allocation_priority,
    )
    _hourly_table(result)
    _period_summary(scenario, result, max_instability_ratio)
    return result
