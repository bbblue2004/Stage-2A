"""Evaluate the hourly games and aggregate them over one day."""

from collections import defaultdict
from typing import Any

from src.core.generate_data import Scenario
from src.core.optimiser import coalition_cost_star
from src.core.game import (
    bondareva_shapley_test,
    build_cost_map,
    build_savings_game,
    build_settlement,
    core_allocation,
    physical_costs,
)


def evaluate_day(scenario: Scenario) -> dict[str, Any]:
    operators, players = scenario.operators, scenario.coalition
    grand = tuple(players)
    daily_costs: dict[tuple[int, ...], float] = defaultdict(float)
    daily_savings: dict[tuple[int, ...], float] = defaultdict(float)
    physical = {i: 0.0 for i in players}
    hourly: list[dict[str, Any]] = []
    previous_guardians: tuple[int, ...] | None = None
    changes = 0

    for hour in range(scenario.num_hours):
        demands = scenario.demands_at_hour(hour)
        costs = build_cost_map(operators, demands, players)
        savings = build_savings_game(costs, players)
        for coalition in costs:
            daily_costs[coalition] += costs[coalition]
            daily_savings[coalition] += savings[coalition]

        grand_cost, guardians, allocation = coalition_cost_star(
            players, operators, demands
        )
        hour_physical = physical_costs(players, guardians, allocation, operators)
        for i in players:
            physical[i] += hour_physical[i]

        guardian_tuple = tuple(guardians)
        changed = previous_guardians is not None and guardian_tuple != previous_guardians
        changes += int(changed)
        previous_guardians = guardian_tuple
        hourly.append(
            {
                "hour": hour,
                "standalone_cost": sum(costs[(i,)] for i in players),
                "coalition_cost": grand_cost,
                "savings": savings[grand],
                "guardians_label": "[" + ", ".join(str(i + 1) for i in guardians) + "]",
                "guardians_changed": changed,
            }
        )

    costs, savings = dict(daily_costs), dict(daily_savings)
    standalone = {i: costs[(i,)] for i in players}
    core = core_allocation(players, savings)
    balancedness = bondareva_shapley_test(players, savings)
    settlement = (
        build_settlement(players, standalone, physical, core.allocation)
        if core.feasible
        else None
    )

    return {
        "hourly": hourly,
        "guardian_changes": changes,
        "cost_map": costs,
        "savings_map": savings,
        "standalone_costs": standalone,
        "physical_costs": physical,
        "savings_allocation": core.allocation,
        "net_costs": settlement.net_costs if settlement else {},
        "transfers": settlement.transfers if settlement else {},
        "total_savings": savings[grand],
        "core_summary": {"status": core.status, "feasible": core.feasible},
        "bondareva_summary": {
            "status": balancedness.status,
            "balanced_value": balancedness.balanced_value,
            "grand_value": balancedness.grand_value,
            "gap": balancedness.gap,
            "core_nonempty": balancedness.core_nonempty,
        },
        "budget_residual": settlement.budget_residual if settlement else None,
    }
