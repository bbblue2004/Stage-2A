"""Evaluate hourly games and aggregate them over a selected period."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from src.core.generate_data import Scenario
from src.core.optimiser import coalition_cost_star
from src.core.game import (
    allocation_check,
    bondareva_shapley_test,
    build_cost_map,
    build_savings_game,
    build_settlement,
    closest_stable_allocation,
    convexity_test,
    core_allocation,
    least_core_allocation,
    normalized_distance,
    nucleolus_allocation,
    physical_costs,
    shapley_value,
)
from src.core.time_window import validate_hours

ALLOCATION_PRIORITIES = ("contribution", "robustness")


def evaluate_period(
    scenario: Scenario,
    hours: Iterable[int],
    allocation_priority: str = "contribution",
) -> dict[str, Any]:
    """Evaluate and aggregate the game over the selected hours."""
    if allocation_priority not in ALLOCATION_PRIORITIES:
        raise ValueError(
            "allocation_priority must be 'contribution' or 'robustness'"
        )
    operators, players = scenario.operators, scenario.coalition
    selected_hours = validate_hours(hours, scenario.num_hours)
    grand = tuple(players)
    period_costs: dict[tuple[int, ...], float] = defaultdict(float)
    period_savings: dict[tuple[int, ...], float] = defaultdict(float)
    physical = {i: 0.0 for i in players}
    hourly: list[dict[str, Any]] = []
    previous_guardians: tuple[int, ...] | None = None
    changes = 0

    for hour in selected_hours:
        demands = scenario.demands_at_hour(hour)
        costs = build_cost_map(operators, demands, players)
        savings = build_savings_game(costs, players)
        for coalition in costs:
            period_costs[coalition] += costs[coalition]
            period_savings[coalition] += savings[coalition]

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

    costs, savings = dict(period_costs), dict(period_savings)
    standalone = {i: costs[(i,)] for i in players}
    convexity = convexity_test(players, savings)
    shapley = shapley_value(players, savings)
    shapley_check = allocation_check(players, savings, shapley)
    core = core_allocation(players, savings)
    balancedness = bondareva_shapley_test(players, savings)
    least_core = least_core_allocation(players, savings)
    if least_core.status != "Optimal":
        raise RuntimeError(
            f"Least-core calculation failed: {least_core.status}"
        )
    nucleolus = nucleolus_allocation(players, savings)
    if nucleolus.status != "Optimal":
        raise RuntimeError(
            f"Nucleolus calculation failed: {nucleolus.status}"
        )

    if shapley_check.in_core:
        fair_label = "Fair stable (equals Shapley)"
        fair_allocation = shapley
        fair_projection_status = "Shapley already belongs to the core"
    else:
        projection_epsilon = 0.0 if core.feasible else least_core.epsilon
        projection_initial = (
            core.allocation if core.feasible else least_core.payoffs
        )
        projection = closest_stable_allocation(
            players,
            savings,
            shapley,
            standalone,
            projection_epsilon,
            projection_initial,
        )
        if not projection.success:
            raise RuntimeError(
                f"Shapley projection failed: {projection.status}"
            )
        fair_label = (
            "Shapley projection on core"
            if core.feasible
            else "Shapley projection on least core"
        )
        fair_allocation = projection.allocation
        fair_projection_status = projection.status

    candidate_allocations = {
        "shapley": ("Shapley", shapley),
        "fair_stable": (fair_label, fair_allocation),
        "nucleolus": ("Nucleolus", nucleolus.allocation),
    }
    candidates: dict[str, dict[str, Any]] = {}
    for key, (label, allocation) in candidate_allocations.items():
        check = allocation_check(players, savings, allocation)
        candidates[key] = {
            "label": label,
            "allocation": allocation,
            "in_core": check.in_core,
            "max_excess": check.max_excess,
            "blocking_count": len(check.blocking_coalitions),
            "blocking_coalitions": check.blocking_coalitions,
            "efficiency_residual": check.efficiency_residual,
            "distance_to_shapley": normalized_distance(
                players,
                allocation,
                shapley,
                standalone,
            ),
        }

    if allocation_priority == "robustness":
        selected_key = "nucleolus"
        selection_reason = (
            "Robustness priority: lexicographic minimisation of objections."
        )
    elif shapley_check.in_core:
        selected_key = "shapley"
        selection_reason = (
            "Shapley belongs to the core: contributive fairness and "
            "coalitional stability."
        )
    else:
        selected_key = "nucleolus"
        selection_reason = (
            "Shapley is outside the core; the nucleolus belongs to the core."
            if core.feasible
            else (
                "The core is empty; the nucleolus lexicographically "
                "minimises coalition objections."
            )
        )
    selected_allocation = candidates[selected_key]["allocation"]
    settlement = build_settlement(
        players,
        standalone,
        physical,
        selected_allocation,
    )
    instability_ratio = max(0.0, least_core.epsilon) / max(
        abs(savings[grand]),
        1e-12,
    )

    return {
        "hours": selected_hours,
        "hourly": hourly,
        "guardian_changes": changes,
        "cost_map": costs,
        "savings_map": savings,
        "standalone_costs": standalone,
        "physical_costs": physical,
        "savings_allocation": selected_allocation,
        "net_costs": settlement.net_costs,
        "transfers": settlement.transfers,
        "total_savings": savings[grand],
        "convexity_summary": {
            "convex": convexity.convex,
            "max_violation": convexity.max_violation,
            "witness": convexity.witness,
        },
        "core_summary": {"status": core.status, "feasible": core.feasible},
        "bondareva_summary": {
            "status": balancedness.status,
            "balanced_value": balancedness.balanced_value,
            "grand_value": balancedness.grand_value,
            "gap": balancedness.gap,
            "core_nonempty": balancedness.core_nonempty,
        },
        "least_core_summary": {
            "status": least_core.status,
            "epsilon": least_core.epsilon,
            "instability_ratio": instability_ratio,
        },
        "shapley_in_core": shapley_check.in_core,
        "nucleolus_stages": nucleolus.stages,
        "fair_projection_status": fair_projection_status,
        "allocation_priority": allocation_priority,
        "allocation_candidates": candidates,
        "selected_allocation_key": selected_key,
        "selected_allocation_label": candidates[selected_key]["label"],
        "selection_reason": selection_reason,
        "budget_residual": settlement.budget_residual,
    }


def evaluate_day(scenario: Scenario) -> dict[str, Any]:
    """Backward-compatible evaluation of all available hours."""
    return evaluate_period(scenario, range(scenario.num_hours))
