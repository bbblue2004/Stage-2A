"""Minimise a coalition's fixed-plus-variable operational cost."""

from itertools import combinations

from src.core.allocate import allocate_greedy
from src.core.generate_data import OperatorParams


def coalition_cost_star(
    coalition: list[int],
    operators: list[OperatorParams],
    demands: dict[int, float],
) -> tuple[float, list[int], dict[int, float]]:
    """Enumerate guardian sets; allocate their traffic greedily."""
    if not coalition:
        return 0.0, [], {}

    total_demand = sum(demands[i] for i in coalition)
    best = (float("inf"), [], {})

    for size in range(1, len(coalition) + 1):
        for guardian_tuple in combinations(sorted(coalition), size):
            guardians = list(guardian_tuple)
            capacities = {i: operators[i].q for i in guardians}
            if sum(capacities.values()) < total_demand - 1e-9:
                continue
            allocation = allocate_greedy(
                guardians,
                capacities,
                {i: operators[i].gamma for i in guardians},
                total_demand,
            )
            cost = sum(
                operators[i].F + operators[i].gamma * allocation.get(i, 0.0)
                for i in guardians
            )
            if cost < best[0] - 1e-12:
                best = (cost, guardians, allocation)

    if not best[1]:
        raise ValueError(f"No feasible guardians for coalition {coalition}")
    return best
