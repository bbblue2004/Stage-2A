from itertools import combinations

from src.core.allocate import allocate_greedy
from src.core.generate_data import OperatorParams


def iter_nonempty_coalitions(num_operators: int) -> list[list[int]]:
    """Return all non-empty subsets of {0, ..., num_operators - 1}."""
    return [
        list(coalition)
        for size in range(1, num_operators + 1)
        for coalition in combinations(range(num_operators), size)
    ]


def _greedy_allocation(
    guardians: list[int],
    operators: list[OperatorParams],
    total_traffic: float,
) -> dict[int, float]:
    capacities = {g: operators[g].capacity_epsilon for g in guardians}
    betas = {g: operators[g].beta for g in guardians}
    return allocate_greedy(guardians, capacities, betas, total_traffic)


def coalition_utility(
    coalition: list[int],
    guardians: list[int],
    operators: list[OperatorParams],
    traffic_at_t: dict[int, float],
    allocation: dict[int, float] | None = None,
    duration_hours: float = 1.0,
) -> float:
    """
    Computes the utility of a coalition with a given guardian set.

    v(s, l_s) = Σ(c_i * T_i for i in coalition)
                - duration_hours * Σ(β_i * ρ_i for i in guardians)
                - duration_hours * Σ(K_i for i in guardians)

    beta and K in OperatorParams are hourly; duration_hours scales costs to
    the accounting period of traffic_at_t (1.0 for one clock hour, 0.25 for 15 min).

    Traffic among guardians is allocated with the greedy rule (optimal for fixed l_s).

    Args:
        coalition: List of operator indices in the coalition
        guardians: List of operator indices serving as guardians
        operators: List of all operator parameters
        traffic_at_t: Dictionary mapping operator index to traffic at current time
        allocation: Precomputed greedy allocation (recomputed if None)

    Returns:
        The coalition's utility value.
    """
    total_revenue = sum(
        operators[i].c * traffic_at_t[i]
        for i in coalition
    )

    total_traffic = sum(traffic_at_t[i] for i in coalition)

    if allocation is None:
        allocation = _greedy_allocation(guardians, operators, total_traffic)

    total_variable_cost = 0.0
    total_fixed_cost = 0.0

    for g in guardians:
        allocated_traffic = allocation.get(g, 0.0)
        rho_g = min(1.0, allocated_traffic / operators[g].capacity_epsilon)
        total_variable_cost += operators[g].beta * rho_g
        total_fixed_cost += operators[g].K

    return (
        total_revenue
        - duration_hours * total_variable_cost
        - duration_hours * total_fixed_cost
    )


def coalition_value_star(
    coalition: list[int],
    operators: list[OperatorParams],
    traffic_at_t: dict[int, float],
    duration_hours: float = 1.0,
) -> tuple[float, list[int], dict[int, float]]:
    """
    Computes the optimal coalition value v*(s) by finding the best guardian set.

    v*(s) = max_{l_s} v(s, l_s) where traffic is split with the greedy rule.

    Enumerates all non-empty subsets of the coalition as candidate guardian sets,
    checks capacity feasibility, and returns the configuration with maximum utility.

    Args:
        coalition: List of operator indices in the coalition
        operators: List of all operator parameters
        traffic_at_t: Dictionary mapping operator index to traffic at current time

    Returns:
        Tuple of:
        - v_star: The maximum coalition value
        - optimal_guardians: The best guardian set
        - optimal_allocation: The greedy traffic allocation for optimal guardians
    """
    if not coalition:
        return 0.0, [], {}

    total_traffic = sum(traffic_at_t[i] for i in coalition)

    best_value = float("-inf")
    best_guardians: list[int] = []
    best_allocation: dict[int, float] = {}

    for r in range(1, len(coalition) + 1):
        for guardian_combo in combinations(coalition, r):
            guardians = list(guardian_combo)

            total_capacity = sum(operators[g].capacity_epsilon for g in guardians)
            if total_capacity < total_traffic - 1e-9:
                continue

            try:
                allocation = _greedy_allocation(guardians, operators, total_traffic)
            except ValueError:
                continue

            value = coalition_utility(
                coalition, guardians, operators, traffic_at_t,
                allocation=allocation,
                duration_hours=duration_hours,
            )

            if value > best_value:
                best_value = value
                best_guardians = guardians
                best_allocation = allocation

    return best_value, best_guardians, best_allocation
