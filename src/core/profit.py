from dataclasses import dataclass
from itertools import combinations, permutations
from math import factorial
from typing import Callable, Optional

import pulp

from src.core.generate_data import OperatorParams
from src.core.optimiser import coalition_value_star, coalition_utility, iter_nonempty_coalitions
from src.core.utility import single_operator_utility


def shapley_values(
    players: list[int],
    value_function: Callable[[list[int]], float]
) -> dict[int, float]:
    """
    Computes Shapley values by enumerating all permutations of players.

    The Shapley value for player i is the average marginal contribution
    when i joins a coalition, averaged over all possible orderings.

    φ_i = (1/N!) * Σ [v(S ∪ {i}) - v(S)]

    where S is the set of players appearing before i in each permutation.

    Args:
        players: List of player indices
        value_function: Function that takes a list of players and returns coalition value

    Returns:
        Dictionary mapping player index to their Shapley value.
    """
    n = len(players)
    if n == 0:
        return {}

    shapley: dict[int, float] = {p: 0.0 for p in players}

    # Enumerate all permutations
    for perm in permutations(players):
        current_coalition: list[int] = []
        current_value = 0.0

        for player in perm:
            # Value of coalition including this player
            new_coalition = current_coalition + [player]
            new_value = value_function(new_coalition)

            # Marginal contribution
            marginal = new_value - current_value
            shapley[player] += marginal

            current_coalition = new_coalition
            current_value = new_value

    # Average over all permutations
    num_perms = factorial(n)
    for p in players:
        shapley[p] /= num_perms

    return shapley


def payoff_rule1(
    coalition: list[int],
    v_star: Callable[[list[int]], float],
    operators: list[OperatorParams],
    traffic_at_t: dict[int, float],
    actual_v_star: Optional[float] = None,
    actual_guardians: Optional[list[int]] = None,
    actual_allocation: Optional[dict[int, float]] = None,
    duration_hours: float = 1.0,
) -> dict[int, float]:
    """
    Gain-sharing rule 1: Equalized costs plus Shapley-based revenues.

    Steps:
    1. Compute guardian costs and average cost share D_s
    2. Use Shapley values to allocate revenue proportionally
    3. Final payoff: g1_i = revenue_share_i - D_s

    This ensures all coalition members share costs equally while revenues
    are distributed based on marginal contributions.

    Args:
        coalition: List of operator indices in the coalition
        v_star: Function computing v*(s) for any coalition s
        operators: List of all operator parameters
        traffic_at_t: Dictionary mapping operator index to traffic at current time
        actual_v_star: Actual coalition value (if None, compute from v_star function)
        actual_guardians: Actual guardians used (if None, compute optimal)
        actual_allocation: Actual traffic allocation (if None, compute optimal)

    Returns:
        Dictionary mapping operator index to their payoff under rule 1.
    """
    if not coalition:
        return {}

    # Get configuration - use actual if provided, otherwise compute optimal
    if actual_v_star is not None and actual_guardians is not None and actual_allocation is not None:
        v_star_coalition = actual_v_star
        guardians = actual_guardians
        allocation = actual_allocation
    else:
        v_star_coalition, guardians, allocation = coalition_value_star(
            coalition, operators, traffic_at_t, duration_hours=duration_hours
        )

    if abs(v_star_coalition) < 1e-9:
        # Avoid division by zero
        return {i: 0.0 for i in coalition}

    # Compute individual guardian costs D_i
    guardian_costs: dict[int, float] = {}
    for g in guardians:
        allocated_traffic = allocation.get(g, 0.0)
        rho_g = min(1.0, allocated_traffic / operators[g].capacity_epsilon)
        # D_i = beta_i * rho_i + K_i
        guardian_costs[g] = (
            operators[g].beta * rho_g + operators[g].K
        ) * duration_hours

    # Average cost share
    total_cost = sum(guardian_costs.values())
    D_s = total_cost / len(coalition)

    # Compute Shapley values (still use the v_star function for marginal contributions)
    phi = shapley_values(coalition, v_star)

    # Total coalition revenue
    R_s = sum(operators[i].c * traffic_at_t[i] for i in coalition)

    # Sum of Shapley values for normalization
    phi_sum = sum(phi.values())
    if abs(phi_sum) < 1e-9:
        phi_sum = 1.0  # Avoid division by zero

    # Allocate revenue proportionally to Shapley values, scaled to actual v_star
    payoffs: dict[int, float] = {}
    for i in coalition:
        # Revenue share based on Shapley proportion, but total scaled to actual v_star + costs
        revenue_share_i = (phi[i] / phi_sum) * (v_star_coalition + len(coalition) * D_s)
        payoffs[i] = revenue_share_i - D_s

    return payoffs


def payoff_rule2(
    coalition: list[int],
    operators: list[OperatorParams],
    traffic_at_t: dict[int, float],
    actual_v_star: Optional[float] = None,
    actual_guardians: Optional[list[int]] = None,
    v_star: Optional[Callable[[list[int]], float]] = None,
    duration_hours: float = 1.0,
) -> dict[int, float]:
    """
    Gain-sharing rule 2: Guard vs non-guard interpolated Shapley.

    For each player:
    1. Compute ordinary Shapley value φ_i (can be guardian)
    2. Compute modified Shapley value ψ_i (forbidden from being guardian)
    3. Interpolate: h_i = ρ_i * φ_i + (1 - ρ_i) * ψ_i
    4. Normalize for efficiency: g2_i = h_i * v*(coalition) / Σh_j

    Args:
        coalition: List of operator indices in the coalition
        operators: List of all operator parameters
        traffic_at_t: Dictionary mapping operator index to traffic at current time
        actual_v_star: Actual coalition value (if None, compute optimal)
        actual_guardians: Actual guardians used (affects rho calculation)

    Returns:
        Dictionary mapping operator index to their payoff under rule 2.
    """
    if not coalition:
        return {}

    # Standard v_star function for Shapley computation
    def v_star_standard(s: list[int]) -> float:
        if not s:
            return 0.0
        if v_star is not None:
            return v_star(s)
        val, _, _ = coalition_value_star(
            s, operators, traffic_at_t, duration_hours=duration_hours
        )
        return val

    # Use actual v_star if provided, otherwise compute optimal
    if actual_v_star is not None:
        v_star_coalition = actual_v_star
    else:
        v_star_coalition = v_star_standard(coalition)

    # Compute ordinary Shapley values
    phi = shapley_values(coalition, v_star_standard)

    # For each player, compute modified Shapley value (player forbidden as guardian)
    psi: dict[int, float] = {}

    for player in coalition:
        guard_memo: dict[tuple[tuple[int, ...], int], float] = {}

        def v_star_without_guard(s: list[int], excluded: int = player) -> float:
            if not s:
                return 0.0

            key = (tuple(sorted(s)), excluded)
            if key in guard_memo:
                return guard_memo[key]

            total_traffic = sum(traffic_at_t[i] for i in s)
            best_value = float("-inf")
            candidates = [i for i in s if i != excluded]

            if not candidates:
                guard_memo[key] = 0.0
                return 0.0

            for r in range(1, len(candidates) + 1):
                for guardian_combo in combinations(candidates, r):
                    guardians = list(guardian_combo)
                    total_capacity = sum(operators[g].capacity_epsilon for g in guardians)

                    if total_capacity < total_traffic - 1e-9:
                        continue

                    value = coalition_utility(
                        s, guardians, operators, traffic_at_t,
                        duration_hours=duration_hours,
                    )
                    if value > best_value:
                        best_value = value

            result = best_value if best_value > float("-inf") else 0.0
            guard_memo[key] = result
            return result

        psi[player] = shapley_values(coalition, v_star_without_guard).get(player, 0.0)

    # Compute load for each player - use actual guardians if provided
    rho: dict[int, float] = {}
    for i in coalition:
        if actual_guardians is not None and i in actual_guardians:
            # If this operator is actually serving as guardian, use higher load
            rho[i] = min(1.0, traffic_at_t[i] / operators[i].capacity_epsilon)
        else:
            rho[i] = min(1.0, traffic_at_t[i] / operators[i].capacity_epsilon)

    # Preliminary payoff
    h: dict[int, float] = {}
    for i in coalition:
        h[i] = rho[i] * phi[i] + (1 - rho[i]) * psi[i]

    # Normalize for efficiency - use actual v_star
    h_sum = sum(h.values())
    if abs(h_sum) < 1e-9:
        # Avoid division by zero
        return {i: v_star_coalition / len(coalition) for i in coalition}

    payoffs: dict[int, float] = {}
    for i in coalition:
        payoffs[i] = h[i] * v_star_coalition / h_sum

    return payoffs


def payoff_rule3(
    coalition: list[int],
    operators: list[OperatorParams],
    traffic_at_t: dict[int, float],
    actual_v_star: Optional[float] = None,
    duration_hours: float = 1.0,
) -> dict[int, float]:
    """
    Gain-sharing rule 3: Proportional to standalone utility.

    Each player receives a share proportional to their standalone utility:
    g3_i = v(A_i) * v*(coalition) / Σv(A_j)

    This is the simplest rule but doesn't account for strategic contributions.

    Args:
        coalition: List of operator indices in the coalition
        operators: List of all operator parameters
        traffic_at_t: Dictionary mapping operator index to traffic at current time
        actual_v_star: Actual coalition value (if None, compute optimal)

    Returns:
        Dictionary mapping operator index to their payoff under rule 3.
    """
    if not coalition:
        return {}

    # Compute standalone utilities
    standalone: dict[int, float] = {}
    for i in coalition:
        T_i = traffic_at_t[i]
        rho_i = min(1.0, T_i / operators[i].capacity_epsilon)
        standalone[i] = single_operator_utility(
            operators[i].c, T_i, operators[i].beta, rho_i, operators[i].K,
            duration_hours=duration_hours,
        )

    # Total standalone utility
    V_total = sum(standalone.values())

    # Use actual v_star if provided, otherwise compute optimal
    if actual_v_star is not None:
        v_star_coalition = actual_v_star
    else:
        v_star_coalition, _, _ = coalition_value_star(
            coalition, operators, traffic_at_t, duration_hours=duration_hours
        )

    if abs(V_total) < 1e-9:
        # Avoid division by zero - equal split
        return {i: v_star_coalition / len(coalition) for i in coalition}

    # Proportional allocation
    payoffs: dict[int, float] = {}
    for i in coalition:
        payoffs[i] = standalone[i] * v_star_coalition / V_total

    return payoffs


# ---------------------------------------------------------------------------
# Cooperative core allocation (max-epsilon LP, "least core" / nucleole approach)
# ---------------------------------------------------------------------------


def iter_coalition_tuples(players: list[int]) -> list[tuple[int, ...]]:
    """All non-empty subsets of players as sorted tuples."""
    return [
        tuple(sorted(coalition))
        for coalition in iter_nonempty_coalitions(len(players))
    ]


def build_v_star_map(
    operators: list[OperatorParams],
    traffic_at_t: dict[int, float],
    players: Optional[list[int]] = None,
    duration_hours: float = 1.0,
) -> dict[tuple[int, ...], float]:
    """
    Compute v*(S) for every non-empty coalition S.

    Uses coalition_value_star (greedy allocation, optimal guardian set).
    """
    if players is None:
        players = list(range(len(operators)))

    v_star_map: dict[tuple[int, ...], float] = {}
    for coalition in iter_nonempty_coalitions(len(players)):
        v_star, _, _ = coalition_value_star(
            coalition, operators, traffic_at_t, duration_hours=duration_hours
        )
        v_star_map[tuple(sorted(coalition))] = v_star
    return v_star_map


@dataclass
class LeastCoreResult:
    """Payoff vector from the max-epsilon core linear program."""

    payoffs: dict[int, float]
    epsilon: float
    status: str


def least_core_allocation(
    players: list[int],
    v_star_map: dict[tuple[int, ...], float],
) -> LeastCoreResult:
    """
    Allocate gains via a linear program that maximises coalition stability margin.

    maximise epsilon
    subject to:
        sum_i x_i = v*(A)                           (efficiency)
        sum_{i in S} x_i >= v*(S) + epsilon           for all S subset A, S != A

    No coalition S should prefer seceding: its members' total payoff must be at
    least v*(S). Epsilon is the uniform margin added above those coalition values.

    Args:
        players: Grand-coalition member indices
        v_star_map: Coalition values keyed by sorted tuples of player indices

    Returns:
        LeastCoreResult with per-player payoffs, epsilon, and solver status.
    """
    grand_coalition = tuple(sorted(players))
    grand_value = v_star_map[grand_coalition]

    model = pulp.LpProblem("LeastCore_RAN_Sharing", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("gain", players, cat=pulp.LpContinuous)
    epsilon_var = pulp.LpVariable("epsilon", cat=pulp.LpContinuous)

    model += epsilon_var, "Maximise_stability_margin"
    model += pulp.lpSum(x[i] for i in players) == grand_value, "Efficiency"

    for coalition in iter_coalition_tuples(players):
        if coalition == grand_coalition:
            continue
        coalition_value = v_star_map[coalition]
        model += (
            pulp.lpSum(x[i] for i in coalition) >= coalition_value + epsilon_var,
            f"Stability_{''.join(str(i) for i in coalition)}",
        )

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[model.status]
    payoffs = {i: float(pulp.value(x[i]) or 0.0) for i in players}
    epsilon = float(pulp.value(epsilon_var) or 0.0)

    return LeastCoreResult(payoffs=payoffs, epsilon=epsilon, status=status)