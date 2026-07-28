"""Savings game, transfers, core and Bondareva--Shapley test."""

from dataclasses import dataclass
from itertools import combinations

import pulp

from src.core.generate_data import OperatorParams
from src.core.optimiser import coalition_cost_star


def iter_coalition_tuples(
    players: list[int],
    include_empty: bool = False,
) -> list[tuple[int, ...]]:
    start = 0 if include_empty else 1
    return [
        coalition
        for size in range(start, len(players) + 1)
        for coalition in combinations(sorted(players), size)
    ]


def build_cost_map(
    operators: list[OperatorParams],
    demands: dict[int, float],
    players: list[int],
) -> dict[tuple[int, ...], float]:
    costs = {(): 0.0}
    for coalition in iter_coalition_tuples(players):
        costs[coalition] = coalition_cost_star(
            list(coalition), operators, demands
        )[0]
    return costs


def build_savings_game(
    costs: dict[tuple[int, ...], float],
    players: list[int],
) -> dict[tuple[int, ...], float]:
    savings = {(): 0.0}
    for coalition in iter_coalition_tuples(players):
        value = sum(costs[(i,)] for i in coalition) - costs[coalition]
        savings[coalition] = 0.0 if abs(value) < 1e-12 else value
    return savings


def physical_costs(
    players: list[int],
    guardians: list[int],
    allocation: dict[int, float],
    operators: list[OperatorParams],
) -> dict[int, float]:
    return {
        i: operators[i].F + operators[i].gamma * allocation.get(i, 0.0)
        if i in guardians
        else 0.0
        for i in players
    }


@dataclass(frozen=True)
class Settlement:
    net_costs: dict[int, float]
    transfers: dict[int, float]
    budget_residual: float


def build_settlement(
    players: list[int],
    standalone: dict[int, float],
    physical: dict[int, float],
    savings: dict[int, float],
) -> Settlement:
    net = {i: standalone[i] - savings[i] for i in players}
    transfers = {i: physical[i] - net[i] for i in players}
    return Settlement(net, transfers, sum(transfers.values()))


@dataclass(frozen=True)
class CoreResult:
    allocation: dict[int, float]
    status: str
    feasible: bool


def core_allocation(
    players: list[int],
    savings: dict[tuple[int, ...], float],
) -> CoreResult:
    """Find one non-negative allocation satisfying all core constraints."""
    grand = tuple(players)
    model = pulp.LpProblem("Core", pulp.LpMinimize)
    z = pulp.LpVariable.dicts("z", players, lowBound=0.0)
    model += pulp.lpSum(0.0 * z[i] for i in players)
    model += pulp.lpSum(z.values()) == savings[grand]
    for coalition in iter_coalition_tuples(players):
        if coalition != grand:
            model += pulp.lpSum(z[i] for i in coalition) >= savings[coalition]
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[model.status]
    feasible = status == "Optimal"
    allocation = {i: float(pulp.value(z[i]) or 0.0) for i in players} if feasible else {}
    return CoreResult(allocation, status, feasible)


@dataclass(frozen=True)
class BondarevaShapleyResult:
    balanced_value: float
    grand_value: float
    gap: float
    status: str
    core_nonempty: bool | None


def bondareva_shapley_test(
    players: list[int],
    savings: dict[tuple[int, ...], float],
    tolerance: float = 1e-7,
) -> BondarevaShapleyResult:
    coalitions, grand = iter_coalition_tuples(players), tuple(players)
    model = pulp.LpProblem("Bondareva_Shapley", pulp.LpMaximize)
    weights = {
        coalition: pulp.LpVariable(
            "lambda_" + "_".join(map(str, coalition)), lowBound=0.0
        )
        for coalition in coalitions
    }
    model += pulp.lpSum(weights[c] * savings[c] for c in coalitions)
    for i in players:
        model += pulp.lpSum(weights[c] for c in coalitions if i in c) == 1.0
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[model.status]
    balanced = (
        float(pulp.value(model.objective) or 0.0)
        if status == "Optimal"
        else float("nan")
    )
    grand_value = savings[grand]
    gap = balanced - grand_value
    nonempty = abs(gap) <= tolerance if status == "Optimal" else None
    return BondarevaShapleyResult(balanced, grand_value, gap, status, nonempty)


# Kept aside for the future least-core work; not called by the current program.


@dataclass
class LeastCoreResult:
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
    """
    grand_coalition = tuple(sorted(players))
    grand_value = v_star_map[grand_coalition]

    model = pulp.LpProblem("LeastCore_RAN_Sharing", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("gain", players, cat=pulp.LpContinuous)
    epsilon_var = pulp.LpVariable("epsilon", cat=pulp.LpContinuous)
    model += epsilon_var
    model += pulp.lpSum(x[i] for i in players) == grand_value

    for coalition in iter_coalition_tuples(players):
        if coalition != grand_coalition:
            model += (
                pulp.lpSum(x[i] for i in coalition)
                >= v_star_map[coalition] + epsilon_var
            )

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return LeastCoreResult(
        {i: float(pulp.value(x[i]) or 0.0) for i in players},
        float(pulp.value(epsilon_var) or 0.0),
        pulp.LpStatus[model.status],
    )
