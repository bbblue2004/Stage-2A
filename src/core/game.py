"""Cooperative-game diagnostics and allocation rules."""

from dataclasses import dataclass
from itertools import combinations
from math import factorial, sqrt

import numpy as np
import pulp
from scipy.optimize import linprog

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
class ConvexityResult:
    convex: bool
    max_violation: float
    witness: tuple[tuple[int, ...], tuple[int, ...], int] | None


def convexity_test(
    players: list[int],
    savings: dict[tuple[int, ...], float],
    tolerance: float = 1e-9,
) -> ConvexityResult:
    """Test increasing marginal contributions for every nested pair."""
    max_violation = 0.0
    witness = None
    for player in players:
        others = [i for i in players if i != player]
        subsets = [
            coalition
            for size in range(len(others) + 1)
            for coalition in combinations(others, size)
        ]
        for smaller in subsets:
            smaller_set = set(smaller)
            smaller_with_player = tuple(sorted(smaller + (player,)))
            smaller_marginal = (
                savings[smaller_with_player] - savings[smaller]
            )
            for larger in subsets:
                if not smaller_set.issubset(larger):
                    continue
                larger_with_player = tuple(sorted(larger + (player,)))
                larger_marginal = (
                    savings[larger_with_player] - savings[larger]
                )
                violation = smaller_marginal - larger_marginal
                if violation > max_violation:
                    max_violation = violation
                    witness = (smaller, larger, player)
    return ConvexityResult(max_violation <= tolerance, max_violation, witness)


def shapley_value(
    players: list[int],
    savings: dict[tuple[int, ...], float],
) -> dict[int, float]:
    """Return the Shapley value from all marginal contributions."""
    n = len(players)
    denominator = factorial(n)
    result = {i: 0.0 for i in players}
    for player in players:
        others = [i for i in players if i != player]
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / denominator
            for coalition in combinations(others, size):
                enlarged = tuple(sorted(coalition + (player,)))
                result[player] += weight * (
                    savings[enlarged] - savings[coalition]
                )
    return result


@dataclass(frozen=True)
class AllocationCheck:
    efficient: bool
    individually_rational: bool
    in_core: bool
    efficiency_residual: float
    max_excess: float
    blocking_coalitions: tuple[tuple[int, ...], ...]


def allocation_check(
    players: list[int],
    savings: dict[tuple[int, ...], float],
    allocation: dict[int, float],
    tolerance: float = 1e-7,
) -> AllocationCheck:
    """Measure efficiency, individual rationality and coalition excesses."""
    grand = tuple(players)
    residual = sum(allocation[i] for i in players) - savings[grand]
    individually_rational = all(
        allocation[i] >= savings[(i,)] - tolerance for i in players
    )
    excesses = {
        coalition: savings[coalition]
        - sum(allocation[i] for i in coalition)
        for coalition in iter_coalition_tuples(players)
        if coalition != grand
    }
    max_excess = max(excesses.values(), default=float("-inf"))
    blocking = tuple(
        coalition
        for coalition, excess in excesses.items()
        if excess > tolerance
    )
    efficient = abs(residual) <= tolerance
    return AllocationCheck(
        efficient,
        individually_rational,
        efficient and individually_rational and not blocking,
        residual,
        max_excess,
        blocking,
    )


def normalized_distance(
    players: list[int],
    allocation: dict[int, float],
    reference: dict[int, float],
    normalizers: dict[int, float],
) -> float:
    """Euclidean distance after scaling by positive standalone costs."""
    return sqrt(
        sum(
            (
                (allocation[i] - reference[i])
                / max(abs(normalizers[i]), 1e-12)
            )
            ** 2
            for i in players
        )
    )


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


@dataclass(frozen=True)
class LeastCoreResult:
    payoffs: dict[int, float]
    epsilon: float
    status: str


def least_core_allocation(
    players: list[int],
    savings: dict[tuple[int, ...], float],
) -> LeastCoreResult:
    """Minimise the largest coalition excess over the imputation set."""
    grand = tuple(players)
    model = pulp.LpProblem("Least_Core", pulp.LpMinimize)
    z = pulp.LpVariable.dicts("least_core_gain", players, lowBound=0.0)
    epsilon = pulp.LpVariable("least_core_epsilon")
    model += epsilon
    model += pulp.lpSum(z.values()) == savings[grand]
    for coalition in iter_coalition_tuples(players):
        if coalition != grand:
            model += (
                savings[coalition]
                - pulp.lpSum(z[i] for i in coalition)
                <= epsilon
            )
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        return LeastCoreResult({}, float("nan"), status)
    return LeastCoreResult(
        {i: float(pulp.value(z[i]) or 0.0) for i in players},
        float(pulp.value(epsilon) or 0.0),
        status,
    )


@dataclass(frozen=True)
class ProjectionResult:
    allocation: dict[int, float]
    objective: float
    status: str
    success: bool


def closest_stable_allocation(
    players: list[int],
    savings: dict[tuple[int, ...], float],
    reference: dict[int, float],
    normalizers: dict[int, float],
    epsilon: float,
    initial: dict[int, float],
) -> ProjectionResult:
    """Project a reference onto the core or a fixed least-core face.

    The number of operators is small, so the convex quadratic projection is
    solved exactly by enumerating the possible active linear constraints.
    """
    grand = tuple(players)
    coalitions = [
        coalition
        for coalition in iter_coalition_tuples(players)
        if coalition != grand
    ]
    player_index = {player: index for index, player in enumerate(players)}
    scales = np.array(
        [max(abs(normalizers[i]), 1e-12) for i in players],
        dtype=float,
    )
    reference_vector = np.array([reference[i] for i in players], dtype=float)
    initial_vector = np.array([initial[i] for i in players], dtype=float)

    coalition_matrix = np.zeros((len(coalitions), len(players)), dtype=float)
    lower_bounds = np.zeros(len(coalitions), dtype=float)
    for row, coalition in enumerate(coalitions):
        for player in coalition:
            coalition_matrix[row, player_index[player]] = 1.0
        lower_bounds[row] = savings[coalition] - epsilon

    inequalities = np.vstack(
        [coalition_matrix, np.eye(len(players), dtype=float)]
    )
    bounds = np.concatenate(
        [lower_bounds, np.zeros(len(players), dtype=float)]
    )
    equality = np.ones((1, len(players)), dtype=float)
    equality_value = np.array([savings[grand]], dtype=float)
    inverse_weights = np.diag(scales**2)
    feasibility_tolerance = 1e-7 * max(1.0, abs(savings[grand]))

    def objective(values: np.ndarray) -> float:
        return float(np.sum(((values - reference_vector) / scales) ** 2))

    best_vector: np.ndarray | None = None
    best_objective = float("inf")
    max_active = max(0, len(players) - 1)
    inequality_indices = range(len(inequalities))

    for active_count in range(max_active + 1):
        for active in combinations(inequality_indices, active_count):
            active_matrix = inequalities[list(active)] if active else np.empty(
                (0, len(players))
            )
            active_values = bounds[list(active)] if active else np.empty(0)
            constraint_matrix = np.vstack([equality, active_matrix])
            constraint_values = np.concatenate(
                [equality_value, active_values]
            )
            gram = constraint_matrix @ inverse_weights @ constraint_matrix.T
            correction, _, _, _ = np.linalg.lstsq(
                gram,
                constraint_values - constraint_matrix @ reference_vector,
                rcond=None,
            )
            candidate = (
                reference_vector
                + inverse_weights @ constraint_matrix.T @ correction
            )
            if np.max(
                np.abs(constraint_matrix @ candidate - constraint_values)
            ) > feasibility_tolerance:
                continue
            if (
                np.min(inequalities @ candidate - bounds)
                < -feasibility_tolerance
            ):
                continue
            candidate_objective = objective(candidate)
            if candidate_objective < best_objective - 1e-12:
                best_vector = candidate
                best_objective = candidate_objective

    if best_vector is None:
        initial_feasible = (
            abs(float(np.sum(initial_vector)) - savings[grand])
            <= feasibility_tolerance
            and np.min(inequalities @ initial_vector - bounds)
            >= -feasibility_tolerance
        )
        if not initial_feasible:
            return ProjectionResult(
                {},
                float("nan"),
                "No feasible projection found",
                False,
            )
        best_vector = initial_vector
        best_objective = objective(initial_vector)

    best_vector[np.abs(best_vector) < 1e-12] = 0.0
    efficiency_residual = savings[grand] - float(np.sum(best_vector))
    best_vector[int(np.argmax(best_vector))] += efficiency_residual
    allocation = {
        player: float(best_vector[player_index[player]]) for player in players
    }
    return ProjectionResult(allocation, best_objective, "Optimal", True)


@dataclass(frozen=True)
class NucleolusResult:
    allocation: dict[int, float]
    status: str
    stages: int


def nucleolus_allocation(
    players: list[int],
    savings: dict[tuple[int, ...], float],
    tolerance: float = 1e-8,
) -> NucleolusResult:
    """Compute the nucleolus by successive minimisation of excesses."""
    grand = tuple(players)
    coalitions = [
        coalition
        for coalition in iter_coalition_tuples(players)
        if coalition != grand
    ]
    fixed: dict[tuple[int, ...], float] = {}
    unresolved = set(coalitions)
    scale = max(1.0, abs(savings[grand]))
    face_tolerance = tolerance * scale

    def solve_stage() -> tuple[str, float, dict[int, float]]:
        model = pulp.LpProblem("Nucleolus_Stage", pulp.LpMinimize)
        z = pulp.LpVariable.dicts("nucleolus_gain", players, lowBound=0.0)
        epsilon = pulp.LpVariable("nucleolus_epsilon")
        model += epsilon
        model += pulp.lpSum(z.values()) == savings[grand]
        for coalition, excess in fixed.items():
            model += (
                savings[coalition]
                - pulp.lpSum(z[i] for i in coalition)
                == excess
            )
        for coalition in unresolved:
            model += (
                savings[coalition]
                - pulp.lpSum(z[i] for i in coalition)
                <= epsilon
            )
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[model.status]
        if status != "Optimal":
            return status, float("nan"), {}
        return (
            status,
            float(pulp.value(epsilon) or 0.0),
            {i: float(pulp.value(z[i]) or 0.0) for i in players},
        )

    def optimise_on_face(
        epsilon: float,
        coefficients: dict[int, float],
        maximize: bool,
    ) -> tuple[str, float]:
        sense = pulp.LpMaximize if maximize else pulp.LpMinimize
        model = pulp.LpProblem("Nucleolus_Face", sense)
        z = pulp.LpVariable.dicts("face_gain", players, lowBound=0.0)
        model += pulp.lpSum(
            coefficients.get(i, 0.0) * z[i] for i in players
        )
        model += pulp.lpSum(z.values()) == savings[grand]
        for coalition, excess in fixed.items():
            model += (
                savings[coalition]
                - pulp.lpSum(z[i] for i in coalition)
                == excess
            )
        for coalition in unresolved:
            model += (
                savings[coalition]
                - pulp.lpSum(z[i] for i in coalition)
                <= epsilon + face_tolerance
            )
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[model.status]
        if status != "Optimal":
            return status, float("nan")
        allocation = {
            i: float(pulp.value(z[i]) or 0.0) for i in players
        }
        value = sum(
            coefficients.get(i, 0.0) * allocation[i] for i in players
        )
        return status, value

    for stage in range(1, len(coalitions) + 2):
        status, epsilon, allocation = solve_stage()
        if status != "Optimal":
            return NucleolusResult({}, status, stage)

        unique = True
        for player in players:
            _, minimum = optimise_on_face(epsilon, {player: 1.0}, False)
            _, maximum = optimise_on_face(epsilon, {player: 1.0}, True)
            if maximum - minimum > 20.0 * face_tolerance:
                unique = False
                break
        if unique:
            return NucleolusResult(allocation, "Optimal", stage)

        newly_fixed: dict[tuple[int, ...], float] = {}
        for coalition in sorted(unresolved):
            coefficients = {i: -1.0 for i in coalition}
            _, minimum = optimise_on_face(epsilon, coefficients, False)
            _, maximum = optimise_on_face(epsilon, coefficients, True)
            minimum_excess = savings[coalition] + minimum
            maximum_excess = savings[coalition] + maximum
            if maximum_excess - minimum_excess <= 20.0 * face_tolerance:
                newly_fixed[coalition] = 0.5 * (
                    minimum_excess + maximum_excess
                )

        if not newly_fixed:
            return NucleolusResult(
                allocation,
                "Numerical failure: no fixed coalition found",
                stage,
            )
        fixed.update(newly_fixed)
        unresolved.difference_update(newly_fixed)

    return NucleolusResult({}, "Iteration limit reached", len(coalitions) + 1)


def nucleolus_allocation_fast(
    players: list[int],
    savings: dict[tuple[int, ...], float],
    tolerance: float = 1e-8,
) -> NucleolusResult:
    """Compute the nucleolus with the same LP cascade, using in-process HiGHS."""
    grand = tuple(players)
    coalitions = [
        coalition
        for coalition in iter_coalition_tuples(players)
        if coalition != grand
    ]
    player_index = {player: index for index, player in enumerate(players)}
    memberships = {
        coalition: np.asarray(
            [float(player in coalition) for player in players], dtype=float
        )
        for coalition in coalitions
    }
    fixed: dict[tuple[int, ...], float] = {}
    unresolved = set(coalitions)
    scale = max(1.0, abs(savings[grand]))
    face_tolerance = tolerance * scale
    bounds = [(0.0, None)] * len(players)

    def fixed_equalities() -> tuple[np.ndarray, np.ndarray]:
        rows = [np.ones(len(players), dtype=float)]
        values = [float(savings[grand])]
        for coalition, excess in fixed.items():
            rows.append(memberships[coalition])
            values.append(float(savings[coalition] - excess))
        return np.asarray(rows, dtype=float), np.asarray(values, dtype=float)

    def solve_stage() -> tuple[str, float, dict[int, float]]:
        objective = np.zeros(len(players) + 1, dtype=float)
        objective[-1] = 1.0
        equality, equality_values = fixed_equalities()
        equality = np.column_stack((equality, np.zeros(equality.shape[0])))
        inequalities = np.asarray(
            [
                np.concatenate((-memberships[coalition], [-1.0]))
                for coalition in unresolved
            ],
            dtype=float,
        )
        inequality_values = np.asarray(
            [-float(savings[coalition]) for coalition in unresolved],
            dtype=float,
        )
        result = linprog(
            objective,
            A_ub=inequalities if unresolved else None,
            b_ub=inequality_values if unresolved else None,
            A_eq=equality,
            b_eq=equality_values,
            bounds=(*bounds, (None, None)),
            method="highs",
        )
        if not result.success:
            return result.message, float("nan"), {}
        return (
            "Optimal",
            float(result.x[-1]),
            {
                player: float(result.x[player_index[player]])
                for player in players
            },
        )

    def optimise_on_face(
        epsilon: float,
        coefficients: dict[int, float],
        maximize: bool,
    ) -> tuple[str, float]:
        objective = np.asarray(
            [coefficients.get(player, 0.0) for player in players],
            dtype=float,
        )
        if maximize:
            objective = -objective
        equality, equality_values = fixed_equalities()
        inequalities = np.asarray(
            [-memberships[coalition] for coalition in unresolved],
            dtype=float,
        )
        inequality_values = np.asarray(
            [
                epsilon
                + face_tolerance
                - float(savings[coalition])
                for coalition in unresolved
            ],
            dtype=float,
        )
        result = linprog(
            objective,
            A_ub=inequalities if unresolved else None,
            b_ub=inequality_values if unresolved else None,
            A_eq=equality,
            b_eq=equality_values,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return result.message, float("nan")
        value = sum(
            coefficients.get(player, 0.0) * result.x[player_index[player]]
            for player in players
        )
        return "Optimal", float(value)

    for stage in range(1, len(coalitions) + 2):
        status, epsilon, allocation = solve_stage()
        if status != "Optimal":
            return NucleolusResult({}, status, stage)

        unique = True
        for player in players:
            status_min, minimum = optimise_on_face(
                epsilon, {player: 1.0}, False
            )
            status_max, maximum = optimise_on_face(
                epsilon, {player: 1.0}, True
            )
            if status_min != "Optimal" or status_max != "Optimal":
                return NucleolusResult({}, status_min, stage)
            if maximum - minimum > 20.0 * face_tolerance:
                unique = False
                break
        if unique:
            return NucleolusResult(allocation, "Optimal", stage)

        newly_fixed: dict[tuple[int, ...], float] = {}
        for coalition in sorted(unresolved):
            coefficients = {player: -1.0 for player in coalition}
            status_min, minimum = optimise_on_face(
                epsilon, coefficients, False
            )
            status_max, maximum = optimise_on_face(
                epsilon, coefficients, True
            )
            if status_min != "Optimal" or status_max != "Optimal":
                return NucleolusResult({}, status_min, stage)
            minimum_excess = savings[coalition] + minimum
            maximum_excess = savings[coalition] + maximum
            if maximum_excess - minimum_excess <= 20.0 * face_tolerance:
                newly_fixed[coalition] = 0.5 * (
                    minimum_excess + maximum_excess
                )

        if not newly_fixed:
            return NucleolusResult(
                allocation,
                "Numerical failure: no fixed coalition found",
                stage,
            )
        fixed.update(newly_fixed)
        unresolved.difference_update(newly_fixed)

    return NucleolusResult({}, "Iteration limit reached", len(coalitions) + 1)
