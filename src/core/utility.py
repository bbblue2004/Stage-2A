"""Single-operator utility v(A_i) = c·T − β·ρ·Δt − K·Δt (see optimiser.coalition_utility for coalitions)."""


def single_operator_utility(
    c: float,
    T: float,
    beta: float,
    rho: float,
    K: float,
    duration_hours: float = 1.0,
) -> float:
    return c * T - beta * rho * duration_hours - K * duration_hours
