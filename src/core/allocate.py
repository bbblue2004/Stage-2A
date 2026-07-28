"""Optimal traffic allocation for fixed guardians."""


def allocate_greedy(
    guardians: list[int],
    capacities: dict[int, float],
    gammas: dict[int, float],
    demand: float,
) -> dict[int, float]:
    """Fill guardians by increasing unit cost, with deterministic tie-breaking."""
    if sum(capacities[i] for i in guardians) < demand - 1e-9:
        raise ValueError("Insufficient guardian capacity")

    allocation: dict[int, float] = {}
    remaining = demand
    for guardian in sorted(guardians, key=lambda i: (gammas[i], i)):
        allocation[guardian] = min(capacities[guardian], remaining)
        remaining -= allocation[guardian]
        if remaining <= 1e-9:
            break
    return allocation
