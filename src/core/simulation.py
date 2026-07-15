"""
Day-long evaluation of the cooperative RAN-sharing game.

Hourly loop: v*(S), rho, surplus, optimal guardians.
Least-core LP runs once at the end on summed hourly coalition values.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from src.core.generate_data import OperatorParams, Scenario
from src.core.optimiser import coalition_value_star
from src.core.profit import build_v_star_map, least_core_allocation
from src.core.utility import single_operator_utility

HOURLY_DURATION_HOURS = 1.0


def _standalone_profit(
    op: OperatorParams,
    traffic: float,
    duration_hours: float,
) -> float:
    rho = min(1.0, traffic / op.capacity_epsilon) if op.capacity_epsilon > 0 else 0.0
    return single_operator_utility(op.c, traffic, op.beta, rho, op.K, duration_hours)


def _mean_rho(
    ops: list[OperatorParams],
    traffic_at_t: dict[int, float],
    coalition: list[int],
) -> float:
    rhos = [
        min(1.0, traffic_at_t[i] / ops[i].capacity_epsilon)
        if ops[i].capacity_epsilon > 0
        else 0.0
        for i in coalition
    ]
    return sum(rhos) / len(rhos) if rhos else 0.0


def _guardian_label(guardians: list[int]) -> str:
    if not guardians:
        return "[]"
    return "[" + ", ".join(str(g + 1) for g in guardians) + "]"


def evaluate_day(
    scenario: Scenario,
    on_hour: Callable[[dict[str, Any]], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """
    Run the grand-coalition game at each clock hour; least-core once at the end.

    ``on_progress(current, total)`` fires after each hour is computed.
    ``on_hour`` fires once per hour after the daily LP (includes LC gain share).
    """
    def phase(msg: str) -> None:
        if on_phase:
            on_phase(msg)

    ops = scenario.operators
    coalition = scenario.coalition
    num_hours = int(scenario.horizon_hours)

    payoffs: dict[str, dict[int, float]] = {
        "standalone": {i: 0.0 for i in coalition},
        "least_core": {i: 0.0 for i in coalition},
    }
    hourly_rows: list[dict[str, Any]] = []
    daily_v_star_map: dict[tuple[int, ...], float] = defaultdict(float)
    prev_guardians: tuple[int, ...] | None = None
    guardian_changes = 0

    phase(f"Evaluating {num_hours} clock hours...")
    for hour in range(num_hours):
        traffic_at_t = scenario.hourly_traffic_means(hour)

        for i in coalition:
            payoffs["standalone"][i] += _standalone_profit(
                ops[i], traffic_at_t[i], HOURLY_DURATION_HOURS
            )

        v_star_map = build_v_star_map(
            ops, traffic_at_t, coalition, duration_hours=HOURLY_DURATION_HOURS
        )
        for key, value in v_star_map.items():
            daily_v_star_map[key] += value

        v_star_h, guardians_h, _ = coalition_value_star(
            coalition, ops, traffic_at_t, duration_hours=HOURLY_DURATION_HOURS
        )
        guardians_key = tuple(sorted(guardians_h))
        changed = prev_guardians is not None and guardians_key != prev_guardians
        if changed:
            guardian_changes += 1
        prev_guardians = guardians_key

        solo_v_star_h = sum(v_star_map[(i,)] for i in coalition)
        hourly_rows.append(
            {
                "hour": hour,
                "rho_mean": _mean_rho(ops, traffic_at_t, coalition),
                "surplus": v_star_h - solo_v_star_h,
                "guardians": guardians_h,
                "guardians_label": _guardian_label(guardians_h),
                "guardians_changed": changed,
            }
        )
        if on_progress:
            on_progress(hour + 1, num_hours)

    phase("Running least-core LP (daily allocation)...")
    lc = least_core_allocation(coalition, dict(daily_v_star_map))
    for i in coalition:
        payoffs["least_core"][i] = lc.payoffs[i]

    total_surplus = sum(r["surplus"] for r in hourly_rows)
    total_lc_gain = sum(payoffs["least_core"].values()) - sum(payoffs["standalone"].values())
    for row in hourly_rows:
        if total_surplus > 1e-9:
            row["lc_gain"] = row["surplus"] / total_surplus * total_lc_gain
        else:
            row["lc_gain"] = 0.0
        if on_hour:
            on_hour(row)

    return {
        "payoffs": payoffs,
        "hourly": hourly_rows,
        "guardian_changes": guardian_changes,
        "total_lc_gain": total_lc_gain,
        "least_core_summary": {
            "status": lc.status,
            "epsilon": lc.epsilon,
            "feasible": lc.status == "Optimal",
        },
    }
