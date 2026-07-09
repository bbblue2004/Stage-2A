from math import sqrt
from typing import Any, Optional

from src.core.allocate import allocate_uniform_until_saturation
from src.core.generate_data import OperatorParams
from src.core.optimiser import coalition_utility, coalition_value_star
from src.core.predict import predict_traffic
from src.core.profit import payoff_rule1, payoff_rule2, payoff_rule3


def simulate_one_hour_oracle(
    operators: list[OperatorParams],
    traffic: dict[int, list[float]],
    coalition: Optional[list[int]] = None,
) -> dict[str, Any]:
    if coalition is None:
        coalition = list(range(len(operators)))

    num_steps = len(next(iter(traffic.values())))
    result: dict[str, Any] = {
        "mode": "oracle",
        "time_steps": num_steps,
        "coalition": coalition,
        "v_star": [],
        "guardians": [],
        "payoffs_rule1": {i: [] for i in coalition},
        "payoffs_rule2": {i: [] for i in coalition},
        "payoffs_rule3": {i: [] for i in coalition},
    }

    def v_star_func(s: list[int], t: int) -> float:
        if not s:
            return 0.0
        traffic_at_t = {i: traffic[i][t] for i in range(len(operators))}
        val, _, _ = coalition_value_star(s, operators, traffic_at_t)
        return val

    for t in range(num_steps):
        traffic_at_t = {i: traffic[i][t] for i in range(len(operators))}
        v_star_t, guardians_t, _ = coalition_value_star(coalition, operators, traffic_at_t)
        result["v_star"].append(v_star_t)
        result["guardians"].append(guardians_t)

        def v_star_for_t(s: list[int], time_step: int = t) -> float:
            return v_star_func(s, time_step)

        payoffs1 = payoff_rule1(coalition, v_star_for_t, operators, traffic_at_t)
        payoffs2 = payoff_rule2(coalition, operators, traffic_at_t)
        payoffs3 = payoff_rule3(coalition, operators, traffic_at_t)

        for i in coalition:
            result["payoffs_rule1"][i].append(payoffs1.get(i, 0.0))
            result["payoffs_rule2"][i].append(payoffs2.get(i, 0.0))
            result["payoffs_rule3"][i].append(payoffs3.get(i, 0.0))

    return result


def simulate_one_hour_online(
    operators: list[OperatorParams],
    traffic: dict[int, list[float]],
    coalition: Optional[list[int]] = None,
    window_size: int = 5,
    safety_margin: float = 0.15,
) -> dict[str, Any]:
    if coalition is None:
        coalition = list(range(len(operators)))

    num_steps = len(next(iter(traffic.values())))
    num_operators = len(operators)
    result: dict[str, Any] = {
        "mode": "online",
        "time_steps": num_steps,
        "coalition": coalition,
        "window_size": window_size,
        "safety_margin": safety_margin,
        "v_star": [],
        "guardians": [],
        "predicted_traffic": [],
        "actual_traffic": [],
        "prediction_errors": {i: [] for i in range(num_operators)},
        "capacity_failures": [],
        "payoffs_rule1": {i: [] for i in coalition},
        "payoffs_rule2": {i: [] for i in coalition},
        "payoffs_rule3": {i: [] for i in coalition},
    }

    history: dict[int, list[float]] = {i: [] for i in range(num_operators)}

    for t in range(num_steps):
        actual_at_t = {i: traffic[i][t] for i in range(num_operators)}
        result["actual_traffic"].append(actual_at_t)

        if t == 0:
            predicted_at_t = actual_at_t.copy()
        else:
            predicted_at_t = predict_traffic(history, window_size)

        result["predicted_traffic"].append(predicted_at_t)

        for i in range(num_operators):
            error = predicted_at_t[i] - actual_at_t[i]
            result["prediction_errors"][i].append(error)

        margined_traffic = {
            i: predicted_at_t[i] * (1.0 + safety_margin) for i in predicted_at_t
        }
        _, guardians_t, _ = coalition_value_star(coalition, operators, margined_traffic)
        result["guardians"].append(guardians_t)

        total_actual_traffic = sum(actual_at_t[i] for i in coalition)
        guardian_capacity = sum(operators[g].capacity_epsilon for g in guardians_t)

        if guardian_capacity >= total_actual_traffic - 1e-9:
            actual_v_star = coalition_utility(coalition, guardians_t, operators, actual_at_t)
            capacities = {g: operators[g].capacity_epsilon for g in guardians_t}
            actual_allocation = allocate_uniform_until_saturation(
                guardians_t, capacities, total_actual_traffic
            )
        else:
            result["capacity_failures"].append(
                {
                    "t": t,
                    "needed": total_actual_traffic,
                    "available": guardian_capacity,
                    "shortfall": total_actual_traffic - guardian_capacity,
                }
            )
            served_fraction = guardian_capacity / total_actual_traffic
            degraded_revenue = sum(
                operators[i].c * actual_at_t[i] * served_fraction for i in coalition
            )
            total_variable_cost = sum(operators[g].beta * 1.0 for g in guardians_t)
            total_fixed_cost = sum(operators[g].K for g in guardians_t)
            actual_v_star = degraded_revenue - total_variable_cost + total_fixed_cost
            actual_allocation = {g: operators[g].capacity_epsilon for g in guardians_t}

        result["v_star"].append(actual_v_star)

        def v_star_for_t(s: list[int]) -> float:
            if not s:
                return 0.0
            val, _, _ = coalition_value_star(s, operators, actual_at_t)
            return val

        payoffs1 = payoff_rule1(
            coalition,
            v_star_for_t,
            operators,
            actual_at_t,
            actual_v_star=actual_v_star,
            actual_guardians=guardians_t,
            actual_allocation=actual_allocation,
        )
        payoffs2 = payoff_rule2(
            coalition,
            operators,
            actual_at_t,
            actual_v_star=actual_v_star,
            actual_guardians=guardians_t,
        )
        payoffs3 = payoff_rule3(
            coalition,
            operators,
            actual_at_t,
            actual_v_star=actual_v_star,
        )

        for i in coalition:
            result["payoffs_rule1"][i].append(payoffs1.get(i, 0.0))
            result["payoffs_rule2"][i].append(payoffs2.get(i, 0.0))
            result["payoffs_rule3"][i].append(payoffs3.get(i, 0.0))

        for i in range(num_operators):
            history[i].append(traffic[i][t])

    return result


def compare_oracle_vs_online(
    oracle_result: dict[str, Any], online_result: dict[str, Any]
) -> dict[str, Any]:
    num_steps = oracle_result["time_steps"]
    num_operators = len(online_result["prediction_errors"])

    agreements = 0
    for t in range(num_steps):
        oracle_guardians = set(oracle_result["guardians"][t])
        online_guardians = set(online_result["guardians"][t])
        if oracle_guardians == online_guardians:
            agreements += 1
    guardian_agreement = agreements / num_steps

    oracle_total = sum(oracle_result["v_star"])
    online_total = sum(online_result["v_star"])
    value_loss_total = oracle_total - online_total
    value_loss_percent = (value_loss_total / oracle_total * 100) if oracle_total > 0 else 0.0

    total_squared_error = 0.0
    total_count = 0
    for i in range(num_operators):
        for error in online_result["prediction_errors"][i]:
            total_squared_error += error**2
            total_count += 1
    prediction_rmse = sqrt(total_squared_error / total_count) if total_count > 0 else 0.0

    capacity_failures = online_result.get("capacity_failures", [])
    capacity_failure_count = len(capacity_failures)

    return {
        "guardian_agreement": guardian_agreement,
        "value_loss_total": value_loss_total,
        "value_loss_percent": value_loss_percent,
        "prediction_rmse": prediction_rmse,
        "oracle_total_value": oracle_total,
        "online_total_value": online_total,
        "capacity_failure_count": capacity_failure_count,
        "capacity_failures": capacity_failures,
    }
