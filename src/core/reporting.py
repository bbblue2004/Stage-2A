from typing import Any

from src.core.generate_data import OperatorParams, Scenario
from src.core.simulation import (
    compare_oracle_vs_online,
    simulate_one_hour_online,
    simulate_one_hour_oracle,
)
from src.core.utility import single_operator_utility


def run_simulation_report(
    scenario: Scenario,
    safety_margin: float = 0.15,
    window_size: int = 5,
) -> dict[str, Any]:
    ops = scenario.operators
    traffic_data = scenario.traffic
    coalition = scenario.coalition
    num_operators = len(ops)
    num_steps = scenario.num_steps

    print("=" * 60)
    print(
        f"RAN Sharing Cooperative Game Simulation - scenario: {scenario.name}, "
        f"{scenario.horizon_label()}"
    )
    print("=" * 60)

    print("\nOperator Parameters:")
    for op in ops:
        print(
            f"  {op.name}: epsilon={op.capacity_epsilon}, c={op.c}, "
            f"beta={op.beta}, K={op.K}"
        )

    sample_ts = scenario.sample_steps(7)
    print(f"\nTraffic at {', '.join(scenario.step_label(t) for t in sample_ts)}:")
    for i in coalition:
        vals = ", ".join(f"T({t})={traffic_data[i][t]:.2f}" for t in sample_ts)
        print(f"  Operator {i}: {vals}")

    print("\n" + "=" * 60)
    print("Running Oracle Mode (god's eye view)...")
    print("=" * 60)
    oracle_result = simulate_one_hour_oracle(ops, traffic_data, coalition)

    print("\n" + "=" * 60)
    print(
        f"Running Online Mode (prediction-based, {safety_margin:.0%} safety margin)..."
    )
    print("=" * 60)
    online_result = simulate_one_hour_online(
        ops, traffic_data, coalition, window_size=window_size, safety_margin=safety_margin
    )

    print("\n" + "=" * 60)
    print("Comparison: Oracle vs Online")
    print("=" * 60)
    comparison = compare_oracle_vs_online(oracle_result, online_result)

    print(f"\n  Safety Margin:           {safety_margin:.0%}")
    print(f"  Guardian Agreement Rate: {comparison['guardian_agreement']:.1%}")
    print(f"  Oracle Total Value:      {comparison['oracle_total_value']:.2f}")
    print(f"  Online Total Value:      {comparison['online_total_value']:.2f}")
    print(
        f"  Value Loss:              {comparison['value_loss_total']:.2f} "
        f"({comparison['value_loss_percent']:.2f}%)"
    )
    print(f"  Prediction RMSE:         {comparison['prediction_rmse']:.4f}")
    print(
        f"  Capacity Failures:       {comparison['capacity_failure_count']} time steps"
    )

    if comparison["capacity_failures"]:
        print("\n" + "-" * 60)
        print("Capacity Failures (prediction underestimated traffic)")
        print("-" * 60)
        for failure in comparison["capacity_failures"]:
            print(
                f"  t={failure['t']:2d}: needed {failure['needed']:.2f}, "
                f"available {failure['available']:.2f}, "
                f"shortfall {failure['shortfall']:.2f}"
            )

    print("\n" + "-" * 60)
    print("Detailed Comparison (selected time steps)")
    print("-" * 60)
    print(
        f"{'t':>3} | {'Oracle Guardians':<18} | {'Online Guardians':<18} | "
        f"{'Match':<5} | {'Oracle v*':>10} | {'Online v*':>10}"
    )
    print("-" * 80)
    for t in sample_ts:
        oracle_g = oracle_result["guardians"][t]
        online_g = online_result["guardians"][t]
        match = "Y" if set(oracle_g) == set(online_g) else "N"
        oracle_v = oracle_result["v_star"][t]
        online_v = online_result["v_star"][t]
        print(
            f"{t:>3} | {str(oracle_g):<18} | {str(online_g):<18} | "
            f"{match:^5} | {oracle_v:>10.2f} | {online_v:>10.2f}"
        )

    print("\n" + "-" * 60)
    print("Prediction Errors (selected time steps)")
    print("-" * 60)
    print(f"{'t':>3} | ", end="")
    for i in range(num_operators):
        print(f"{'Op' + str(i) + ' Pred':>10} {'Actual':>10} {'Error':>8} | ", end="")
    print()

    pred_sample_ts = [t for t in scenario.sample_steps(5) if t > 0]
    for t in pred_sample_ts:
        print(f"{t:>3} | ", end="")
        for i in range(num_operators):
            pred = online_result["predicted_traffic"][t][i]
            actual = online_result["actual_traffic"][t][i]
            error = online_result["prediction_errors"][i][t]
            print(f"{pred:>10.2f} {actual:>10.2f} {error:>+8.2f} | ", end="")
        print()

    standalone_profits = _compute_standalone_profits(ops, coalition, traffic_data, num_steps)
    standalone_total = sum(standalone_profits.values())

    print("\n" + "=" * 60)
    print(
        f"Per-Operator Total Profit (summed over {num_steps} steps, "
        f"{scenario.horizon_label()})"
    )
    print("=" * 60)

    print("\n--- Non-Cooperative (each operator alone) ---")
    print(f"{'Operator':<12} | {'Standalone':>12}")
    print("-" * 28)
    for i in coalition:
        print(f"{ops[i].name:<12} | {standalone_profits[i]:>12.2f}")
    print("-" * 28)
    print(f"{'Total':<12} | {standalone_total:>12.2f}")

    _print_mode_profit_table("Oracle Mode", coalition, ops, oracle_result, standalone_profits)
    _print_mode_profit_table("Online Mode", coalition, ops, online_result, standalone_profits)

    return {
        "oracle_result": oracle_result,
        "online_result": online_result,
        "comparison": comparison,
        "standalone_profits": standalone_profits,
    }


def _compute_standalone_profits(
    ops: list[OperatorParams],
    coalition: list[int],
    traffic_data: dict[int, list[float]],
    num_steps: int,
) -> dict[int, float]:
    standalone_profits: dict[int, float] = {}
    for i in coalition:
        total = 0.0
        for t in range(num_steps):
            t_i = traffic_data[i][t]
            rho_i = min(1.0, t_i / ops[i].capacity_epsilon)
            total += single_operator_utility(ops[i].c, t_i, ops[i].beta, rho_i, ops[i].K)
        standalone_profits[i] = total
    return standalone_profits


def _print_mode_profit_table(
    mode_name: str,
    coalition: list[int],
    ops: list[OperatorParams],
    result: dict[str, Any],
    standalone_profits: dict[int, float],
) -> None:
    print(f"\n--- {mode_name} ---")
    print(
        f"{'Operator':<12} | {'Rule 1':>12} | {'Rule 2':>12} | {'Rule 3':>12} | "
        f"{'vs Alone':>12}"
    )
    print("-" * 70)
    for i in coalition:
        r1 = sum(result["payoffs_rule1"][i])
        r2 = sum(result["payoffs_rule2"][i])
        r3 = sum(result["payoffs_rule3"][i])
        gain = r1 - standalone_profits[i]
        print(
            f"{ops[i].name:<12} | {r1:>12.2f} | {r2:>12.2f} | "
            f"{r3:>12.2f} | {gain:>+12.2f}"
        )
    print("-" * 70)
    total_r1 = sum(sum(result["payoffs_rule1"][i]) for i in coalition)
    total_r2 = sum(sum(result["payoffs_rule2"][i]) for i in coalition)
    total_r3 = sum(sum(result["payoffs_rule3"][i]) for i in coalition)
    standalone_total = sum(standalone_profits.values())
    print(
        f"{'Total':<12} | {total_r1:>12.2f} | {total_r2:>12.2f} | {total_r3:>12.2f} | "
        f"{total_r1 - standalone_total:>+12.2f}"
    )
