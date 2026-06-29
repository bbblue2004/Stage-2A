"""
RAN Sharing Cooperative Game Simulation

This module implements a cooperative game theory model for RAN (Radio Access Network) sharing
among multiple mobile operators. It uses Shapley values and various gain-sharing rules to
determine fair allocation of benefits when operators form coalitions and share infrastructure.

Key concepts:
- Coalition: A group of operators who agree to share their network resources
- Guardian set (l_s): Operators within a coalition who keep their equipment running
- Uniform-until-saturation: An allocation rule that distributes traffic evenly among guardians
- Shapley value: A fair division method based on marginal contributions

Two simulation modes:
- Oracle mode: Full knowledge of traffic at each time step (upper bound on performance)
- Online mode: Predict traffic using historical data (realistic scenario)
"""

from itertools import permutations, combinations
from typing import Any, Callable, Optional
from math import factorial, sqrt

from src.core.utility import single_operator_utility
from src.core.generate_data import OperatorParams, load_scenario
from src.core.optimiser import coalition_value_star, coalition_utility
from src.core.allocate import allocate_uniform_until_saturation
from src.core.profit import shapley_values, payoff_rule1, payoff_rule2, payoff_rule3
from src.core.predict import predict_traffic
from src.data_processing.antenna_metrics import compute_antenna_metrics


def simulate_one_hour_oracle(
    operators: list[OperatorParams],
    traffic: dict[int, list[float]],
    coalition: Optional[list[int]] = None
) -> dict[str, Any]:
    """
    Oracle mode simulation: full knowledge of traffic at each time step.

    This is the "god's eye view" where we know the exact traffic at time t
    when making decisions at time t. Used as an upper bound for comparison.

    For each time step, computes:
    - Coalition value v*(s)
    - Payoffs under all three gain-sharing rules

    Args:
        operators: List of all operator parameters
        traffic: Dictionary mapping operator index to traffic time series
        coalition: Coalition to simulate (default: all operators)

    Returns:
        Dictionary containing:
        - 'time_steps': Number of time steps
        - 'coalition': The coalition being simulated
        - 'v_star': Time series of coalition values
        - 'guardians': Time series of optimal guardian sets
        - 'payoffs_rule1': Per-operator time series under rule 1
        - 'payoffs_rule2': Per-operator time series under rule 2
        - 'payoffs_rule3': Per-operator time series under rule 3
    """
    if coalition is None:
        coalition = list(range(len(operators)))

    num_steps = len(next(iter(traffic.values())))

    # Initialize result structure
    result: dict[str, Any] = {
        'mode': 'oracle',
        'time_steps': num_steps,
        'coalition': coalition,
        'v_star': [],
        'guardians': [],
        'payoffs_rule1': {i: [] for i in coalition},
        'payoffs_rule2': {i: [] for i in coalition},
        'payoffs_rule3': {i: [] for i in coalition},
    }

    # Define v_star callable for Shapley computation
    def v_star_func(s: list[int], t: int) -> float:
        if not s:
            return 0.0
        traffic_at_t = {i: traffic[i][t] for i in range(len(operators))}
        val, _, _ = coalition_value_star(s, operators, traffic_at_t)
        return val

    # Simulate each time step
    for t in range(num_steps):
        # Get traffic at this time step
        traffic_at_t = {i: traffic[i][t] for i in range(len(operators))}

        # Compute v*(coalition)
        v_star_t, guardians_t, _ = coalition_value_star(
            coalition, operators, traffic_at_t
        )
        result['v_star'].append(v_star_t)
        result['guardians'].append(guardians_t)

        # Create v_star callable for this time step
        def v_star_for_t(s: list[int], time_step: int = t) -> float:
            return v_star_func(s, time_step)

        # Compute payoffs under each rule
        payoffs1 = payoff_rule1(coalition, v_star_for_t, operators, traffic_at_t)
        payoffs2 = payoff_rule2(coalition, operators, traffic_at_t)
        payoffs3 = payoff_rule3(coalition, operators, traffic_at_t)

        for i in coalition:
            result['payoffs_rule1'][i].append(payoffs1.get(i, 0.0))
            result['payoffs_rule2'][i].append(payoffs2.get(i, 0.0))
            result['payoffs_rule3'][i].append(payoffs3.get(i, 0.0))

    return result


def simulate_one_hour_online(
    operators: list[OperatorParams],
    traffic: dict[int, list[float]],
    coalition: Optional[list[int]] = None,
    window_size: int = 5,
    safety_margin: float = 0.15
) -> dict[str, Any]:
    """
    Online mode simulation: predict traffic using historical data (0..t-1).

    This is the realistic scenario where we don't know the exact traffic at time t.
    Instead, we use a moving average of past observations to predict future traffic,
    then make decisions based on the prediction.

    To avoid capacity failures (losing traffic is more severe than suboptimal profit),
    predicted traffic is inflated by `safety_margin` before guardian selection.

    At each time step t:
    1. Use history (0..t-1) to predict traffic at t
    2. Inflate prediction by safety_margin for guardian selection
    3. Select guardians based on inflated traffic
    4. Compute actual coalition value using real traffic

    Args:
        operators: List of all operator parameters
        traffic: Dictionary mapping operator index to traffic time series
        coalition: Coalition to simulate (default: all operators)
        window_size: Number of past observations for moving average prediction
        safety_margin: Fraction to inflate predicted traffic for guardian selection (default 0.15 = 15%)

    Returns:
        Dictionary containing:
        - 'time_steps': Number of time steps
        - 'coalition': The coalition being simulated
        - 'v_star': Time series of coalition values (actual, based on decisions)
        - 'guardians': Time series of guardian sets (chosen based on prediction)
        - 'predicted_traffic': Predicted traffic at each time step
        - 'actual_traffic': Actual traffic at each time step
        - 'prediction_errors': Per-operator prediction errors
        - 'payoffs_rule1/2/3': Per-operator time series under each rule
    """
    if coalition is None:
        coalition = list(range(len(operators)))

    num_steps = len(next(iter(traffic.values())))
    num_operators = len(operators)

    # Initialize result structure
    result: dict[str, Any] = {
        'mode': 'online',
        'time_steps': num_steps,
        'coalition': coalition,
        'window_size': window_size,
        'safety_margin': safety_margin,
        'v_star': [],
        'guardians': [],
        'predicted_traffic': [],
        'actual_traffic': [],
        'prediction_errors': {i: [] for i in range(num_operators)},
        'capacity_failures': [],  # Time steps where chosen guardians couldn't handle actual traffic
        'payoffs_rule1': {i: [] for i in coalition},
        'payoffs_rule2': {i: [] for i in coalition},
        'payoffs_rule3': {i: [] for i in coalition},
    }

    # Build history incrementally
    history: dict[int, list[float]] = {i: [] for i in range(num_operators)}

    # Simulate each time step
    for t in range(num_steps):
        # Actual traffic at this time step
        actual_at_t = {i: traffic[i][t] for i in range(num_operators)}
        result['actual_traffic'].append(actual_at_t)

        # Predict traffic based on history (0..t-1)
        if t == 0:
            # No history, use actual traffic for first step (cold start)
            predicted_at_t = actual_at_t.copy()
        else:
            predicted_at_t = predict_traffic(history, window_size)

        result['predicted_traffic'].append(predicted_at_t)

        # Record prediction errors
        for i in range(num_operators):
            error = predicted_at_t[i] - actual_at_t[i]
            result['prediction_errors'][i].append(error)

        # Inflate predicted traffic by safety margin for guardian selection
        margined_traffic = {
            i: predicted_at_t[i] * (1.0 + safety_margin)
            for i in predicted_at_t
        }

        # Make decision based on MARGINED predicted traffic
        _, guardians_t, _ = coalition_value_star(
            coalition, operators, margined_traffic
        )
        result['guardians'].append(guardians_t)

        # But compute actual value using REAL traffic with chosen guardians
        # This represents what actually happens when we apply the decision
        from src.core.optimiser import coalition_utility
        from src.core.allocate import allocate_uniform_until_saturation

        # Check if guardians can handle actual traffic
        total_actual_traffic = sum(actual_at_t[i] for i in coalition)
        guardian_capacity = sum(operators[g].capacity_epsilon for g in guardians_t)

        if guardian_capacity >= total_actual_traffic - 1e-9:
            # Guardians can handle the actual traffic
            actual_v_star = coalition_utility(coalition, guardians_t, operators, actual_at_t)
            # Compute actual allocation for payoff calculation
            capacities = {g: operators[g].capacity_epsilon for g in guardians_t}
            actual_allocation = allocate_uniform_until_saturation(
                guardians_t, capacities, total_actual_traffic
            )
        else:
            # CAPACITY FAILURE: Guardians can't handle actual traffic!
            # In reality, this means service degradation - we can only serve up to capacity
            # Revenue is reduced proportionally to the traffic we can actually serve
            result['capacity_failures'].append({
                't': t,
                'needed': total_actual_traffic,
                'available': guardian_capacity,
                'shortfall': total_actual_traffic - guardian_capacity,
            })
            
            # Compute degraded value: scale revenue by served fraction, keep full costs
            served_fraction = guardian_capacity / total_actual_traffic
            
            # Revenue only for traffic we can serve
            degraded_revenue = sum(
                operators[i].c * actual_at_t[i] * served_fraction
                for i in coalition
            )
            
            # Guardians are fully loaded (rho = 1.0 for all)
            total_variable_cost = sum(operators[g].beta * 1.0 for g in guardians_t)
            total_fixed_cost = sum(operators[g].K for g in guardians_t)
            
            actual_v_star = degraded_revenue - total_variable_cost + total_fixed_cost
            
            # Allocation is at full capacity for each guardian
            actual_allocation = {g: operators[g].capacity_epsilon for g in guardians_t}

        result['v_star'].append(actual_v_star)

        # Compute payoffs using ACTUAL v_star, guardians, and allocation
        def v_star_for_t(s: list[int]) -> float:
            if not s:
                return 0.0
            val, _, _ = coalition_value_star(s, operators, actual_at_t)
            return val

        # Pass actual values to payoff functions
        payoffs1 = payoff_rule1(
            coalition, v_star_for_t, operators, actual_at_t,
            actual_v_star=actual_v_star,
            actual_guardians=guardians_t,
            actual_allocation=actual_allocation
        )
        payoffs2 = payoff_rule2(
            coalition, operators, actual_at_t,
            actual_v_star=actual_v_star,
            actual_guardians=guardians_t
        )
        payoffs3 = payoff_rule3(
            coalition, operators, actual_at_t,
            actual_v_star=actual_v_star
        )

        for i in coalition:
            result['payoffs_rule1'][i].append(payoffs1.get(i, 0.0))
            result['payoffs_rule2'][i].append(payoffs2.get(i, 0.0))
            result['payoffs_rule3'][i].append(payoffs3.get(i, 0.0))

        # Update history for next iteration
        for i in range(num_operators):
            history[i].append(traffic[i][t])

    return result


def compare_oracle_vs_online(
    oracle_result: dict[str, Any],
    online_result: dict[str, Any]
) -> dict[str, Any]:
    """
    Compare oracle mode vs online mode results.

    Args:
        oracle_result: Result from simulate_one_hour_oracle
        online_result: Result from simulate_one_hour_online

    Returns:
        Dictionary containing:
        - 'guardian_agreement': Fraction of time steps with same guardian selection
        - 'value_loss_total': Total value loss (oracle - online)
        - 'value_loss_percent': Value loss as percentage
        - 'prediction_rmse': Root mean square error of traffic prediction
        - 'capacity_failure_count': Number of time steps with capacity failures
        - 'capacity_failures': List of capacity failure details
    """
    num_steps = oracle_result['time_steps']
    num_operators = len(online_result['prediction_errors'])

    # Guardian agreement rate
    agreements = 0
    for t in range(num_steps):
        oracle_guardians = set(oracle_result['guardians'][t])
        online_guardians = set(online_result['guardians'][t])
        if oracle_guardians == online_guardians:
            agreements += 1
    guardian_agreement = agreements / num_steps

    # Value comparison
    oracle_total = sum(oracle_result['v_star'])
    online_total = sum(online_result['v_star'])
    value_loss_total = oracle_total - online_total
    value_loss_percent = (value_loss_total / oracle_total * 100) if oracle_total > 0 else 0.0

    # Prediction RMSE
    total_squared_error = 0.0
    total_count = 0
    for i in range(num_operators):
        for error in online_result['prediction_errors'][i]:
            total_squared_error += error ** 2
            total_count += 1
    prediction_rmse = sqrt(total_squared_error / total_count) if total_count > 0 else 0.0

    # Capacity failures
    capacity_failures = online_result.get('capacity_failures', [])
    capacity_failure_count = len(capacity_failures)

    return {
        'guardian_agreement': guardian_agreement,
        'value_loss_total': value_loss_total,
        'value_loss_percent': value_loss_percent,
        'prediction_rmse': prediction_rmse,
        'oracle_total_value': oracle_total,
        'online_total_value': online_total,
        'capacity_failure_count': capacity_failure_count,
        'capacity_failures': capacity_failures,
    }


def verify_super_additivity(
    operators: list[OperatorParams],
    traffic: dict[int, list[float]],
    max_coalition_size: Optional[int] = None,
    test_times: Optional[list[int]] = None,
    debug: bool = False,
    tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """
    Numerically verify super-additivity of v*(s).

    Tests the full definition: for all disjoint non-empty coalitions S and T,
        v*(S union T) >= v*(S) + v*(T)
    at each selected time step.

    When a violation is found, also evaluates the merged-guardian configuration
    (union of optimal guardian sets for S and T separately).

    Args:
        operators: List of all operator parameters
        traffic: Dictionary mapping operator index to traffic time series
        max_coalition_size: Maximum size of S and T to test (default: all)
        test_times: Time steps to test (default: all steps in traffic series)
        debug: If True, print detailed guardian/allocation info for violations
        tolerance: Numerical tolerance for the inequality check

    Returns:
        List of violation records (empty if super-additivity holds).
    """
    num_operators = len(operators)
    num_steps = len(next(iter(traffic.values())))
    if max_coalition_size is None:
        max_coalition_size = num_operators
    if test_times is None:
        test_times = list(range(num_steps))

    all_coalitions = [
        list(coalition)
        for size in range(1, max_coalition_size + 1)
        for coalition in combinations(range(num_operators), size)
    ]

    disjoint_pairs = [
        (s, t)
        for s in all_coalitions
        for t in all_coalitions
        if not set(s) & set(t) and s < t
    ]

    print("\n" + "=" * 80)
    print("SUPER-ADDITIVITY VERIFICATION")
    print("=" * 80)
    print("Testing property: v*(S U T) >= v*(S) + v*(T) for all disjoint S, T")
    print(
        f"Coalition sizes: 1..{max_coalition_size}, "
        f"disjoint pairs: {len(disjoint_pairs)}, time steps: {test_times}"
    )
    print()

    violations: list[dict[str, Any]] = []

    for t in test_times:
        traffic_at_t = {j: traffic[j][t] for j in range(num_operators)}
        value_cache: dict[tuple[int, ...], tuple[float, list[int], dict[int, float]]] = {}

        def cached_value_star(coalition: list[int]) -> tuple[float, list[int], dict[int, float]]:
            key = tuple(coalition)
            if key not in value_cache:
                value_cache[key] = coalition_value_star(coalition, operators, traffic_at_t)
            return value_cache[key]

        for s, t_coalition in disjoint_pairs:
            s_union_t = sorted(set(s) | set(t_coalition))

            v_s, guardians_s, alloc_s = cached_value_star(s)
            v_t, guardians_t, alloc_t = cached_value_star(t_coalition)
            v_st, guardians_st, alloc_st = cached_value_star(s_union_t)

            if v_st < v_s + v_t - tolerance:
                union_guardians = sorted(set(guardians_s) | set(guardians_t))
                total_traffic_union = sum(traffic_at_t[j] for j in s_union_t)

                try:
                    alloc_union = (
                        allocate_uniform_until_saturation(
                            union_guardians,
                            {g: operators[g].capacity_epsilon for g in union_guardians},
                            total_traffic_union,
                        )
                        if union_guardians
                        else {}
                    )
                except ValueError:
                    alloc_union = {}

                try:
                    union_value = coalition_utility(
                        s_union_t, union_guardians, operators, traffic_at_t
                    )
                except ValueError:
                    union_value = None

                violations.append({
                    'time': t,
                    'S': s,
                    'T': t_coalition,
                    'S_union_T': s_union_t,
                    'v(S)': v_s,
                    'v(T)': v_t,
                    'v(S_union_T)': v_st,
                    'v(S) + v(T)': v_s + v_t,
                    'deficit': v_s + v_t - v_st,
                    'guardians_S': guardians_s,
                    'alloc_S': alloc_s,
                    'guardians_T': guardians_t,
                    'alloc_T': alloc_t,
                    'guardians_S_union_T': guardians_st,
                    'alloc_S_union_T': alloc_st,
                    'union_guardians': union_guardians,
                    'alloc_union_guardians': alloc_union,
                    'union_value': union_value,
                })

    total_tests = len(disjoint_pairs) * len(test_times)
    print(f"Disjoint coalition pairs tested: {len(disjoint_pairs)}")
    print(f"Total inequality checks: {total_tests}")
    print()

    if not violations:
        print("SUCCESS: Super-additivity holds for all tested pairs!")
        print()
    else:
        print(f"VIOLATIONS FOUND: {len(violations)} super-additivity violations")
        print()
        print(
            f"{'t':>3} | {'S':>12} | {'T':>12} | {'v(S)':>10} | {'v(T)':>10} | "
            f"{'v(S U T)':>12} | {'v(S)+v(T)':>12} | {'Deficit':>10}"
        )
        print("-" * 100)
        for v in violations[:20]:
            print(
                f"{v['time']:>3} | {str(v['S']):>12} | {str(v['T']):>12} | "
                f"{v['v(S)']:>10.2f} | {v['v(T)']:>10.2f} | "
                f"{v['v(S_union_T)']:>12.2f} | {v['v(S) + v(T)']:>12.2f} | "
                f"{v['deficit']:>10.2f}"
            )
        if len(violations) > 20:
            print(f"... and {len(violations) - 20} more violations")
        print()

        if debug:
            print("Detailed debug for first violations:")
            for v in violations[:5]:
                print("-" * 60)
                print(
                    f"t={v['time']}, S={v['S']}, T={v['T']}, "
                    f"S U T={v['S_union_T']}"
                )
                print(
                    f"v(S)={v['v(S)']:.6f}, v(T)={v['v(T)']:.6f}, "
                    f"v(S U T)={v['v(S_union_T)']:.6f}, "
                    f"v(S)+v(T)={v['v(S) + v(T)']:.6f}, deficit={v['deficit']:.6f}"
                )
                print(f" guardians_S: {v['guardians_S']}\n alloc_S: {v['alloc_S']}")
                print(f" guardians_T: {v['guardians_T']}\n alloc_T: {v['alloc_T']}")
                print(
                    f" guardians_S U T (optimal): {v['guardians_S_union_T']}\n"
                    f" alloc_S U T: {v['alloc_S_union_T']}"
                )
                print(
                    f" union_guardians (guardians_S | guardians_T): {v['union_guardians']}"
                )
                print(f" alloc_union_guardians: {v['alloc_union_guardians']}")
                print(
                    " union_value (coalition_utility with union_guardians): "
                    f"{v['union_value']}"
                )
            print()

        deficits = [v['deficit'] for v in violations]
        print("Violation statistics:")
        print(f"  Min deficit:     {min(deficits):>10.4f}")
        print(f"  Max deficit:     {max(deficits):>10.4f}")
        print(f"  Mean deficit:    {sum(deficits) / len(deficits):>10.4f}")
        print()

    return violations



def main():
    scenario = load_scenario("realistic")
    ops = scenario.operators
    traffic_data = scenario.traffic
    coalition = scenario.coalition
    num_operators = len(ops)
    num_steps = scenario.num_steps

    print("=" * 60)
    print(f"RAN Sharing Cooperative Game Simulation — scenario: {scenario.name}, {scenario.horizon_label()}")
    print("=" * 60)

    print("\nOperator Parameters:")
    for i, op in enumerate(ops):
        print(f"  {op.name}: ε={op.capacity_epsilon}, c={op.c}, "
              f"β={op.beta}, K={op.K}")

    sample_ts = scenario.sample_steps(7)
    print(f"\nTraffic at {', '.join(scenario.step_label(t) for t in sample_ts)}:")
    for i in coalition:
        vals = ", ".join(f"T({t})={traffic_data[i][t]:.2f}" for t in sample_ts)
        print(f"  Operator {i}: {vals}")

    # Run both simulation modes
    print("\n" + "=" * 60)
    print("Running Oracle Mode (god's eye view)...")
    print("=" * 60)
    oracle_result = simulate_one_hour_oracle(ops, traffic_data, coalition)

    safety_margin = 0.15
    print("\n" + "=" * 60)
    print(f"Running Online Mode (prediction-based, {safety_margin:.0%} safety margin)...")
    print("=" * 60)
    online_result = simulate_one_hour_online(
        ops, traffic_data, coalition, window_size=5, safety_margin=safety_margin
    )

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison: Oracle vs Online")
    print("=" * 60)
    comparison = compare_oracle_vs_online(oracle_result, online_result)

    print(f"\n  Safety Margin:           {safety_margin:.0%}")
    print(f"  Guardian Agreement Rate: {comparison['guardian_agreement']:.1%}")
    print(f"  Oracle Total Value:      {comparison['oracle_total_value']:.2f}")
    print(f"  Online Total Value:      {comparison['online_total_value']:.2f}")
    print(f"  Value Loss:              {comparison['value_loss_total']:.2f} ({comparison['value_loss_percent']:.2f}%)")
    print(f"  Prediction RMSE:         {comparison['prediction_rmse']:.4f}")
    print(f"  Capacity Failures:       {comparison['capacity_failure_count']} time steps")
    
    # Show capacity failure details if any
    if comparison['capacity_failures']:
        print("\n" + "-" * 60)
        print("Capacity Failures (prediction underestimated traffic)")
        print("-" * 60)
        for failure in comparison['capacity_failures']:
            print(f"  t={failure['t']:2d}: needed {failure['needed']:.2f}, "
                  f"available {failure['available']:.2f}, "
                  f"shortfall {failure['shortfall']:.2f}")

    # Show detailed comparison for selected time steps
    print("\n" + "-" * 60)
    print("Detailed Comparison (selected time steps)")
    print("-" * 60)
    print(f"{'t':>3} | {'Oracle Guardians':<18} | {'Online Guardians':<18} | {'Match':<5} | {'Oracle v*':>10} | {'Online v*':>10}")
    print("-" * 80)

    for t in sample_ts:
        oracle_g = oracle_result['guardians'][t]
        online_g = online_result['guardians'][t]
        match = "✓" if set(oracle_g) == set(online_g) else "✗"
        oracle_v = oracle_result['v_star'][t]
        online_v = online_result['v_star'][t]
        print(f"{t:>3} | {str(oracle_g):<18} | {str(online_g):<18} | {match:^5} | {oracle_v:>10.2f} | {online_v:>10.2f}")

    # Show prediction errors
    print("\n" + "-" * 60)
    print("Prediction Errors (selected time steps)")
    print("-" * 60)
    print(f"{'t':>3} | ", end="")
    for i in range(num_operators):
        print(f"{'Op'+str(i)+' Pred':>10} {'Actual':>10} {'Error':>8} | ", end="")
    print()

    pred_sample_ts = [t for t in scenario.sample_steps(5) if t > 0]
    for t in pred_sample_ts:
        print(f"{t:>3} | ", end="")
        for i in range(num_operators):
            pred = online_result['predicted_traffic'][t][i]
            actual = online_result['actual_traffic'][t][i]
            error = online_result['prediction_errors'][i][t]
            print(f"{pred:>10.2f} {actual:>10.2f} {error:>+8.2f} | ", end="")
        print()

    # Compute non-cooperative standalone profits (each operator alone)
    standalone_profits: dict[int, float] = {}
    for i in coalition:
        total = 0.0
        for t in range(num_steps):
            T_i = traffic_data[i][t]
            rho_i = min(1.0, T_i / ops[i].capacity_epsilon)
            total += single_operator_utility(ops[i].c, T_i, ops[i].beta, rho_i, ops[i].K)
        standalone_profits[i] = total
    standalone_total = sum(standalone_profits.values())

    # Show per-operator total profit
    print("\n" + "=" * 60)
    print(f"Per-Operator Total Profit (summed over {num_steps} steps, {scenario.horizon_label()})")
    print("=" * 60)

    print("\n--- Non-Cooperative (each operator alone) ---")
    print(f"{'Operator':<12} | {'Standalone':>12}")
    print("-" * 28)
    for i in coalition:
        print(f"{ops[i].name:<12} | {standalone_profits[i]:>12.2f}")
    print("-" * 28)
    print(f"{'Total':<12} | {standalone_total:>12.2f}")

    print("\n--- Oracle Mode ---")
    print(f"{'Operator':<12} | {'Rule 1':>12} | {'Rule 2':>12} | {'Rule 3':>12} | {'vs Alone':>12}")
    print("-" * 70)
    for i in coalition:
        r1 = sum(oracle_result['payoffs_rule1'][i])
        r2 = sum(oracle_result['payoffs_rule2'][i])
        r3 = sum(oracle_result['payoffs_rule3'][i])
        gain = r1 - standalone_profits[i]
        print(f"{ops[i].name:<12} | {r1:>12.2f} | {r2:>12.2f} | {r3:>12.2f} | {gain:>+12.2f}")
    print("-" * 70)
    total_r1 = sum(sum(oracle_result['payoffs_rule1'][i]) for i in coalition)
    total_r2 = sum(sum(oracle_result['payoffs_rule2'][i]) for i in coalition)
    total_r3 = sum(sum(oracle_result['payoffs_rule3'][i]) for i in coalition)
    print(f"{'Total':<12} | {total_r1:>12.2f} | {total_r2:>12.2f} | {total_r3:>12.2f} | {total_r1 - standalone_total:>+12.2f}")

    print("\n--- Online Mode ---")
    print(f"{'Operator':<12} | {'Rule 1':>12} | {'Rule 2':>12} | {'Rule 3':>12} | {'vs Alone':>12}")
    print("-" * 70)
    for i in coalition:
        r1 = sum(online_result['payoffs_rule1'][i])
        r2 = sum(online_result['payoffs_rule2'][i])
        r3 = sum(online_result['payoffs_rule3'][i])
        gain = r1 - standalone_profits[i]
        print(f"{ops[i].name:<12} | {r1:>12.2f} | {r2:>12.2f} | {r3:>12.2f} | {gain:>+12.2f}")
    print("-" * 70)
    total_r1 = sum(sum(online_result['payoffs_rule1'][i]) for i in coalition)
    total_r2 = sum(sum(online_result['payoffs_rule2'][i]) for i in coalition)
    total_r3 = sum(sum(online_result['payoffs_rule3'][i]) for i in coalition)
    print(f"{'Total':<12} | {total_r1:>12.2f} | {total_r2:>12.2f} | {total_r3:>12.2f} | {total_r1 - standalone_total:>+12.2f}")



# Example usage
if __name__ == "__main__":
    scenario = load_scenario("realistic")
    ops = scenario.operators
    traffic_data = scenario.traffic
    num_operators = len(ops)

    # Verify super-additivity of v*
    print("\n" + "=" * 60)
    print("Verifying Super-Additivity Property...")
    print("=" * 60)
    verify_super_additivity(
        ops,
        traffic_data,
        max_coalition_size=num_operators,
        test_times=scenario.sample_steps(7),
        debug=True,
    )