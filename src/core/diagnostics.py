from itertools import combinations
from typing import Any, Optional

from src.core.allocate import allocate_uniform_until_saturation
from src.core.generate_data import OperatorParams
from src.core.optimiser import coalition_utility, coalition_value_star


def verify_super_additivity(
    operators: list[OperatorParams],
    traffic: dict[int, list[float]],
    max_coalition_size: Optional[int] = None,
    test_times: Optional[list[int]] = None,
    debug: bool = False,
    tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
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

                violations.append(
                    {
                        "time": t,
                        "S": s,
                        "T": t_coalition,
                        "S_union_T": s_union_t,
                        "v(S)": v_s,
                        "v(T)": v_t,
                        "v(S_union_T)": v_st,
                        "v(S) + v(T)": v_s + v_t,
                        "deficit": v_s + v_t - v_st,
                        "guardians_S": guardians_s,
                        "alloc_S": alloc_s,
                        "guardians_T": guardians_t,
                        "alloc_T": alloc_t,
                        "guardians_S_union_T": guardians_st,
                        "alloc_S_union_T": alloc_st,
                        "union_guardians": union_guardians,
                        "alloc_union_guardians": alloc_union,
                        "union_value": union_value,
                    }
                )

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

        deficits = [v["deficit"] for v in violations]
        print("Violation statistics:")
        print(f"  Min deficit:     {min(deficits):>10.4f}")
        print(f"  Max deficit:     {max(deficits):>10.4f}")
        print(f"  Mean deficit:    {sum(deficits) / len(deficits):>10.4f}")
        print()

    return violations
