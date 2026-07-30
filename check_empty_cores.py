"""Count empty period cores among field antennas."""

import argparse
from collections import defaultdict
from statistics import fmean

from src.core.game import (
    allocation_check,
    bondareva_shapley_test,
    build_cost_map,
    build_savings_game,
    convexity_test,
    core_allocation,
    shapley_value,
)
from src.core.generate_data import (
    DEFAULT_ELECTRICITY_PRICE_PER_KWH,
    Scenario,
    build_scenario_from_days,
)
from src.core.time_window import (
    DEFAULT_END_HOUR,
    DEFAULT_START_HOUR,
    inclusive_hour_window,
    validate_hours,
)
from src.data_processing.data_loader import FULL_CSV_PATH, iter_records


def load_first_antenna_days(
    count: int | None = None,
    num_days: int = 5,
) -> tuple[
    int,
    list[tuple[int, str, list[list[float]], list[list[float]]]],
    list[tuple[int, str, str]],
]:
    """Load the requested antennas, or all antennas, in one CSV pass."""
    cells: dict[
        str,
        dict[tuple[object, int], list[tuple[float, float]]],
    ] = {}

    for dt, antenna_id, traffic, power in iter_records(FULL_CSV_PATH):
        if antenna_id not in cells:
            if count is not None and len(cells) >= count:
                continue
            cells[antenna_id] = defaultdict(list)
        cells[antenna_id][(dt.date(), dt.hour)].append((traffic, power))

    if count is not None and len(cells) < count:
        raise ValueError(f"Only {len(cells)} antennas found; {count} requested")

    result = []
    errors = []
    for index, (antenna_id, samples) in enumerate(cells.items(), start=1):
        days = sorted({day for day, _ in samples})[:num_days]
        if len(days) < num_days:
            errors.append(
                (index, antenna_id, f"only {len(days)} days available")
            )
            continue
        if any((day, hour) not in samples for day in days for hour in range(24)):
            errors.append((index, antenna_id, "missing hourly observations"))
            continue

        traffic_days = [
            [fmean(value[0] for value in samples[(day, hour)]) for hour in range(24)]
            for day in days
        ]
        power_days = [
            [fmean(value[1] for value in samples[(day, hour)]) for hour in range(24)]
            for day in days
        ]
        result.append((index, antenna_id, traffic_days, power_days))
    return len(cells), result, errors


def period_savings_game(
    scenario: Scenario,
    hours: tuple[int, ...],
) -> dict[tuple[int, ...], float]:
    """Aggregate coalition costs before constructing the period savings game."""
    players = scenario.coalition
    selected_hours = validate_hours(hours, scenario.num_hours)
    costs: dict[tuple[int, ...], float] = defaultdict(float)
    for hour in selected_hours:
        hourly_costs = build_cost_map(
            scenario.operators,
            scenario.demands_at_hour(hour),
            players,
        )
        for coalition, value in hourly_costs.items():
            costs[coalition] += value
    return build_savings_game(dict(costs), players)


def daily_savings_game(scenario: Scenario) -> dict[tuple[int, ...], float]:
    """Backward-compatible construction of the full-day savings game."""
    return period_savings_game(scenario, tuple(range(scenario.num_hours)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count empty cores among field antennas.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="number of first antennas to test (default: all antennas)",
    )
    parser.add_argument(
        "--traffic-mode",
        choices=("average", "daily"),
        default="average",
        help="same traffic configuration as main.py (default: average)",
    )
    parser.add_argument(
        "--price-per-kwh",
        type=float,
        default=DEFAULT_ELECTRICITY_PRICE_PER_KWH,
    )
    parser.add_argument(
        "--hours",
        nargs=2,
        type=int,
        default=(DEFAULT_START_HOUR, DEFAULT_END_HOUR),
        metavar=("START", "END"),
        help=(
            "inclusive study window, with overnight wrap allowed "
            f"(default: {DEFAULT_START_HOUR} {DEFAULT_END_HOUR})"
        ),
    )
    args = parser.parse_args()
    if args.count is not None and args.count <= 0:
        parser.error("--count must be positive")
    try:
        hours = inclusive_hour_window(*args.hours)
    except ValueError as error:
        parser.error(str(error))

    requested = (
        "all antennas"
        if args.count is None
        else f"the first {args.count} antennas"
    )
    print(
        f">> Loading {requested} in one CSV pass...",
        flush=True,
    )
    print(
        f">> Study window: {hours[0]:02d}:00--{hours[-1]:02d}:00 "
        f"inclusive ({len(hours)} hours).",
        flush=True,
    )
    total, antenna_days, errors = load_first_antenna_days(args.count)
    empty: list[tuple[int, str, float]] = []
    convex_count = 0
    shapley_in_core_count = 0
    nonconvex_shapley_in_core_count = 0
    nonconvex_shapley_outside_core_count = 0
    shapley_outside_nonempty_core_count = 0
    convexity_inconsistencies: list[tuple[int, str]] = []

    for index, antenna_id, error in errors:
        print(
            f"   antenna {index:2d} ({antenna_id}): ERROR — {error}",
            flush=True,
        )

    for processed, (
        index,
        antenna_id,
        traffic_days,
        power_days,
    ) in enumerate(
        antenna_days,
        start=1,
    ):
        try:
            scenario = build_scenario_from_days(
                antenna_id,
                traffic_days,
                power_days,
                traffic_mode=args.traffic_mode,
                price_per_kwh=args.price_per_kwh,
                data_source="full_csv",
            )
            savings = period_savings_game(scenario, hours)
            convex = convexity_test(
                scenario.coalition,
                savings,
            ).convex
            shapley = shapley_value(scenario.coalition, savings)
            shapley_in_core = allocation_check(
                scenario.coalition,
                savings,
                shapley,
            ).in_core
            core = core_allocation(scenario.coalition, savings)

            convex_count += int(convex)
            shapley_in_core_count += int(shapley_in_core)
            nonconvex_shapley_in_core_count += int(
                not convex and shapley_in_core
            )
            nonconvex_shapley_outside_core_count += int(
                not convex and not shapley_in_core
            )
            shapley_outside_nonempty_core_count += int(
                not shapley_in_core and core.feasible
            )
            if convex and not shapley_in_core:
                convexity_inconsistencies.append((index, antenna_id))

            if not core.feasible:
                test = bondareva_shapley_test(scenario.coalition, savings)
                empty.append((index, antenna_id, test.gap))
                print(
                    f"   antenna {index:2d} ({antenna_id}): EMPTY "
                    f"[Bondareva gap = {test.gap:.6g}]",
                    flush=True,
                )
        except (RuntimeError, ValueError) as error:
            errors.append((index, antenna_id, str(error)))
            print(
                f"   antenna {index:2d} ({antenna_id}): ERROR — {error}",
                flush=True,
            )

        if processed % 10 == 0:
            print(
                f">> {processed}/{len(antenna_days)} datasets tested.",
                flush=True,
            )

    evaluated = total - len(errors)
    nonempty = evaluated - len(empty)
    print("\nResults:")
    print(f"Total antennas found: {total}")
    print(f"Calculation errors: {len(errors)}")
    print(f"Empty cores: {len(empty)}")
    print(f"Non-empty cores: {nonempty}")
    print(f"Convex games: {convex_count}")
    print(f"Non-convex games: {evaluated - convex_count}")
    print(f"Shapley in the core: {shapley_in_core_count}")
    print(f"Shapley outside the core: {evaluated - shapley_in_core_count}")
    print(
        "Non-convex games with Shapley in the core: "
        f"{nonconvex_shapley_in_core_count}"
    )
    print(
        "Non-convex games with Shapley outside the core: "
        f"{nonconvex_shapley_outside_core_count}"
    )
    print(
        "Non-empty cores with Shapley outside: "
        f"{shapley_outside_nonempty_core_count}"
    )
    print(
        "Convexity/Shapley consistency anomalies: "
        f"{len(convexity_inconsistencies)}"
    )
    if empty:
        print(
            "Empty cores: "
            + ", ".join(f"{index} ({antenna_id})" for index, antenna_id, _ in empty)
        )
    if errors:
        print(f"Unusable datasets: {len(errors)}.")


if __name__ == "__main__":
    main()
