"""Shared handling of inclusive hourly study windows."""

from collections.abc import Iterable

HOURS_PER_DAY = 24
# Defaults for the current numerical protocol only. All callers may override
# both bounds; no simulation logic should assume this particular window.
DEFAULT_START_HOUR = 0
DEFAULT_END_HOUR = 6


def inclusive_hour_window(
    start: int = DEFAULT_START_HOUR,
    end: int = DEFAULT_END_HOUR,
    total_hours: int = HOURS_PER_DAY,
) -> tuple[int, ...]:
    """Return an inclusive hour window, allowing an overnight wrap."""
    if total_hours <= 0:
        raise ValueError("total_hours must be positive")
    if not 0 <= start < total_hours or not 0 <= end < total_hours:
        raise ValueError(
            f"hours must be between 0 and {total_hours - 1}"
        )
    if start <= end:
        return tuple(range(start, end + 1))
    return tuple(range(start, total_hours)) + tuple(range(0, end + 1))


def validate_hours(
    hours: Iterable[int],
    total_hours: int = HOURS_PER_DAY,
) -> tuple[int, ...]:
    """Validate and materialise an ordered collection of study hours."""
    selected = tuple(hours)
    if not selected:
        raise ValueError("at least one study hour is required")
    if any(not 0 <= hour < total_hours for hour in selected):
        raise ValueError(
            f"hours must be between 0 and {total_hours - 1}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("study hours must not contain duplicates")
    return selected


def describe_hours(hours: Iterable[int]) -> str:
    """Return a concise label for an already ordered hour window."""
    selected = tuple(hours)
    if not selected:
        raise ValueError("at least one study hour is required")
    overnight = " (overnight)" if selected[0] > selected[-1] else ""
    return (
        f"{selected[0]:02d}:00--{selected[-1]:02d}:00 inclusive"
        f"{overnight} ({len(selected)} hourly periods)"
    )
