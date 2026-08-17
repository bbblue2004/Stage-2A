"""Tests for the shared numerical study window."""

import unittest

from src.core.time_window import inclusive_hour_window


class TimeWindowTests(unittest.TestCase):
    def test_default_window_is_midnight_to_seven_exclusive(self) -> None:
        self.assertEqual(inclusive_hour_window(), tuple(range(7)))

    def test_explicit_overnight_window_wraps_at_midnight(self) -> None:
        self.assertEqual(
            inclusive_hour_window(22, 2),
            (22, 23, 0, 1, 2),
        )

    def test_explicit_daytime_window_overrides_the_default(self) -> None:
        self.assertEqual(
            inclusive_hour_window(9, 14),
            (9, 10, 11, 12, 13, 14),
        )

    def test_single_hour_window_is_accepted(self) -> None:
        self.assertEqual(inclusive_hour_window(17, 17), (17,))

    def test_full_day_window_is_accepted(self) -> None:
        self.assertEqual(inclusive_hour_window(0, 23), tuple(range(24)))

    def test_out_of_range_bound_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inclusive_hour_window(0, 24)


if __name__ == "__main__":
    unittest.main()
