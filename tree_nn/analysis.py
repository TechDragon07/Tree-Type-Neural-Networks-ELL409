from __future__ import annotations

import math


def seed_supports_double_descent(rows: list[dict]) -> bool:
    """Return whether one seed has a peak at interpolation and a later descent."""
    ordered = sorted(rows, key=lambda row: (row["width"], row["parameters"]))
    first = next(
        (index for index, row in enumerate(ordered) if row["interpolates"]),
        None,
    )
    if first is None or first == 0 or first == len(ordered) - 1:
        return False
    interpolation_error = ordered[first]["test_error"]
    rises_at_interpolation = (
        min(row["test_error"] for row in ordered[:first]) < interpolation_error - 1e-4
    )
    descends_afterward = (
        min(row["test_error"] for row in ordered[first + 1 :]) < interpolation_error - 1e-4
    )
    return rises_at_interpolation and descends_afterward


def repeated_seed_double_descent(rows: list[dict]) -> tuple[bool, int, int]:
    """Apply the predeclared three-seed, two-thirds-agreement rule."""
    seeds = sorted({int(row["seed"]) for row in rows})
    supporting = sum(
        seed_supports_double_descent([row for row in rows if int(row["seed"]) == seed])
        for seed in seeds
    )
    required = math.ceil(2 * len(seeds) / 3)
    established = len(seeds) >= 3 and supporting >= required
    return established, supporting, len(seeds)
