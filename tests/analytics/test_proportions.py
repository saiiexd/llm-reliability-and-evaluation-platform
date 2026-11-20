"""
Tests for the shared proportion statistics (Wilson interval, exact McNemar).
"""

from __future__ import annotations

import pytest

from llm_reliability.analytics.proportions import mcnemar_exact, wilson


def test_wilson_undefined_for_empty_sample() -> None:
    assert wilson(0, 0) == (None, None, None)


def test_wilson_known_value() -> None:
    # 5/10: center 0.5, half-width 1.96*sqrt(0.025 + 0.009604)/1.38416 = 0.26341.
    p, lo, hi = wilson(5, 10)
    assert p == 0.5
    assert lo == pytest.approx(0.23659, abs=1e-5)
    assert hi == pytest.approx(0.76341, abs=1e-5)


def test_wilson_stays_inside_unit_interval_at_extremes() -> None:
    p, lo, hi = wilson(0, 4)
    assert p == 0.0 and lo == 0.0 and hi is not None and 0.0 < hi < 1.0
    p, lo, hi = wilson(4, 4)
    assert p == 1.0 and hi == 1.0 and lo is not None and 0.0 < lo < 1.0


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_exact(1, 4) == pytest.approx(2 * 6 / 32)
    assert mcnemar_exact(3, 3) == 1.0
