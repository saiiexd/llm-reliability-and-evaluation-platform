"""
Small-sample statistics for proportions and paired binary outcomes.

Shared by the study analysis scripts (``scripts/analyze_reliability_study.py``,
``scripts/analyze_human_agreement.py``). Pure functions, no dependencies.
"""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None, float | None]:
    """Point estimate and Wilson score interval for ``k`` successes out of ``n``.

    Returns ``(p, lower, upper)``; all three are ``None`` when ``n == 0`` (the
    proportion is undefined, not zero). ``z = 1.96`` gives a 95% interval
    (Wilson, 1927). Unlike the normal approximation it stays inside [0, 1] and
    behaves sensibly for small ``n`` and for ``p`` near 0 or 1.
    """
    if n == 0:
        return None, None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
