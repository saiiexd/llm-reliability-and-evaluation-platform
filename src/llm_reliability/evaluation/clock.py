"""
Clock abstraction for execution timing.

Provides injectable monotonic timing so execution durations can be measured
accurately in production while remaining 100% deterministic in unit tests
without relying on fragile sleep calls or timing thresholds.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Protocol for monotonic time sources."""

    def __call__(self) -> float:
        """Return the current time in seconds as a float."""
        ...


default_clock: Clock = time.monotonic


class FakeClock:
    """A deterministic, controllable clock for testing duration measurements."""

    def __init__(self, current_time: float = 0.0, step: float = 0.0) -> None:
        self._current_time = float(current_time)
        self._step = float(step)

    def __call__(self) -> float:
        val = self._current_time
        self._current_time += self._step
        return val

    def advance(self, delta: float) -> None:
        """Advance the simulated clock by ``delta`` seconds."""
        if delta < 0:
            raise ValueError("Clock cannot move backward.")
        self._current_time += delta

    @property
    def current_time(self) -> float:
        return self._current_time
