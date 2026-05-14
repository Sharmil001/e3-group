"""Shared benchmark utilities."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable


@dataclass
class TimingStats:
    name: str
    samples_ms: list[float]

    @property
    def n(self) -> int:
        return len(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def p95_ms(self) -> float:
        if self.n < 20:
            # Fall back to max for very small samples
            return max(self.samples_ms)
        s = sorted(self.samples_ms)
        return s[int(0.95 * (self.n - 1))]

    def format(self) -> str:
        return (
            f"{self.name:<32} n={self.n:<4} "
            f"min={self.min_ms:8.2f}  median={self.median_ms:8.2f}  "
            f"p95={self.p95_ms:8.2f}  mean={self.mean_ms:8.2f}   ms"
        )


def summarize(name: str, samples_ms: Iterable[float]) -> TimingStats:
    return TimingStats(name=name, samples_ms=list(samples_ms))


def print_table(rows: list[TimingStats]) -> None:
    width = max((len(r.name) for r in rows), default=20) + 4
    print("=" * (width + 60))
    print(f"{'measurement':<{width}} n    min       median    p95       mean       (ms)")
    print("-" * (width + 60))
    for r in rows:
        print(
            f"{r.name:<{width}} "
            f"{r.n:<4} "
            f"{r.min_ms:8.2f}  {r.median_ms:8.2f}  "
            f"{r.p95_ms:8.2f}  {r.mean_ms:8.2f}"
        )
    print("=" * (width + 60))
