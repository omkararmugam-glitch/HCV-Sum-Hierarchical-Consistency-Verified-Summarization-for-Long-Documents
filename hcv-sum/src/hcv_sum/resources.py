"""Per-stage wall-clock and peak memory, so a long run reports numbers instead of impressions.

Memory is the process's resident set size (RSS) from psutil, sampled on a background thread while a
stage runs. RSS is used, not tracemalloc, because tracemalloc only sees allocations made through
Python's allocator and misses PyTorch tensors, which dominate this pipeline's memory.

Limits of the measurement: sampling every ``interval`` seconds can miss a spike shorter than that,
and RSS includes memory the allocator has freed but not returned to the OS, so a stage's peak can be
inflated by a previous stage. Treat the numbers as upper bounds at ~50 ms resolution.
"""

from __future__ import annotations

import threading
import time

import psutil

_MB = 1024 * 1024


class StageMonitor:
    """Context manager: ``with StageMonitor() as m: ...`` then ``m.seconds``, ``m.peak_mb``, ``m.start_mb``."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_mb = self.peak_mb = self.end_mb = 0.0
        self.seconds = 0.0

    def _rss(self) -> float:
        return self._process.memory_info().rss / _MB

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_mb = max(self.peak_mb, self._rss())

    def __enter__(self) -> "StageMonitor":
        self.start_mb = self.peak_mb = self._rss()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.seconds = time.perf_counter() - self._t0
        self.end_mb = self._rss()
        self.peak_mb = max(self.peak_mb, self.end_mb)


def process_peak_mb() -> float | None:
    """Lifetime peak RSS where the OS reports it (Windows: peak working set); None elsewhere."""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    return round(peak / _MB, 1) if peak else None
