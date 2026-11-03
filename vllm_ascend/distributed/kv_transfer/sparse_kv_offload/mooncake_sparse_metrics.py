# SPDX-License-Identifier: Apache-2.0
"""Bounded, opt-in timings; collection belongs outside graph capture/replay.

Host intervals measure submission plus any waits inside the measured body.
Device intervals use two events on the same current stream and include queued
work between them, not isolated DMA duration. Effective bytes are supplied by
the caller from the actual transfer plan, never inferred from pool capacity.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import isfinite
from operator import index as integer_index
from threading import RLock
from time import perf_counter_ns
from typing import Any


@dataclass(frozen=True)
class _Sample:
    phase: str
    effective_bytes: int
    host_ns: int
    device_ms: float | None
    host_includes_wait: bool
    failed: bool


@dataclass(frozen=True)
class _Pending:
    sample: _Sample
    start: Any
    end: Any


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return dict(p50=None, p95=None, p99=None)
    ordered = sorted(values)
    result = {}
    for name, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        result[name] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return result


class SparseOffloadMetrics:
    """Per-instance bounded metrics, with no NPU work when disabled.

    Pass the worker's NPU runtime when measuring Host phases too, so capture
    calls can be skipped. Capture only records a graph; measure the actual
    replay span from its outer Python caller, not capture-time layer calls.
    Pending and completed samples each have a max_samples bound. Cache counts
    and dropped/evicted counts are lifetime totals; percentiles cover the
    retained completed window. No cache selection/replacement policy is added.
    """

    def __init__(
        self,
        enabled: bool = False,
        max_samples: int = 4096,
        *,
        npu_runtime: Any = None,
        clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        maximum = integer_index(max_samples)
        if maximum <= 0:
            raise ValueError("max_samples must be positive")
        self.enabled = enabled
        self.max_samples = maximum
        self._npu = npu_runtime
        self._clock = clock_ns
        self._lock = RLock()
        self._completed: deque[_Sample] = deque(maxlen=maximum)
        self._pending: deque[_Pending] = deque()
        self._active = 0
        self._dropped = 0
        self._evicted = 0
        self._capture_skipped = 0
        self._timing_errors = 0
        self._hits = 0
        self._misses = 0

    def _runtime(self):
        if self._npu is None:
            # Optional profiling runtime; CPU-only summaries never import it.
            import torch

            self._npu = torch.npu
        return self._npu

    def measure(
        self,
        phase: str,
        effective_bytes: int = 0,
        *,
        device: bool = False,
        host_includes_wait: bool = False,
        capturing: bool = False,
    ):
        if not self.enabled:
            return nullcontext()
        if not isinstance(phase, str) or not phase:
            raise ValueError("Metric phase must be a nonempty name")
        byte_count = integer_index(effective_bytes)
        if byte_count < 0:
            raise ValueError("Effective transfer bytes must be nonnegative")
        runtime = self._runtime() if device else self._npu
        if capturing or (runtime is not None and runtime.is_current_stream_capturing()):
            with self._lock:
                self._capture_skipped += 1
            return nullcontext()
        return self._measure(phase, byte_count, runtime if device else None, host_includes_wait)

    def _append(self, sample: _Sample) -> None:
        if len(self._completed) == self.max_samples:
            self._evicted += 1
        self._completed.append(sample)

    @contextmanager
    def _measure(self, phase, effective_bytes, runtime, host_includes_wait):
        with self._lock:
            if self._active + len(self._pending) >= self.max_samples:
                self._dropped += 1
                admitted = False
            else:
                self._active += 1
                admitted = True
        if not admitted:
            yield
            return
        start_event = end_event = stream = None
        try:
            if runtime is not None:
                stream = runtime.current_stream()
                start_event = runtime.Event(enable_timing=True)
                end_event = runtime.Event(enable_timing=True)
                start_event.record(stream)
            start_ns = self._clock()
        except BaseException:
            with self._lock:
                self._active -= 1
                self._timing_errors += 1
            raise
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            event_error = None
            try:
                host_ns = max(self._clock() - start_ns, 0)
            except BaseException as error:
                host_ns = 0
                event_error = error
            if runtime is not None:
                try:
                    if runtime.current_stream() != stream:
                        raise RuntimeError("A metric interval must start/end on the same NPU stream")
                    end_event.record(stream)
                except BaseException as error:
                    if event_error is None:
                        event_error = error
            sample = _Sample(phase, effective_bytes, host_ns, None, host_includes_wait, failed)
            with self._lock:
                self._active -= 1
                if event_error is not None:
                    self._timing_errors += 1
                elif runtime is None:
                    self._append(sample)
                else:
                    self._pending.append(_Pending(sample, start_event, end_event))
            if event_error is not None and not failed:
                raise event_error

    def collect(self, *, wait: bool = False) -> int:
        """Collect ready timing events; wait=True synchronizes only for profiling.

        No output tensors or request ownership are managed here. The operation
        implementation must independently retain its actual copy resources.
        """
        if not self.enabled:
            return 0
        with self._lock:
            if not self._pending:
                return 0
            if self._runtime().is_current_stream_capturing():
                raise RuntimeError("Collect metrics only outside graph capture/replay")
            collected = 0
            for _ in range(len(self._pending)):
                pending = self._pending[0]
                if wait:
                    pending.end.synchronize()
                elif not pending.end.query():
                    self._pending.rotate(-1)
                    continue
                device_ms = float(pending.start.elapsed_time(pending.end))
                if not isfinite(device_ms) or device_ms < 0:
                    raise RuntimeError("NPU timing returned an invalid duration")
                sample = pending.sample
                self._append(_Sample(
                    sample.phase, sample.effective_bytes, sample.host_ns, device_ms,
                    sample.host_includes_wait, sample.failed,
                ))
                self._pending.popleft()
                collected += 1
            return collected

    def record_cache(self, hits: int, misses: int) -> None:
        if not self.enabled:
            return
        hits, misses = map(integer_index, (hits, misses))
        if hits < 0 or misses < 0:
            raise ValueError("Cache counts must be nonnegative")
        with self._lock:
            self._hits += hits
            self._misses += misses

    def summary(self) -> dict[str, Any]:
        """Snapshot completed samples only; never queries or synchronizes NPU."""
        with self._lock:
            phases = {}
            for phase in sorted({sample.phase for sample in self._completed}):
                samples = [sample for sample in self._completed if sample.phase == phase]
                successful = [sample for sample in samples if not sample.failed]
                device_samples = [sample for sample in successful if sample.device_ms is not None]
                device_seconds = sum(sample.device_ms for sample in device_samples) / 1000
                device_bytes = sum(sample.effective_bytes for sample in device_samples)
                phases[phase] = {
                    "samples": len(samples),
                    "failed_samples": len(samples) - len(successful),
                    "host_includes_wait_samples": sum(sample.host_includes_wait for sample in samples),
                    "host_ms": _percentiles([sample.host_ns / 1_000_000 for sample in successful]),
                    "device_stream_ms": _percentiles([sample.device_ms for sample in device_samples]),
                    "effective_bytes": sum(sample.effective_bytes for sample in successful),
                    "device_effective_bytes": device_bytes,
                    "device_bandwidth_bytes_per_second": device_bytes / device_seconds if device_seconds > 0 else None,
                }
            cache_total = self._hits + self._misses
            return {
                "enabled": self.enabled,
                "phases": phases,
                "cache": {
                    "hits": self._hits,
                    "misses": self._misses,
                    "hit_rate": self._hits / cache_total if cache_total else None,
                },
                "pending": len(self._pending),
                "active": self._active,
                "dropped_samples": self._dropped,
                "evicted_samples": self._evicted,
                "capture_skipped": self._capture_skipped,
                "timing_errors": self._timing_errors,
            }


def benchmark_sparse_copy(
    submit: Callable[[], None],
    *,
    effective_bytes: int,
    npu_runtime: Any,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, Any]:
    """Explicit serialized latency benchmark; never called by normal serving.

    The caller prepares valid, registered/mapped descriptors, owns their data
    and completion, and supplies actual nonzero-length bytes for one invocation.
    submit may call mooncake.sparse_copy.sparse_copy with those stable buffers.
    Warmup is excluded. This intentionally waits per sample and therefore does
    not measure overlapped throughput or model end-to-end performance.
    """
    warmup, iterations, effective_bytes = map(integer_index, (warmup, iterations, effective_bytes))
    if warmup < 0 or iterations <= 0 or effective_bytes < 0:
        raise ValueError("Benchmark needs nonnegative warmup/bytes and positive iterations")
    if npu_runtime.is_current_stream_capturing():
        raise RuntimeError("Run copy benchmarks outside graph capture")
    for _ in range(warmup):
        submit()
    npu_runtime.current_stream().synchronize()
    metrics = SparseOffloadMetrics(enabled=True, max_samples=iterations, npu_runtime=npu_runtime)
    for _ in range(iterations):
        with metrics.measure("copy", effective_bytes=effective_bytes, device=True):
            submit()
        metrics.collect(wait=True)
    return metrics.summary()
