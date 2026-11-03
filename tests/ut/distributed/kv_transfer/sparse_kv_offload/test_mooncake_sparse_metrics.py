# SPDX-License-Identifier: Apache-2.0
"""Timing/window contracts with fake events; no NPU work is executed."""

from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_metrics import (
    SparseOffloadMetrics,
    benchmark_sparse_copy,
)


class FakeEvent:
    def __init__(self, runtime):
        self.runtime = runtime
        self.recorded = []
        self.query_count = 0
        self.synchronize_count = 0
        self.error = None

    def record(self, stream):
        if self.error is not None:
            raise self.error
        self.recorded.append(stream)

    def query(self):
        self.query_count += 1
        return self.runtime.ready

    def synchronize(self):
        self.synchronize_count += 1
        self.runtime.ready = True

    def elapsed_time(self, other):
        assert self.runtime is other.runtime
        return self.runtime.duration_ms


class FakeNPU:
    def __init__(self):
        self.stream = object()
        self.events = []
        self.ready = False
        self.capturing = False
        self.duration_ms = 2.0

    def Event(self, *, enable_timing):
        assert enable_timing
        event = FakeEvent(self)
        self.events.append(event)
        return event

    def current_stream(self):
        return self.stream

    def is_current_stream_capturing(self):
        return self.capturing


def test_disabled_does_not_touch_runtime_clock_or_validate_unused_arguments():
    runtime = MagicMock()
    clock = MagicMock()
    metrics = SparseOffloadMetrics(npu_runtime=runtime, clock_ns=clock)
    with metrics.measure("", -1, device=True):
        pass
    metrics.record_cache(-1, -1)
    assert metrics.collect(wait=True) == 0
    assert metrics.summary()["phases"] == {}
    assert runtime.mock_calls == []
    clock.assert_not_called()


def test_host_percentiles_are_completed_window_and_wait_flag_is_explicit():
    ticks = iter([0, 1_000_000, 10_000_000, 12_000_000, 20_000_000, 23_000_000])
    metrics = SparseOffloadMetrics(enabled=True, clock_ns=lambda: next(ticks))
    for _ in range(3):
        with metrics.measure("submit", effective_bytes=128, host_includes_wait=True):
            pass
    phase = metrics.summary()["phases"]["submit"]
    assert phase["host_ms"] == pytest.approx(dict(p50=2, p95=2.9, p99=2.98))
    assert phase["host_includes_wait_samples"] == 3
    assert phase["effective_bytes"] == 384
    assert phase["device_bandwidth_bytes_per_second"] is None


def test_nonblocking_collect_only_accepts_ready_events_and_summary_never_queries():
    runtime = FakeNPU()
    metrics = SparseOffloadMetrics(enabled=True, npu_runtime=runtime)
    with metrics.measure("h2d", 1024, device=True):
        pass
    assert metrics.summary()["pending"] == 1
    assert runtime.events[1].query_count == 0
    assert metrics.collect() == 0
    assert runtime.events[1].synchronize_count == 0
    runtime.ready = True
    assert metrics.collect() == 1
    phase = metrics.summary()["phases"]["h2d"]
    assert phase["device_stream_ms"]["p50"] == 2.0
    assert phase["device_bandwidth_bytes_per_second"] == 1024 / 0.002
    assert runtime.events[0].recorded == runtime.events[1].recorded == [runtime.stream]


def test_explicit_wait_collect_synchronizes_timing_events_only():
    runtime = FakeNPU()
    metrics = SparseOffloadMetrics(enabled=True, npu_runtime=runtime)
    with metrics.measure("d2h", 64, device=True):
        pass
    assert metrics.collect(wait=True) == 1
    assert runtime.events[1].synchronize_count == 1


def test_pending_limit_drops_measurement_without_skipping_body_or_allocating_events():
    runtime = FakeNPU()
    metrics = SparseOffloadMetrics(enabled=True, max_samples=2, npu_runtime=runtime)
    bodies = []
    for index in range(3):
        with metrics.measure("copy", device=True):
            bodies.append(index)
    assert bodies == [0, 1, 2]
    assert len(runtime.events) == 4
    assert metrics.summary()["pending"] == 2
    assert metrics.summary()["dropped_samples"] == 1
    runtime.ready = True
    assert metrics.collect() == 2


def test_completed_window_has_a_bound_and_reports_eviction():
    metrics = SparseOffloadMetrics(enabled=True, max_samples=2)
    for size in (1, 2, 3):
        with metrics.measure("host", size):
            pass
    summary = metrics.summary()
    assert summary["evicted_samples"] == 1
    assert summary["phases"]["host"]["effective_bytes"] == 5


def test_capture_skips_recording_and_forbids_collecting_existing_events():
    runtime = FakeNPU()
    metrics = SparseOffloadMetrics(enabled=True, npu_runtime=runtime)
    with metrics.measure("replay", device=True):
        pass
    runtime.capturing = True
    with metrics.measure("capture", device=True):
        pass
    with metrics.measure("host_capture", capturing=True):
        pass
    assert len(runtime.events) == 2
    assert metrics.summary()["capture_skipped"] == 2
    with pytest.raises(RuntimeError, match="outside"):
        metrics.collect(wait=True)
    assert runtime.events[1].synchronize_count == 0


def test_failed_body_is_not_counted_as_successful_bandwidth():
    runtime = FakeNPU()
    metrics = SparseOffloadMetrics(enabled=True, npu_runtime=runtime)
    with pytest.raises(ValueError, match="operation failed"):
        with metrics.measure("copy", 4096, device=True):
            raise ValueError("operation failed")
    metrics.collect(wait=True)
    phase = metrics.summary()["phases"]["copy"]
    assert phase["failed_samples"] == 1
    assert phase["effective_bytes"] == 0
    assert phase["device_bandwidth_bytes_per_second"] is None


def test_timing_error_does_not_mask_body_exception_or_leak_active_admission():
    runtime = FakeNPU()
    metrics = SparseOffloadMetrics(enabled=True, npu_runtime=runtime)
    with pytest.raises(ValueError, match="body"):
        with metrics.measure("copy", device=True):
            runtime.events[-1].error = RuntimeError("event failed")
            raise ValueError("body")
    assert metrics.summary()["active"] == 0
    assert metrics.summary()["timing_errors"] == 1


@pytest.mark.parametrize("body_fails", [False, True])
def test_final_clock_error_releases_admission_and_preserves_original_body_error(body_fails):
    clock = MagicMock(side_effect=[0, RuntimeError("clock failed")])
    metrics = SparseOffloadMetrics(enabled=True, clock_ns=clock)
    expected = ValueError if body_fails else RuntimeError
    with pytest.raises(expected, match="body" if body_fails else "clock"):
        with metrics.measure("host"):
            if body_fails:
                raise ValueError("body failed")
    assert metrics.summary()["active"] == 0
    assert metrics.summary()["timing_errors"] == 1


def test_stream_switch_cannot_produce_a_misleading_elapsed_interval():
    runtime = FakeNPU()
    metrics = SparseOffloadMetrics(enabled=True, npu_runtime=runtime)
    with pytest.raises(RuntimeError, match="same NPU stream"):
        with metrics.measure("copy", device=True):
            runtime.stream = object()
    assert metrics.summary()["timing_errors"] == 1
    assert metrics.summary()["pending"] == 0


def test_cache_counts_are_observations_of_existing_strategy():
    metrics = SparseOffloadMetrics(enabled=True)
    metrics.record_cache(80, 20)
    metrics.record_cache(0, 0)
    assert metrics.summary()["cache"] == dict(hits=80, misses=20, hit_rate=0.8)
    with pytest.raises(ValueError):
        metrics.record_cache(-1, 2)


@pytest.mark.parametrize("duration", [-1, float("inf"), float("nan")])
def test_invalid_device_duration_is_not_reported_as_bandwidth(duration):
    runtime = FakeNPU()
    runtime.duration_ms = duration
    metrics = SparseOffloadMetrics(enabled=True, npu_runtime=runtime)
    with metrics.measure("copy", device=True):
        pass
    with pytest.raises(RuntimeError, match="invalid duration"):
        metrics.collect(wait=True)
    assert metrics.summary()["pending"] == 1


def test_explicit_benchmark_excludes_warmup_and_reports_only_measured_bytes():
    runtime = FakeNPU()
    runtime.stream = MagicMock()
    submit = MagicMock()
    result = benchmark_sparse_copy(
        submit, effective_bytes=256, npu_runtime=runtime, warmup=2, iterations=3,
    )
    assert submit.call_count == 5
    runtime.stream.synchronize.assert_called_once()
    assert result["phases"]["copy"]["samples"] == 3
    assert result["phases"]["copy"]["device_effective_bytes"] == 768


def test_benchmark_cannot_submit_inside_capture():
    runtime = FakeNPU()
    runtime.capturing = True
    submit = MagicMock()
    with pytest.raises(RuntimeError, match="outside"):
        benchmark_sparse_copy(submit, effective_bytes=1, npu_runtime=runtime)
    submit.assert_not_called()
