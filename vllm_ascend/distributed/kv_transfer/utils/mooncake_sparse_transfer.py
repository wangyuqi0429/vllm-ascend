# SPDX-License-Identifier: Apache-2.0
"""Mooncake-only registrations and drained reads for sparse PD/Store users.

No MemFabric initialization or register-once singleton is used here. The engine
connection itself is reused from the existing process-wide Mooncake TE.
"""

from __future__ import annotations

from contextlib import nullcontext
from threading import Lock

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_metrics import SparseOffloadMetrics

BACKEND_MOONCAKE = "mooncake"


class MooncakeCompletionUnknown(RuntimeError):
    """No terminal completion proof; neither side may recycle its KV blocks."""


class MooncakeSparseTransfer:
    def __init__(self, hostname: str, device_name: str | None = None, *, metrics_enabled: bool = False):
        from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

        self._engine = global_te.get_transfer_engine(hostname, device_name)
        if global_te.local_hostname is None or global_te.local_hostname != hostname:
            raise ValueError("Existing Mooncake engine identity does not match the sparse PD endpoint")
        if global_te.local_device_name not in ("", None, device_name):
            raise ValueError("Existing Mooncake engine device selection differs from sparse PD")
        self._hostname = global_te.local_hostname
        self._lock = Lock()
        self._regions: dict[int, tuple[int, object]] = {}
        self._ticket = None
        self._unknown = False
        self._closed = False
        self.metrics = SparseOffloadMetrics(enabled=metrics_enabled)

    @property
    def unique_id(self) -> str:
        host = f"[{self._hostname}]" if ":" in self._hostname else self._hostname
        return f"{host}:{self.get_rpc_port()}"

    def get_rpc_port(self) -> int:
        return self._engine.get_rpc_port()

    def register_memory(self, ptr: int, size: int, *, owner: object) -> int:
        """Register actual NPU or device-mapped DDR VA, retaining its owner."""
        with self._lock:
            if self._closed or self._unknown:
                raise MooncakeCompletionUnknown("Mooncake sparse TE is closed or quarantined")
            if not isinstance(ptr, int) or not isinstance(size, int) or ptr <= 0 or size <= 0:
                raise ValueError("Mooncake registration requires a positive address and capacity")
            if owner is None:
                raise ValueError("Mooncake registrations require allocation ownership")
            if ptr in self._regions:
                if self._regions[ptr][0] != size:
                    raise ValueError("Mooncake registration changed capacity at a live address")
                return 0
            if any(ptr < base + capacity and base < ptr + size for base, (capacity, _) in self._regions.items()):
                raise ValueError("Mooncake sparse registrations must not overlap")
            result = self._engine.register_memory(ptr, size, "npu", True)
            if result == 0:
                self._regions[ptr] = (size, owner)
            return result

    def batch_transfer_sync_read(self, session_id, local_buffers, peer_buffers, length_list) -> int:
        """Return success/failure only after a ticket proves physical drain."""
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            metrics.collect(wait=False)
        span = (
            metrics.measure("pd_read", sum(length_list) if metrics.enabled else 0, device=True, host_includes_wait=True)
            if metrics is not None else nullcontext()
        )
        class CompletedReadFailure(Exception):
            pass

        try:
            with span:
                result = self._batch_read(session_id, local_buffers, peer_buffers, length_list)
                if result != 0:
                    raise CompletedReadFailure()
        except CompletedReadFailure:
            return result
        finally:
            if metrics is not None:
                metrics.collect(wait=False)
        return result

    def metrics_summary(self, *, wait: bool = False) -> dict:
        self.metrics.collect(wait=wait)
        return self.metrics.summary()

    def _batch_read(self, session_id, local_buffers, peer_buffers, length_list) -> int:
        with self._lock:
            if self._closed or self._unknown:
                raise MooncakeCompletionUnknown("Mooncake sparse TE is closed or quarantined")
            if not (len(local_buffers) == len(peer_buffers) == len(length_list)):
                raise ValueError("Mooncake read descriptor capacities differ")
            descriptors = []
            for local, peer, size in zip(local_buffers, peer_buffers, length_list):
                if not all(isinstance(value, int) for value in (local, peer, size)):
                    raise TypeError("Mooncake read descriptors must contain integers")
                if local <= 0 or peer <= 0 or size < 0 or max(local, peer) + size > (1 << 64) - 1:
                    raise ValueError("Mooncake read range exceeds uint64 capacity")
                if not any(base <= local and local + size <= base + capacity for base, (capacity, _) in self._regions.items()):
                    raise ValueError("Mooncake read target is outside registered device memory")
                if size:
                    if any(local < previous + length and previous < local + size
                           for previous, _, length in descriptors):
                        raise ValueError("Mooncake read targets overlap")
                    descriptors.append((local, peer, size))
            if not descriptors:
                return 0
            local, peer, sizes = map(list, zip(*descriptors))
            try:
                self._ticket = self._engine.submit_scatter_read(
                    session_id, local, sizes, peer, sizes,
                    [[0] for _ in sizes], [[0] for _ in sizes], [[size] for size in sizes],
                )
            except BaseException as error:
                self._unknown = True
                raise MooncakeCompletionUnknown("Mooncake read submission has no completion proof") from error
            interrupted = None
            while True:
                try:
                    if self._ticket.drained:
                        break
                    self._ticket.drain(1000)
                except (KeyboardInterrupt, SystemExit) as error:
                    if interrupted is None:
                        interrupted = error
                except BaseException as error:
                    self._unknown = True
                    raise MooncakeCompletionUnknown("Mooncake ticket wait lost its completion proof") from error
            success = self._ticket.status.name == "COMPLETED"
            self._ticket = None
            if interrupted is not None:
                raise interrupted
            return 0 if success else -1

    def close(self) -> None:
        """Call after connector threads and their remote consumers stop."""
        with self._lock:
            if self._closed:
                return
            if self._unknown or self._ticket is not None:
                raise MooncakeCompletionUnknown("Cannot unregister memory without a completion proof")
            # Remove one entry only after its unregister succeeds; retries keep
            # the other owners and never silently skip a partially closed set.
            for ptr in tuple(self._regions):
                if self._engine.unregister_memory(ptr) != 0:
                    raise RuntimeError(f"Mooncake sparse memory unregister failed: {ptr:#x}")
                del self._regions[ptr]
            self._closed = True
