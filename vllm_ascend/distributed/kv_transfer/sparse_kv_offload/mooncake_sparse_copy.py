"""Host-described, synchronous tensor/pool copies using an existing Mooncake TE.

The caller owns engine initialization/registration and must supply its actual
advertised self segment name. The Python TE has no local-name getter, so this
wrapper cannot verify routing to LocalCopyEngine. That route additionally
requires the Ascend direct transport with fabric memory disabled.

Before calling, synchronize tensor producers and previous destination consumers.
The current NPU device/context must match the initialized TE and all tensors.
Pool ranges must come from MooncakeSparsePool.resolve_range: the total-capacity
check below does not authenticate fabricated ranges or exclude entry padding.
This wrapper does not accept streams or support capture/replay. Tensors and the
host pool must not physically alias, including through different virtual VAs.
Numeric interval checks cannot prove that contract.

This ticket path requires the classic Ascend engine and the matching native
scatter exception-safety changes. It waits through timeout/interrupt until the
ticket reports physical drain before releasing allocations. No automatic retry
is made after failure. If submission raises before returning a ticket, resources
remain quarantined in this executor: retain it in long-lived instance state;
garbage collection cannot supply a missing completion proof.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from operator import index as integer_index
from threading import Lock
from typing import Any

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_adapter import SharedKVByteRange

_UINT64_MAX = (1 << 64) - 1
_DRAIN_POLL_MS = 1000


@dataclass(frozen=True)
class TensorPoolCopy:
    """One registered tensor slice paired with an owner-retaining host range."""

    tensor: torch.Tensor
    pool_range: SharedKVByteRange


def _checked_interval(address: int, length: int) -> tuple[int, int]:
    address, length = map(integer_index, (address, length))
    if address <= 0 or address > _UINT64_MAX or length < 0 or length > _UINT64_MAX - address:
        raise ValueError("Copy range must fit a nonnull uint64 address interval")
    return address, address + length


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


class MooncakeEagerCopy:
    """Serialize direction-homogeneous batches through native scatter tickets.

    The engine and all addressed allocations must remain initialized/registered
    throughout use. This class never registers, unregisters or destroys them.
    """

    def __init__(self, engine: Any, self_segment_name: str) -> None:
        if not isinstance(self_segment_name, str) or not self_segment_name or "\x00" in self_segment_name:
            raise ValueError("self_segment_name must be the engine's advertised local name")
        self._engine = engine
        self._self_segment_name = self_segment_name
        self._lock = Lock()
        self._poisoned = False
        self._retained: tuple[TensorPoolCopy, ...] = ()
        self._ticket = None
        self._closed = False

    def _drain_ticket(self) -> None:
        """Never release owners on a timeout or an interrupted wait."""
        interrupted = None
        while True:
            try:
                if self._ticket.drained:
                    break
                self._ticket.drain(_DRAIN_POLL_MS)
            except BaseException as error:
                if interrupted is None:
                    interrupted = error
        status = self._ticket.status
        self._ticket = None
        self._retained = ()
        if interrupted is not None:
            raise interrupted
        if status.name != "COMPLETED":
            raise RuntimeError(f"Mooncake copy failed after physical drain: {status}")

    def close(self) -> None:
        """Wait for active work, then release this wrapper's engine reference.

        Registrations belong to the caller and may be removed only afterwards.
        Repeated close is safe. A non-returning physical transfer blocks close;
        a timeout cannot authorize resource release.
        """
        with self._lock:
            if self._closed:
                return
            if self._ticket is not None:
                self._drain_ticket()
            if self._retained:
                raise RuntimeError("Submission state is unknown; external recovery is required")
            self._closed = True
            self._engine = None

    def copy_to_pool(self, pairs: Iterable[TensorPoolCopy]) -> None:
        """Synchronously write NPU tensor bytes to registered Host pool ranges."""
        self._copy(pairs, to_pool=True)

    def copy_from_pool(self, pairs: Iterable[TensorPoolCopy]) -> None:
        """Synchronously write registered Host pool bytes to NPU tensor slices."""
        self._copy(pairs, to_pool=False)

    def _copy(self, pairs: Iterable[TensorPoolCopy], *, to_pool: bool) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Copy executor is closed")
            if self._poisoned:
                raise RuntimeError("Copy executor is poisoned; retain resources pending external recovery")
            batch = tuple(pairs)
            sources: list[tuple[int, int]] = []
            destinations: list[tuple[int, int]] = []
            device = None
            for pair in batch:
                tensor, pool_range = pair.tensor, pair.pool_range
                if not isinstance(tensor, torch.Tensor) or tensor.device.type != "npu":
                    raise ValueError("Copy tensors must be NPU tensors")
                if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                    raise ValueError("Copy tensors must be contiguous BF16")
                if device is not None and tensor.device != device:
                    raise ValueError("Copy tensors in one batch must use the same NPU")
                device = tensor.device
                if not isinstance(pool_range, SharedKVByteRange) or pool_range.address_domain != "host":
                    raise ValueError("Pool ranges must be owner-retaining Host ranges")
                if pool_range._segment is None:
                    raise ValueError("Pool ranges require a live segment owner")
                length = integer_index(pool_range.length)
                capacity = tensor.numel() * tensor.element_size()
                if length != capacity:
                    raise ValueError("Pool range must match the complete tensor byte length")
                pool_interval = _checked_interval(pool_range.address, length)
                segment_base, segment_end = _checked_interval(
                    pool_range._segment.base_addr(), pool_range._segment.total_size()
                )
                if pool_interval[0] < segment_base or pool_interval[1] > segment_end:
                    raise ValueError("Pool range is outside its shared segment")
                if length == 0:
                    continue
                tensor_interval = _checked_interval(tensor.data_ptr(), length)
                source, destination = (tensor_interval, pool_interval) if to_pool else (pool_interval, tensor_interval)
                sources.append(source)
                destinations.append(destination)
            for index, destination in enumerate(destinations):
                if any(_overlaps(destination, other) for other in destinations[:index]):
                    raise ValueError("Copy destinations must not overlap")
                if any(_overlaps(destination, source) for source in sources):
                    raise ValueError("Copy source and destination ranges must not overlap")
            if not sources:
                return
            self._retained = batch
            try:
                sizes = [source[1] - source[0] for source in sources]
                self._ticket = self._engine.submit_scatter_write(
                    self._self_segment_name,
                    [source[0] for source in sources],
                    sizes,
                    [destination[0] for destination in destinations],
                    sizes,
                    [[0] for _ in sources],
                    [[0] for _ in sources],
                    [[size] for size in sizes],
                )
                self._drain_ticket()
            except BaseException:
                self._poisoned = True
                if self._ticket is not None:
                    self._drain_ticket()
                raise
            self._retained = ()
