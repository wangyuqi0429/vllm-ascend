# SPDX-License-Identifier: Apache-2.0
"""Import/seal KV ranges through the existing Mooncake Store sessions.

One StoreSessionGate must be shared by every user of these keys on a native
client. It serializes whole sessions, not individual calls. Existing backend
calls that bypass this gate are outside that contract. The caller owns layout
binding, producer/consumer completion and request/slot reservations; a successful
import result is the earliest point at which it may publish the target ready.

A native exception leaves completion unknown. The shared gate then refuses all
new operations and retains mappings and registrations. Neither close nor GC is a
drain/recovery mechanism: keep the gate with the client until instance teardown.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from operator import index as integer_index
from threading import RLock
from typing import Any

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.layerwise_keys import LayerwiseKeyBuilder
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_adapter import SharedKVByteRange

MAX_STORE_RANGE_BYTES = (1 << 31) - 1
_MAX_ADDRESS = (1 << 64) - 1


class StoreOperationError(RuntimeError):
    """A completed Store operation failed; its output must not be published."""


class StoreCompletionUnknown(StoreOperationError):
    """Native completion or session state is unknown; retain all resources."""


@dataclass(frozen=True)
class StoreRange:
    pool_range: SharedKVByteRange
    object_offset: int


@dataclass(frozen=True)
class StoreObject:
    group_id: int
    block_hash: str
    rank: int
    stage: int
    object_size: int
    layout_token: str
    ranges: tuple[StoreRange, ...]


@dataclass(frozen=True)
class StoreTransferResult:
    keys: tuple[str, ...]
    byte_counts: tuple[int, ...]
    layout_token: str


class StoreSessionGate:
    """Per-backend shared serialization and fail-stop ownership, never global.

    Keep this gate in the same long-lived state as its backend. Retaining a
    tensor alone does not reserve its framework slot against reuse.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.lock = RLock()
        self.poisoned = False
        self.retained: list[Any] = []
        self.registrations: list[tuple[int, int, MooncakeSparseStore]] = []
        self.client = None


class MooncakeSparseStore:
    """A bound, synchronous Store bridge for registered Host pool ranges.

    Construct one shared gate per backend/client. Keys come only from the
    existing bound LayerwiseKeyBuilder; layout_token is an explicit additional
    caller assertion, not a replacement wire-key or a new naming protocol.
    The native session client must be dedicated to this gate; do not reuse a
    live layerwise worker's client whose start/end calls bypass the gate.
    A returned registration error retains earlier successful registrations so
    the caller can explicitly close or retry; it never starts a session.
    """

    def __init__(
        self,
        backend: Any,
        regions: Iterable[SharedKVByteRange],
        key_builder: LayerwiseKeyBuilder,
        layout_token: str,
        *,
        session_gate: StoreSessionGate,
        exclusive_session_client: bool = False,
        max_transfer_bytes: int = MAX_STORE_RANGE_BYTES,
    ) -> None:
        if exclusive_session_client is not True:
            raise ValueError("Sparse Store bridge requires a dedicated session client without bypass callers")
        if session_gate.backend is not backend:
            raise ValueError("Store session gate belongs to another backend")
        if not isinstance(layout_token, str) or not layout_token:
            raise ValueError("A bound model/dtype/TP/group layout token is required")
        limit = integer_index(max_transfer_bytes)
        if not 0 < limit <= MAX_STORE_RANGE_BYTES:
            raise ValueError("Store transfer limit must be positive and fit INT_MAX")
        self.backend = backend
        self.gate = session_gate
        self.key_builder = key_builder
        self.layout_token = layout_token
        self.max_transfer_bytes = limit
        self.regions = tuple(regions)
        self._registered: list[SharedKVByteRange] = []
        self._closed = False
        self._client = None
        if not self.regions:
            raise ValueError("Store bridge requires at least one Host region")
        intervals = []
        for region in self.regions:
            start, end = self._host_interval(region)
            if end == start or any(start < other_end and other_start < end for other_start, other_end in intervals):
                raise ValueError("Store registration regions must be nonempty and disjoint")
            intervals.append((start, end))

    @staticmethod
    def _host_interval(region: SharedKVByteRange) -> tuple[int, int]:
        if not isinstance(region, SharedKVByteRange) or region.address_domain != "host" or region._segment is None:
            raise ValueError("Store ranges must retain a live Host mapping")
        address, length = map(integer_index, (region.address, region.length))
        if address <= 0 or length < 0 or address > _MAX_ADDRESS or length > _MAX_ADDRESS - address:
            raise ValueError("Store range does not fit a nonnull uint64 interval")
        base = integer_index(region._segment.base_addr())
        capacity = integer_index(region._segment.total_size())
        if (
            base <= 0 or base > _MAX_ADDRESS or capacity < 0 or capacity > _MAX_ADDRESS - base
            or address < base or address + length > base + capacity
        ):
            raise ValueError("Store range exceeds its shared mapping")
        return address, address + length

    def _check_open(self) -> None:
        if self._closed:
            raise StoreOperationError("Store bridge is closed")
        if self.gate.poisoned:
            raise StoreCompletionUnknown("Store gate is isolated; retain client, slots and registrations")

    def _native(self, method, *args):
        try:
            return method(*args)
        except BaseException as error:
            self._isolate()
            raise StoreCompletionUnknown("Native Store call raised; completion is unknown") from error

    def _isolate(self) -> None:
        self.gate.poisoned = True
        if not any(owner is self for owner in self.gate.retained):
            self.gate.retained.append(self)

    def _register(self) -> None:
        self._native(self.backend.ensure_initialized)
        self._native(self.backend.validate_layerwise_support)
        client = self.backend.store
        if client is None:
            raise StoreOperationError("Store backend has no initialized client")
        if self.gate.client is not None and self.gate.client is not client:
            self._isolate()
            raise StoreCompletionUnknown("Shared Store session gate cannot switch native clients")
        self.gate.client = client
        if self._client is not None and client is not self._client:
            self._isolate()
            raise StoreCompletionUnknown("Store client changed while pool registrations are live")
        self._client = client
        for region in self.regions[len(self._registered) :]:
            start, end = self._host_interval(region)
            if any(start < other_end and other_start < end for other_start, other_end, _ in self.gate.registrations):
                raise StoreOperationError("Host range is already registered by another Store bridge")
            status = self._native(client.register_buffer, start, end - start)
            if status != 0:
                raise StoreOperationError(f"Store buffer registration failed: {status}")
            self._registered.append(region)
            self.gate.registrations.append((start, end, self))

    def _prepare(self, objects: Iterable[StoreObject], *, seal: bool):
        items = tuple(objects)
        keys, totals = [], []
        target_intervals = []
        for item in items:
            if item.layout_token != self.layout_token:
                raise ValueError("Store object layout differs from the bound active pool")
            group, rank, stage, object_size = map(
                integer_index, (item.group_id, item.rank, item.stage, item.object_size)
            )
            if (
                group < 0 or rank < 0 or not 0 <= stage < self.key_builder.pp_size
                or not 0 < object_size <= _MAX_ADDRESS
            ):
                raise ValueError("Invalid Store object coordinates or size")
            if not isinstance(item.block_hash, str) or not item.block_hash:
                raise ValueError("Store object needs its framework block hash")
            key = self.key_builder.make_full_key(group, item.block_hash, rank, stage)
            if not isinstance(key, str) or not key or key in keys:
                raise ValueError("Store object keys must be nonempty and unique in a batch")
            keys.append(key)
            object_intervals, total = [], 0
            for fragment in item.ranges:
                region = fragment.pool_range
                start, end = self._host_interval(region)
                offset = integer_index(fragment.object_offset)
                length = end - start
                if offset < 0 or offset > object_size or length > object_size - offset:
                    raise ValueError("Store range exceeds its object")
                if not any(
                    parent._segment is region._segment
                    and parent.address <= start and end <= parent.address + parent.length
                    for parent in self.regions
                ):
                    raise ValueError("Store range is outside the bridge's registered regions")
                if not length:
                    continue
                if not seal:
                    if any(start < other_end and other_start < end for other_start, other_end in target_intervals):
                        raise ValueError("Store import destinations overlap")
                    target_intervals.append((start, end))
                object_intervals.append((offset, offset + length))
                total += length
            if not total:
                raise ValueError("Store object requires a nonempty range plan")
            if seal:
                cursor = 0
                for start, end in sorted(object_intervals):
                    if start != cursor:
                        raise ValueError("Sealed Store objects require complete nonoverlapping coverage")
                    cursor = end
                if cursor != object_size:
                    raise ValueError("Sealed Store objects must cover the full object")
            totals.append(total)
        return items, keys, totals

    def _chunks(self, item: StoreObject):
        buffers, sizes, offsets, used = [], [], [], 0
        for fragment in item.ranges:
            remaining = integer_index(fragment.pool_range.length)
            address = integer_index(fragment.pool_range.address)
            offset = integer_index(fragment.object_offset)
            while remaining:
                take = min(remaining, self.max_transfer_bytes - used)
                buffers.append(address)
                sizes.append(take)
                offsets.append(offset)
                address, offset, remaining, used = address + take, offset + take, remaining - take, used + take
                if used == self.max_transfer_bytes:
                    yield buffers, sizes, offsets
                    buffers, sizes, offsets, used = [], [], [], 0
        if sizes:
            yield buffers, sizes, offsets

    def _batch_result(self, result, keys: list[str], operation: str) -> list[int]:
        if not isinstance(result, (list, tuple)) or len(result) != len(keys):
            self._isolate()
            raise StoreCompletionUnknown(f"{operation} returned unaligned session results")
        try:
            return [integer_index(value) for value in result]
        except TypeError as error:
            self._isolate()
            raise StoreCompletionUnknown(f"{operation} returned invalid session status") from error

    def _finish(self, keys: list[str], *, seal: bool) -> None:
        if not keys:
            return
        if seal:
            status = self._batch_result(self._native(self.backend.batch_revoke, keys), keys, "revoke")
            ok = all(value == 0 for value in status)
        else:
            ok = self._native(self.backend.batch_get_end, keys) == 0
        if not ok:
            self._isolate()
            raise StoreCompletionUnknown("Store session cleanup failed; retain its resources")

    def import_objects(self, objects: Iterable[StoreObject]) -> StoreTransferResult:
        """Copy all requested object ranges; return only after successful end."""
        return self._transfer(objects, seal=False)

    def seal_objects(self, objects: Iterable[StoreObject]) -> StoreTransferResult:
        """Copy complete objects and commit; ordinary failures revoke sessions."""
        return self._transfer(objects, seal=True)

    def _transfer(self, objects: Iterable[StoreObject], *, seal: bool) -> StoreTransferResult:
        with self.gate.lock:
            self._check_open()
            items, keys, totals = self._prepare(objects, seal=seal)
            if not items:
                return StoreTransferResult((), (), self.layout_token)
            self._register()
            # Hold every mapping and reservation object until all native calls
            # return; unknown completion leaves this reference quarantined.
            self._active = items
            started = []
            try:
                result = (
                    self._native(self.backend.batch_put_start, keys, [integer_index(item.object_size) for item in items])
                    if seal else self._native(self.backend.batch_get_start, keys)
                )
                status = self._batch_result(result, keys, "start")
                started = [key for key, value in zip(keys, status) if value == 0]
                if len(started) != len(keys):
                    raise StoreOperationError(f"Store session start failed: {status}")
                copy_method = self.backend.batch_copy_put if seal else self.backend.batch_copy_get
                for key, item in zip(keys, items):
                    for buffers, sizes, offsets in self._chunks(item):
                        result = self._native(copy_method, [key], [buffers], [sizes], [offsets])
                        status = self._batch_result(result, [key], "range copy")
                        if status[0] != sum(sizes):
                            raise StoreOperationError(f"Store range copy failed or was short: {status[0]}")
                if seal:
                    status = self._batch_result(self._native(self.backend.batch_commit, keys), keys, "commit")
                    if any(value != 0 for value in status):
                        # Successfully committed objects have no live put
                        # session left. Only unsuccessful sessions may revoke.
                        started = [key for key, value in zip(keys, status) if value != 0]
                        raise StoreOperationError(f"Store commit failed: {status}")
                else:
                    if self._native(self.backend.batch_get_end, keys) != 0:
                        self._isolate()
                        raise StoreCompletionUnknown("Store read session end failed; no automatic cleanup retry")
                started = []
            except BaseException:
                if not self.gate.poisoned:
                    self._finish(started, seal=seal)
                raise
            finally:
                if not self.gate.poisoned:
                    self._active = ()
            return StoreTransferResult(tuple(keys), tuple(totals), self.layout_token)

    def close(self) -> None:
        """Unregister only after ordinary calls completed; never recover unknown."""
        with self.gate.lock:
            if self._closed:
                return
            self._check_open()
            while self._registered:
                region = self._registered[-1]
                status = self._native(self._client.unregister_buffer, region.address)
                if status != 0:
                    raise StoreOperationError(f"Store buffer unregistration failed: {status}")
                self._registered.pop()
                self.gate.registrations[:] = [
                    entry for entry in self.gate.registrations
                    if entry[2] is not self or entry[0] != region.address
                ]
            self._closed = True
            self.regions = ()
            self._active = ()
