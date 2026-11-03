"""Internal memory-only building block for native Mooncake sparse offload.

This module does not enable offload, submit copies, or implement attention.
The caller must retain the pool (or returned views/ranges) until all users,
including external copies and graphs, have finished. Derived views/slices and
bare addresses do not inherit this ownership: retain the pool, original view,
or range explicitly. No drain is implied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from operator import index as integer_index
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from mooncake.shared_segment import SharedSegment


@dataclass(frozen=True)
class SharedKVBlock:
    """A group of identically shaped contiguous BF16 entries (e.g. layers)."""

    count: int
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.bfloat16


@dataclass(frozen=True)
class SharedKVByteRange:
    """Resolved bytes in an explicit address domain, retaining their mapping."""

    address: int
    length: int
    address_domain: Literal["host", "device"]
    _segment: SharedSegment = field(repr=False, compare=False)


class MooncakeSparsePool:
    """Single-rank shared pool; uses Mooncake's own layout and tensor wrappers.

    Shapes describe contiguous storage, not arbitrary existing tensor strides.
    Block names carry framework layer/group/K/V identity; this class neither
    interprets slots nor allocates them. Returned ranges are not copy commands.
    """

    def __init__(self, segment: SharedSegment, blocks: tuple[str, ...]) -> None:
        """Internal wrapper; use create() for the supported TP/BF16 checks."""
        self._segment = segment
        self._views = {name: tuple(segment.tensors(name)) for name in blocks}

    @classmethod
    def create(
        cls,
        name: str,
        blocks: Mapping[str, SharedKVBlock],
        *,
        device_id: int,
        world_size: int = 1,
        rank_id: int = 0,
        owner_rank: int = 0,
        comm_group=None,
        mmap: bool = True,
        host_register: bool = False,
    ) -> MooncakeSparsePool:
        """Allocate one physical segment using Mooncake's TP blob exchange.

        Sizes and indices follow Python's integer protocol (including bool).
        """
        world_size, rank_id, owner_rank = map(integer_index, (world_size, rank_id, owner_rank))
        if world_size <= 0 or not 0 <= rank_id < world_size or not 0 <= owner_rank < world_size:
            raise ValueError("Invalid shared-pool world_size/rank_id/owner_rank")
        if world_size > 1 and comm_group is None:
            raise ValueError("comm_group is required for a TP shared pool")
        if not blocks:
            raise ValueError("At least one KV block is required")
        declarations = {}
        for block_name, spec in blocks.items():
            if not isinstance(block_name, str) or not block_name:
                raise ValueError("KV block names must be nonempty strings")
            count = integer_index(spec.count)
            shape = tuple(integer_index(dim) for dim in spec.shape)
            if count <= 0 or not shape or any(dim <= 0 for dim in shape):
                raise ValueError(f"KV block {block_name!r} requires positive count and shape")
            if spec.dtype != torch.bfloat16:
                raise ValueError("MooncakeSparsePool currently supports only BF16 KV blocks")
            declarations[block_name] = dict(count=count, shape=shape, dtype=spec.dtype)
        if host_register and not mmap:
            raise ValueError("host_register requires mmap=True")
        device_id = integer_index(device_id)
        if device_id < 0:
            raise ValueError("device_id must be nonnegative")

        # Optional runtime dependency: importing this module must not initialize
        # Mooncake or any other offload backend.
        from mooncake.shared_segment import create_shared_segment

        segment = create_shared_segment(
            name,
            declarations,
            world_size=world_size,
            rank_id=rank_id,
            owner_rank=owner_rank,
            comm_group=comm_group,
            device_id=device_id,
            mmap=mmap,
            host_register=host_register,
        )
        pool = None
        error = None
        try:
            pool = cls(segment, tuple(declarations))
        except BaseException as exception:
            error = exception
        if world_size > 1:
            errors = [None] * world_size
            torch.distributed.all_gather_object(
                errors, str(error) if error is not None else None,
                group=getattr(comm_group, "cpu_group", comm_group),
            )
            if any(item is not None for item in errors):
                raise RuntimeError(f"Mooncake TP device-view construction failed: {errors}") from error
        elif error is not None:
            raise error
        return pool

    def tensors(self, block: str) -> tuple[torch.Tensor, ...]:
        """Return original contiguous views retaining the mapping.

        Keep these original views or the pool alive when deriving tensors or
        extracting addresses; derived objects do not inherit the holder.
        """
        return self._views[block]

    def host_tensors(self, block: str) -> tuple[torch.Tensor, ...]:
        """Original Host views, distinct from a possible device-address view."""
        if block not in self._views:
            raise KeyError(block)
        return tuple(self._segment.host_tensors(block))

    def resolve_range(
        self,
        block: str,
        entry: int,
        byte_offset: int,
        length: int,
        *,
        address_domain: Literal["host", "device"],
    ) -> SharedKVByteRange:
        """Resolve bytes within one entry; zero bytes at its end are legal.

        Valid ranges cannot include inter-entry alignment padding. Device and
        host VAs are queried independently, even if they happen to be equal.
        """
        if block not in self._views:
            raise KeyError(block)
        layout = self._segment.block_layout(block)
        entry, byte_offset, length = map(integer_index, (entry, byte_offset, length))
        if entry < 0 or entry >= layout.count:
            raise ValueError("KV entry index is outside the block")
        if (
            byte_offset < 0
            or length < 0
            or byte_offset > layout.nbytes
            or length > layout.nbytes - byte_offset
        ):
            raise ValueError("KV byte range is outside the entry")
        if address_domain == "host":
            base = self._segment.base_addr()
        elif address_domain == "device":
            base = self._segment.device_addr()
        else:
            raise ValueError("address_domain must be 'host' or 'device'")
        if not base:
            raise ValueError(f"Shared segment has no {address_domain} address")
        offset = layout.offset + entry * layout.entry_stride + byte_offset
        if offset < 0 or offset + length > self._segment.total_size():
            raise ValueError("KV byte range is outside the shared segment")
        return SharedKVByteRange(base + offset, length, address_domain, self._segment)
