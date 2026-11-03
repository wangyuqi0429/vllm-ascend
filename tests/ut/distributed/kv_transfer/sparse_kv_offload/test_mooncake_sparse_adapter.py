"""Memory-only contract checks; no NPU allocation or copies are performed."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_adapter import (
    MooncakeSparsePool,
    SharedKVBlock,
)


def make_segment(device_base=8192):
    segment = MagicMock()
    segment.tensors.return_value = [torch.empty((2, 4), dtype=torch.bfloat16)] * 2
    segment.block_layout.return_value = SimpleNamespace(count=2, offset=4096, entry_stride=8192, nbytes=16)
    segment.base_addr.return_value = 1024
    segment.device_addr.return_value = device_base
    segment.total_size.return_value = 16384
    return segment


def test_ranges_use_mooncake_layout_and_distinct_address_domains():
    segment = make_segment()
    pool = MooncakeSparsePool(segment, ("k",))
    host = pool.resolve_range("k", 1, 4, 8, address_domain="host")
    device = pool.resolve_range("k", 1, 4, 8, address_domain="device")
    assert host.address == 1024 + 4096 + 8192 + 4
    assert device.address == 8192 + 4096 + 8192 + 4
    assert host.length == device.length == 8
    assert pool.resolve_range("k", 1, 16, 0, address_domain="host").length == 0


@pytest.mark.parametrize(
    "entry,offset,length",
    [(-1, 0, 1), (2, 0, 1), (0, -1, 1), (0, 0, -1), (0, 16, 1), (0, 17, 0)],
)
def test_ranges_reject_padding_and_out_of_bounds(entry, offset, length):
    pool = MooncakeSparsePool(make_segment(), ("k",))
    with pytest.raises(ValueError):
        pool.resolve_range("k", entry, offset, length, address_domain="host")


def test_ranges_reject_missing_mapping_nonintegers_and_unknown_blocks():
    pool = MooncakeSparsePool(make_segment(device_base=0), ("k",))
    with pytest.raises(ValueError, match="no device address"):
        pool.resolve_range("k", 0, 0, 0, address_domain="device")
    with pytest.raises(TypeError):
        pool.resolve_range("k", 0, 0.5, 1, address_domain="host")
    with pytest.raises(KeyError):
        pool.resolve_range("v", 0, 0, 1, address_domain="host")
    with pytest.raises(ValueError, match="address_domain"):
        pool.resolve_range("k", 0, 0, 1, address_domain="cpu")


def test_resolved_range_retains_mapping_without_pool():
    segment = make_segment()
    reference = weakref.ref(segment)
    pool = MooncakeSparsePool(segment, ("k",))
    byte_range = pool.resolve_range("k", 0, 0, 4, address_domain="host")
    del pool, segment
    gc.collect()
    assert reference() is not None
    assert byte_range.length == 4


def test_factory_reuses_existing_shared_segment_declaration():
    factory = MagicMock(return_value=make_segment())
    with patch.dict("sys.modules", {"mooncake.shared_segment": SimpleNamespace(create_shared_segment=factory)}):
        pool = MooncakeSparsePool.create("kv", {"k": SharedKVBlock(2, (2, 4))}, device_id=0)
    factory.assert_called_once_with(
        "kv",
        {"k": dict(count=2, shape=(2, 4), dtype=torch.bfloat16)},
        world_size=1,
        rank_id=0,
        owner_rank=0,
        comm_group=None,
        device_id=0,
        mmap=True,
        host_register=False,
    )
    assert len(pool.tensors("k")) == 2


@pytest.mark.parametrize(
    "block",
    [
        SharedKVBlock(0, (2, 4)),
        SharedKVBlock(1, ()),
        SharedKVBlock(1, (2, -1)),
        SharedKVBlock(1, (2, 4), torch.float32),
    ],
)
def test_factory_rejects_unsupported_layout_before_runtime_import(block):
    with pytest.raises(ValueError):
        MooncakeSparsePool.create("kv", {"k": block}, device_id=0)


def test_factory_rejects_tp_before_runtime_import():
    with pytest.raises(ValueError, match="comm_group"):
        MooncakeSparsePool.create("kv", {"k": SharedKVBlock(1, (2, 4))}, device_id=0, world_size=2)


def test_tp_device_wrapper_failure_is_reported_before_any_rank_continues():
    segment = make_segment()
    segment.tensors.side_effect = RuntimeError("wrapper failed")
    factory = MagicMock(return_value=segment)
    group = SimpleNamespace(cpu_group=object())

    def gather(output, error, **_kwargs):
        output[:] = [None, error]

    with patch.dict("sys.modules", {"mooncake.shared_segment": SimpleNamespace(create_shared_segment=factory)}):
        with patch("torch.distributed.all_gather_object", side_effect=gather):
            with pytest.raises(RuntimeError, match="device-view"):
                MooncakeSparsePool.create(
                    "kv", {"k": SharedKVBlock(1, (2, 4))}, device_id=1,
                    world_size=2, rank_id=1, comm_group=group,
                )


@pytest.mark.parametrize("block", [SharedKVBlock(1.5, (2, 4)), SharedKVBlock(1, (2.5, 4))])
def test_factory_rejects_noninteger_dimensions(block):
    with pytest.raises(TypeError):
        MooncakeSparsePool.create("kv", {"k": block}, device_id=0)


def test_range_checks_total_segment_capacity():
    segment = make_segment()
    segment.total_size.return_value = 4096
    pool = MooncakeSparsePool(segment, ("k",))
    with pytest.raises(ValueError, match="outside the shared segment"):
        pool.resolve_range("k", 0, 0, 1, address_domain="host")


def test_factory_rejects_host_registration_without_mmap():
    with pytest.raises(ValueError, match="requires mmap"):
        MooncakeSparsePool.create(
            "kv", {"k": SharedKVBlock(1, (2, 4))}, device_id=0, mmap=False, host_register=True
        )


def test_factory_propagates_view_creation_failure():
    segment = make_segment()
    segment.tensors.side_effect = RuntimeError("view failed")
    factory = MagicMock(return_value=segment)
    with patch.dict("sys.modules", {"mooncake.shared_segment": SimpleNamespace(create_shared_segment=factory)}):
        with pytest.raises(RuntimeError, match="view failed"):
            MooncakeSparsePool.create("kv", {"k": SharedKVBlock(2, (2, 4))}, device_id=0)
