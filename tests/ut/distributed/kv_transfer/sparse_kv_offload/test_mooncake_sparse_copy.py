"""Mock eager-copy contract tests; not hardware validation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_adapter import SharedKVByteRange
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_copy import (
    MooncakeEagerCopy,
    TensorPoolCopy,
)


def make_pair(tensor_address=4096, host_address=8192, length=16):
    tensor = MagicMock(spec=torch.Tensor)
    tensor.device = SimpleNamespace(type="npu")
    tensor.dtype = torch.bfloat16
    tensor.is_contiguous.return_value = True
    tensor.numel.return_value = length // 2
    tensor.element_size.return_value = 2
    tensor.data_ptr.return_value = tensor_address
    segment = MagicMock()
    segment.base_addr.return_value = host_address
    segment.total_size.return_value = length
    return TensorPoolCopy(tensor, SharedKVByteRange(host_address, length, "host", segment))


def make_engine(status="COMPLETED"):
    engine = MagicMock()
    ticket = engine.submit_scatter_write.return_value
    ticket.drained = False
    ticket.status = SimpleNamespace(name=status)

    def drain(_timeout):
        ticket.drained = True
        return ticket.status

    ticket.drain.side_effect = drain
    return engine


@pytest.mark.parametrize("to_pool", [True, False])
def test_direction_uses_existing_synchronous_write(to_pool):
    engine = make_engine()
    executor = MooncakeEagerCopy(engine, "self")
    pair = make_pair()
    if to_pool:
        executor.copy_to_pool([pair])
    else:
        executor.copy_from_pool([pair])
    engine.submit_scatter_write.assert_called_once_with(
        "self", [4096 if to_pool else 8192], [16], [8192 if to_pool else 4096], [16], [[0]], [[0]], [[16]]
    )
    assert executor._retained == ()


def test_empty_batches_and_valid_zero_length_ranges_do_not_submit():
    engine = MagicMock()
    executor = MooncakeEagerCopy(engine, "self")
    executor.copy_to_pool([])
    executor.copy_from_pool([make_pair(tensor_address=0, length=0)])
    engine.submit_scatter_write.assert_not_called()


@pytest.mark.parametrize("failure", ["FAILED_DRAINED", RuntimeError("transport raised")])
def test_failure_poison_retains_owners_and_rejects_later_calls(failure):
    engine = make_engine()
    if isinstance(failure, Exception):
        engine.submit_scatter_write.side_effect = failure
    else:
        engine.submit_scatter_write.return_value.status = SimpleNamespace(name=failure)
    executor = MooncakeEagerCopy(engine, "self")
    pair = make_pair()
    with pytest.raises(RuntimeError):
        executor.copy_to_pool([pair])
    assert executor._retained == ((pair,) if isinstance(failure, Exception) else ())
    with pytest.raises(RuntimeError, match="poisoned"):
        executor.copy_from_pool([])
    assert engine.submit_scatter_write.call_count == 1


@pytest.mark.parametrize(
    "pairs",
    [
        [make_pair(), make_pair(tensor_address=16384, host_address=8196)],
        [make_pair(), make_pair(tensor_address=8196, host_address=16384)],
        [make_pair(host_address=0)],
        [make_pair(host_address=(1 << 64) - 8)],
    ],
)
def test_invalid_intervals_and_cross_item_hazards_never_submit(pairs):
    engine = MagicMock()
    with pytest.raises(ValueError):
        MooncakeEagerCopy(engine, "self").copy_to_pool(pairs)
    engine.submit_scatter_write.assert_not_called()


def test_validation_failure_does_not_poison_executor():
    engine = make_engine()
    executor = MooncakeEagerCopy(engine, "self")
    pair = make_pair()
    pair.tensor.is_contiguous.return_value = False
    with pytest.raises(ValueError, match="contiguous"):
        executor.copy_to_pool([pair])
    pair.tensor.is_contiguous.return_value = True
    executor.copy_to_pool([pair])
    assert engine.submit_scatter_write.call_count == 1


def test_range_domain_length_and_owner_are_checked():
    executor = MooncakeEagerCopy(MagicMock(), "self")
    tensor = make_pair().tensor
    for byte_range in (
        SharedKVByteRange(8192, 16, "device", object()),
        SharedKVByteRange(8192, 14, "host", object()),
        SharedKVByteRange(8192, 16, "host", None),
        SharedKVByteRange(8192, 16.0, "host", object()),
    ):
        with pytest.raises((ValueError, TypeError)):
            executor.copy_to_pool([TensorPoolCopy(tensor, byte_range)])


def test_pool_capacity_and_batch_device_are_checked():
    executor = MooncakeEagerCopy(MagicMock(), "self")
    pair = make_pair()
    pair.pool_range._segment.total_size.return_value = 8
    with pytest.raises(ValueError, match="outside its shared segment"):
        executor.copy_to_pool([pair])
    first, second = make_pair(), make_pair(tensor_address=16384, host_address=32768)
    first.tensor.device = SimpleNamespace(type="npu", index=0)
    second.tensor.device = SimpleNamespace(type="npu", index=1)
    with pytest.raises(ValueError, match="same NPU"):
        executor.copy_to_pool([first, second])


def test_ticket_wait_retries_unknown_and_preserves_interrupt_until_drained():
    engine = make_engine()
    ticket = engine.submit_scatter_write.return_value
    calls = []

    def drain(_timeout):
        calls.append(len(calls))
        if len(calls) == 1:
            raise KeyboardInterrupt()
        if len(calls) == 3:
            ticket.drained = True
        return SimpleNamespace(name="COMPLETION_UNKNOWN")

    ticket.drain.side_effect = drain
    executor = MooncakeEagerCopy(engine, "self")
    with pytest.raises(KeyboardInterrupt):
        executor.copy_to_pool([make_pair()])
    assert len(calls) == 3
    assert executor._retained == ()
    executor.close()
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        executor.copy_to_pool([])
