"""Native manager byte-layout/plan checks, with execution mocked."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager


def make_manager():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.is_mooncake = True
    manager._closed = False
    manager.tp_size = 1
    manager.tp_rank = 0
    manager._mooncake_inflight = None
    manager.mooncake_raw = torch.empty(4096, dtype=torch.int8)
    manager._mooncake_layout_offset = 0
    manager._mooncake_regions = []
    manager.mooncake_pool = MagicMock()
    manager.mooncake_pool.resolve_range.side_effect = (
        lambda _block, _entry, offset, length, **_kw: SimpleNamespace(address=1048576 + offset)
    )
    return manager


def test_initial_layout_uses_actual_base_alignment_and_is_atomic_on_capacity_error():
    manager = make_manager()
    k, v = manager.allocate_mooncake_views((16, 32), 128)
    assert k.data_ptr() % 128 == v.data_ptr() % 128 == 0
    previous_offset = manager._mooncake_layout_offset
    previous_regions = list(manager._mooncake_regions)
    with pytest.raises(ValueError, match="exceeds"):
        manager.allocate_mooncake_views((8192,), 128)
    assert manager._mooncake_layout_offset == previous_offset
    assert manager._mooncake_regions == previous_regions


def test_address_conversion_excludes_alignment_padding():
    manager = make_manager()
    (raw,) = manager.allocate_mooncake_views((16,), 128)
    offset = raw.data_ptr() - manager.mooncake_raw.data_ptr()
    assert manager._mooncake_device_address(raw, 4, 8) == 1048576 + offset + 4
    with pytest.raises(ValueError, match="exceeds"):
        manager._mooncake_device_address(raw, 12, 8)
    with pytest.raises(ValueError, match="declared"):
        manager._mooncake_device_address(manager.mooncake_raw, 2048, 8)


def test_lim_request_local_ids_resolve_through_both_block_tables_and_deduplicate():
    manager = make_manager()
    k, v = manager.allocate_mooncake_views((32, 32), 128)
    manager.k_caches_cpu = [k.view(torch.bfloat16)]
    manager.v_caches_cpu = [v.view(torch.bfloat16)]
    manager.topk_buffers_k = [torch.empty(16, dtype=torch.bfloat16)]
    manager.topk_buffers_v = [torch.empty(16, dtype=torch.bfloat16)]
    manager.block_size = 2
    manager.token_size_bytes_k = manager.token_size_bytes_v = 4
    manager.layer_name_to_offload_id = {"layer": 0}
    manager._mooncake_copy = MagicMock()
    manager.reload_mooncake_lim(
        "layer", torch.tensor([[2, 2]]), torch.tensor([[1, 1]]), torch.tensor([2]),
        torch.tensor([[1, 0]]), torch.tensor([[0, 1]]),
    )
    sources, destinations, lengths, _owners = manager._mooncake_copy.call_args.args
    assert sources == [
        manager._mooncake_device_address(manager.k_caches_cpu[0], 0, 4),
        manager._mooncake_device_address(manager.v_caches_cpu[0], 0, 4),
    ]
    assert destinations == [manager.topk_buffers_k[0].data_ptr() + 4, manager.topk_buffers_v[0].data_ptr() + 4]
    assert lengths == [4, 4]


def test_malformed_descriptors_are_rejected_without_launch():
    manager = make_manager()
    with pytest.raises(ValueError, match="overlap"):
        manager._mooncake_copy([4096, 8192], [16384, 16388], [8, 8], ())
    with pytest.raises(ValueError, match="int64"):
        manager._mooncake_copy([-1], [16384], [8], ())


def fake_npu_rows(address, rows):
    value = MagicMock(spec=torch.Tensor)
    value.dtype = torch.bfloat16
    value.device = SimpleNamespace(type="npu")
    value.element_size.return_value = 2
    value.contiguous.return_value = value
    value.view.return_value = value
    value.shape = (rows, 2)
    value.__getitem__.side_effect = lambda row: SimpleNamespace(data_ptr=lambda: address + row * 4)
    return value


@pytest.mark.parametrize("prefill", [True, False])
def test_writeback_resolves_physical_prefill_and_sequential_current_rows(prefill):
    manager = make_manager()
    k, v = manager.allocate_mooncake_views((32, 32), 128)
    manager.token_size_bytes_k = manager.token_size_bytes_v = 4
    manager.sparse_kv_offload_config = SimpleNamespace(keep_device_kv_cache=True)
    manager._mooncake_copy = MagicMock()
    source_k = fake_npu_rows(4096, 8 if prefill else 3)
    source_v = fake_npu_rows(8192, 8 if prefill else 3)
    manager._mooncake_writeback(
        torch.tensor([3, -1, 5]), k.view(torch.bfloat16), v.view(torch.bfloat16),
        source_k if prefill else None, source_v if prefill else None,
        None if prefill else source_k, None if prefill else source_v, prefill,
    )
    sources, destinations, lengths, _owners = manager._mooncake_copy.call_args.args
    rows = [3, 5] if prefill else [0, 2]
    assert sources == [4096 + row * 4 for row in rows] + [8192 + row * 4 for row in rows]
    assert destinations == [
        manager._mooncake_device_address(host, slot * 4, 4)
        for host in (k, v) for slot in (3, 5)
    ]
    assert lengths == [4] * 4


@pytest.mark.parametrize("slots", [[1, 1], [0, 8]])
def test_writeback_rejects_duplicate_and_out_of_range_slots(slots):
    manager = make_manager()
    k, v = manager.allocate_mooncake_views((32, 32), 128)
    manager.token_size_bytes_k = manager.token_size_bytes_v = 4
    manager._mooncake_copy = MagicMock()
    with pytest.raises(ValueError):
        manager._mooncake_writeback(
            torch.tensor(slots), k.view(torch.bfloat16), v.view(torch.bfloat16),
            None, None, fake_npu_rows(4096, 2), fake_npu_rows(8192, 2), False,
        )
    manager._mooncake_copy.assert_not_called()


def test_close_preserves_inflight_owners_after_sync_failure_and_can_retry():
    manager = make_manager()
    stream = MagicMock()
    stream.synchronize.side_effect = [RuntimeError("device failure"), None]
    owner = object()
    operation = (stream, owner)
    manager._mooncake_inflight = operation
    pool = manager.mooncake_pool
    with pytest.raises(RuntimeError):
        manager.close()
    assert manager._mooncake_inflight is operation
    assert manager.mooncake_pool is pool
    with patch("torch_npu.npu.synchronize"):
        manager.close()
        manager.close()
    assert manager._mooncake_inflight is None
    assert manager.mooncake_pool is None


def test_tp_peer_reuses_owner_relative_layout_instead_of_its_virtual_base_alignment():
    owner, peer = make_manager(), make_manager()
    owner.tp_size = peer.tp_size = 2
    peer.tp_rank = 1
    owner.tp_group = MagicMock()
    peer.tp_group = MagicMock()

    def gather(output, declaration, **_kwargs):
        output[:] = [declaration, declaration]

    owner.tp_group.broadcast_object.side_effect = lambda payload, **_kwargs: payload
    with patch("torch.distributed.all_gather_object", side_effect=gather):
        owner.allocate_mooncake_views((16, 32), 128, layer_name="layer")
        payload = owner.tp_group.broadcast_object.call_args.args[0]
        peer.tp_group.broadcast_object.return_value = payload
        peer.allocate_mooncake_views((16, 32), 128, layer_name="layer")
    assert peer._mooncake_regions == owner._mooncake_regions
    assert peer._mooncake_layout_offset == owner._mooncake_layout_offset


def test_tp_allocation_order_mismatch_fails_on_every_rank_before_owner_plan():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = MagicMock()

    def mismatch(output, declaration, **_kwargs):
        output[:] = [declaration, ("other_layer", *declaration[1:])]

    with patch("torch.distributed.all_gather_object", side_effect=mismatch):
        with pytest.raises(ValueError, match="allocation order"):
            manager.allocate_mooncake_views((16,), 128, layer_name="layer")
    manager.tp_group.broadcast_object.assert_not_called()


def test_tp_close_peer_sync_error_preserves_owner_pool_before_any_barrier():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = MagicMock()
    pool = manager.mooncake_pool
    operation = (MagicMock(), object())
    manager._mooncake_inflight = operation

    def gather(output, _error, **_kwargs):
        output[:] = [None, "peer sync failed"]

    with patch("torch_npu.npu.synchronize"):
        with patch("torch.distributed.all_gather_object", side_effect=gather):
            with pytest.raises(RuntimeError, match="peer sync failed"):
                manager.close()
    assert manager.mooncake_pool is pool
    assert manager._mooncake_inflight is operation
    manager.tp_group.barrier.assert_not_called()


def test_tp0_writeback_completion_is_published_after_copy_returns():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = MagicMock()
    manager._mooncake_writeback = MagicMock()

    def gather(output, error, **_kwargs):
        manager._mooncake_writeback.assert_called_once()
        output[:] = [error, None]

    with patch("torch.distributed.all_gather_object", side_effect=gather):
        manager.offload_new_kv("layer", torch.tensor([0]), None, None, None, None, None, None)


def test_tp_peer_never_writes_replicated_current_kv():
    manager = make_manager()
    manager.tp_rank = 1
    manager.tp_size = 2
    manager.tp_group = MagicMock()
    manager._mooncake_writeback = MagicMock()

    def gather(output, _error, **_kwargs):
        output[:] = [None, None]

    with patch("torch.distributed.all_gather_object", side_effect=gather):
        manager.offload_new_kv("layer", torch.tensor([0]), None, None, None, None, None, None)
    manager._mooncake_writeback.assert_not_called()
