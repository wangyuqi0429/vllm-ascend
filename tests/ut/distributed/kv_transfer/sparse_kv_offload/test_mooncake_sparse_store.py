# SPDX-License-Identifier: Apache-2.0
"""Store session contracts with native I/O mocked; not hardware validation."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.layerwise_keys import LayerwiseKeyBuilder
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_adapter import SharedKVByteRange
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_store import (
    MAX_STORE_RANGE_BYTES,
    MooncakeSparseStore,
    StoreCompletionUnknown,
    StoreObject,
    StoreOperationError,
    StoreRange,
    StoreSessionGate,
)


def make_bridge(*, limit=8, capacity=64):
    backend = MagicMock()
    backend.store.register_buffer.return_value = 0
    backend.store.unregister_buffer.return_value = 0
    backend.batch_get_start.side_effect = lambda keys: [0] * len(keys)
    backend.batch_put_start.side_effect = lambda keys, sizes: [0] * len(keys)
    backend.batch_copy_get.side_effect = lambda keys, buffers, sizes, offsets: [sum(row) for row in sizes]
    backend.batch_copy_put.side_effect = lambda keys, buffers, sizes, offsets: [sum(row) for row in sizes]
    backend.batch_get_end.return_value = 0
    backend.batch_commit.side_effect = lambda keys: [0] * len(keys)
    backend.batch_revoke.side_effect = lambda keys: [0] * len(keys)
    segment = SimpleNamespace(base_addr=lambda: 4096, total_size=lambda: capacity)
    region = SharedKVByteRange(4096, capacity, "host", segment)
    builder = LayerwiseKeyBuilder(lambda group, block_hash, rank, stage: f"bound/{group}/{block_hash}/{rank}/{stage}", 1)
    gate = StoreSessionGate(backend)
    bridge = MooncakeSparseStore(
        backend, (region,), builder, "bf16/tp1/group0", session_gate=gate,
        exclusive_session_client=True, max_transfer_bytes=limit,
    )
    return bridge, backend, region


def make_object(region, *, block_hash="hash", size=16, address=None, offset=0, length=None):
    payload = SharedKVByteRange(
        region.address if address is None else address,
        size if length is None else length, "host", region._segment,
    )
    return StoreObject(0, block_hash, 0, 0, size, "bf16/tp1/group0", (StoreRange(payload, offset),))


def test_import_keeps_one_session_across_chunks_and_checks_byte_results():
    bridge, backend, region = make_bridge()
    result = bridge.import_objects([make_object(region)])
    key = "bound/0/hash/0/0"
    assert result.keys == (key,)
    assert result.byte_counts == (16,)
    backend.batch_get_start.assert_called_once_with([key])
    assert backend.batch_copy_get.call_args_list == [
        call([key], [[4096]], [[8]], [[0]]),
        call([key], [[4104]], [[8]], [[8]]),
    ]
    backend.batch_get_end.assert_called_once_with([key])
    backend.store.register_buffer.assert_called_once_with(4096, 64)
    bridge.close()
    bridge.close()
    backend.store.unregister_buffer.assert_called_once_with(4096)


def test_put_copies_logical_bytes_then_commits_full_object():
    bridge, backend, region = make_bridge()
    result = bridge.seal_objects([make_object(region)])
    key = result.keys[0]
    backend.batch_put_start.assert_called_once_with([key], [16])
    assert backend.batch_copy_put.call_count == 2
    backend.batch_commit.assert_called_once_with([key])
    backend.batch_revoke.assert_not_called()


@pytest.mark.parametrize("seal", [False, True])
def test_short_completed_copy_does_not_publish_and_cleans_session(seal):
    bridge, backend, region = make_bridge()
    method = backend.batch_copy_put if seal else backend.batch_copy_get
    method.side_effect = None
    method.return_value = [0]
    with pytest.raises(StoreOperationError, match="short"):
        (bridge.seal_objects if seal else bridge.import_objects)([make_object(region)])
    assert not bridge.gate.poisoned
    if seal:
        backend.batch_revoke.assert_called_once()
        backend.batch_commit.assert_not_called()
    else:
        backend.batch_get_end.assert_called_once()
    bridge.close()


def test_unknown_native_exception_keeps_session_and_registration_isolated():
    bridge, backend, region = make_bridge()
    backend.batch_copy_get.side_effect = RuntimeError("native completion unknown")
    with pytest.raises(StoreCompletionUnknown):
        bridge.import_objects([make_object(region)])
    assert bridge.gate.poisoned
    assert any(owner is bridge for owner in bridge.gate.retained)
    assert bridge._active
    backend.batch_get_end.assert_not_called()
    with pytest.raises(StoreCompletionUnknown):
        bridge.close()
    with pytest.raises(StoreCompletionUnknown):
        bridge.import_objects([])
    backend.store.unregister_buffer.assert_not_called()


def test_partial_start_only_cleans_successfully_opened_keys():
    bridge, backend, region = make_bridge()
    backend.batch_get_start.side_effect = None
    backend.batch_get_start.return_value = [0, -1]
    first = make_object(region)
    second = make_object(region, block_hash="second", address=4128)
    with pytest.raises(StoreOperationError, match="start"):
        bridge.import_objects([first, second])
    backend.batch_get_end.assert_called_once_with(["bound/0/hash/0/0"])
    backend.batch_copy_get.assert_not_called()


def test_partial_commit_only_revokes_failed_put_sessions():
    bridge, backend, region = make_bridge()
    backend.batch_commit.side_effect = None
    backend.batch_commit.return_value = [0, -1]
    with pytest.raises(StoreOperationError, match="commit"):
        bridge.seal_objects([make_object(region), make_object(region, block_hash="second", address=4128)])
    backend.batch_revoke.assert_called_once_with(["bound/0/second/0/0"])


def test_registration_failure_keeps_successful_registration_for_explicit_close():
    bridge, backend, region = make_bridge()
    # Split this one mapping into two disjoint registrations.
    bridge.regions = (
        replace(region, length=16), replace(region, address=4128, length=16),
    )
    backend.store.register_buffer.side_effect = [0, -1]
    with pytest.raises(StoreOperationError, match="registration"):
        bridge.import_objects([make_object(region)])
    backend.batch_get_start.assert_not_called()
    bridge.close()
    backend.store.unregister_buffer.assert_called_once_with(4096)


@pytest.mark.parametrize("change", ["layout", "bounds", "coverage", "key", "stage"])
def test_invalid_object_plans_fail_before_registration(change):
    bridge, backend, region = make_bridge()
    item = make_object(region)
    if change == "layout":
        item = replace(item, layout_token="different-model-layout")
    elif change == "bounds":
        item = make_object(region, offset=15)
    elif change == "coverage":
        item = make_object(region, length=8)
    elif change == "key":
        item = replace(item, block_hash="")
    else:
        item = replace(item, stage=1)
    with pytest.raises(ValueError):
        bridge.seal_objects([item])
    backend.store.register_buffer.assert_not_called()


def test_import_destinations_and_duplicate_keys_cannot_overlap():
    bridge, backend, region = make_bridge()
    first = make_object(region)
    with pytest.raises(ValueError, match="overlap"):
        bridge.import_objects([first, make_object(region, block_hash="second", address=4100)])
    with pytest.raises(ValueError, match="unique"):
        bridge.import_objects([first, first])
    backend.batch_get_start.assert_not_called()


def test_real_int_max_boundary_splits_without_allocating_a_large_buffer():
    size = MAX_STORE_RANGE_BYTES + 1
    bridge, backend, region = make_bridge(limit=MAX_STORE_RANGE_BYTES, capacity=size)
    result = bridge.import_objects([make_object(region, size=size)])
    assert result.byte_counts == (size,)
    assert [args.args[2] for args in backend.batch_copy_get.call_args_list] == [[[MAX_STORE_RANGE_BYTES]], [[1]]]


def test_dedicated_client_is_required_and_unknown_gate_is_shared():
    bridge, backend, region = make_bridge()
    with pytest.raises(ValueError, match="dedicated"):
        MooncakeSparseStore(backend, (region,), bridge.key_builder, bridge.layout_token, session_gate=bridge.gate)
    backend.batch_get_start.side_effect = RuntimeError("session state unknown")
    with pytest.raises(StoreCompletionUnknown):
        bridge.import_objects([make_object(region)])
    other = MooncakeSparseStore(
        backend, (region,), bridge.key_builder, bridge.layout_token,
        session_gate=bridge.gate, exclusive_session_client=True,
    )
    with pytest.raises(StoreCompletionUnknown):
        other.import_objects([make_object(region)])


def test_unaligned_native_results_isolate_without_guessed_cleanup():
    bridge, backend, region = make_bridge()
    backend.batch_get_start.side_effect = None
    backend.batch_get_start.return_value = []
    with pytest.raises(StoreCompletionUnknown):
        bridge.import_objects([make_object(region)])
    backend.batch_get_end.assert_not_called()
    backend.store.unregister_buffer.assert_not_called()


def test_read_end_error_isolates_without_retrying_session_cleanup():
    bridge, backend, region = make_bridge()
    backend.batch_get_end.return_value = -1
    with pytest.raises(StoreCompletionUnknown, match="no automatic"):
        bridge.import_objects([make_object(region)])
    assert bridge.gate.poisoned
    backend.batch_get_end.assert_called_once()
    with pytest.raises(StoreCompletionUnknown):
        bridge.close()
    backend.store.unregister_buffer.assert_not_called()
