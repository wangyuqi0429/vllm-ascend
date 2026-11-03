"""Registration and ticket terminal-state contracts, without TE/NPU execution."""

from threading import Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.utils.mooncake_sparse_transfer import (
    MooncakeCompletionUnknown,
    MooncakeSparseTransfer,
)


def make_transfer(status="COMPLETED"):
    transfer = MooncakeSparseTransfer.__new__(MooncakeSparseTransfer)
    transfer._engine = MagicMock()
    transfer._hostname = "127.0.0.1"
    transfer._lock = Lock()
    transfer._regions = {4096: (1024, object())}
    transfer._ticket = None
    transfer._unknown = False
    transfer._closed = False
    ticket = transfer._engine.submit_scatter_read.return_value
    ticket.status = SimpleNamespace(name=status)
    ticket.drained = False

    def drain(_timeout):
        ticket.drained = True

    ticket.drain.side_effect = drain
    return transfer


@pytest.mark.parametrize("status,result", [("COMPLETED", 0), ("FAILED_DRAINED", -1)])
def test_reads_report_result_only_after_drained_ticket(status, result):
    transfer = make_transfer(status)
    assert transfer.batch_transfer_sync_read("peer", [4096], [8192], [16]) == result
    ticket = transfer._engine.submit_scatter_read.return_value
    ticket.drain.assert_called_once_with(1000)
    assert transfer._ticket is None


def test_binding_error_is_quarantined_without_busy_retry_or_unregister():
    transfer = make_transfer()
    ticket = transfer._engine.submit_scatter_read.return_value
    ticket.drain.side_effect = AttributeError("incompatible binding")
    with pytest.raises(MooncakeCompletionUnknown):
        transfer.batch_transfer_sync_read("peer", [4096], [8192], [16])
    assert transfer._unknown
    assert transfer._ticket is ticket
    ticket.drain.assert_called_once()
    with pytest.raises(MooncakeCompletionUnknown):
        transfer.close()
    transfer._engine.unregister_memory.assert_not_called()


def test_registration_uses_device_domain_and_retains_owner_until_unregistered():
    transfer = make_transfer()
    owner = object()
    transfer._engine.register_memory.return_value = 0
    transfer._engine.unregister_memory.return_value = 0
    assert transfer.register_memory(8192, 256, owner=owner) == 0
    transfer._engine.register_memory.assert_called_once_with(8192, 256, "npu", True)
    assert transfer._regions[8192][1] is owner
    transfer.close()
    assert transfer._regions == {}


def test_read_rejects_unregistered_or_overlapping_destinations():
    transfer = make_transfer()
    for local, peer, lengths in (([3072], [8192], [16]), ([4096, 4100], [8192, 16384], [16, 16])):
        with pytest.raises(ValueError):
            transfer.batch_transfer_sync_read("peer", local, peer, lengths)
    transfer._engine.submit_scatter_read.assert_not_called()


def test_failed_drained_read_does_not_contribute_success_bytes_and_collects():
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_metrics import SparseOffloadMetrics

    transfer = make_transfer("FAILED_DRAINED")
    runtime = MagicMock()
    runtime.is_current_stream_capturing.return_value = False
    runtime.Event.return_value.query.return_value = True
    runtime.Event.return_value.elapsed_time.return_value = 1.0
    transfer.metrics = SparseOffloadMetrics(enabled=True, npu_runtime=runtime)
    result = transfer.batch_transfer_sync_read("peer", [4096], [8192], [16])
    assert result == -1
    summary = transfer.metrics_summary()
    phase = summary["phases"]["pd_read"]
    assert phase["failed_samples"] == 1
    assert phase["effective_bytes"] == 0
