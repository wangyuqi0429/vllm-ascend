"""PD block ownership follows drained ACKs, never thread liveness alone."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.send_thread import MembPullSendingThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler import SFAPDRD2HScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import (
    SFAPDRD2HConsumerWorker,
    SFAPDRD2HProducerWorker,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_sparse_transfer import (
    MooncakeCompletionUnknown,
    MooncakeSparseTransfer,
)


def pending_sender():
    sender = MembPullSendingThread.__new__(MembPullSendingThread)
    sender._state = SimpleNamespace(strict_completion=True, layer_storage_slots={1: (0,)})
    sender._pending_reads_by_layer = {1: 1}
    sender._pending_read_paths_by_layer = {1: {"peer"}}
    sender._storage_read_errors = {}
    sender.storage_send_done_events = [Event()]
    sender.quarantined_error = None
    return sender


def test_control_plane_failure_does_not_release_source_storage():
    sender = pending_sender()
    sender._fail_layer(1, "peer disconnected")
    assert not sender.storage_send_done_events[0].is_set()
    assert sender._pending_reads_by_layer == {1: 1}
    assert sender.quarantined_error == "peer disconnected"


def test_drained_failed_ack_releases_storage_but_keeps_failure_report():
    sender = pending_sender()
    sender._complete_peer_read("peer", 1, "FAILED_DRAINED")
    assert sender.storage_send_done_events[0].is_set()
    assert sender._pending_reads_by_layer == {}
    assert sender.get_storage_error(0) == "FAILED_DRAINED"


def test_forward_end_waits_for_all_published_source_layers():
    worker = SimpleNamespace(
        _backend="mooncake", _pd_dispatched_layers={1, 2},
        wait_for_layer_send=MagicMock(), kv_send_layer_thread=SimpleNamespace(quarantined_error=None),
    )
    SFAPDRD2HProducerWorker.wait_for_save(worker)
    assert {call.args[0] for call in worker.wait_for_layer_send.call_args_list} == {1, 2}


def test_unknown_peer_rank_is_gathered_before_consumer_leaves_collective_phase():
    worker = SimpleNamespace(
        engine=MooncakeSparseTransfer.__new__(MooncakeSparseTransfer),
        _mf_read_thread=SimpleNamespace(completion_unknown_error=None), tp_size=2,
    )

    def gather(output, _local, **_kwargs):
        output[:] = [None, "peer completion unknown"]

    with patch("torch.distributed.all_gather_object", side_effect=gather):
        with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker.get_tp_group"):
            with pytest.raises(MooncakeCompletionUnknown, match="peer completion unknown"):
                SFAPDRD2HConsumerWorker.get_finished(worker)


def test_cancelled_consumer_request_keeps_scheduler_blocks_until_receive_finishes():
    scheduler = SimpleNamespace(
        _dspark_pending_recv=set(), _mooncake_pending_recv={"request"},
        _deferred_finished_req_ids=set(), _cleanup_finished_request=MagicMock(),
    )
    assert SFAPDRD2HScheduler.request_finished_all_groups(
        scheduler, SimpleNamespace(request_id="request"), ([0, 1],)
    ) == (True, None)
    scheduler._cleanup_finished_request.assert_not_called()
    SFAPDRD2HScheduler.update_connector_output(
        scheduler, SimpleNamespace(finished_recving={"request"})
    )
    scheduler._cleanup_finished_request.assert_called_once_with("request")
