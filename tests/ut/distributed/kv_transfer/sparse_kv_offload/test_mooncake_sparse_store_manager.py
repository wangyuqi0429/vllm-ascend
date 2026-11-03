"""Store runner transactions: mock-only, not executed in this workspace."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_metadata import attach_sparse_store_transactions
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager


def fixture():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.is_mooncake = True
    manager._closed = manager._mooncake_store_failed = False
    manager._mooncake_store_blocks = {}
    manager._mooncake_store_computed = {}
    manager._mooncake_store_ready = set()
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.block_size = 2
    manager._mooncake_store_main_group = 0
    manager._mooncake_store_sidecars = [(0, "index", 0, "torch.int8", 4, torch.empty((8, 4), dtype=torch.uint8))]
    manager._mooncake_store_sidecar_ranges = [object()]
    manager._prepare_mooncake_store_sidecars = MagicMock()
    manager._copy_mooncake_store_sidecars = MagicMock()
    manager._hydrate_mooncake_store_main = MagicMock()
    manager.mooncake_device_execution = False
    manager.vllm_config = SimpleNamespace(kv_transfer_config=None)
    manager.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()], num_blocks=8)
    manager._mooncake_store = SimpleNamespace(gate=SimpleNamespace(poisoned=False), layout_token="layout",
                                             import_objects=MagicMock(), seal_objects=MagicMock())
    manager.mooncake_raw = torch.empty(32, dtype=torch.bfloat16)
    manager.k_caches_cpu = [manager.mooncake_raw[:16]]
    manager.v_caches_cpu = [manager.mooncake_raw[16:]]
    manager.cpu_block_lens = [(4, 4)]
    manager.mooncake_pool = SimpleNamespace(resolve_range=lambda *a, **k: object())
    manager.mooncake_metrics = SimpleNamespace(measure=lambda *a, **k: nullcontext())
    output = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[
        SimpleNamespace(req_id="r", block_ids=([3, 4],), num_computed_tokens=2)
    ], scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()),
                             kv_connector_metadata=None)
    return manager, output


def test_import_publishes_ready_only_after_successful_session():
    manager, output = fixture()
    manager._mooncake_store.import_objects.side_effect = lambda objects: (
        pytest.fail("published before completion") if "r" in manager._mooncake_store_ready else None
    )
    attach_sparse_store_transactions(output, imports=[dict(req_id="r", block_hash="h", block_index=0)])
    with patch("torch_npu.npu.synchronize"):
        manager.process_mooncake_store_metadata(output)
    assert manager._mooncake_store_ready == {"r"}
    assert manager._mooncake_store.import_objects.call_count == 1
    manager._copy_mooncake_store_sidecars.assert_called_once_with((3,), to_device=True)
    manager._hydrate_mooncake_store_main.assert_called_once_with(3)


def test_partial_import_failure_stops_future_steps_and_retains_reservation():
    manager, output = fixture()
    manager._mooncake_store.import_objects.side_effect = RuntimeError("partial")
    attach_sparse_store_transactions(output, imports=[dict(req_id="r", block_hash="h", block_index=0)])
    with patch("torch_npu.npu.synchronize"), pytest.raises(RuntimeError, match="partial"):
        manager.process_mooncake_store_metadata(output)
    assert not manager._mooncake_store_ready
    assert manager._mooncake_store_blocks["r"][0] == [3, 4]
    with pytest.raises(RuntimeError, match="another step"):
        manager.process_mooncake_store_metadata(output)


def test_seal_rejects_allocated_but_incomplete_block():
    manager, output = fixture()
    attach_sparse_store_transactions(output, seals=[dict(req_id="r", block_hash="h", block_index=1)])
    with patch("torch_npu.npu.synchronize"), pytest.raises(ValueError, match="fully computed"):
        manager.process_mooncake_store_metadata(output)
    manager._mooncake_store.seal_objects.assert_not_called()


def test_pd_conflict_rejected_before_native_session():
    manager, output = fixture()
    manager.vllm_config.kv_transfer_config = object()
    attach_sparse_store_transactions(output, imports=[dict(req_id="r", block_hash="h", block_index=0)])
    with patch("torch_npu.npu.synchronize") as sync, pytest.raises(ValueError, match="PD connector"):
        manager.process_mooncake_store_metadata(output)
    sync.assert_not_called()


def test_unknown_gate_blocks_shutdown():
    manager, _ = fixture()
    manager._mooncake_store.gate.poisoned = True
    with pytest.raises(RuntimeError, match="unknown"):
        manager.assert_mooncake_store_shutdown_safe()


def test_model_hydration_copies_complete_main_and_sidecar_pages():
    manager, _ = fixture()
    # Invoke real hydration helpers on CPU tensors; NPU synchronization mocked.
    del manager._copy_mooncake_store_sidecars
    del manager._hydrate_mooncake_store_main
    manager.offload_layer_names = ["layer"]
    main_k = torch.zeros_like(manager.k_caches_cpu[0])
    main_v = torch.zeros_like(manager.v_caches_cpu[0])
    manager._mooncake_store_main_hbm = {"layer": (main_k, main_v)}
    manager.k_caches_cpu[0].fill_(2)
    manager.v_caches_cpu[0].fill_(3)
    staging = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    manager._mooncake_store_sidecar_views = [staging]
    with patch("torch_npu.npu.current_stream"):
        manager._copy_mooncake_store_sidecars((3,), to_device=True)
        manager._hydrate_mooncake_store_main(3)
    assert manager._mooncake_store_sidecars[0][5][3].tolist() == [1, 2, 3, 4]
    assert main_k.view(torch.uint8).reshape(-1, 4)[3].tolist() == manager.k_caches_cpu[0].view(torch.uint8).reshape(-1, 4)[3].tolist()
    assert main_v.view(torch.uint8).reshape(-1, 4)[3].tolist() == manager.v_caches_cpu[0].view(torch.uint8).reshape(-1, 4)[3].tolist()


def test_producer_normalizes_framework_bytes_hash():
    _, output = fixture()
    attach_sparse_store_transactions(output, imports=[dict(req_id="r", block_hash=b"\x01\xab", block_index=0)])
    assert output.kv_connector_metadata.sparse_store_imports[0]["block_hash"] == "01ab"


def test_v2_store_restore_runs_after_upstream_zero_and_copy():
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    runner = NPUModelRunner.__new__(NPUModelRunner)
    order = []
    runner.sparse_kv_offload_manager = SimpleNamespace(
        process_mooncake_store_metadata=lambda output: order.append("restore")
    )
    parent = NPUModelRunner.__mro__[1]
    with patch.object(parent, "update_requests", side_effect=lambda output: order.append("zero_and_cow")):
        runner.update_requests(object())
    assert order == ["zero_and_cow", "restore"]


def test_peer_healthy_gate_rejects_owner_unknown_before_shutdown():
    manager, _ = fixture()
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(cpu_group=object())
    def owner_unknown(output, local, **kwargs):
        output[:] = ["owner Store completion unknown", None]
    with patch("torch.distributed.all_gather_object", side_effect=owner_unknown):
        with pytest.raises(RuntimeError, match="shutdown preflight"):
            manager.assert_mooncake_store_shutdown_safe()
    assert manager._mooncake_store.gate.poisoned is False


def test_import_must_cover_all_skipped_prefix_blocks():
    manager, output = fixture()
    output.scheduled_new_reqs[0].num_computed_tokens = 4
    attach_sparse_store_transactions(output, imports=[dict(req_id="r", block_hash="h", block_index=0)])
    with patch("torch_npu.npu.synchronize"), pytest.raises(ValueError, match="complete scheduler"):
        manager.process_mooncake_store_metadata(output)
    manager._mooncake_store.import_objects.assert_not_called()


def test_cached_import_is_rejected_before_native_copy():
    manager, output = fixture()
    manager._mooncake_store_blocks = {"r": ([3, 4],)}
    manager._mooncake_store_computed = {"r": 2}
    output.scheduled_new_reqs = []
    attach_sparse_store_transactions(output, imports=[dict(req_id="r", block_hash="h", block_index=0)])
    with patch("torch_npu.npu.synchronize"), pytest.raises(ValueError, match="newly admitted"):
        manager.process_mooncake_store_metadata(output)
    manager._mooncake_store.import_objects.assert_not_called()
