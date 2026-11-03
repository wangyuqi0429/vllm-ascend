"""Device descriptor generation/status contracts; kernels remain mocked."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ascend_config import SparseKVOffloadConfig
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager


def native_manager():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.mooncake_metrics = SimpleNamespace(measure=lambda *args, **kwargs: __import__("contextlib").nullcontext())
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.mooncake_device_execution = True
    manager.token_size_bytes_k = manager.token_size_bytes_v = 4
    manager.max_num_tokens = 4
    manager.d2h_token_indices_npu = torch.arange(4, dtype=torch.int64)
    manager.d2h_src_ptrs_npu = torch.empty(8, dtype=torch.int64)
    manager.d2h_dst_ptrs_npu = torch.empty(8, dtype=torch.int64)
    manager.d2h_lengths_npu = torch.empty(8, dtype=torch.int32)
    manager.d2h_size_npu = torch.empty(1, dtype=torch.int32)
    manager.mooncake_status = torch.zeros((1, 2, 1), dtype=torch.int32)
    manager._mooncake_native_copy = MagicMock()
    return manager


@pytest.mark.parametrize("slots,bit", [([1, 1], 16), ([1, 9], 4)])
def test_graph_slot_errors_stick_and_mask_unsafe_writes(slots, bit):
    manager = native_manager()
    manager.mooncake_status[0, 0].fill_(2)
    host_k, host_v = torch.empty((4, 2), dtype=torch.bfloat16), torch.empty((4, 2), dtype=torch.bfloat16)
    current_k, current_v = torch.empty((2, 2), dtype=torch.bfloat16), torch.empty((2, 2), dtype=torch.bfloat16)
    manager._offload_new_kv_on_current_stream(
        torch.tensor(slots), host_k, host_v, None, None, current_k, current_v,
        capturing=True, layer_id=0,
    )
    assert int(manager.mooncake_status[0, 0, 0]) & bit
    assert int(manager.mooncake_status[0, 0, 0]) & 2
    lengths = manager.d2h_lengths_npu[:4].tolist()
    assert lengths == ([0, 0, 0, 0] if bit == 16 else [4, 0, 4, 0])
    assert manager._mooncake_native_copy.call_args.kwargs["reset_status"] is False


def test_capture_warmup_compiles_both_fixed_descriptor_signatures_at_count_zero():
    manager = native_manager()
    manager.mooncake_h2d_src = torch.empty(12, dtype=torch.int64)
    manager.mooncake_h2d_dst = torch.empty(12, dtype=torch.int64)
    manager.mooncake_h2d_lengths = torch.empty(12, dtype=torch.int64)
    manager.mooncake_h2d_count = torch.ones(1, dtype=torch.int32)
    with patch("torch.npu.is_current_stream_capturing", return_value=False):
        with patch("torch_npu.npu.current_stream"):
            manager.warmup_mooncake_native_copy()
    assert manager._mooncake_native_copy.call_count == 2
    assert manager.d2h_size_npu.tolist() == [0]
    assert manager.mooncake_h2d_count.tolist() == [0]


def test_graph_config_requires_actual_fused_device_path_and_vmm():
    config = SparseKVOffloadConfig(
        enabled=True, backend="mooncake", mooncake_execution_mode="graph",
        mooncake_attention_mode="fused_copy", mooncake_memory_mode="vmm",
        keep_device_kv_cache=True, topk_buffer_size=2048,
    )
    inputs = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False, hf_text_config=SimpleNamespace(index_topk=2048, qk_rope_head_dim=64)),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, prefill_context_parallel_size=1,
                                       decode_context_parallel_size=1, pipeline_parallel_size=1),
        kv_transfer_config=None, speculative_config=None,
    )
    with patch("vllm_ascend.ascend_config.is_950", return_value=False):
        config._validate_preconditions(inputs)
        config.mooncake_memory_mode = "mmap_registered"
        with pytest.raises(ValueError, match="VMM"):
            config._validate_preconditions(inputs)


def test_status_waits_side_streams_and_reports_sticky_error():
    manager = native_manager()
    manager.mooncake_status[0, 1].fill_(8)
    manager.current_kv_save_stream = object()
    manager.fused_plan_stream = object()
    stream = MagicMock()
    with patch("torch.npu.is_current_stream_capturing", return_value=False), patch(
        "torch_npu.npu.current_stream", return_value=stream
    ):
        with pytest.raises(RuntimeError, match="replay descriptor"):
            manager.check_mooncake_status()
    assert stream.wait_stream.call_count == 2
    assert manager.mooncake_status[0, 1].item() == 8


def test_prepare_clears_only_at_step_boundary_and_closed_pool_cannot_replay():
    manager = native_manager()
    manager._closed = False
    manager._mooncake_metric_step = 7
    manager._mooncake_cache_metric_refs = {0: object()}
    manager.check_mooncake_status = MagicMock()
    with patch("torch.npu.is_current_stream_capturing", return_value=False):
        manager.prepare_mooncake_step()
        assert manager._mooncake_cache_metric_refs == {}
        assert manager._mooncake_metric_step == 8
        manager._closed = True
        with pytest.raises(RuntimeError, match="closed"):
            manager.prepare_mooncake_step()
    manager.check_mooncake_status.assert_called_once()


def test_h2d_uses_fixed_arena_clamps_count_and_preserves_sticky_status():
    manager = native_manager()
    manager.mooncake_metrics = SimpleNamespace(measure=lambda *args, **kwargs: __import__("contextlib").nullcontext())
    manager.mooncake_h2d_src = torch.empty(8, dtype=torch.int64)
    manager.mooncake_h2d_dst = torch.empty(8, dtype=torch.int64)
    manager.mooncake_h2d_lengths = torch.empty(8, dtype=torch.int64)
    manager.mooncake_h2d_count = torch.zeros(1, dtype=torch.int32)
    manager.mooncake_status[0, 1].fill_(2)
    with patch("torch.npu.is_current_stream_capturing", return_value=False):
        manager.copy_sfa_kv(torch.ones(2, dtype=torch.int64), torch.ones(2, dtype=torch.int64),
                            torch.ones(2, dtype=torch.int64), torch.tensor([5], dtype=torch.int32), layer_id=0)
    assert manager.mooncake_h2d_count.item() == 2
    assert manager.mooncake_status[0, 1].item() == 10
    args = manager._mooncake_native_copy.call_args
    assert args.args[0] is manager.mooncake_h2d_src
    assert args.kwargs["reset_status"] is False


def test_graph_lease_keeps_all_pool_owners():
    manager = native_manager()
    manager.mooncake_pool, manager.mooncake_raw = object(), object()
    manager.mooncake_membership_segment = object()
    manager.warmup_mooncake_native_copy = MagicMock()
    lease = manager.retain_mooncake_graph_resources()
    assert lease == (manager, manager.mooncake_pool, manager.mooncake_raw, manager.mooncake_membership_segment)
    manager.warmup_mooncake_native_copy.assert_called_once()


def test_cache_counts_once_per_step_and_graph_never_uses_python_refs():
    manager = native_manager()
    manager.mooncake_metrics = SimpleNamespace(enabled=True, record_cache=MagicMock(), collect=MagicMock())
    manager.sparse_kv_offload_config = SimpleNamespace(mooncake_execution_mode="eager")
    manager._mooncake_metric_step = 1
    manager._mooncake_metric_recorded_step = -1
    manager.topk = 2
    manager._mooncake_cache_metric_refs = {0: (
        torch.tensor([1]), torch.tensor([2]), torch.tensor([5]), torch.tensor([1])
    )}
    with patch("torch.npu.is_current_stream_capturing", return_value=False), patch("torch_npu.npu.current_stream"):
        manager.check_mooncake_status()
        manager.check_mooncake_status()
        manager.mooncake_metrics.record_cache.assert_called_once_with(1, 1)
        manager._closed = False
        manager.prepare_mooncake_step()
        manager.check_mooncake_status()  # prefill with no decode refs
        assert manager.mooncake_metrics.record_cache.call_count == 1
        manager.sparse_kv_offload_config.mooncake_execution_mode = "graph"
        manager._mooncake_metric_step += 1
        manager._mooncake_cache_metric_refs = {0: object()}  # must not inspect capture refs
        manager.check_mooncake_status()
        assert manager.mooncake_metrics.record_cache.call_count == 1
