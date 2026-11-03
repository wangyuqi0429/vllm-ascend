"""The eager bridge must preserve selected multiplicity and ordinary SFA dispatch."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl


@pytest.mark.parametrize("cache,tail", [(0, 127), (0, 128), (2048, 1), (2048, 128)])
def test_native_bridge_uses_existing_sfa_and_dense_or_selected_plus_tail(cache, tail):
    logical = tail if cache == 0 else cache + tail
    # Repeated destination intentionally tests multiplicity, not a set rewrite.
    selected_slots = torch.arange(2048, dtype=torch.int32) % 1024
    owner = SimpleNamespace(
        lim_topk_dst=selected_slots.view(1, 1, -1),
        lim_topk_misses=torch.tensor([0]),
        lim_misses=torch.tensor([0]),
        lim_miss_src=torch.empty((1, 32768), dtype=torch.int32),
        lim_miss_dst=torch.empty((1, 32768), dtype=torch.int32),
    )
    impl = SimpleNamespace(
        lim_indexer_owner=owner, skip_topk=False, has_indexer=True,
        kv_lora_rank=2, qk_rope_head_dim=1, scale=0.5,
        _in_graph_runtime=lambda: False,
    )
    manager = SimpleNamespace(
        is_mooncake=True,
        _get_offload_layer_id=lambda _: 0,
        topk_buffers_k=[torch.empty((32, 128, 1, 2))],
        topk_buffers_v=[torch.empty((32, 128, 1, 1))],
        k_caches_cpu=[torch.empty((32, 128, 1, 2))],
        v_caches_cpu=[torch.empty((32, 128, 1, 1))],
        reload_mooncake_lim=MagicMock(),
    )
    metadata = SimpleNamespace(
        num_decode_tokens=1,
        copy_sfa_pool_entries=torch.tensor([0]),
        copy_sfa_cache_tokens=torch.tensor([cache]),
        copy_sfa_logical_lens=torch.tensor([logical]),
        copy_sfa_query_ends=torch.tensor([1]),
        copy_sfa_source_block_table=torch.arange(32).view(1, -1),
        copy_sfa_hbm_block_table=torch.arange(32).view(1, -1),
    )
    query, rope = torch.empty((1, 1, 2)), torch.empty((1, 1, 1))
    ordinary, fused = MagicMock(return_value=(query,)), MagicMock()
    with patch.object(torch.ops._C_ascend, "npu_sparse_flash_attention", ordinary, create=True):
        with patch.object(torch.ops._C_ascend, "npu_fused_scatter_copy_sparse_flash_attention", fused, create=True):
            result = AscendSFAKVOffloadImpl._copy_sfa_attention(
                impl, query, rope, None, metadata, manager, "layer"
            )
    expected = list(range(logical)) if cache == 0 else selected_slots.tolist() + list(range(cache, logical))
    assert ordinary.call_args.kwargs["sparse_indices"].reshape(-1).tolist() == expected
    assert ordinary.call_args.kwargs["key"] is ordinary.call_args.kwargs["value"]
    assert result is query
    manager.reload_mooncake_lim.assert_called_once()
    fused.assert_not_called()
