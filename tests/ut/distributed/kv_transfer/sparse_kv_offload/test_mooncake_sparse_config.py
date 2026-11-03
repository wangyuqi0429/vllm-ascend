"""Configuration boundary checks for the native eager integration."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.ascend_config import SparseKVOffloadConfig


def config_input():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_text_config=SimpleNamespace(index_topk=2048, qk_rope_head_dim=64),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, prefill_context_parallel_size=1,
            decode_context_parallel_size=1, pipeline_parallel_size=1,
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )


def test_mooncake_reuses_layout_without_selecting_fused_execution():
    config = SparseKVOffloadConfig(enabled=True, backend="mooncake", keep_device_kv_cache=True, topk_buffer_size=2048)
    assert config.use_fused_copy_sfa
    assert not config.execute_fused_copy_sfa
    with patch("vllm_ascend.ascend_config.is_950", return_value=False):
        config._validate_preconditions(config_input())


@pytest.mark.parametrize("option", ["graph", "pd", "speculative"])
def test_unconnected_modes_fail_before_resource_allocation(option):
    inputs = config_input()
    if option == "graph":
        inputs.model_config.enforce_eager = False
    elif option == "pd":
        inputs.kv_transfer_config = SimpleNamespace(kv_connector_extra_config={"transfer_backend": "memfabric"})
    else:
        inputs.speculative_config = object()
    config = SparseKVOffloadConfig(enabled=True, backend="mooncake", keep_device_kv_cache=True, topk_buffer_size=2048)
    with pytest.raises(ValueError):
        config._validate_preconditions(inputs)


def test_legacy_default_retains_fused_execution_selection():
    config = SparseKVOffloadConfig(enabled=True)
    assert config.backend == "memfabric"
    assert config.execute_fused_copy_sfa


def test_mooncake_tp_debug_configuration_is_no_longer_rejected():
    inputs = config_input()
    inputs.parallel_config.tensor_parallel_size = 2
    config = SparseKVOffloadConfig(enabled=True, backend="mooncake", keep_device_kv_cache=True, topk_buffer_size=2048)
    with patch("vllm_ascend.ascend_config.is_950", return_value=False):
        config._validate_preconditions(inputs)


def test_pd_consumer_can_disable_complete_device_main_kv_allocation():
    inputs = config_input()
    inputs.kv_transfer_config = SimpleNamespace(
        kv_connector_extra_config={"transfer_backend": "mooncake"},
        is_kv_consumer=True,
        has_connector=lambda _name: False,
    )
    config = SparseKVOffloadConfig(enabled=True, backend="mooncake", keep_device_kv_cache=False, topk_buffer_size=2048)
    with patch("vllm_ascend.ascend_config.is_950", return_value=False):
        config._validate_preconditions(inputs)
