import contextlib
import typing
from dataclasses import dataclass
from zlib import adler32

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.ascend_config import SparseKVOffloadConfig, get_ascend_config
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import copy_sfa_pool_capacity
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_metrics import SparseOffloadMetrics
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

# Main BF16 cache:
# [k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v].
# Sparse LI C8 indexer caches are separate and
# remain device-resident, so they are not registered with this manager.
OFFLOAD_KV_CACHE_TUPLE_LEN = 6
OFFLOAD_K_CACHE_NPU_INDEX = 0
OFFLOAD_V_CACHE_NPU_INDEX = 1
OFFLOAD_K_CACHE_CPU_INDEX = 2
OFFLOAD_V_CACHE_CPU_INDEX = 3
OFFLOAD_TOPK_BUFFER_K_INDEX = 4
OFFLOAD_TOPK_BUFFER_V_INDEX = 5
OFFLOAD_STORE_PORT_BASE = 8500

# Optional legacy runtime, initialized only when that backend is selected.
# Kept as a module reference for the worker lifetime and existing test patches.
offload = None


def _legacy_offload():
    global offload
    if offload is None:
        from memfabric_hybrid import offload as legacy_offload

        offload = legacy_offload
    return offload

FSA_EXTERNAL_PLAN_READY_MARKER = 0x5A45
FSA_PAIRED_SELECTION_COPY_MARKER = 0x5A56
FSA_SELECTION_MEMBERSHIP_MAP_INT16_COUNT = 16376
FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT = 16
FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT = 8
FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT = (
    (FSA_SELECTION_MEMBERSHIP_MAP_INT16_COUNT + FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT - 1)
    // FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
    * FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
)
FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT = (
    (
        FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT
        + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
        + FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
        - 1
    )
    // FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
    * FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
)
FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS = (
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
)


_SPARSE_KV_OFFLOAD_OPS = None
_SPARSE_KV_OFFLOAD_OP_NAMES = (
    "sparse_kv_warmup_lru_resident_threads",
    "sparse_kv_lru_resident_compact_with_plan_stable_rows",
    "sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows",
    "sparse_kv_restore_bfloat16_tensor",
    "sparse_kv_restore_int16_tensor",
)


def _sparse_kv_ops():
    global _SPARSE_KV_OFFLOAD_OPS
    if _SPARSE_KV_OFFLOAD_OPS is not None:
        return _SPARSE_KV_OFFLOAD_OPS
    if not enable_custom_op():
        raise RuntimeError("Sparse KV offload requires custom ops to be enabled.")

    missing_ops = [name for name in _SPARSE_KV_OFFLOAD_OP_NAMES if not hasattr(torch.ops._C_ascend, name)]
    if missing_ops:
        raise RuntimeError(
            "Sparse KV offload requires _C_ascend sparse KV ops, "
            f"missing={missing_ops}. Rebuild vllm_ascend_C with Atlas A3 support."
        )
    _SPARSE_KV_OFFLOAD_OPS = torch.ops._C_ascend
    return _SPARSE_KV_OFFLOAD_OPS


def allocate_kv_offload_topk_buffer_pair(
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_width = 1
    if vllm_config.speculative_config is not None:
        decode_width += vllm_config.speculative_config.num_speculative_tokens

    topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
    num_kv_heads = 1  # sparse kv offload only support sfa(mla) now.
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    max_num_topk_rows = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.scheduler_config.max_num_seqs * decode_width,
    )
    if sparse_kv_offload_config.use_fused_copy_sfa:
        # The backend imports this manager; defer its layout constant import.
        from vllm_ascend.attention.sfa_kv_offload import COPY_SFA_TAIL_BLOCKS

        # LIM merges all query selections into one request-owned hot row.
        # Graph padding uses a second, private arena of the same capacity.
        # Query-row metadata still needs max_num_seqs * decode_width, but
        # multiplying the large hot K/V allocation by that width wastes HBM.
        request_rows = 2 * copy_sfa_pool_capacity(vllm_config.scheduler_config.max_num_seqs)
        max_num_topk_rows = request_rows
        topk_buffer_size += COPY_SFA_TAIL_BLOCKS * vllm_config.cache_config.block_size
    topk_buffer_k_size_bytes = max_num_topk_rows * topk_buffer_size * num_kv_heads * k_dim * torch.bfloat16.itemsize
    topk_buffer_v_size_bytes = max_num_topk_rows * topk_buffer_size * num_kv_heads * v_dim * torch.bfloat16.itemsize
    # NOTE make sure to allocate k+v together and split them after allocate.
    # Refer to the comment in empty_aligned_int8_cpu_tensors for reason.
    topk_buffer_raw = torch.empty([topk_buffer_k_size_bytes + topk_buffer_v_size_bytes], dtype=torch.int8, device="npu")
    topk_buffer_k = (
        topk_buffer_raw[:topk_buffer_k_size_bytes]
        .view(torch.bfloat16)
        .view([max_num_topk_rows, topk_buffer_size, num_kv_heads, k_dim])
    )
    topk_buffer_v = (
        topk_buffer_raw[topk_buffer_k_size_bytes : topk_buffer_k_size_bytes + topk_buffer_v_size_bytes]
        .view(torch.bfloat16)
        .view([max_num_topk_rows, topk_buffer_size, num_kv_heads, v_dim])
    )
    if sparse_kv_offload_config.use_fused_copy_sfa:
        # Dummy copy-SFA reads its private hot rows during capture/replay.
        topk_buffer_raw.zero_()
    return (topk_buffer_k, topk_buffer_v)


def allocate_kv_offload_topk_profile_buffers(
    kv_cache_spec: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    num_offload_layers = sum(bool(getattr(spec, "store_on_host", False)) for spec in kv_cache_spec.values())
    if num_offload_layers == 0:
        raise RuntimeError("Sparse KV offload profile did not find any host-resident SFA KV cache layers.")

    buffers = [
        allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config) for _ in range(num_offload_layers)
    ]
    buffer_bytes = sum(tensor.numel() * tensor.element_size() for buffer_pair in buffers for tensor in buffer_pair)
    logger.info_once(
        "Sparse KV offload reserved %.2f GiB of KV offload topk buffers across %d layers for profile run.",
        buffer_bytes / (1 << 30),
        num_offload_layers,
    )
    return buffers


_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024
# Worst-case 2 MiB-alignment waste per host layer (see empty_aligned_int8_cpu_tensors):
# 1x for rounding the raw buffer base address up to the alignment boundary, plus 1x tail
# padding for each of the two aligned tensors (k_cache_cpu, v_cache_cpu) whose sizes are
# rounded up to whole alignment chunks.
_CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER = 3 * _CPU_CACHE_ALIGNMENT
_VLLM_NULL_BLOCK_COUNT = 1


def empty_aligned_int8_cpu_tensors(
    sizes: list[int],
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> list[torch.Tensor]:
    """
    Allocate multiple int8 tensors with specified sizes,
    each aligned to specified alignment,
    and minimize the gap between each tensor's data_ptr.
    Keep paired K/V allocations aligned within the shared host pool.
    """
    chunk_nums = [cdiv(size, alignment) for size in sizes]
    total_chunk_num = 1 + sum(chunk_nums)
    raw_tensor = _legacy_offload().empty([total_chunk_num * alignment], dtype=torch.int8, pin_memory=True)
    base_addr = raw_tensor.data_ptr()
    if base_addr % alignment:
        base_addr = (base_addr // alignment + 1) * alignment
    base_offset = base_addr - raw_tensor.data_ptr()
    allocate_tensors = []
    for size, chunk_num in zip(sizes, chunk_nums):
        allocate_tensors.append(raw_tensor[base_offset : base_offset + size])
        base_offset += chunk_num * alignment
    return allocate_tensors


@dataclass(frozen=True)
class SparseKVOffloadMemoryBudget:
    npu_limit_blocks: int
    dram_limit_blocks: int
    workload_limit_blocks: int
    final_num_blocks: int
    final_planner_bytes: int
    planned_host_bytes: int
    planned_device_bytes: int
    host_alignment_reserve_bytes: int
    limiting_factor: str


def _split_host_device_kv_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> tuple[list[KVCacheSpec], list[KVCacheSpec]]:
    if not kv_cache_spec:
        raise ValueError("Sparse KV offload requires a non-empty kv_cache_spec")
    host_specs: list[KVCacheSpec] = []
    device_specs: list[KVCacheSpec] = []
    for spec in kv_cache_spec.values():
        if getattr(spec, "store_on_host", False):
            host_specs.append(spec)
        else:
            device_specs.append(spec)
    if not host_specs:
        raise ValueError("Sparse KV offload requires at least one host KV cache spec")
    if not device_specs:
        raise ValueError("Sparse KV offload requires at least one device KV cache spec")
    block_sizes = {spec.block_size for spec in host_specs + device_specs}
    if len(block_sizes) != 1:
        raise ValueError(f"Sparse KV offload memory planning requires one shared block size, got {sorted(block_sizes)}")
    return host_specs, device_specs


def plan_sparse_kv_offload_memory(
    kv_cache_spec: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    available_device_memory_bytes: int,
    dram_limit_bytes: int,
    keep_device_kv_cache: bool,
) -> SparseKVOffloadMemoryBudget:
    """Bound sparse KV offload blocks by NPU, DRAM, and active demand."""
    host_specs, device_specs = _split_host_device_kv_specs(kv_cache_spec)
    host_page_size_bytes = sum(spec.page_size_bytes for spec in host_specs)
    device_page_size_bytes = sum(spec.page_size_bytes for spec in device_specs)
    total_page_size_bytes = host_page_size_bytes + device_page_size_bytes

    host_alignment_reserve_bytes = len(host_specs) * _CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    sparse_options = (getattr(vllm_config, "additional_config", None) or {}).get("sparse_kv_offload_config", {})
    if sparse_options.get("backend") == "mooncake" and sparse_options.get("mooncake_attention_mode") == "fused_overlap":
        rows = min(vllm_config.scheduler_config.max_num_batched_tokens, vllm_config.scheduler_config.max_num_seqs)
        host_alignment_reserve_bytes += rows * FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT * 2
    usable_dram_bytes = max(dram_limit_bytes - host_alignment_reserve_bytes, 0)
    dram_limit_blocks = usable_dram_bytes // host_page_size_bytes

    npu_page_size_bytes = total_page_size_bytes if keep_device_kv_cache else device_page_size_bytes
    npu_limit_blocks = max(available_device_memory_bytes, 0) // npu_page_size_bytes

    max_blocks_per_request = max(
        cdiv(
            spec.max_memory_usage_bytes(vllm_config),
            spec.page_size_bytes,
        )
        for spec in host_specs + device_specs
    )
    workload_limit_blocks = max_blocks_per_request * vllm_config.scheduler_config.max_num_seqs + _VLLM_NULL_BLOCK_COUNT

    limits = {
        "npu": npu_limit_blocks,
        "dram": dram_limit_blocks,
        "workload": workload_limit_blocks,
    }
    if dram_limit_blocks < npu_limit_blocks:
        logger.warning_once(
            "Sparse KV offload: host DRAM budget allows only %d KV blocks while NPU "
            "memory could hold %d, so batch size / max_model_len capacity is limited "
            "by host memory. Consider increasing sparse_kv_offload_config."
            "dram_size_per_dp_GB (current budget %.2f GiB) to use more of the NPU "
            "capacity.",
            dram_limit_blocks,
            npu_limit_blocks,
            dram_limit_bytes / (1 << 30),
        )
    limiting_factor = min(limits, key=lambda name: limits[name])
    final_num_blocks = limits[limiting_factor]
    final_planner_bytes = final_num_blocks * total_page_size_bytes
    planned_host_bytes = final_num_blocks * host_page_size_bytes
    planned_device_bytes = final_num_blocks * npu_page_size_bytes
    return SparseKVOffloadMemoryBudget(
        npu_limit_blocks=npu_limit_blocks,
        dram_limit_blocks=dram_limit_blocks,
        workload_limit_blocks=workload_limit_blocks,
        final_num_blocks=final_num_blocks,
        final_planner_bytes=final_planner_bytes,
        planned_host_bytes=planned_host_bytes,
        planned_device_bytes=planned_device_bytes,
        host_alignment_reserve_bytes=host_alignment_reserve_bytes,
        limiting_factor=limiting_factor,
    )


def get_sparse_kv_offload_cpu_pool_size_bytes(
    kv_cache_config: KVCacheConfig,
) -> int:
    """Return a safe upper bound for aligned host KV allocations."""
    layer_specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            layer_specs.update(group.kv_cache_spec.kv_cache_specs)
        else:
            layer_specs.update((layer_name, group.kv_cache_spec) for layer_name in group.layer_names)
    host_specs = [spec for spec in layer_specs.values() if getattr(spec, "store_on_host", False)]
    if not host_specs:
        raise ValueError("Sparse KV offload requires host-resident KV cache specs")
    raw_host_bytes = kv_cache_config.num_blocks * sum(spec.page_size_bytes for spec in host_specs)
    alignment_reserve_bytes = len(host_specs) * _CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    return raw_host_bytes + alignment_reserve_bytes


def allocate_kv_cache_tensors_for_sparse_kv_offload(
    k_tensor_size: int,
    v_tensor_size: int,
    alignment: int,
    tp_rank: int,
    keep_device_kv_cache: bool,
    npu_kv_cache_allocate_func: typing.Callable,
    *,
    layer_name: str | None = None,
):
    if getattr(get_ascend_config().sparse_kv_offload_config, "backend", "memfabric") == "mooncake":
        k_tensor_cpu, v_tensor_cpu = get_sparse_kv_offload_manager().allocate_mooncake_views(
            (k_tensor_size, v_tensor_size), alignment, layer_name=layer_name
        )
    elif tp_rank == 0:
        [k_tensor_cpu, v_tensor_cpu] = empty_aligned_int8_cpu_tensors(
            [k_tensor_size, v_tensor_size], alignment
        )
    else:
        k_tensor_cpu = None
        v_tensor_cpu = None

    if keep_device_kv_cache:
        k_tensor = npu_kv_cache_allocate_func(
            k_tensor_size,
            alignment,
        )
        v_tensor = npu_kv_cache_allocate_func(
            v_tensor_size,
            alignment,
        )
    else:
        k_tensor = None
        v_tensor = None

    return (k_tensor, v_tensor, k_tensor_cpu, v_tensor_cpu, k_tensor_size + v_tensor_size)


def reshape_kv_cache_tensors_for_sparse_kv_offload(
    raw_cache_tensors: tuple,
    current_kv_cache_spec: AttentionSpec,
    attn_backend: type[AttentionBackend],
    tp_rank: int,
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    raw_k_tensor, raw_v_tensor, raw_k_tensor_cpu, raw_v_tensor_cpu, sum_page_size_bytes = raw_cache_tensors
    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes
    kv_cache_shape = attn_backend.get_kv_cache_shape(
        num_blocks,
        current_kv_cache_spec.block_size,
        current_kv_cache_spec.num_kv_heads,
        current_kv_cache_spec.head_size,
    )
    mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    k_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        k_dim,
    )
    v_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        v_dim,
    )
    k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype

    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape) if raw_k_tensor is not None else None
    v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape) if raw_v_tensor is not None else None

    if tp_rank == 0 or sparse_kv_offload_config.backend == "mooncake":
        k_cache_cpu = raw_k_tensor_cpu.view(k_cache_dtype).view(k_shape)
        v_cache_cpu = raw_v_tensor_cpu.view(v_cache_dtype).view(v_shape)
        if sparse_kv_offload_config.backend == "mooncake":
            owner = get_sparse_kv_offload_manager().mooncake_pool
            k_cache_cpu._mooncake_pool = owner
            v_cache_cpu._mooncake_pool = owner
    else:
        k_cache_cpu = None
        v_cache_cpu = None

    topk_buffer_k, topk_buffer_v = allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config)
    return (k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v)


def update_sparse_kv_offload_metadata(
    num_tokens: int,
    num_reqs: int,
    num_tokens_padded: int,
    num_reqs_padded: int,
    req_ids: list[str],
    query_start_loc: CpuGpuBuffer | torch.Tensor | np.ndarray,
    offload_req_ids_tensor: CpuGpuBuffer,
    offload_token_to_req: CpuGpuBuffer,
) -> None:
    """Populate per-request identity tensors for the Sparse KV offload LRU.

    Request ids are adler32 hashes of the scheduler request id,
    rows are reset whenever the occupying request changes.
    """
    offload_req_ids_tensor.np[:num_reqs_padded].fill(0)
    effective_num_reqs = min(num_reqs, len(req_ids))
    req_id_values = np.asarray(
        [
            adler32(req_id.encode("utf-8")) if isinstance(req_id, str) else row + 1
            for row, req_id in enumerate(req_ids[:effective_num_reqs])
        ],
        dtype=np.int64,
    )
    offload_req_ids_tensor.np[:effective_num_reqs] = req_id_values
    offload_req_ids_tensor.copy_to_gpu(num_reqs_padded)

    if isinstance(query_start_loc, CpuGpuBuffer):
        query_start_loc_cpu = query_start_loc.cpu[: num_reqs + 1].numpy()
    elif isinstance(query_start_loc, torch.Tensor):
        if query_start_loc.device.type != "cpu":
            raise ValueError("Sparse KV offload query boundaries must have a CPU mirror")
        query_start_loc_cpu = query_start_loc[: num_reqs + 1].numpy()
    else:
        query_start_loc_cpu = query_start_loc[: num_reqs + 1]
    query_lens = np.diff(query_start_loc_cpu).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        raise RuntimeError(
            "KV offload token_to_req metadata is shorter than the scheduled token batch: "
            f"metadata={token_to_req.shape[0]}, tokens={num_tokens}"
        )
    offload_token_to_req.np[:num_tokens] = token_to_req[:num_tokens]
    if num_tokens_padded > num_tokens:
        offload_token_to_req.np[num_tokens:num_tokens_padded].fill(0)
    offload_token_to_req.copy_to_gpu(num_tokens_padded)


def prepare_sparse_kv_offload_mtp_dummy_metadata(
    num_tokens: int,
    num_reqs: int,
    query_start_loc_cpu: torch.Tensor,
    req_ids_buffer: CpuGpuBuffer,
    token_to_req_buffer: CpuGpuBuffer,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not get_ascend_config().sparse_kv_offload_config.enabled:
        return None, None
    if req_ids_buffer is None or token_to_req_buffer is None:
        raise RuntimeError("Sparse KV offload metadata buffers are not initialized")

    query_lens = np.diff(query_start_loc_cpu[: num_reqs + 1].numpy()).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        token_to_req = np.pad(token_to_req, (0, num_tokens - token_to_req.shape[0]))

    req_ids_buffer.np[:num_reqs] = np.arange(1, num_reqs + 1, dtype=np.int64)
    req_ids_buffer.copy_to_gpu(num_reqs)
    token_to_req_buffer.np[:num_tokens] = token_to_req[:num_tokens]
    token_to_req_buffer.copy_to_gpu(num_tokens)
    return (
        req_ids_buffer.gpu[:num_reqs],
        token_to_req_buffer.gpu[:num_tokens],
    )


class SparseKVOffloadManager:
    """
    A manager responsible to the Sparse KV cache Offload.
    It enlarge the available memory that scheduler can see,
    so we can schedule longer max_model_len or larger decode batch size.
    No more scheduling logic: we reuse the original block_table/slot_mapping.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        sparse_kv_offload_config: SparseKVOffloadConfig,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.sparse_kv_offload_config = sparse_kv_offload_config
        self.is_mooncake = sparse_kv_offload_config.backend == "mooncake"
        self.mooncake_device_execution = self.is_mooncake and sparse_kv_offload_config.mooncake_attention_mode != "unfused"
        self.mooncake_fused_copy = self.is_mooncake and sparse_kv_offload_config.execute_fused_copy_sfa
        self._closed = False
        self._mooncake_inflight = None
        self.mooncake_metrics = SparseOffloadMetrics(
            enabled=self.is_mooncake and sparse_kv_offload_config.mooncake_metrics,
            npu_runtime=torch_npu.npu,
        )
        self._mooncake_cache_metric_refs = {}
        self._mooncake_metric_step = 0
        self._mooncake_metric_recorded_step = -1
        self._mooncake_store = None
        self._mooncake_store_blocks = {}
        self._mooncake_store_ready = set()
        self._mooncake_store_computed = {}
        self._mooncake_store_failed = False

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.num_target_layers = model_config.get_num_layers(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.block_size = self._infer_group_block_sizes(self.kv_cache_config)
        self.topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
        self.topk = sparse_kv_offload_config.topk
        self.use_fused_overlap = sparse_kv_offload_config.use_fused_overlap
        self.use_fused_copy_sfa = sparse_kv_offload_config.use_fused_copy_sfa

        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        decode_width = 1
        if vllm_config.speculative_config is not None:
            decode_width += vllm_config.speculative_config.num_speculative_tokens
        self.max_num_topk_rows = min(
            self.max_num_tokens,
            self.max_num_reqs * decode_width,
        )
        self.fused_overlap_membership_map: torch.Tensor | None = None
        self.fused_overlap_membership_map_rows = 0
        self.fused_overlap_plan_owner_layer_id: int | None = None
        self.fused_overlap_plan_topk: int | None = None
        self.fused_overlap_plan_num_tokens = 0
        self.fused_overlap_plan_membership_map: torch.Tensor | None = None
        if not self.is_mooncake or self.use_fused_overlap:
            _sparse_kv_ops()

        dram_limit_bytes = int(sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30))
        planned_pool_size_bytes = get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config)
        self._mooncake_aux_bytes = (
            self.max_num_topk_rows * FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT * 2
            if self.is_mooncake and self.use_fused_overlap else 0
        )
        if planned_pool_size_bytes + self._mooncake_aux_bytes > dram_limit_bytes:
            raise ValueError(
                "Sparse KV offload planned CPU pool exceeds DRAM limit after "
                "alignment: "
                f"planned={planned_pool_size_bytes / (1 << 30):.2f} GiB, "
                f"limit={sparse_kv_offload_config.dram_size_per_dp_GB} GiB, "
                f"num_blocks={kv_cache_config.num_blocks}"
            )
        actual_pool_size_bytes = min(planned_pool_size_bytes, dram_limit_bytes)
        logger.info(
            "SparseKVOffloadManager starts CPU KV pool initialization: "
            "planned=%.2f GiB, configured_limit=%s GiB, num_blocks=%s.",
            actual_pool_size_bytes / (1 << 30),
            sparse_kv_offload_config.dram_size_per_dp_GB,
            kv_cache_config.num_blocks,
        )
        if self.is_mooncake:
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_adapter import (
                MooncakeSparsePool,
                SharedKVBlock,
            )

            mode = sparse_kv_offload_config.mooncake_memory_mode
            self.mooncake_pool = MooncakeSparsePool.create(
                "vllm_sparse_kv",
                {"raw": SharedKVBlock(1, (actual_pool_size_bytes // 2,))},
                device_id=torch_npu.npu.current_device(),
                world_size=self.tp_size,
                rank_id=self.tp_rank,
                comm_group=self.tp_group if self.tp_size > 1 else None,
                mmap=mode == "mmap_registered",
                host_register=mode == "mmap_registered",
            )
            view_error = None
            try:
                self.mooncake_raw = self.mooncake_pool.host_tensors("raw")[0].view(torch.int8)
                self.mooncake_pool.resolve_range("raw", 0, 0, 0, address_domain="device")
            except BaseException as exception:
                view_error = exception
            try:
                self._mooncake_check_phase(view_error, "Host/device mapping")
            except BaseException:
                self.mooncake_raw = None
                self.mooncake_pool = None
                self._closed = True
                raise
            self._mooncake_layout_offset = 0
            self._mooncake_regions: list[tuple[int, int]] = []
            if self.mooncake_device_execution:
                host_address = self.mooncake_pool.resolve_range("raw", 0, 0, 0, address_domain="host").address
                device_address = self.mooncake_pool.resolve_range("raw", 0, 0, 0, address_domain="device").address
                if host_address != device_address:
                    raise ValueError("Mooncake fused kernels require verified equal Host/device VMM addresses")
                from mooncake.sparse_copy import sparse_copy

                self._mooncake_native_copy = sparse_copy
            return

        config = _legacy_offload().OffloadConfig()
        config.device_id = torch_npu.npu.current_device()
        config.reserve_size = actual_pool_size_bytes
        config.alloc_size = actual_pool_size_bytes if self.tp_rank == 0 else 0
        config.world_size = self.tp_size
        config.rank_id = self.tp_rank
        config.scene = _legacy_offload().Scene.SHARED
        store_port = OFFLOAD_STORE_PORT_BASE + parallel_config.data_parallel_index
        config.store_url = f"tcp://127.0.0.1:{store_port}"
        assert _legacy_offload().initialize(config) == 0, "Sparse KV offload offload.initialize failed."
        self.tp_group.barrier()

    def _mooncake_check_phase(self, error: BaseException | None, phase: str) -> None:
        """All ranks leave an ordinary failure phase together, before reuse."""
        if self.tp_size > 1:
            errors = [None] * self.tp_size
            dist.all_gather_object(
                errors, str(error) if error is not None else None, group=self.tp_group.cpu_group
            )
            if any(item is not None for item in errors):
                raise RuntimeError(f"Mooncake TP {phase} failed: {errors}") from error
        elif error is not None:
            raise error

    def allocate_mooncake_views(
        self, sizes: tuple[int, ...], alignment: int, *, layer_name: str | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Carve initialization-time byte layout; this never assigns KV slots."""
        if self._closed or not self.is_mooncake:
            raise RuntimeError("Mooncake pool is unavailable")
        if alignment <= 0 and self.tp_size == 1:
            raise ValueError("KV byte alignment must be positive")
        base = self.mooncake_raw.data_ptr()
        offset = self._mooncake_layout_offset
        if self.tp_size > 1:
            declarations = [None] * self.tp_size
            dist.all_gather_object(
                declarations, (layer_name, offset, tuple(sizes), alignment), group=self.tp_group.cpu_group
            )
            if any(declaration != declarations[0] for declaration in declarations):
                raise ValueError("TP ranks disagree on Mooncake KV allocation order or byte layout")
            payload = None
            if self.tp_rank == 0:
                try:
                    if alignment <= 0:
                        raise ValueError("KV byte alignment must be positive")
                    planned = []
                    for size in sizes:
                        if size < 0 or size % 2:
                            raise ValueError("BF16 KV allocation byte sizes must be nonnegative and even")
                        offset = cdiv(base + offset, alignment) * alignment - base
                        if offset + size > self.mooncake_raw.numel():
                            raise ValueError("Mooncake KV layout exceeds the planned shared pool")
                        planned.append((offset, size))
                        offset += size
                    payload = ("", planned, offset)
                except Exception as error:
                    payload = (str(error), [], 0)
            error, regions, next_offset = self.tp_group.broadcast_object(payload, src=0)
            if error:
                raise ValueError(error)
            # Byte kernels need BF16 alignment, not the owner's optional
            # huge-page packing hint. All ranks reuse the same relative layout.
            mapping_error = (
                "TP mapping cannot represent the agreed BF16 byte layout"
                if any((base + start) % 2 or start + size > self.mooncake_raw.numel() for start, size in regions)
                else None
            )
            mapping_errors = [None] * self.tp_size
            dist.all_gather_object(mapping_errors, mapping_error, group=self.tp_group.cpu_group)
            if any(item is not None for item in mapping_errors):
                raise ValueError(str(mapping_errors))
            self._mooncake_layout_offset = next_offset
            self._mooncake_regions.extend(regions)
            return tuple(self.mooncake_raw[start : start + size] for start, size in regions)
        regions = []
        for size in sizes:
            if size < 0 or size % 2:
                raise ValueError("BF16 KV allocation byte sizes must be nonnegative and even")
            offset = cdiv(base + offset, alignment) * alignment - base
            if offset + size > self.mooncake_raw.numel():
                raise ValueError("Mooncake KV layout exceeds the planned shared pool")
            regions.append((offset, size))
            offset += size
        self._mooncake_layout_offset = offset
        self._mooncake_regions.extend(regions)
        return tuple(self.mooncake_raw[start : start + size] for start, size in regions)

    def _prepare_mooncake_store_sidecars(self) -> None:
        if self._mooncake_store_sidecar_pool is not None:
            return
        from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_adapter import (
            MooncakeSparsePool, SharedKVBlock,
        )
        sizes = [entry[4] for entry in self._mooncake_store_sidecars]
        total = sum(sizes)
        if self.mooncake_raw.numel() + total + self._mooncake_aux_bytes > int(
            self.sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30)
        ):
            raise ValueError("Sparse Store sidecar staging exceeds configured DRAM budget")
        if any(size % 2 for size in sizes):
            raise ValueError("Sparse Store sidecar page bytes must have BF16 storage alignment")
        self._mooncake_store_sidecar_pool = MooncakeSparsePool.create(
            "vllm_sparse_store_sidecars", {"raw": SharedKVBlock(1, (total // 2,))},
            device_id=torch_npu.npu.current_device(), world_size=self.tp_size, rank_id=self.tp_rank,
            comm_group=self.tp_group if self.tp_size > 1 else None, mmap=False, host_register=False,
        )
        self._mooncake_store_sidecar_raw = self._mooncake_store_sidecar_pool.host_tensors("raw")[0].view(torch.uint8)
        self._mooncake_store_sidecar_ranges, self._mooncake_store_sidecar_views = [], []
        offset = 0
        for size in sizes:
            self._mooncake_store_sidecar_ranges.append(self._mooncake_store_sidecar_pool.resolve_range(
                "raw", 0, offset, size, address_domain="host"))
            self._mooncake_store_sidecar_views.append(self._mooncake_store_sidecar_raw[offset:offset + size])
            offset += size

    def _copy_mooncake_store_sidecars(self, physical_blocks, *, to_device: bool) -> None:
        for entry, staging, physical in zip(
            self._mooncake_store_sidecars, self._mooncake_store_sidecar_views, physical_blocks
        ):
            device_page = entry[5][physical]
            if to_device:
                device_page.copy_(staging, non_blocking=False)
            elif self.tp_rank == 0:
                staging.copy_(device_page, non_blocking=False)
        torch_npu.npu.current_stream().synchronize()

    def _hydrate_mooncake_store_main(self, physical: int) -> None:
        for name, k_host, v_host, sizes in zip(
            self.offload_layer_names, self.k_caches_cpu, self.v_caches_cpu, self.cpu_block_lens
        ):
            for destination, host, size in zip(self._mooncake_store_main_hbm[name], (k_host, v_host), sizes):
                if destination is None:  # decode-only consumer uses the Host pool
                    continue
                rows = destination.view(torch.uint8).reshape(-1, size)
                source = host.view(torch.uint8).reshape(-1, size)
                rows[physical].copy_(source[physical], non_blocking=False)
        torch_npu.npu.current_stream().synchronize()

    def assert_mooncake_store_shutdown_safe(self) -> None:
        if not self.is_mooncake:
            return
        bridge = getattr(self, "_mooncake_store", None)
        error = None
        if bridge is not None and bridge.gate.poisoned:
            error = RuntimeError("Store completion unknown: retain client, slots and registrations")
        self._mooncake_check_phase(error, "Store shutdown preflight")

    def process_mooncake_store_metadata(self, scheduler_output) -> None:
        """Execute synchronous Store transactions before the framework forward.

        Scheduler block assignments are authoritative; metadata supplies only
        request/hash coordinates. No raw pointer or arbitrary range is accepted.
        A failed transaction is instance fail-stop, so no partial pool can run.
        """
        if not self.is_mooncake:
            return
        if self._closed or self._mooncake_store_failed:
            raise RuntimeError("Mooncake Store instance cannot admit another step")
        if self._mooncake_store is not None and self._mooncake_store.gate.poisoned:
            raise RuntimeError("Mooncake Store completion is unknown; retain slots and client")
        error = None
        try:
            # Prior graph and side-stream accesses must retire before any pool write
            # or scheduler assignment can be reused by the synchronous transaction.
            for req_id in getattr(scheduler_output, "finished_req_ids", ()):
                self._mooncake_store_blocks.pop(req_id, None)
                self._mooncake_store_computed.pop(req_id, None)
                self._mooncake_store_ready.discard(req_id)
            for request in scheduler_output.scheduled_new_reqs:
                self._mooncake_store_blocks[request.req_id] = tuple(list(g) for g in request.block_ids)
                self._mooncake_store_ready.discard(request.req_id)
                self._mooncake_store_computed[request.req_id] = request.num_computed_tokens
            cached = scheduler_output.scheduled_cached_reqs
            if len(cached.req_ids) != len(cached.new_block_ids) or len(cached.req_ids) != len(cached.num_computed_tokens):
                raise ValueError("Store scheduler cached vectors differ in length")
            for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
                self._mooncake_store_computed[req_id] = computed
            for req_id, additions in zip(cached.req_ids, cached.new_block_ids):
                if additions is None:
                    continue
                if req_id in cached.resumed_req_ids:
                    self._mooncake_store_blocks[req_id] = tuple(list(g) for g in additions)
                    self._mooncake_store_ready.discard(req_id)
                else:
                    current = self._mooncake_store_blocks.get(req_id)
                    if current is None:
                        raise ValueError("Store scheduler snapshot is missing a cached request")
                    if len(current) != len(additions):
                        raise ValueError("Store scheduler group count changed")
                    for group, extra in zip(current, additions):
                        group.extend(extra)
            metadata = scheduler_output.kv_connector_metadata
            imports = tuple(getattr(metadata, "sparse_store_imports", ()))
            seals = tuple(getattr(metadata, "sparse_store_seals", ()))
        except BaseException as exception:
            error = exception
        try:
            self._mooncake_check_phase(error, "Store scheduler metadata")
            if self.tp_size > 1:
                signatures = [None] * self.tp_size
                dist.all_gather_object(signatures, repr((imports, seals)), group=self.tp_group.cpu_group)
                if any(value != signatures[0] for value in signatures):
                    raise ValueError("TP ranks received different Store transactions")
        except BaseException:
            self._mooncake_store_failed = True
            raise
        error = None
        transactions = []
        try:
            if imports or seals:
                if self.vllm_config.kv_transfer_config is not None:
                    raise ValueError("Sparse Store metadata cannot share a step/client with a PD connector")
                main_group = self._mooncake_store_main_group
                if self.tp_size > 1 and self._mooncake_store_nonreplicated_sidecars:
                    raise ValueError("TP Sparse Store supports only proven replicated DeepseekV32IndexerCache sidecars")
                if main_group is None:
                    raise ValueError("Sparse Store main group was not registered")
                if not self._mooncake_store_sidecars:
                    raise ValueError("Sparse Store must register the model indexer sidecars")
                self._prepare_mooncake_store_sidecars()
                # Explicit drain at this external transaction boundary, never inside
                # graph capture. It proves old hot readers have stopped before import.
                torch_npu.npu.synchronize()
                if self.mooncake_device_execution:
                    self.check_mooncake_status()
                from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_sparse_store import (
                    MooncakeSparseStore, StoreSessionGate, StoreObject, StoreRange,
                )
                transactions = []
                occupied = {}
                for owner, groups in self._mooncake_store_blocks.items():
                    for group_id, blocks in enumerate(groups):
                        for block in blocks:
                            coordinate = (group_id, block)
                            if coordinate in occupied:
                                raise ValueError("Store active block aliases another logical/request block")
                            occupied[coordinate] = owner
                fresh_ids = {request.req_id for request in scheduler_output.scheduled_new_reqs}
                import_indices = {}
                for item in imports:
                    if not isinstance(item, dict) or item.get("req_id") not in fresh_ids:
                        raise ValueError("Store imports require a newly admitted request with fresh hot residency")
                    import_indices.setdefault(item["req_id"], []).append(item.get("block_index"))
                for req_id, indices in import_indices.items():
                    computed = self._mooncake_store_computed[req_id]
                    if computed % self.block_size or sorted(indices) != list(range(computed // self.block_size)):
                        raise ValueError("Store import must cover the complete scheduler skipped full-block prefix")
                for seal, items in ((False, imports), (True, seals)):
                    objects, requests = [], set()
                    for item in items:
                        if not isinstance(item, dict) or set(item) != {"req_id", "block_hash", "block_index"}:
                            raise ValueError("Store item requires req_id/block_hash/block_index only")
                        req_id, logical = item["req_id"], item["block_index"]
                        if isinstance(logical, bool) or not isinstance(logical, int) or logical < 0:
                            raise ValueError("Store block_index must be nonnegative integer")
                        blocks = self._mooncake_store_blocks.get(req_id)
                        if blocks is None or logical >= len(blocks[main_group]):
                            raise ValueError("Store target is not reserved by the scheduler for this request")
                        if seal and (logical + 1) * self.block_size > self._mooncake_store_computed.get(req_id, 0):
                            raise ValueError("Store seal requires a fully computed immutable prefix block")
                        physical = blocks[main_group][logical]
                        if not 0 <= physical < self.kv_cache_config.num_blocks:
                            raise ValueError("Store physical block exceeds active pool")
                        fragments, offset = [], 0
                        for k, v, sizes in zip(self.k_caches_cpu, self.v_caches_cpu, self.cpu_block_lens):
                            for tensor, size in zip((k, v), sizes):
                                pool_offset = tensor.data_ptr() - self.mooncake_raw.data_ptr() + physical * size
                                region = self.mooncake_pool.resolve_range("raw", 0, pool_offset, size, address_domain="host")
                                fragments.append(StoreRange(region, offset))
                                offset += size
                        sidecar_targets = []
                        for entry, staging in zip(self._mooncake_store_sidecars, self._mooncake_store_sidecar_ranges):
                            group_id, name, component, dtype, page_bytes, device_pages = entry
                            if group_id >= len(blocks) or logical >= len(blocks[group_id]):
                                raise ValueError("Sparse Store sidecar block was not allocated to the request")
                            sidecar_block = blocks[group_id][logical]
                            if not 0 <= sidecar_block < device_pages.shape[0]:
                                raise ValueError("Sparse Store sidecar physical block is invalid")
                            fragments.append(StoreRange(staging, offset))
                            offset += page_bytes
                            sidecar_targets.append(sidecar_block)
                        # Existing bound key builder includes model/topology/layout.
                        objects.append((item["block_hash"], offset, tuple(fragments), physical, tuple(sidecar_targets)))
                        requests.add(req_id)
                    transactions.append((seal, objects, requests))
                if self._mooncake_store is None:
                    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend
                    backend = MooncakeBackend(
                        self.vllm_config.parallel_config, contribute_memory=False, dedicated_transfer_engine=True
                    )
                    builder = backend.bind_layerwise_keys(
                        vllm_config=self.vllm_config, kv_cache_config=self.kv_cache_config,
                        model_name=self.vllm_config.model_config.model + ":native-sparse-full-v1", use_hybrid=True,
                        grouped_block_size=[self.block_size] * len(self.kv_cache_config.kv_cache_groups),
                    )
                    token = repr((tuple(self.offload_layer_names), tuple(self.cpu_block_lens),
                                  "bfloat16", self.tp_size, self.block_size,
                                      tuple(entry[:5] for entry in self._mooncake_store_sidecars)))
                    regions = [self.mooncake_pool.resolve_range("raw", 0, start, size, address_domain="host")
                               for start, size in self._mooncake_regions]
                    regions.extend(self._mooncake_store_sidecar_ranges)
                    self._mooncake_store = MooncakeSparseStore(
                        backend, regions, builder, token, session_gate=StoreSessionGate(backend),
                        exclusive_session_client=True,
                    )
        except BaseException as exception:
            error = exception
        try:
            self._mooncake_check_phase(error, "Store validation/setup")
            for seal, objects, requests in transactions:
                if not objects:
                    continue
                bridge = self._mooncake_store
                for hash_value, size, ranges, physical, sidecar_blocks in objects:
                    bound = (StoreObject(self._mooncake_store_main_group, hash_value, 0, 0, size,
                                         bridge.layout_token, ranges),)
                    error = None
                    try:
                        if seal:
                            self._copy_mooncake_store_sidecars(sidecar_blocks, to_device=False)
                        with self.mooncake_metrics.measure("store_seal" if seal else "store_import", host_includes_wait=True):
                            if self.tp_rank == 0:
                                (bridge.seal_objects if seal else bridge.import_objects)(bound)
                    except BaseException as exception:
                        error = exception
                    self._mooncake_check_phase(error, "Store transaction")
                    error = None
                    try:
                        if not seal:
                            self._copy_mooncake_store_sidecars(sidecar_blocks, to_device=True)
                            self._hydrate_mooncake_store_main(physical)
                    except BaseException as exception:
                        error = exception
                    self._mooncake_check_phase(error, "Store model hydration")
                if not seal:
                    self._mooncake_store_ready.update(requests)
        except BaseException:
            # Snapshot reservations and the bridge remain held even on ordinary
            # failure. The instance cannot consume partial data or reuse slots.
            self._mooncake_store_failed = True
            raise

    def _mooncake_device_address(self, tensor: torch.Tensor, offset: int, length: int) -> int:
        capacity = tensor.numel() * tensor.element_size()
        if offset < 0 or length < 0 or offset > capacity or length > capacity - offset:
            raise ValueError("Mooncake KV range exceeds its tensor view")
        pool_offset = tensor.data_ptr() - self.mooncake_raw.data_ptr() + offset
        if not any(start <= pool_offset and pool_offset + length <= start + size for start, size in self._mooncake_regions):
            raise ValueError("Mooncake address is outside the declared KV byte layout")
        return self.mooncake_pool.resolve_range("raw", 0, pool_offset, length, address_domain="device").address

    def _mooncake_copy(self, sources, destinations, lengths, owners) -> None:
        """Eager bytes: retain every source until stream completion is proven."""
        if self._closed or self._mooncake_inflight is not None:
            raise RuntimeError("Mooncake copy is closed or has an unresolved operation")
        if not (len(sources) == len(destinations) == len(lengths)):
            raise ValueError("Mooncake copy descriptor sizes differ")
        if not sources:
            return
        target_ranges = []
        for source, destination, length in zip(sources, destinations, lengths):
            if (
                not isinstance(source, int)
                or not isinstance(destination, int)
                or not isinstance(length, int)
                or source <= 0
                or destination <= 0
                or length < 0
                or source + length > (1 << 63) - 1
                or destination + length > (1 << 63) - 1
            ):
                raise ValueError("Mooncake byte descriptors must fit nonnegative int64 address intervals")
            if length and any(destination < end and start < destination + length for start, end in target_ranges):
                raise ValueError("Mooncake byte copy targets overlap")
            if length:
                target_ranges.append((destination, destination + length))
        from mooncake.sparse_copy import sparse_copy

        device = self.topk_buffers_k[0].device
        with torch_npu.npu.device(device):
            with self.mooncake_metrics.measure("descriptor"):
                src = torch.tensor(sources, dtype=torch.int64, device=device)
                dst = torch.tensor(destinations, dtype=torch.int64, device=device)
                size = torch.tensor(lengths, dtype=torch.int64, device=device)
                count = torch.tensor([len(sources)], dtype=torch.int32, device=device)
                status = torch.zeros(1, dtype=torch.int32, device=device)
            stream = torch_npu.npu.current_stream()
            self._mooncake_inflight = (stream, owners, src, dst, size, count, status)
            with self.mooncake_metrics.measure("submit", sum(lengths), device=True):
                sparse_copy(src, dst, size, count, status)
            with self.mooncake_metrics.measure("wait", host_includes_wait=True):
                stream.synchronize()
            self._mooncake_inflight = None
            if status.item() != 0:
                raise RuntimeError("Mooncake sparse byte execution rejected descriptors")

    def warmup_mooncake_native_copy(self) -> None:
        """Compile the fixed D2H descriptor signature before graph capture."""
        if not getattr(self, "mooncake_device_execution", False):
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Mooncake byte-kernel warmup must precede graph capture")
        self.d2h_size_npu.zero_()
        self._mooncake_native_copy(
            self.d2h_src_ptrs_npu, self.d2h_dst_ptrs_npu, self.d2h_lengths_npu,
            self.d2h_size_npu, self.mooncake_status[0, 0],
        )
        self.mooncake_h2d_count.zero_()
        self._mooncake_native_copy(
            self.mooncake_h2d_src, self.mooncake_h2d_dst, self.mooncake_h2d_lengths,
            self.mooncake_h2d_count, self.mooncake_status[0, 1],
        )
        torch_npu.npu.current_stream().synchronize()
        self.mooncake_status.zero_()

    def check_mooncake_status(self) -> None:
        """Check independent sticky layer statuses before publishing outputs."""
        if not getattr(self, "mooncake_device_execution", False):
            return
        if torch.npu.is_current_stream_capturing():
            return
        current = torch_npu.npu.current_stream()
        for name in ("current_kv_save_stream", "fused_plan_stream"):
            stream = getattr(self, name, None)
            if stream is not None:
                current.wait_stream(stream)
        if bool(torch.any(self.mooncake_status != 0).item()):
            raise RuntimeError("Mooncake native copy rejected a replay descriptor; results are invalid")
        plan_status = getattr(self, "fused_plan_status_npu", None)
        if plan_status is not None and int(plan_status.item()) != 0:
            raise RuntimeError("Mooncake overlap planner failed; results are invalid")
        if (
            self.mooncake_metrics.enabled
            and self.sparse_kv_offload_config.mooncake_execution_mode == "eager"
            and self._mooncake_metric_recorded_step != self._mooncake_metric_step
        ):
            for misses, cache, logical, query_ends in self._mooncake_cache_metric_refs.values():
                end_values = query_ends.cpu().tolist()
                cache_values, logical_values = cache.cpu().tolist(), logical.cpu().tolist()
                active = int(end_values[-1]) if end_values else 0
                miss_count = int(misses[:active].sum().item())
                total = 0
                begin = 0
                for row, end in enumerate(end_values):
                    total += (end - begin) * (logical_values[row] if cache_values[row] == 0 else self.topk)
                    begin = end
                self.mooncake_metrics.record_cache(max(total - miss_count, 0), miss_count)
            self._mooncake_metric_recorded_step = self._mooncake_metric_step
        self.mooncake_metrics.collect(wait=False)

    def prepare_mooncake_step(self) -> None:
        if not getattr(self, "mooncake_device_execution", False) or torch.npu.is_current_stream_capturing():
            return
        if self._closed:
            raise RuntimeError("Mooncake graphs cannot replay after their pool is closed")
        self.check_mooncake_status()
        self.mooncake_status.zero_()
        self._mooncake_metric_step += 1
        self._mooncake_cache_metric_refs.clear()

    def run_mooncake_forward(self, forward, *args, **kwargs):
        """Measure actual replay/eager execution from the outer Python caller."""
        with self.mooncake_metrics.measure("model_forward", device=True):
            return forward(*args, **kwargs)

    def mooncake_metrics_summary(self, *, wait: bool = False) -> dict:
        self.mooncake_metrics.collect(wait=wait)
        result = self.mooncake_metrics.summary()
        result["coverage"] = {
            "cache_counts": "eager LIM only; graph and fused_overlap cache counts are not collected",
            "native_copy": "current-stream event spans; graph capture skips per-operation timing",
            "graph_bytes": "not inferred from reserved capacity; graph bandwidth is not reported",
            "model_forward": "actual outer eager/replay call; capture timing is skipped",
        }
        return result

    def retain_mooncake_graph_resources(self) -> tuple:
        """Attach this lease to the graph wrapper, not a temporary tensor view."""
        if not getattr(self, "mooncake_device_execution", False):
            return ()
        self.warmup_mooncake_native_copy()
        return (self, self.mooncake_pool, self.mooncake_raw, getattr(self, "mooncake_membership_segment", None))

    def close(self) -> None:
        """Release resources only after successful device completion."""
        if self._closed:
            return
        if self.is_mooncake:
            # Refuse future forwards before draining. Final CPU tensor views
            # retain their pool independently of manager/worker teardown.
            self._closed = True
            error = None
            try:
                if self._mooncake_inflight is not None:
                    self._mooncake_inflight[0].synchronize()
                torch_npu.npu.synchronize()
                bridge = getattr(self, "_mooncake_store", None)
                if bridge is not None:
                    bridge.close()
            except BaseException as exception:
                error = exception
            try:
                self._mooncake_check_phase(error, "resource drain")
            except BaseException:
                self._closed = False
                raise
            self._mooncake_inflight = None
            if self.tp_size > 1:
                # Every rank has stopped all sanctioned users before the
                # owner's resources can retire. Closed pools admit no new work.
                self.tp_group.barrier()
                if self.tp_rank != 0:
                    self._mooncake_store_sidecar_pool = None
                    self._mooncake_store_sidecar_raw = None
                    self._mooncake_store_sidecar_views = []
                    self._mooncake_store_sidecar_ranges = []
                    self.mooncake_raw = None
                    self.mooncake_pool = None
                self.tp_group.barrier()
            self._mooncake_store_sidecar_pool = None
            self._mooncake_store_sidecar_raw = None
            self._mooncake_store_sidecar_views = []
            self._mooncake_store_sidecar_ranges = []
            self.mooncake_raw = None
            self.mooncake_pool = None
        self._closed = True

    def reload_mooncake_lim(self, layer_name, source_ids, destination_slots, counts, source_table, target_table):
        """Execute the existing LIM request miss plan without changing selection."""
        layer_id = self._get_offload_layer_id(layer_name)
        sources_by_request = source_ids.cpu().tolist()
        targets_by_request = destination_slots.cpu().tolist()
        request_counts = counts.cpu().tolist()
        host_table = source_table.cpu().tolist()
        hot_table = target_table.cpu().tolist()
        plan = {}
        for request, count in enumerate(request_counts):
            if count < 0 or count > len(sources_by_request[request]):
                raise ValueError("LIM miss count exceeds descriptor capacity")
            for source, target in zip(sources_by_request[request][:count], targets_by_request[request][:count]):
                if source < 0 or target < 0:
                    raise ValueError("Active LIM misses must have nonnegative source and target")
                src_block, src_token = divmod(source, self.block_size)
                dst_block, dst_token = divmod(target, self.block_size)
                if src_block >= len(host_table[request]) or dst_block >= len(hot_table[request]):
                    raise ValueError("LIM miss exceeds its block table")
                physical_src = host_table[request][src_block] * self.block_size + src_token
                physical_dst = hot_table[request][dst_block] * self.block_size + dst_token
                if physical_src < 0 or physical_dst < 0:
                    raise ValueError("LIM miss references an invalid physical block")
                if physical_dst in plan and plan[physical_dst] != physical_src:
                    raise ValueError("LIM plan maps different sources to one hot slot")
                plan[physical_dst] = physical_src
        srcs, dsts, sizes = [], [], []
        for host, hot, token_bytes in (
            (self.k_caches_cpu[layer_id], self.topk_buffers_k[layer_id], self.token_size_bytes_k),
            (self.v_caches_cpu[layer_id], self.topk_buffers_v[layer_id], self.token_size_bytes_v),
        ):
            for target, source in plan.items():
                if target * token_bytes + token_bytes > hot.numel() * hot.element_size():
                    raise ValueError("LIM destination exceeds its hot KV buffer")
                srcs.append(self._mooncake_device_address(host, source * token_bytes, token_bytes))
                dsts.append(hot.data_ptr() + target * token_bytes)
                sizes.append(token_bytes)
        self._mooncake_copy(srcs, dsts, sizes, (self.mooncake_pool,))

    def _warmup_external_lru_planner_threads(self) -> int:
        if not (self.use_fused_overlap and self.tp_rank == 0):
            return 0
        warmed_threads = _sparse_kv_ops().sparse_kv_warmup_lru_resident_threads(self.lru_workspace_threads)
        logger.info(
            "Warmed external LRU planner OpenMP team with %s threads",
            warmed_threads,
        )
        return warmed_threads

    def _infer_group_block_sizes(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> int:
        block_sizes = {group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups}
        if len(block_sizes) != 1:
            raise ValueError("Sparse KV offload requires one shared block size across target and draft groups.")
        return block_sizes.pop()

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _register_offload_layers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        host_layer_names = set()
        for group in self.kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[layer_name]
                if getattr(spec, "store_on_host", False):
                    host_layer_names.add(layer_name)
        self.offload_layer_names = [layer_name for layer_name in kv_caches if layer_name in host_layer_names]
        if not self.offload_layer_names:
            raise ValueError("Sparse KV offload did not find SFA KV cache layers.")

        self.num_layers = len(self.offload_layer_names)
        self.layer_name_to_offload_id = {
            layer_name: layer_id for layer_id, layer_name in enumerate(self.offload_layer_names)
        }

        logger.info_once(
            "Sparse KV offload registered %s layers (%s target layers).",
            self.num_layers,
            self.num_target_layers,
        )
        self.mtp_layer_id = self.num_layers - 1 if self.num_layers != self.num_target_layers else -1
        if self.tp_rank == 0:
            preview_layer_names = self.offload_layer_names[:4]
            if len(self.offload_layer_names) > 4:
                preview_layer_names += ["..."] + self.offload_layer_names[-4:]
            logger.info("Sparse KV offload layer names: %s", preview_layer_names)

    def _get_offload_layer_id(self, layer_name: str) -> int:
        layer_id = self.layer_name_to_offload_id.get(layer_name)
        if layer_id is None:
            registered_layers = ", ".join(self.offload_layer_names[:8])
            if len(self.offload_layer_names) > 8:
                registered_layers += ", ..."
            raise KeyError(
                "Sparse KV offload layer is not registered, "
                f"layer_name={layer_name}, registered_layers=[{registered_layers}]"
            )
        return layer_id

    def _restore_bfloat16_tensor(self, ptr: int, shape: list[int]) -> torch.Tensor:
        view = _sparse_kv_ops().sparse_kv_restore_bfloat16_tensor(ptr, shape)
        if int(view.data_ptr()) != int(ptr):
            raise RuntimeError(
                "restore_bfloat16_tensor returned a tensor with unexpected data_ptr: "
                f"expected={ptr}, got={view.data_ptr()}"
            )
        if list(view.shape) != list(shape):
            raise RuntimeError(
                f"restore_bfloat16_tensor returned unexpected shape: expected={shape}, got={list(view.shape)}"
            )
        if view.dtype != torch.bfloat16:
            raise RuntimeError(f"restore_bfloat16_tensor returned unexpected dtype: {view.dtype}")
        if not view.is_contiguous():
            raise RuntimeError("restore_bfloat16_tensor requires a contiguous view")
        return view

    def _restore_int16_tensor(self, ptr: int, shape: list[int]) -> torch.Tensor:
        view = _sparse_kv_ops().sparse_kv_restore_int16_tensor(ptr, shape)
        if int(view.data_ptr()) != int(ptr):
            raise RuntimeError(
                "restore_int16_tensor returned a tensor with an unexpected data_ptr: "
                f"expected={ptr}, got={view.data_ptr()}"
            )
        if list(view.shape) != list(shape):
            raise RuntimeError(
                f"restore_int16_tensor returned an unexpected shape: expected={shape}, got={list(view.shape)}"
            )
        if view.dtype != torch.int16:
            raise RuntimeError(f"restore_int16_tensor returned an unexpected dtype: {view.dtype}")
        if not view.is_contiguous():
            raise RuntimeError("restore_int16_tensor requires a contiguous view")
        return view

    def allocate_fused_overlap_membership_map(
        self,
        row_capacity: int,
    ) -> torch.Tensor:
        if not self.use_fused_overlap:
            raise RuntimeError("mapped membership allocation requires kv_offload_decode_config.use_fused_overlap=true")
        if row_capacity <= 0:
            raise ValueError(f"mapped membership row capacity must be positive, got {row_capacity}")
        if self.topk >= FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT:
            raise ValueError(
                "SFA sparse topk exceeds external plan storage: "
                f"topk={self.topk} "
                f"control_offset={FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT}"
            )

        if getattr(self, "fused_overlap_membership_map", None) is not None:
            if row_capacity > self.fused_overlap_membership_map_rows:
                raise RuntimeError(
                    "fused_overlap membership storage was allocated for fewer "
                    f"rows: allocated={self.fused_overlap_membership_map_rows}, "
                    f"required={row_capacity}"
                )
            return self.fused_overlap_membership_map

        shape = [row_capacity, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT]
        if self.is_mooncake:
            if row_capacity > self.max_num_topk_rows:
                raise ValueError("Mooncake overlap membership exceeds its planned DRAM capacity")
            from mooncake.shared_segment import create_shared_segment

            segment = create_shared_segment(
                "vllm_sparse_membership",
                {"membership": dict(count=1, shape=shape, dtype=torch.int16)},
                world_size=self.tp_size, rank_id=self.tp_rank,
                device_id=torch_npu.npu.current_device(), comm_group=self.tp_group if self.tp_size > 1 else None,
                mmap=False,
            )
            membership_map = None
            error = None
            try:
                if segment.base_addr() != segment.device_addr():
                    raise ValueError("Mooncake overlap membership requires equal Host/device addresses")
                membership_map = segment.host_tensors("membership")[0]
                if self.tp_rank == 0:
                    membership_map.fill_(-1)
                    control = membership_map[:, FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT:]
                    control[:, 1] = FSA_EXTERNAL_PLAN_READY_MARKER
                    control[:, 2] = self.topk
                    control[:, 3] = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - self.topk
                    control[:, 7] = FSA_PAIRED_SELECTION_COPY_MARKER
            except BaseException as exception:
                error = exception
            self._mooncake_check_phase(error, "overlap membership mapping")
            self.tp_group.barrier()
            membership_map._mooncake_segment = segment
            self.fused_overlap_membership_map = membership_map
            self.fused_overlap_membership_map_rows = row_capacity
            self.mooncake_membership_segment = segment
            return membership_map
        owner_ptr = torch.zeros(1, dtype=torch.int64, device="npu")
        membership_map = None
        if self.tp_rank == 0:
            membership_map = _legacy_offload().empty(
                [row_capacity * FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT],
                dtype=torch.int16,
                pin_memory=True,
            ).view(shape)
            membership_map.fill_(-1)
            control = membership_map[
                :,
                FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT
                + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
            ]
            control[:, 1] = FSA_EXTERNAL_PLAN_READY_MARKER
            control[:, 2] = self.topk
            control[:, 3] = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - self.topk
            control[:, 7] = FSA_PAIRED_SELECTION_COPY_MARKER
            owner_ptr[0] = membership_map.data_ptr()
        self.tp_group.broadcast(owner_ptr, src=0)
        shared_ptr = int(owner_ptr.item())
        if self.tp_rank != 0:
            membership_map = self._restore_int16_tensor(shared_ptr, shape)
        if membership_map is None:
            raise RuntimeError("mapped membership storage was not initialized")
        self.tp_group.barrier()
        self.fused_overlap_membership_map = membership_map
        self.fused_overlap_membership_map_rows = row_capacity
        return membership_map

    def get_fused_overlap_cpu_kv_inputs(
        self,
        layer_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CPU full KV tensors used by fused_overlap decode.

        On TP0 these are the owned CPU pools; on other ranks they are non-owning
        GVA views restored after broadcast.
        """
        if not self.use_fused_overlap:
            raise RuntimeError(
                "get_fused_overlap_cpu_kv_inputs requires kv_offload_decode_config.use_fused_overlap=true"
            )
        layer_id = self._get_offload_layer_id(layer_name)
        if layer_id >= len(self.k_caches_cpu) or layer_id >= len(self.v_caches_cpu):
            raise RuntimeError(
                "fused_overlap CPU KV views are not registered: "
                f"layer_id={layer_id}, k_len={len(self.k_caches_cpu)}, "
                f"v_len={len(self.v_caches_cpu)}"
            )
        return self.k_caches_cpu[layer_id], self.v_caches_cpu[layer_id]

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ):
        if self.is_mooncake and self.tp_size > 1:
            error = ""
            signature = None
            try:
                signature = tuple(
                    (name, tuple((tuple(t.shape), str(t.dtype)) if t is not None else None
                                 for t in self._as_cache_tuple(caches)))
                    for name, caches in kv_caches.items()
                )
            except Exception as exception:
                error = str(exception)
            signatures = [None] * self.tp_size
            dist.all_gather_object(signatures, (error, signature), group=self.tp_group.cpu_group)
            if any(item[0] or item[1] != signatures[0][1] for item in signatures):
                raise ValueError("Mooncake TP ranks disagree on KV layer shapes/order/dtypes")
        self._register_offload_layers(kv_caches)
        if self.is_mooncake:
            self._mooncake_store_main_hbm = {
                name: self._as_cache_tuple(kv_caches[name])[:2] for name in self.offload_layer_names
            }
            self._mooncake_store_sidecars = []
            self._mooncake_store_nonreplicated_sidecars = []
            self._mooncake_store_main_group = None
            for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
                for name in group.layer_names:
                    spec = group.kv_cache_spec
                    if isinstance(spec, UniformTypeKVCacheSpecs):
                        spec = spec.kv_cache_specs[name]
                    if name in self.offload_layer_names:
                        if self._mooncake_store_main_group not in (None, group_id):
                            raise ValueError("Sparse Store main layers must share a block group")
                        self._mooncake_store_main_group = group_id
                    elif name in kv_caches:
                        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
                        context = self.vllm_config.compilation_config.static_forward_context
                        if not isinstance(context.get(name), DeepseekV32IndexerCache):
                            self._mooncake_store_nonreplicated_sidecars.append(name)
                        if spec.block_size != self.block_size:
                            raise ValueError("Sparse Store sidecar token block sizes must match MLA")
                        for component_id, tensor in enumerate(self._as_cache_tuple(kv_caches[name])):
                            if tensor is None or not tensor.is_contiguous():
                                raise ValueError("Sparse Store requires contiguous complete device sidecars")
                            raw = tensor.view(torch.uint8).reshape(-1)
                            if raw.numel() % self.kv_cache_config.num_blocks:
                                raise ValueError("Sparse Store sidecar bytes do not form complete framework pages")
                            page_bytes = raw.numel() // self.kv_cache_config.num_blocks
                            self._mooncake_store_sidecars.append(
                                (group_id, name, component_id, str(tensor.dtype), page_bytes,
                                 raw.view(self.kv_cache_config.num_blocks, page_bytes))
                            )
            self._mooncake_store_sidecar_pool = None


        # register topk_buffer and cpu kv_cache
        self.topk_buffers_k: list[torch.Tensor] = []
        self.topk_buffers_v: list[torch.Tensor] = []
        self.k_caches_cpu: list[torch.Tensor] = []
        self.v_caches_cpu: list[torch.Tensor] = []
        for layer_name in self.offload_layer_names:
            cache_or_caches = self._as_cache_tuple(kv_caches[layer_name])
            tuple_len = len(cache_or_caches)
            if tuple_len not in [OFFLOAD_KV_CACHE_TUPLE_LEN]:
                raise ValueError(
                    f"Sparse KV offload layer {layer_name}: expected tuple length "
                    f"{OFFLOAD_KV_CACHE_TUPLE_LEN}, got {tuple_len}"
                )
            self.topk_buffers_k.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_K_INDEX])
            self.topk_buffers_v.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_V_INDEX])
            if self.tp_rank == 0 or self.is_mooncake:
                self.k_caches_cpu.append(cache_or_caches[OFFLOAD_K_CACHE_CPU_INDEX])
                self.v_caches_cpu.append(cache_or_caches[OFFLOAD_V_CACHE_CPU_INDEX])

        kv_head_num = self.topk_buffers_k[0].size(-2)
        head_dim_k = self.topk_buffers_k[0].size(-1)
        head_dim_v = self.topk_buffers_v[0].size(-1)
        dtype = self.topk_buffers_k[0].dtype
        assert kv_head_num == 1, "Sparse KV offload only support sfa(mla)"
        if dtype != torch.bfloat16:
            raise ValueError(
                "Sparse KV offload requires a BF16 main SFA cache; sparse LI "
                "C8 is supported only for the device-resident indexer cache."
            )
        self.token_size_bytes_k = kv_head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = kv_head_num * head_dim_v * dtype.itemsize
        if self.topk_buffer_size % self.block_size != 0:
            raise ValueError(
                "Sparse KV offload topk_buffer_size must be divisible by "
                f"block_size, got {self.topk_buffer_size} and {self.block_size}"
            )

        # D2H descriptors commit newly produced KV into the shared host pool.
        # Both prefill (colocate debug only, gated by keep_device_kv_cache)
        # and decode can produce up to max_num_tokens rows.
        d2h_descriptor_rows = self.max_num_tokens * 2
        device = self.topk_buffers_k[0].device
        self.d2h_src_ptrs_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int64, device=device)
        self.d2h_dst_ptrs_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int64, device=device)
        self.d2h_lengths_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int32, device=device)
        if self.use_fused_overlap:
            self.current_kv_save_stream = torch_npu.npu.Stream()
            self.fused_plan_stream = torch_npu.npu.Stream()
            self.fused_plan_metadata_npu = torch.zeros(
                1 + self.max_num_topk_rows,
                dtype=torch.int32,
                device=device,
            )
            self.fused_plan_status_npu = self.fused_plan_metadata_npu[:1]
            self.fused_plan_current_linear_slots_npu = self.fused_plan_metadata_npu[1:]
            self.current_kv_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.d2h_size_npu = torch.empty(1, dtype=torch.int32, device=device)
        self.d2h_token_indices_npu = torch.arange(self.max_num_tokens, dtype=torch.int64, device=device)

        if self.is_mooncake:
            self.gvas_k_bases = [view.data_ptr() for view in self.k_caches_cpu]
            self.gvas_v_bases = [view.data_ptr() for view in self.v_caches_cpu]
            self.cpu_block_lens = [
                (k.numel() * k.element_size() // self.kv_cache_config.num_blocks,
                 v.numel() * v.element_size() // self.kv_cache_config.num_blocks)
                for k, v in zip(self.k_caches_cpu, self.v_caches_cpu)
            ]
            if self.use_fused_copy_sfa:
                self._bind_copy_sfa_bases()
            if self.use_fused_overlap:
                self._init_fused_overlap_planner()
            if self.mooncake_device_execution:
                self.mooncake_status = torch.zeros((self.num_layers, 2, 1), dtype=torch.int32, device=device)
                h2d_capacity = 4 * (self.max_num_reqs + 2)
                self.mooncake_h2d_src = torch.empty(h2d_capacity, dtype=torch.int64, device=device)
                self.mooncake_h2d_dst = torch.empty_like(self.mooncake_h2d_src)
                self.mooncake_h2d_lengths = torch.empty_like(self.mooncake_h2d_src)
                self.mooncake_h2d_count = torch.zeros(1, dtype=torch.int32, device=device)
                self.warmup_mooncake_native_copy()
            return

        # Broadcast the shared host KV pool bases and shapes.
        self.gvas_k_bases: list[int] = []
        self.gvas_v_bases: list[int] = []
        # RD2H consumers need byte strides for each TP-shared host KV block.
        self.cpu_block_lens: list[tuple[int, int]] = []
        gvas_k_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device="npu")
        gvas_v_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device="npu")
        cpu_block_lens_tensor = torch.zeros([self.num_layers, 2], dtype=torch.int64, device="npu")
        shape_k_tensor = torch.zeros([4], dtype=torch.int64, device="npu")
        shape_v_tensor = torch.zeros([4], dtype=torch.int64, device="npu")
        if self.tp_rank == 0:
            for layer_id in range(self.num_layers):
                k_cpu = self.k_caches_cpu[layer_id]
                v_cpu = self.v_caches_cpu[layer_id]
                gvas_k_tensor[layer_id] = k_cpu.data_ptr()
                gvas_v_tensor[layer_id] = v_cpu.data_ptr()
                cpu_block_lens_tensor[layer_id, 0] = (
                    k_cpu.numel() * k_cpu.element_size() // self.kv_cache_config.num_blocks
                )
                cpu_block_lens_tensor[layer_id, 1] = (
                    v_cpu.numel() * v_cpu.element_size() // self.kv_cache_config.num_blocks
                )
            shape_k_tensor.copy_(torch.tensor(self.k_caches_cpu[0].shape, dtype=torch.int64, device="npu"))
            shape_v_tensor.copy_(torch.tensor(self.v_caches_cpu[0].shape, dtype=torch.int64, device="npu"))
        self.tp_group.broadcast(gvas_k_tensor, src=0)
        self.tp_group.broadcast(gvas_v_tensor, src=0)
        self.tp_group.broadcast(cpu_block_lens_tensor, src=0)
        self.tp_group.broadcast(shape_k_tensor, src=0)
        self.tp_group.broadcast(shape_v_tensor, src=0)
        for layer_id in range(self.num_layers):
            self.gvas_k_bases.append(gvas_k_tensor[layer_id].item())
            self.gvas_v_bases.append(gvas_v_tensor[layer_id].item())

            self.cpu_block_lens.append(
                (
                    cpu_block_lens_tensor[layer_id, 0].item(),
                    cpu_block_lens_tensor[layer_id, 1].item(),
                )
            )

        if self.tp_rank != 0:
            cpu_k_shape = [int(x) for x in shape_k_tensor.tolist()]
            cpu_v_shape = [int(x) for x in shape_v_tensor.tolist()]
            self.k_caches_cpu = [self._restore_bfloat16_tensor(ptr, cpu_k_shape) for ptr in self.gvas_k_bases]
            self.v_caches_cpu = [self._restore_bfloat16_tensor(ptr, cpu_v_shape) for ptr in self.gvas_v_bases]
            logger.info(
                "[fused_overlap_offload][init] restored shared CPU KV views on "
                "tp_rank=%s layer_count=%s k_shape=%s v_shape=%s",
                self.tp_rank,
                self.num_layers,
                cpu_k_shape,
                cpu_v_shape,
            )
        if self.use_fused_copy_sfa:
            self._bind_copy_sfa_bases()

        if self.use_fused_overlap:
            self._init_fused_overlap_planner()

    def _init_fused_overlap_planner(self) -> None:
        """Allocate the CPU LRU workspace used by the overlap operator."""
        self.lru_workspace_threads = 8
        self._warmup_external_lru_planner_threads()
        self.lru_topk_indices_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_slot_to_token_cpu_list = [
            torch.full(
                [self.max_num_topk_rows, self.topk_buffer_size],
                -1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_slots_cpu_list = [
            torch.arange(
                self.topk_buffer_size,
                dtype=torch.int32,
                device="cpu",
            )
            .view(1, -1)
            .repeat(self.max_num_topk_rows, 1)
            .pin_memory()
            for _ in range(self.num_layers)
        ]
        self.lru_current_slots_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_miss_count_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_miss_tokens_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_miss_slots_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_req_ids_cpu = torch.empty([self.max_num_topk_rows], dtype=torch.int64, device="cpu", pin_memory=True)
        self.lru_stable_prefix_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_last_req_ids_cpu_list = [
            torch.full(
                [self.max_num_topk_rows],
                -1,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_visible_seq_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        self.lru_token_mark_workspace = torch.zeros(
            [self.lru_workspace_threads, self.max_model_len],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_token_pos_workspace = torch.full(
            [self.lru_workspace_threads, self.max_model_len],
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_slot_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk_buffer_size * 3],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_miss_position_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_epochs = torch.zeros(
            [self.lru_workspace_threads],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_physical_row_workspace = torch.empty(
            [self.max_num_topk_rows * 3],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        self.lru_req_ids_ptr = self.lru_req_ids_cpu.data_ptr()
        self.lru_stable_prefix_lens_ptr = self.lru_stable_prefix_lens_cpu.data_ptr()
        self.lru_last_req_ids_ptrs = [
            lru_last_req_ids_cpu.data_ptr() for lru_last_req_ids_cpu in self.lru_last_req_ids_cpu_list
        ]
        self.lru_visible_seq_lens_ptr = self.lru_visible_seq_lens_cpu.data_ptr()
        self.lru_topk_indices_ptr = self.lru_topk_indices_cpu.data_ptr()
        self.lru_slot_to_token_ptrs = [
            lru_slot_to_token_cpu.data_ptr() for lru_slot_to_token_cpu in self.lru_slot_to_token_cpu_list
        ]
        self.lru_slots_ptrs = [lru_slots_cpu.data_ptr() for lru_slots_cpu in self.lru_slots_cpu_list]
        self.lru_current_slots_ptr = self.lru_current_slots_cpu.data_ptr()
        self.lru_miss_count_ptrs = [
            lru_miss_count_cpu.data_ptr() for lru_miss_count_cpu in self.lru_miss_count_cpu_list
        ]
        self.lru_miss_tokens_ptrs = [
            lru_miss_tokens_cpu.data_ptr() for lru_miss_tokens_cpu in self.lru_miss_tokens_cpu_list
        ]
        self.lru_miss_slots_ptrs = [
            lru_miss_slots_cpu.data_ptr() for lru_miss_slots_cpu in self.lru_miss_slots_cpu_list
        ]
        self.lru_token_mark_workspace_ptr = self.lru_token_mark_workspace.data_ptr()
        self.lru_token_pos_workspace_ptr = self.lru_token_pos_workspace.data_ptr()
        self.lru_slot_workspace_ptr = self.lru_slot_workspace.data_ptr()
        self.lru_miss_position_workspace_ptr = self.lru_miss_position_workspace.data_ptr()
        self.lru_epochs_ptr = self.lru_epochs.data_ptr()
        self.lru_physical_row_workspace_ptr = self.lru_physical_row_workspace.data_ptr()

    def _bind_copy_sfa_bases(self) -> None:
        """Pin host/device row bases used by eager prefix-rollback tail restore."""
        if not self.k_caches_cpu or not self.topk_buffers_k:
            raise RuntimeError("fused_copy_sfa tail restore requires host and device KV bases")
        device = self.topk_buffers_k[0].device
        descriptor_rows = 4 * (self.max_num_reqs + 2)
        self.copy_sfa_copy_src = torch.empty(descriptor_rows, dtype=torch.int64, device=device)
        self.copy_sfa_copy_dst = torch.empty_like(self.copy_sfa_copy_src)
        self.copy_sfa_host_bases = []
        self.copy_sfa_device_bases = []
        for layer_id in range(len(self.topk_buffers_k)):
            self.copy_sfa_host_bases.append(
                torch.tensor(
                    [self.k_caches_cpu[layer_id].data_ptr(), self.v_caches_cpu[layer_id].data_ptr()],
                    dtype=torch.int64,
                    device=device,
                ).view(2, 1)
            )
            self.copy_sfa_device_bases.append(
                torch.tensor(
                    [self.topk_buffers_k[layer_id].data_ptr(), self.topk_buffers_v[layer_id].data_ptr()],
                    dtype=torch.int64,
                    device=device,
                ).view(2, 1)
            )

    def dense_fill_copy_sfa_rows(
        self,
        dense_fills: dict[int, tuple[int, int]],
        *,
        block_size: int,
        block_table: np.ndarray | torch.Tensor,
    ) -> None:
        """Restore whole short rows from the host pool after prefix rollback.

        A rollback across the hot boundary leaves a sparse-layout row under a
        dense (-3) reader. Restore it using the input batch's block table
        (a CPU mirror in V1 or a device tensor in V2) and block size, outside
        graph capture at metadata-build time. Host and device bases are bound
        during cache registration.
        """
        topk_k = self.topk_buffers_k[0]
        stride_tokens = topk_k.shape[1]
        token_bytes = torch.tensor(
            [
                [topk_k.element_size() * topk_k.shape[-1]],
                [self.topk_buffers_v[0].element_size() * self.topk_buffers_v[0].shape[-1]],
            ],
            dtype=torch.int64,
            device=topk_k.device,
        )
        block_bytes = token_bytes * block_size
        for slot, (row, computed) in dense_fills.items():
            nblocks = (computed + block_size - 1) // block_size
            if isinstance(block_table, torch.Tensor):
                if block_table.device != topk_k.device:
                    raise ValueError("Tensor block table must be on the same device as the top-k buffers")
                # block_bytes is a dimensioned int64 tensor, not a Python scalar.
                # The product is int64 before adding the host base, including offsets above 2 GiB.
                src_offsets = block_table[row, :nblocks].view(1, -1) * block_bytes
            else:
                src_tokens = torch.tensor(
                    [int(block_table[row, b]) * block_size for b in range(nblocks)], dtype=torch.int64
                ).to(topk_k.device)
                src_offsets = src_tokens.view(1, -1) * token_bytes
            lengths = torch.clamp(
                torch.arange(nblocks, dtype=torch.int64, device=topk_k.device) * (-block_size) + computed,
                min=0,
                max=block_size,
            )
            dst_block_tokens = torch.arange(nblocks, dtype=torch.int64, device=topk_k.device) * block_size
            dst_offsets = (slot * stride_tokens + dst_block_tokens.view(1, -1).expand(2, -1)) * token_bytes
            lengths_bytes = lengths.view(1, -1).expand(2, -1) * token_bytes
            count = torch.full((1,), 2 * nblocks, dtype=torch.int32, device=topk_k.device)
            for layer_id, (host_bases, device_bases) in enumerate(zip(self.copy_sfa_host_bases, self.copy_sfa_device_bases)):
                sources = (src_offsets + host_bases).reshape(-1)
                destinations = (dst_offsets + device_bases).reshape(-1)
                self.copy_sfa_kv(sources, destinations, lengths_bytes.reshape(-1), count, layer_id=layer_id)

    def restore_copy_sfa_tails(self, metadata) -> None:
        """Copy the current tail descriptors for every layer. Used on prefix rollback."""
        src_off = getattr(metadata, "copy_sfa_copy_src_offsets", None)
        dst_off = getattr(metadata, "copy_sfa_copy_dst_offsets", None)
        lengths = getattr(metadata, "copy_sfa_copy_lengths", None)
        count = getattr(metadata, "copy_sfa_copy_count", None)
        if src_off is None or dst_off is None or lengths is None or count is None:
            return
        if not getattr(self, "copy_sfa_host_bases", None):
            raise RuntimeError("fused_copy_sfa KV base addresses must be bound before tail restore")
        n = src_off.numel()
        for layer_id, (host_bases, device_bases) in enumerate(zip(self.copy_sfa_host_bases, self.copy_sfa_device_bases)):
            torch.add(src_off.view(2, -1), host_bases, out=self.copy_sfa_copy_src[:n].view(2, -1))
            torch.add(dst_off.view(2, -1), device_bases, out=self.copy_sfa_copy_dst[:n].view(2, -1))
            self.copy_sfa_kv(self.copy_sfa_copy_src[:n], self.copy_sfa_copy_dst[:n], lengths, count, layer_id=layer_id)

    def copy_sfa_kv(self, sources, destinations, lengths, count, *, layer_id: int | None = None) -> None:
        """Enqueue bounded descriptor copies on the current compute stream."""
        if getattr(self, "mooncake_device_execution", False):
            if layer_id is None:
                raise ValueError("Mooncake native H2D requires its layer status slot")
            capacity = self.mooncake_h2d_src.numel()
            number = sources.numel()
            status = self.mooncake_status[layer_id, 1]
            if number > capacity:
                if torch.npu.is_current_stream_capturing():
                    raise ValueError("Mooncake captured H2D capacity exceeds its prewarmed descriptor arena")
                self._mooncake_native_copy(sources, destinations, lengths, count, status, reset_status=False)
                self.check_mooncake_status()
                return
            self.mooncake_h2d_src[:number].copy_(sources.reshape(-1))
            self.mooncake_h2d_dst[:number].copy_(destinations.reshape(-1))
            self.mooncake_h2d_lengths[:number].copy_(lengths.reshape(-1))
            torch.bitwise_or(status, torch.any((count < 0) | (count > number)).to(torch.int32).reshape(1) * 8, out=status)
            self.mooncake_h2d_count.copy_(count.clamp(0, number))
            with self.mooncake_metrics.measure("native_h2d", device=True):
                self._mooncake_native_copy(
                    self.mooncake_h2d_src, self.mooncake_h2d_dst, self.mooncake_h2d_lengths,
                    self.mooncake_h2d_count, status, reset_status=False,
                )
            return
        if getattr(self, "is_mooncake", False):
            number = int(count.reshape(-1)[0].item())
            if number < 0 or number > min(sources.numel(), destinations.numel(), lengths.numel()):
                raise ValueError("Mooncake H2D descriptor count exceeds capacity")
            srcs = sources.reshape(-1)[:number].cpu().tolist()
            dsts = destinations.reshape(-1)[:number].cpu().tolist()
            sizes = lengths.reshape(-1)[:number].cpu().tolist()
            resolved_sources, resolved_destinations, resolved_sizes = [], [], []
            target_ranges = []
            pool_base = self.mooncake_raw.data_ptr()
            device_buffers = self.topk_buffers_k + self.topk_buffers_v
            for source, destination, length in zip(srcs, dsts, sizes):
                if length < 0:
                    raise ValueError("Mooncake H2D length must be nonnegative")
                offset = source - pool_base
                if not any(start <= offset and offset + length <= start + size for start, size in self._mooncake_regions):
                    raise ValueError("Mooncake H2D source is outside a declared KV region")
                if not any(
                    buffer.data_ptr() <= destination
                    and destination + length <= buffer.data_ptr() + buffer.numel() * buffer.element_size()
                    for buffer in device_buffers
                ):
                    raise ValueError("Mooncake H2D destination is outside a hot buffer")
                if length == 0:
                    continue
                if any(destination < end and start < destination + length for start, end in target_ranges):
                    raise ValueError("Mooncake H2D destinations overlap")
                target_ranges.append((destination, destination + length))
                resolved_sources.append(
                    self.mooncake_pool.resolve_range("raw", 0, offset, length, address_domain="device").address
                )
                resolved_destinations.append(destination)
                resolved_sizes.append(length)
            self._mooncake_copy(resolved_sources, resolved_destinations, resolved_sizes, tuple(device_buffers))
            return
        result = _legacy_offload().sparse_copy(sources, destinations, lengths, count, sources.device)
        if result not in (None, 0):
            raise RuntimeError(f"memfabric fused_copy_sfa tail H2D failed with result={result}")

    def offload_new_kv(
        self,
        layer_name: str,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,
        v_cache_npu: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        has_prefill: bool = False,
        capturing: bool = False,
    ) -> None:
        if getattr(self, "is_mooncake", False):
            if getattr(self, "mooncake_device_execution", False):
                if self._closed:
                    raise RuntimeError("Mooncake sparse pool is closed")
                layer_id = self._get_offload_layer_id(layer_name)
                if self.use_fused_overlap and k is not None and v is not None:
                    self.current_kv_by_layer[layer_id] = (k, v)
                error = None
                try:
                    self._offload_new_kv_on_current_stream(
                        slot_mapping, k_cache_cpu, v_cache_cpu, k_cache_npu, v_cache_npu,
                        k, v, has_prefill, capturing=capturing, layer_id=layer_id,
                    )
                    if self.tp_size > 1:
                        torch_npu.npu.current_stream().synchronize()
                        if bool(torch.any(self.mooncake_status[layer_id, 0] != 0).item()):
                            raise RuntimeError("Mooncake TP native writeback rejected a descriptor")
                except BaseException as exception:
                    error = exception
                if self.tp_size > 1:
                    self._mooncake_check_phase(error, "native writeback")
                elif error is not None:
                    raise error
                return
            if capturing:
                raise RuntimeError("Mooncake eager writeback cannot execute inside graph capture/replay")
            error = None
            if self.tp_rank == 0:
                try:
                    self._mooncake_writeback(
                        slot_mapping, k_cache_cpu, v_cache_cpu, k_cache_npu, v_cache_npu, k, v, has_prefill
                    )
                except BaseException as exception:
                    error = exception
            if self.tp_size > 1:
                # The writer synchronizes its NPU stream before publishing this
                # completion. Peers cannot reload a partly written DDR row.
                errors = [None] * self.tp_size
                dist.all_gather_object(
                    errors, str(error) if error is not None else None, group=self.tp_group.cpu_group
                )
                if any(item is not None for item in errors):
                    raise RuntimeError(f"Mooncake TP0 writeback failed; all ranks must stop: {errors}") from error
            elif error is not None:
                raise error
            return
        if not has_prefill and k is not None and v is not None and self.use_fused_overlap:
            layer_id = self._get_offload_layer_id(layer_name)
            self.current_kv_by_layer[layer_id] = (k, v)
        use_side_stream = self.tp_rank == 0 and self.use_fused_overlap and capturing and not has_prefill
        if not use_side_stream:
            self._offload_new_kv_on_current_stream(
                slot_mapping,
                k_cache_cpu,
                v_cache_cpu,
                k_cache_npu,
                v_cache_npu,
                k,
                v,
                has_prefill,
            )
            return

        current_kv_ready = torch_npu.npu.current_stream().record_event()
        with torch_npu.npu.stream(self.current_kv_save_stream):
            self.current_kv_save_stream.wait_event(current_kv_ready)
            self._offload_new_kv_on_current_stream(
                slot_mapping,
                k_cache_cpu,
                v_cache_cpu,
                k_cache_npu,
                v_cache_npu,
                k,
                v,
                has_prefill,
            )

    def _mooncake_writeback(
        self, slot_mapping, k_cpu, v_cpu, k_npu, v_npu, k, v, has_prefill
    ) -> None:
        if k_cpu is None or v_cpu is None:
            raise RuntimeError("Mooncake Host KV views are not registered")
        if has_prefill and not self.sparse_kv_offload_config.keep_device_kv_cache:
            raise RuntimeError("Mooncake colocated prefill requires device KV staging")
        if slot_mapping.dtype not in (torch.int32, torch.int64):
            raise ValueError("Mooncake slot_mapping must contain integer slots")
        slots = slot_mapping.reshape(-1).cpu().tolist()
        valid_slots = [slot for slot in slots if slot >= 0]
        if len(set(valid_slots)) != len(valid_slots):
            raise ValueError("Mooncake writeback has overlapping destination slots")
        sources, destinations, lengths, owners = [], [], [], []
        for host, paged, current, token_bytes in (
            (k_cpu, k_npu, k, self.token_size_bytes_k),
            (v_cpu, v_npu, v, self.token_size_bytes_v),
        ):
            if not host.is_contiguous() or token_bytes <= 0:
                raise ValueError("Mooncake writeback requires contiguous row-addressable Host KV")
            source = paged if has_prefill else current
            if source is None or source.dtype != torch.bfloat16 or source.device.type != "npu":
                raise ValueError("Mooncake writeback requires BF16 current/paged KV")
            rows = source.contiguous().view(-1, token_bytes // source.element_size())
            owners.append(rows)
            if not has_prefill and rows.shape[0] != len(slots):
                raise ValueError("Mooncake current KV row count differs from slot_mapping")
            capacity = host.numel() * host.element_size() // token_bytes
            for row, slot in enumerate(slots):
                if slot < 0:
                    continue
                source_row = slot if has_prefill else row
                if slot >= capacity or source_row >= rows.shape[0]:
                    raise ValueError("Mooncake writeback slot is outside its KV allocation")
                sources.append(rows[source_row].data_ptr())
                destinations.append(self._mooncake_device_address(host, slot * token_bytes, token_bytes))
                lengths.append(token_bytes)
        self._mooncake_copy(sources, destinations, lengths, tuple(owners))

    def _offload_new_kv_on_current_stream(
        self,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        v_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        k: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        v: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        has_prefill: bool = False,
        capturing: bool = False,
        layer_id: int | None = None,
    ) -> None:
        # the has_prefill path (NPU paged cache -> CPU pool D2H) only exists
        # for single-node PD-colocate debug.
        if self.tp_rank != 0:
            # Decode-produced K/V is replicated across TP ranks, so TP0 alone
            # writes new decode tokens. PD pull fills disjoint parts of this
            # shared pool from all TP ranks through the broadcast GVA.
            return
        if k_cache_cpu is None or v_cache_cpu is None:
            raise RuntimeError("Sparse KV offload TP0 CPU cache is not registered")
        if has_prefill and not self.sparse_kv_offload_config.keep_device_kv_cache:
            raise RuntimeError(
                "Sparse KV offload prefill offload requires "
                "keep_device_kv_cache=True; a PD-disaggregated decode node "
                "never stages prefill KV in an NPU paged cache"
            )

        if has_prefill:
            if k_cache_npu is None or v_cache_npu is None:
                raise ValueError("prefill offload requires NPU paged K/V caches")
            device = k_cache_npu.device
        else:
            if k is None or v is None:
                raise ValueError("decode offload requires current-token K/V")
            device = k.device

        slots = slot_mapping.reshape(-1).to(device=device, dtype=torch.int64)
        token_count = slots.numel()
        if token_count > self.max_num_tokens:
            raise ValueError(
                "Sparse KV offload rows exceed D2H descriptor capacity, "
                f"got {token_count}, capacity={self.max_num_tokens}"
            )

        num_k_slots = k_cache_cpu.numel() * k_cache_cpu.element_size() // self.token_size_bytes_k
        num_v_slots = v_cache_cpu.numel() * v_cache_cpu.element_size() // self.token_size_bytes_v
        if num_k_slots != num_v_slots or num_k_slots <= 0:
            raise ValueError(
                f"Sparse KV offload CPU K/V pools have incompatible token capacities: k={num_k_slots}, v={num_v_slots}"
            )
        valid = (slots >= 0) & (slots < num_k_slots)
        safe_slots = slots.clamp(min=0, max=num_k_slots - 1)
        duplicate = None
        if getattr(self, "mooncake_device_execution", False):
            sorted_slots = torch.sort(slots).values
            duplicate = torch.any((sorted_slots[1:] == sorted_slots[:-1]) & (sorted_slots[1:] >= 0))
            # Reject the whole batch before any concurrent duplicate writes.
            valid = valid & ~duplicate

        if has_prefill:
            assert k_cache_npu is not None and v_cache_npu is not None
            src_k = int(k_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_k
            src_v = int(v_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_v
        else:
            assert k is not None and v is not None
            k_rows = k.reshape(-1, self.token_size_bytes_k // k.element_size())
            v_rows = v.reshape(-1, self.token_size_bytes_v // v.element_size())
            if k_rows.shape[0] != token_count or v_rows.shape[0] != token_count:
                raise ValueError("decode K/V row counts must match slot_mapping")
            if not k_rows.is_contiguous():
                k_rows = k_rows.contiguous()
            if not v_rows.is_contiguous():
                v_rows = v_rows.contiguous()
            token_indices = self.d2h_token_indices_npu[:token_count]
            src_k = int(k_rows.data_ptr()) + token_indices * self.token_size_bytes_k
            src_v = int(v_rows.data_ptr()) + token_indices * self.token_size_bytes_v

        dst_k = int(k_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_k
        dst_v = int(v_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_v
        self.d2h_src_ptrs_npu[:token_count].copy_(src_k)
        self.d2h_src_ptrs_npu[token_count : 2 * token_count].copy_(src_v)
        self.d2h_dst_ptrs_npu[:token_count].copy_(dst_k)
        self.d2h_dst_ptrs_npu[token_count : 2 * token_count].copy_(dst_v)
        self.d2h_lengths_npu[:token_count].fill_(self.token_size_bytes_k)
        self.d2h_lengths_npu[token_count : 2 * token_count].fill_(self.token_size_bytes_v)
        self.d2h_lengths_npu[:token_count].masked_fill_(~valid, 0)
        self.d2h_lengths_npu[token_count : 2 * token_count].masked_fill_(~valid, 0)
        self.d2h_size_npu.fill_(2 * token_count)

        if getattr(self, "mooncake_device_execution", False):
            if layer_id is None:
                raise ValueError("Mooncake native writeback requires a stable layer status slot")
            status = self.mooncake_status[layer_id, 0]
            torch.bitwise_or(status, torch.any(slots >= num_k_slots).to(torch.int32).reshape(1) * 4, out=status)
            torch.bitwise_or(status, duplicate.to(torch.int32).reshape(1) * 16, out=status)
            with self.mooncake_metrics.measure("native_d2h", device=True):
                self._mooncake_native_copy(
                    self.d2h_src_ptrs_npu, self.d2h_dst_ptrs_npu, self.d2h_lengths_npu,
                    self.d2h_size_npu, status, reset_status=False,
                )
            return
        result = _legacy_offload().sparse_copy(
            self.d2h_src_ptrs_npu,
            self.d2h_dst_ptrs_npu,
            self.d2h_lengths_npu,
            self.d2h_size_npu,
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"memfabric D2H sparse_copy failed with result={result}")

    def prepare_fused_overlap_external_plan(
        self,
        layer_name: str,
        num_tokens: int,
        topk_indices_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        visible_seq_lens_npu: torch.Tensor,
        selection_membership_map: torch.Tensor,
        capturing: bool = False,
        skip_topk: bool = False,
    ) -> bool:
        self._validate_fused_overlap_external_plan_inputs(
            num_tokens,
            topk_indices_npu,
            req_ids_npu,
            stable_prefix_lens_npu,
            visible_seq_lens_npu,
            selection_membership_map,
        )
        layer_id = self._get_offload_layer_id(layer_name)
        plan_start = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - self.topk
        plan_storage = selection_membership_map[
            :num_tokens,
            plan_start:FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS,
        ]
        encoded_plan_stride = selection_membership_map.stride(0)

        owner_layer_id = self.fused_overlap_plan_owner_layer_id
        owner_map = self.fused_overlap_plan_membership_map
        can_reuse_owner_plan = (
            skip_topk
            and owner_layer_id is not None
            and layer_id > owner_layer_id
            and self.fused_overlap_plan_topk == self.topk
            and self.fused_overlap_plan_num_tokens == num_tokens
            and owner_map is not None
        )
        if can_reuse_owner_plan:
            assert owner_map is not None
            if selection_membership_map.data_ptr() != owner_map.data_ptr():
                if capturing:
                    torch_npu.npu.current_stream().wait_stream(self.fused_plan_stream)
                plan_storage.copy_(
                    owner_map[:num_tokens, plan_start:FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS],
                    non_blocking=capturing,
                )
            return True

        def run_planner(enqueue: bool) -> None:
            sparse_kv_ops = _sparse_kv_ops()
            planner = (
                sparse_kv_ops.sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows
                if enqueue
                else sparse_kv_ops.sparse_kv_lru_resident_compact_with_plan_stable_rows
            )
            planner(
                self.lru_req_ids_ptr,
                self.lru_last_req_ids_ptrs[layer_id],
                self.lru_topk_indices_ptr,
                self.lru_stable_prefix_lens_ptr,
                self.lru_slot_to_token_ptrs[layer_id],
                self.lru_slots_ptrs[layer_id],
                self.lru_current_slots_ptr,
                self.lru_miss_count_ptrs[layer_id],
                self.lru_miss_tokens_ptrs[layer_id],
                self.lru_miss_slots_ptrs[layer_id],
                self.lru_token_mark_workspace_ptr,
                self.lru_token_pos_workspace_ptr,
                self.lru_slot_workspace_ptr,
                self.lru_miss_position_workspace_ptr,
                self.lru_epochs_ptr,
                self.lru_physical_row_workspace_ptr,
                self.max_num_topk_rows,
                plan_storage.data_ptr(),
                encoded_plan_stride,
                num_tokens,
                self.topk,
                self.topk_buffer_size,
                self.max_model_len,
                self.lru_workspace_threads,
                self.lru_workspace_threads,
                self.lru_visible_seq_lens_ptr,
            )

        if capturing:
            plan_inputs_ready = torch_npu.npu.current_stream().record_event()
            with torch_npu.npu.stream(self.fused_plan_stream):
                self.fused_plan_stream.wait_event(plan_inputs_ready)
                if self.tp_rank == 0:
                    self.lru_topk_indices_cpu[:num_tokens].copy_(topk_indices_npu, non_blocking=True)
                    self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu, non_blocking=True)
                    self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(stable_prefix_lens_npu, non_blocking=True)
                    self.lru_visible_seq_lens_cpu[:num_tokens].copy_(visible_seq_lens_npu, non_blocking=True)
                    run_planner(enqueue=True)
                    self.fused_plan_current_linear_slots_npu[:num_tokens].copy_(
                        self.lru_physical_row_workspace[
                            self.max_num_topk_rows * 2 : self.max_num_topk_rows * 2 + num_tokens
                        ],
                        non_blocking=True,
                    )
                self.tp_group.broadcast(self.fused_plan_metadata_npu, src=0)
            self.fused_overlap_plan_owner_layer_id = layer_id
            self.fused_overlap_plan_topk = self.topk
            self.fused_overlap_plan_num_tokens = num_tokens
            self.fused_overlap_plan_membership_map = selection_membership_map
            return True

        planner_error = None
        if self.tp_rank == 0:
            self.fused_plan_status_npu.zero_()
            try:
                self.lru_topk_indices_cpu[:num_tokens].copy_(topk_indices_npu)
                self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu)
                self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(stable_prefix_lens_npu)
                self.lru_visible_seq_lens_cpu[:num_tokens].copy_(visible_seq_lens_npu)
                run_planner(enqueue=False)
                self.fused_plan_current_linear_slots_npu[:num_tokens].copy_(
                    self.lru_physical_row_workspace[
                        self.max_num_topk_rows * 2 : self.max_num_topk_rows * 2 + num_tokens
                    ]
                )
            except Exception as exc:
                planner_error = exc
                self.fused_plan_status_npu.fill_(1)
        self.tp_group.broadcast(self.fused_plan_metadata_npu, src=0)
        if int(self.fused_plan_status_npu.item()) != 0:
            detail = (
                f"{type(planner_error).__name__}: {planner_error}"
                if planner_error is not None
                else "TP0 external planner failed"
            )
            raise RuntimeError(
                f"SFA fused_overlap external planner failed: layer={layer_name}, tp_rank={self.tp_rank}, {detail}"
            ) from planner_error
        self.fused_overlap_plan_owner_layer_id = layer_id
        self.fused_overlap_plan_topk = self.topk
        self.fused_overlap_plan_num_tokens = num_tokens
        self.fused_overlap_plan_membership_map = selection_membership_map
        return True

    def _validate_fused_overlap_external_plan_inputs(
        self,
        num_tokens: int,
        topk_indices_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        visible_seq_lens_npu: torch.Tensor,
        selection_membership_map: torch.Tensor,
    ) -> None:
        if not self.use_fused_overlap:
            raise RuntimeError("external FSA plan requires fused_overlap mode")
        if num_tokens <= 0 or num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "external FSA plan rows exceed configured workspace: "
                f"num_tokens={num_tokens}, max_rows={self.max_num_topk_rows}"
            )
        expected_topk_shape = (num_tokens, self.topk)
        if tuple(topk_indices_npu.shape) != expected_topk_shape:
            raise ValueError(
                "external FSA plan requires flattened single-head TopK input: "
                f"expected={expected_topk_shape}, actual={tuple(topk_indices_npu.shape)}"
            )
        if tuple(req_ids_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one request id per logical row: "
                f"expected=({num_tokens},), actual={tuple(req_ids_npu.shape)}"
            )
        if tuple(stable_prefix_lens_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one stable prefix length per logical row: "
                f"expected=({num_tokens},), "
                f"actual={tuple(stable_prefix_lens_npu.shape)}"
            )
        if tuple(visible_seq_lens_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one visible KV length per logical row: "
                f"expected=({num_tokens},), "
                f"actual={tuple(visible_seq_lens_npu.shape)}"
            )
        if (
            selection_membership_map.dim() != 2
            or selection_membership_map.shape[0] < num_tokens
            or selection_membership_map.shape[1] < FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS
            or selection_membership_map.dtype != torch.int16
            or selection_membership_map.device.type != "cpu"
        ):
            raise ValueError(
                "external FSA plan requires mapped CPU int16 membership storage: "
                f"min_shape=({num_tokens}, {FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS}), "
                f"actual_shape={tuple(selection_membership_map.shape)}, "
                f"dtype={selection_membership_map.dtype}, "
                f"device={selection_membership_map.device}"
            )

    def inject_current_kv_into_selection(
        self,
        layer_name: str,
        num_tokens: int,
        selection_kv_cache: torch.Tensor,
        selection_k_rope: torch.Tensor,
        capturing: bool = False,
    ) -> None:
        layer_id = self._get_offload_layer_id(layer_name)
        current_kv = self.current_kv_by_layer.get(layer_id)
        if current_kv is None:
            raise RuntimeError(f"current decode K/V is unavailable for fused layer {layer_name}")
        if capturing:
            torch_npu.npu.current_stream().wait_stream(self.fused_plan_stream)
        current_k, current_rope = current_kv
        current_k = current_k.reshape(-1, selection_kv_cache.shape[-1])[:num_tokens]
        current_rope = current_rope.reshape(-1, selection_k_rope.shape[-1])[:num_tokens]
        linear_slots = self.fused_plan_current_linear_slots_npu[:num_tokens]
        indices = linear_slots.view(-1, 1)
        flat_kv = selection_kv_cache.reshape(-1, selection_kv_cache.shape[-1])
        flat_rope = selection_k_rope.reshape(-1, selection_k_rope.shape[-1])
        torch_npu.npu_scatter_nd_update_(flat_kv, indices, current_k)
        torch_npu.npu_scatter_nd_update_(flat_rope, indices, current_rope)

    def wait_for_current_kv_writeback(self, capturing: bool = False) -> None:
        if self.use_fused_overlap and capturing and self.tp_rank == 0:
            torch_npu.npu.current_stream().wait_stream(self.current_kv_save_stream)


_SPARSE_KV_OFFLOAD_MANAGER: SparseKVOffloadManager | None = None


def init_sparse_kv_offload_manager(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    global _SPARSE_KV_OFFLOAD_MANAGER
    if _SPARSE_KV_OFFLOAD_MANAGER is None:
        if get_ascend_device_type() != AscendDeviceType.A3:
            raise RuntimeError("Sparse KV offload is only support on A3")
        _SPARSE_KV_OFFLOAD_MANAGER = SparseKVOffloadManager(
            vllm_config,
            kv_cache_config,
            sparse_kv_offload_config,
        )
    return _SPARSE_KV_OFFLOAD_MANAGER


def get_sparse_kv_offload_manager():
    assert _SPARSE_KV_OFFLOAD_MANAGER is not None, "KV offload manager is not initialized."
    return _SPARSE_KV_OFFLOAD_MANAGER
