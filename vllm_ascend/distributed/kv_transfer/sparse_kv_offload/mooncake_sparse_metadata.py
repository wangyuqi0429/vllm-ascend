# SPDX-License-Identifier: Apache-2.0
"""Explicit scheduler-side envelope for native sparse Store transactions.

Invoke after framework block allocation and before worker dispatch. This is an
opt-in programmatic integration, independent of a PD connector. No hash lookup,
new allocator, scheduling decision or automatic cache-hit claim is introduced.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SparseStoreMetadata:
    sparse_store_imports: tuple[dict, ...] = ()
    sparse_store_seals: tuple[dict, ...] = ()


def attach_sparse_store_transactions(scheduler_output, *, imports=(), seals=()) -> None:
    """Queue explicit full-block imports/seals on an allocated scheduler step.

    Items have exactly req_id, block_hash, block_index. The producer must use the
    framework's content hash for the full block; worker verifies slot ownership
    and complete-prefix eligibility for seals. A caller may dispatch this output
    only after this function returns. Do not attach alongside a PD connector.
    """
    if scheduler_output.kv_connector_metadata is not None:
        raise ValueError("Sparse Store envelope cannot overwrite connector metadata")
    copied = []
    for batch in (imports, seals):
        items = []
        for item in batch:
            if not isinstance(item, dict) or set(item) != {"req_id", "block_hash", "block_index"}:
                raise ValueError("Sparse Store item requires req_id/block_hash/block_index")
            if not isinstance(item["req_id"], str) or not item["req_id"]:
                raise ValueError("Sparse Store request id must be nonempty")
            item = dict(item)
            if isinstance(item["block_hash"], bytes):
                item["block_hash"] = item["block_hash"].hex()
            if not isinstance(item["block_hash"], str) or not item["block_hash"]:
                raise ValueError("Sparse Store framework content hash must be nonempty")
            index = item["block_index"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("Sparse Store block index must be nonnegative")
            items.append(dict(item))
        copied.append(tuple(items))
    scheduler_output.kv_connector_metadata = SparseStoreMetadata(*copied)
