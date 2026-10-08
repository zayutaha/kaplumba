# Technical Plan: Persistent Green Residency & Duplicate Memory Elimination

## 1. Architecture & Data Flow

### 1.1 Memory Separation
```
+----------------------------------------------------------------------------------+
|                            Unified Memory (24 GiB)                              |
+----------------------------------------------------------------------------------+
| System + Desktop + Apps: ~4.5 GiB                                                |
|                                                                                  |
| Kaplumba Weight Allocations:                                                     |
|  - WeightArena (Dedicated Metal Buffers): ~12.04 GiB (Trellis) + ~0.6 GiB (Sides) |
|    * StorageModeShared, dirty anonymous pages.                                   |
|    * Wrapped in MTLResidencySet -> attached to MTLCommandQueue.                  |
|    * NEVER mlocked; kept resident via Metal GPU driver residency contract.       |
|                                                                                  |
|  - File-backed On-Disk Mapping:                                                  |
|    * embed_tokens.weight: 2.82 GiB mapped on demand (only read 1 row / token).  |
|    * ZERO MADV_WILLNEED pre-faulting on shard files.                            |
|    * Unused shards dropped immediately after arena copy phase.                  |
|                                                                                  |
|  - KV Cache & Activations: ~1.5 - 2.5 GiB at 16k context.                        |
|                                                                                  |
| Total Physical RAM Used: ~12.6 + 4.5 + 2.0 = ~19.1 GiB <= 24 GiB (SOLID GREEN)   |
+----------------------------------------------------------------------------------+
```

### 1.2 Metal 3 `MTLResidencySet` Integration
In macOS 14+ / Metal 3:
- An `MTLResidencySetDescriptor` is instantiated.
- `[device newResidencySetWithDescriptor:error:]` creates the set.
- Every `Buffer` created in `WeightArena` (all segments) is registered via `[set addAllocation:buffer]`.
- `[set commit]` finalizes the resource set.
- `[set requestResidency]` requests standing resident priority.
- `[command_queue addResidencySet:set]` attaches the residency set to the active execution queue.
- This ensures the `iogpu` wired collector does not evict or unwire the weights when the GPU is temporarily idle waiting for user prompts in `--chat`.

---

## 2. File / Module Touchpoints

1. `crates/engine/src/weights.rs`:
   - Remove `libc::madvise(..., MADV_WILLNEED)` in `ShardReader::open`.
   - Update `load_internal`: only retain shards in `self.shards` if they actually contain zero-copy/mapped tensors (e.g. `skip_copy`).
   - Add `residency_set` management to `WeightArena` or `TextWeights`.
2. `crates/engine/src/metal.rs`:
   - Add Objective-C FFI bindings for `MTLResidencySetDescriptor`, `MTLResidencySet`, and `addResidencySet` on `MTLCommandQueue`.
   - Expose safe Rust methods: `create_residency_set`, `add_allocation`, `commit`, `request_residency`, `attach_to_queue`.
3. `crates/engine/src/model.rs`:
   - Ensure the command queue used for forward passes has the weights' `MTLResidencySet` attached.
4. `crates/engine/tests/`:
   - New unit tests verifying `MTLResidencySet` creation and attachment.
   - Verification that `ShardReader` does not force `MADV_WILLNEED`.
   - Sabotage tests to prove test failure before passing.

---

## 3. Requirement Mapping
- Requirement 1 (`MADV_WILLNEED` removal) -> Task T1.
- Requirement 2 (Selective shard retention) -> Task T2.
- Requirement 3 (`MTLResidencySet` implementation & attachment) -> Task T3.
- Requirement 4 (Isolated zero-copy embedding) -> Task T4.
- Requirements 5 & 6 (Green memory pressure & 0 swap at 16k) -> Task T5.
