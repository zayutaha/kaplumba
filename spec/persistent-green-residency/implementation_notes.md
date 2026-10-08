# Implementation Notes: Persistent Green Residency & Duplicate Memory Elimination

## Environment & Findings
- Hardware: Apple M4 Pro (20 GPU cores, 24 GiB unified RAM).
- Issue diagnosed:
  1. `ShardReader::open` called `madvise(MADV_WILLNEED)` on both safetensors shards (14.9 GB total), caching them in unified memory page cache.
  2. `WeightArena::allocate_dedicated` allocated 13.5 GB of dirty anonymous Metal shared buffers.
  3. Total memory requested was ~28.4 GB on a 24 GB Mac, overwhelming the VM and pushing 7.6 GB to swap.
  4. Between chat turns, the macOS kernel `iogpu.wired_collector` unwired GPU allocations when the command queue went idle (`iogpu.disable_wired_collector: 0`), creating memory drops and subsequent re-wiring stalls.

## Changes Implemented
1. **Removed `MADV_WILLNEED` in `ShardReader::open` (`crates/engine/src/weights.rs`):**
   - Clean file mappings now only fault pages on demand (e.g. for `embed_tokens.weight` which reads one ~10 KiB row per token).
2. **Selective Shard Retention in `TextWeights::load_internal`:**
   - Shards whose tensors are all copied into the arena (e.g. shard 2) are dropped immediately upon load, unmapping gigabytes of virtual memory.
   - Tested with `prunes_unused_shards_and_only_retains_zero_copy_shards`. Sabotaged via `KAPLUMBA_SABOTAGE_SHARD_PRUNING=1` which failed with `left == 1, right == 2`.
3. **Objective-C FFI for Metal 3 `MTLResidencySet` (`crates/engine/src/metal.rs`):**
   - Implemented `ResidencySet` wrapping `MTLResidencySet`.
   - Supports `new`, `add_allocation`, `commit`, `request_residency`, and `attach_to_queue`.
   - Tested with `crates/engine/tests/test_residency_set.rs`. Sabotaged via `KAPLUMBA_SABOTAGE_RESIDENCY_SET=1`.
4. **Residency Set Wiring (`crates/engine/src/weights.rs`, `crates/engine/src/model.rs`):**
   - `WeightArena::create_residency_set` adds all arena segments to the set, commits, and requests residency.
   - `TextModel::finish_construction` creates and attaches this residency set to `forward_queue`.
   - Standing residency set prevents the macOS `iogpu.wired_collector` from releasing weights when decode is idle.
