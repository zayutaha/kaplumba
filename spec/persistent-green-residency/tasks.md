# Tasks: Persistent Green Residency & Duplicate Memory Elimination

- [x] **T1: Remove `MADV_WILLNEED` and full-shard page faulting**
  - Files: `crates/engine/src/weights.rs`
  - Intent: Prevent `ShardReader::open` from pre-faulting 14.9 GiB into OS file cache.
  - Test: Clean file mapping verified without `MADV_WILLNEED`.

- [x] **T2: Prune unused shard mappings from `TextWeights.shards`**
  - Files: `crates/engine/src/weights.rs`
  - Intent: Only keep shards in `shards: BTreeMap<String, ShardReader>` if a tensor within them actually requires zero-copy mapping (e.g. `embed_tokens.weight`). Shards whose tensors are all copied into the arena are dropped immediately.
  - Test: `weights::tests::prunes_unused_shards_and_only_retains_zero_copy_shards` passed. Sabotage verified (`KAPLUMBA_SABOTAGE_SHARD_PRUNING=1` failed with `left == 1, right == 2`).

- [x] **T3: Implement Metal 3 `MTLResidencySet` FFI in `crates/engine/src/metal.rs`**
  - Files: `crates/engine/src/metal.rs`, `Cargo.toml`, `crates/engine/Cargo.toml`
  - Intent: Implement Objective-C FFI for `MTLResidencySetDescriptor`, `MTLResidencySet`, `addAllocation`, `commit`, `requestResidency`, and `MTLCommandQueue.addResidencySet`.
  - Test: `test_residency_set_creation_and_queue_attachment` in `tests/test_residency_set.rs` passed. Sabotage verified (`KAPLUMBA_SABOTAGE_RESIDENCY_SET=1` failed with panic).

- [x] **T4: Attach `WeightArena` buffers to `MTLResidencySet` and command queue**
  - Files: `crates/engine/src/weights.rs`, `crates/engine/src/model.rs`
  - Intent: In `TextWeights` or `TextModel`, register all arena buffers in the residency set and attach to the forward queue.
  - Test: `WeightArena::create_residency_set` and `TextModel::finish_construction` verified via integration and lib tests.

- [x] **T5: End-to-end memory residency verification without loading 15 GB model**
  - Files: `crates/engine/tests/test_residency_set.rs`, `crates/engine/tests/probe_shared_weights_footprint.rs`
  - Intent: End-to-end test verifying that standing residency sets prevent idle memory release without `mlock`, and verifying no swap usage growth.
  - Test: All 136 tests pass cleanly with 0 failures.
