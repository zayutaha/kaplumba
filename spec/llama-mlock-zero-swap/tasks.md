# Task Checklist: `llama.cpp` `mlock` Architecture & Zero-Swap Residency (1x Weights)

- [x] **Task 1: Fix KV cache sizing and position wipe in `gpu_cache.rs`**
  - **Files Touched**: `crates/engine/src/gpu_cache.rs`
  - **Intent**: Fix line 234 in `ensure_capacity` from `(target * self.stream_len * 2 * 2)` to `(target * self.stream_len * 2)` so `new_keys` and `new_values` allocate exact 1x capacity. Fix line 283 to wipe `(target - live) * 4` bytes with `0xFF`.
  - **Commit**: `becd9c2`
  - **Testing**: `cargo test --package kaplumba-engine --lib gpu_cache`.

- [x] **Task 2: Implement KV cache capacity verification and sabotage test**
  - **Files Touched**: `crates/engine/tests/probe_kv_cache_capacity.rs`
  - **Intent**: Write unit probe asserting exact buffer sizes across ladder growth steps.
  - **Commit**: `becd9c2`
  - **Testing**: `cargo test --package kaplumba-engine --test probe_kv_cache_capacity`.

- [x] **Task 3: Shard-level Metal buffer mapping, direct I/O via `F_NOCACHE`, and removal of `drop_cache` / `MADV_DONTNEED`**
  - **Files Touched**: `crates/engine/src/weights.rs`
  - **Intent**: Wrap each shard `mmap` into a 16 KiB page-aligned Metal buffer (`StorageModeShared | HazardTrackingModeUntracked`) like `llama.cpp`. Remove Phase 2.5 and `reader.drop_cache` (`MADV_DONTNEED`). Use Darwin's `fcntl(fd, F_NOCACHE, 1)` for arena `pread` file reads so the OS unified buffer cache does not duplicate weights.
  - **Commit**: `373924e`
  - **Testing**: `cargo test --package kaplumba-engine --lib weights` (12/12 passed).

- [x] **Task 4: Integrate `LlamaMlock` on file `mmap` pages and eliminate `pin_realigned_copy`**
  - **Files Touched**: `crates/engine/src/weights.rs`, `crates/engine/src/model.rs`
  - **Intent**: Use `LlamaMlock` only on host `mmap` memory ranges. Remove `pin_realigned_copy` completely so POSIX `mlock` is never called on `id<MTLBuffer>` pointers. Ensure `embed_tokens` remains unpinned.
  - **Commit**: `373924e`
  - **Testing**: `cargo test --package kaplumba-engine --test probe_llama_mlock` (3/3 passed).

- [x] **Task 5: Fix residency set handling in `model.rs` (remove `attach_to_queue`)**
  - **Files Touched**: `crates/engine/src/model.rs`, `crates/engine/src/metal.rs`
  - **Intent**: Remove `set.attach_to_queue(&forward_queue)` in `model.rs:3478`. Ensure residency sets are created with all shard buffers and committed with `request_residency()` without duplicate command-queue registration.
  - **Commit**: `373924e`
  - **Testing**: `cargo check --workspace` & `cargo test --workspace --lib` (140/140 passed).

- [x] **Task 6: Sabotage verification and unit test suite**
  - **Files Touched**: `crates/engine/tests/probe_llama_mlock.rs`, `spec/llama-mlock-zero-swap/implementation_notes.md`
  - **Intent**: Run sabotage tests on `probe_llama_mlock` and verify all unit tests pass without errors.
  - **Commit**: `373924e`
  - **Testing**: Sabotage confirmed (2 failures observed when ignoring return code); restored and verified 3/3 passed.

- [ ] **Task 7: Memory footprint & swap behavior verification**
  - **Files Touched**: `spec/llama-mlock-zero-swap/implementation_notes.md`
  - **Intent**: Verify 1x resident model footprint (~13-14 GiB total, 0 swap growth, Green memory pressure) when model is loaded.
