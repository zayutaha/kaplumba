# Specification: `llama.cpp` `mlock` Architecture & Zero-Swap Residency (1x Weights)

## Goal
Port `llama.cpp`'s battle-tested `llama_mlock` memory locking engine into Kaplumba, pair it with zero-copy Metal shared buffers and `MTLResidencySet`, fix the 2x over-allocation bug in `GpuKvCache::ensure_capacity`, and keep `embed_tokens` unpinned to remain safely below Darwin's 18.24 GiB user-wire limit. Trellis weights remain strictly **zero-copy clean file mappings**, eliminating the 2x model weights loading bug (no arena copying or pread duplication). Use `fcntl(fd, F_NOCACHE, 1)` for arena reads to prevent OS file cache duplication, enforce Apple Silicon 16 KiB page alignment for Metal zero-copy shard buffers, avoid calling POSIX `mlock` on Metal shared buffers, and completely eliminate `MADV_DONTNEED` cache-dropping hacks that trigger fatal Apple Silicon DART IOMMU faults. This guarantees **Green** memory pressure, **1x weight residency**, stability without kernel panics, and **0 MiB swap growth** up to at least 17 GiB of resident load.

---

## Non-Goals / Restrictions
- **No arena copying of trellis weights**: Trellis weights must NOT be copied into `WeightArena` via `pread`, which duplicates the 12 GiB model into a 24 GiB resident footprint.
- **No `madvise(MADV_DONTNEED)` cache hacks**: Zero-copy mappings must remain clean and intact without ad-hoc cache dropping passes that invalidate active GPU/Metal IOMMU pages (causes fatal `AppleT8110DART` panics).
- **No calling `mlock` on Metal shared buffers**: `mlock` must strictly apply to CPU host / `mmap` file pages (`ggml_backend_buffer_is_host` parity in `llama.cpp`). It must never be called on `id<MTLBuffer>` pointers.
- **No double residency registration**: Do not call `[queue addResidencySet:]` on a residency set that already called `[set requestResidency]`.
- **No sub-16KB page assumptions**: Metal `newBufferWithBytesNoCopy` pointer and length must strictly conform to Apple Silicon's 16 KiB page granularity (`sysconf(_SC_PAGESIZE)` = 16384).
- **No KV cache quantization**: KV cache elements remain full-precision FP16 (AGENTS.md §8).
- **No quality loss**: Output tokens remain bit-exact against baseline greedy reference.
- **No wiring `embed_tokens`**: The 2.82 GB embedding table must remain clean, unpinned `mmap` to avoid blowing past Darwin's `vm.user_wire_limit` of 18.24 GiB.
- **No Python code**: Pure Rust and Metal only.

---

## Requirements

1. **R1: Zero-Copy 1x Trellis Residency**
   `.trellis` weights (12.04 GiB) must remain clean zero-copy `mmap` views directly from safetensors shards. Each shard is wrapped into a 16 KiB page-aligned Metal buffer, avoiding 2x memory duplication.
2. **R2: Port `llama_mlock` Engine**
   Implement `LlamaMlock` in Rust mirroring `llama.cpp`'s `llama_mlock` (`reference/llama.cpp/src/llama-mmap.cpp`). It must support incremental `grow_to(target_size)`, enforce 16 KiB page-alignment, lock only host `mmap` pages, and gracefully catch `ENOMEM` / wire limits without aborting the model load.
3. **R3: Dedicated Unpinned Embedding Offload**
   `embed_tokens` (2.82 GB) must remain mapped as clean zero-copy `mmap` with `MADV_RANDOM`, reading single rows on demand without consuming wired memory.
4. **R4: Metal Residency Set Registration Without Double Accounting**
   All zero-copy shard weight buffers mapped into Metal (`StorageModeShared`) must be registered with `ResidencySet` (`metal.rs`), issuing `add_allocation`, `commit`, and `request_residency` as done in `llama.cpp` (`ggml-metal-device.m`), with no duplicate `addResidencySet:` on the command queue.
5. **R5: Fix KV Cache Double-Allocation Bug**
   In `crates/engine/src/gpu_cache.rs:234`, allocate exact 1x capacity (`target * stream_len * 2` bytes for FP16), eliminating the redundant `* 2` multiplier.
6. **R6: Fix Position Buffer Wipe**
   In `crates/engine/src/gpu_cache.rs:283`, wipe `(target - live) * 4` bytes with `0xFF` so all 4 bytes of each uninitialized slot are covered.
7. **R7: Direct I/O via `F_NOCACHE` for Arena Reads**
   For all `!skip_copy` tensors read via `pread` into `WeightArena`, set `fcntl(fd, F_NOCACHE, 1)` to prevent Darwin from duplicating file pages in the OS buffer cache, without calling `MADV_DONTNEED`.
8. **R8: Memory Pressure Green up to 17 GiB**
   System memory pressure must remain in the **Green** tier (`PhysMem` / `vm_stat` headroom $\ge 20\%$) during active inference up to 17 GiB resident usage.
9. **R9: Zero Swap Growth**
   System swap usage (`sysctl -n vm.swapusage`) must show 0 MiB growth during inference.

---

## Acceptance Criteria

1. Model load resident footprint is strictly **1x** (~13.1 GiB weights), confirming absence of the 2x loading bug.
2. System loads model without freezing, kernel panic, or `AppleT8110DART` IOMMU fault.
3. `probe_llama_mlock` passes, verifying incremental locking and soft clamping on `ENOMEM`.
4. `probe_kv_cache_capacity` passes, confirming exact 1x byte sizing across all ladder steps.
5. Generation at 16.5k context completes with 0 MiB swap growth and Green memory pressure.
6. Greedy generation remains byte-identical to reference outputs.

---

## Open Questions
*(None; all clarified and confirmed by user)*
