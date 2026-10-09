# Implementation Notes: `llama.cpp` `mlock` & Zero-Swap Architecture (1x Weights)

Append as work proceeds. Factual and dense.

## 0. Hardware & Environment Facts
- **Machine:** Apple M4 Pro (20 GPU cores, 24 GiB unified RAM, 273 GB/s peak bandwidth).
- **OS:** macOS 15+ (Darwin kernel 27.0).
- **Darwin Kernel Wire Limit Limits:**
  - `sysctl vm.user_wire_limit`: 19,585,050,869 bytes (~18.24 GiB).
  - `sysctl vm.global_no_user_wire_amount`: 6,184,752,907 bytes (~5.76 GiB reserved unwired).
  - OS baseline wired memory: ~3.1–3.5 GiB.
- **Model:** Gemma 4 31B dense (`~/.kaplumba/models/Gxma-repacked`), EXL3 ~3.0 bpw.
  - Trellis layers: 12.04 GiB.
  - `embed_tokens`: 2.82 GB.
  - `lm_head.trellis`: 1.06 GB.
  - Arena (non-trellis side tensors): ~1.1 GiB.

## 1. Mathematical Breakdown of System Memory (1x Residency)
- Trellis weights remain **zero-copy clean file mappings** (`no_copy_view`), pinned via `LlamaMlock`: 12.04 GiB wired.
- OS baseline: ~3.5 GiB wired.
- Total system wired: ~15.54 GiB.
- Remaining unwired RAM: $24.0 - 15.54 = \mathbf{8.46\text{ GiB}}$.
- Allocations within unwired RAM:
  - System desktop apps: ~4.5 GiB.
  - Non-trellis arena: ~1.1 GiB.
  - `ForwardScratch` buffers: ~1.2 GiB.
  - 1x KV Cache at 16k context (fixed from 2x bug): ~1.35 GiB.
  - Total unwired required: $4.5 + 1.1 + 1.2 + 1.35 = \mathbf{8.15\text{ GiB}} \le 8.46\text{ GiB}$.
  - System memory pressure remains **Green** with 0 swap.

## 2. Why the 2x Weights Bug Happened and Why the Stash Was Flawed
1. **The 2x Model Weights Bug:**
   - In `spec/omlx-memory-parity`, setting `skip_copy = false` for trellis caused `copy_tensor` to `pread` 12 GiB of weights into `WeightArena` Metal shared buffers.
   - On Darwin, `pread` populates the kernel's unified page cache (UBC), while `WeightArena` holds dirty anonymous Metal buffers. Both exist concurrently: $12\text{ GiB} + 12\text{ GiB} = \mathbf{24\text{ GiB}}$ resident!
2. **Why `stash@{0}` Flawed Approach Failed:**
   - `stash@{0}` tried to call `madvise(MADV_DONTNEED)` in phase 2.5 on the mapped shards to drop the duplicate source cache.
   - On Darwin, `madvise(MADV_DONTNEED)` on file mappings does not flush the UBC `pread` buffer, and dropping pages of shared files risked evicting zero-copy tensors (`lm_head.trellis`, `embed_tokens`), while leaving dirty anonymous arena pages that macOS compressed at 16k context (0.37 tok/s collapse).
3. **The Clean `llama.cpp` Solution:**
   - Never copy trellis into `WeightArena`!
   - Keep trellis as clean zero-copy `mmap` views (`no_copy_view`, `skip_copy = true`).
   - Weights exist in physical memory **strictly 1x**.
   - Pin them with `LlamaMlock::grow_to` and register with `MTLResidencySet`.

## 3. Crash Post-Mortem & Root Cause Analysis (2026-10-09)
- **Symptom:** System froze completely upon model load (even with 11 GiB resident). Hard reboot required.
- **Diagnostics:** `/Library/Logs/DiagnosticReports/forceReset-full-2026-10-09-112830.0002.diag` & `ResetCounter-2026-10-09-112831.diag`.
  - Panic frame: `AppleT8110DART` IOMMU Translation Fault.
  - Reset reason: `target_off_restart` / `dblclick_timeout` (SoC hardware Watchdog ForceReset).
- **The 4 Interacting Root Causes:**
  1. `PAGE_ALIGN = 4096` in `weights.rs` vs. Apple Silicon's 16 KiB hardware page size (`sysconf(_SC_PAGESIZE) = 16384`). Passing 4KB-aligned pointers and unaligned lengths to Metal's `newBufferWithBytesNoCopy` violated Metal runtime requirements.
  2. Fatal DART IOMMU Fault via `MADV_DONTNEED`: `reader.drop_cache()` called `madvise(..., MADV_DONTNEED)` on file mappings that shared 16 KiB pages with active zero-copy Metal buffers. When the GPU accessed these pages, the hardware translation table was invalidated, causing an instant SoC force reset.
  3. Calling `libc::mlock` on Metal shared buffers (`pin_realigned_copy`): Mach VM memory managed by `IOGPUFamily` cannot be locked via POSIX `mlock`. In `llama.cpp`, `mlock` is strictly guarded by `ggml_backend_buffer_is_host(buf)`.
  4. Double residency registration: Calling `[queue addResidencySet:]` on a residency set that already called `[set requestResidency]` caused driver assertion panics.
- **The Permanent Fix:**
  - Map each shard file once as a single page-aligned Metal buffer (`StorageModeShared`).
  - Completely eliminate Phase 2.5 and `reader.drop_cache` (`MADV_DONTNEED`).
  - Use `fcntl(fd, F_NOCACHE, 1)` on file descriptors during arena `pread` to bypass the unified page cache cleanly.
  - Apply `LlamaMlock` only to host file `mmap` ranges, never to `id<MTLBuffer>`.
  - Clean residency registration via `set.request_residency()` without queue attachment.

## 4. Non-Vacuity & Sabotage Verification (AGENTS.md §3)
- **Sabotage 1 (`probe_kv_cache_capacity.rs`):**
  - Verified in `becd9c2`. Removing the 1x multiplier or position buffer wipe causes `kv_buffers_are_exactly_1x_at_every_ladder_rung` and `the_position_tail_is_wiped_4_bytes_per_slot` to fail.
- **Sabotage 2 (`probe_llama_mlock.rs`):**
  - Sabotaged `LlamaMlock::grow_to` in `crates/engine/src/weights.rs` to ignore the `libc::mlock` return code and update `self.size = target_size` unconditionally.
  - Test run output:
    ```
    running 3 tests
    test grow_to_is_incremental_and_page_aligned ... ok
    test a_failed_grow_keeps_the_successful_prefix ... FAILED
    test oversized_grow_fails_soft_without_panicking ... FAILED

    failures:
    ---- a_failed_grow_keeps_the_successful_prefix stdout ----
    thread 'a_failed_grow_keeps_the_successful_prefix' panicked at crates/engine/tests/probe_llama_mlock.rs:155:5:
    usize::MAX must not succeed and must not panic

    ---- oversized_grow_fails_soft_without_panicking stdout ----
    thread 'oversized_grow_fails_soft_without_panicking' panicked at crates/engine/tests/probe_llama_mlock.rs:118:5:
    a 100 GiB mlock must set failed_already; it returned success, which means grow_to is not checking mlock's return code
    ```
  - Reverted sabotage and re-ran tests: 3/3 passed cleanly (`probe_llama_mlock.rs`).

