# File-Backed Weight Buffers — Tasks

## T1: Audit CPU-read tensor paths
- [ ] **Status:** Not started
- **Files:** `crates/engine/src/model.rs`, `crates/engine/src/weights.rs`
- **Intent:** Identify every tensor name passed to `read_f16()`,
  `packed_u16()`, or any CPU-side read. These must stay in the arena.
  Everything else can be file-backed.
- **Output:** A function `needs_arena_copy(name: &str, converted: bool) -> bool`
  or equivalent decision in the `skip_copy` block.
- **Test:** Manual audit; the list is documented in `implementation_notes.md`.

## T2: Expand `skip_copy` to all eligible tensors
- [ ] **Status:** Not started
- **Files:** `crates/engine/src/weights.rs` (lines ~574-593)
- **Intent:** Change the `skip_copy` decision so all non-converted,
  non-CPU-read packed EXL3 tensors (`.trellis`, `.su`, `.sv`) are file-backed
  via `no_copy_view()`. CPU-read side-tensors (`.suh`, `.svh`, `.mcg`) and
  BF16-converted norms/scalars stay in the arena.
- **Preserve:** `KAPLUMBA_RESIDENT_WEIGHTS=1` forces everything into the arena.
  `KAPLUMBA_ZERO_COPY_WEIGHTS` maps everything (existing behavior).
- **Test:** `cargo test` passes. Bit-exact output at `--temperature 0` vs
  baseline (same prompt, same model). Compare with `KAPLUMBA_RESIDENT_WEIGHTS=1`.

## T3: Propagate alignment flag for file-backed trellis
- [ ] **Status:** Not started
- **Files:** `crates/engine/src/model.rs` — `QuantizedLinear` construction
- **Intent:** Ensure that `aligned = false` is set for all file-backed trellis
  tensors, not just `lm_head.trellis`. The `no_copy_view()` delta might be
  non-4-byte-aligned (measured: 396/796 packed tensors are 2-byte misaligned).
- **Test:** Bit-exact output at `--temperature 0`. The existing `lm_head.trellis`
  unaligned path is the reference — extend the same logic to layer trellis.
  Sabotage: force `aligned=true` on a known-misaligned layer trellis and
  verify output is wrong (garbled/NaN), then revert.

## T4: Add `mlock()` after file-backed buffer creation
- [ ] **Status:** Not started
- **Files:** `crates/engine/src/weights.rs` — ShardReader, after Phase 1
- **Intent:** Pin all shard mmap pages in RAM via `mlock()` so there are zero
  page faults during inference. Pages remain file-backed (external) in macOS VM
  classification → no memory pressure contribution.
- **Sub-tasks:**
  - Check `sysctl vm.user_wire_limit` to verify the limit allows ~15 GB of
    wired pages.
  - Add `mlock()` call after all `no_copy_view()` calls complete.
  - Add `munlock()` in `ShardReader::drop()` before `munmap()`.
  - Handle `mlock()` failure gracefully (log warning, continue).
- **Test:** Verify pages are wired via `vm_stat` or `top` (look at "wired"
  count increasing by ~11 GB). Verify no page faults during first decode pass.

## T5: Measure and validate
- [ ] **Status:** Not started
- **Files:** None (measurement only)
- **Intent:** A/B interleaved measurement:
  - **A** = `KAPLUMBA_RESIDENT_WEIGHTS=1` (old arena path, baseline)
  - **B** = default (new file-backed + mlock path)
  - Metrics: `phys_footprint` via `/usr/bin/time -l`, Activity Monitor
    pressure color, decode tg/s, TTFT, `vm.swapusage` during 2500-token decode.
  - Pattern: A B A B A B, report medians with spread.
- **Acceptance:**
  - Green pressure in Activity Monitor (B)
  - `phys_footprint` < 14 GB (B), ideally < 5 GB
  - tg/s within measurement spread of (A)
  - 0 swap (B)
  - Bit-identical output (A) == (B) at `--temperature 0`
- **Test:** Record results in `implementation_notes.md`.

## T6: Clean up / documentation
- [ ] **Status:** Not started
- **Files:** `crates/engine/src/weights.rs` comments
- **Intent:** Update the comments in the `skip_copy` block (lines 508-573)
  to reflect the new default. Document the memory pressure rationale. Update
  `AGENTS.md` §8 if the measured `phys_footprint` baseline changes.
- **Test:** Code review.