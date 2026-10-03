# File-Backed Weight Buffers — Technical Plan

## Architecture

### Current flow (arena path — the problem)

```
shard.safetensors → mmap(MAP_PRIVATE, PROT_READ) → ShardReader
                                                        │
    ┌───────────────────────────────────────────────────┘
    │
    ├─ [skip_copy=false] arena.allocate() → device.new_buffer(SEGMENT_BYTES)
    │                                           ↓
    │                    copy_tensor() → read_at(file, …) into buffer contents
    │                                           ↓
    │                    Buffer is ANONYMOUS (internal pages) ← YELLOW PRESSURE
    │
    └─ [skip_copy=true]  no_copy_view() → newBufferWithBytesNoCopy()
                                           ↓
                         File-backed (external pages) ← GREEN PRESSURE
```

### Target flow (file-backed + mlocked)

```
shard.safetensors → mmap(MAP_PRIVATE, PROT_READ) → ShardReader
                                                        │
    ┌───────────────────────────────────────────────────┘
    │
    ├─ [needs_cpu_read OR converted, ~200 MB]
    │       arena.allocate() → device.new_buffer() → copy_tensor()
    │       (BF16 norms/scalars + CPU-read side-tensors)
    │
    └─ [everything else, ~11 GB]
            no_copy_view() → newBufferWithBytesNoCopy()
                ↓
            mlock() on ShardReader mmap regions
                ↓
            Pages are: FILE-BACKED (no pressure) + PINNED (no faults)
                ↓
            GREEN PRESSURE + ZERO TTFT PENALTY
```

## Key Insight: Why mlock + file-backed = green pressure

macOS classifies memory into:
- **Internal** (anonymous): counts toward pressure, causes yellow/red
- **External** (file-backed): does NOT count toward pressure

`mlock()` pins pages in RAM but does NOT reclassify them. A file-backed page
that is mlocked is still "external" in the VM system — it just can't be evicted.
macOS sees 11 GB of external pinned pages and says "green" because those pages
have a backing store (the file) and are not anonymous dirty pages.

This is exactly what MLX does (via its buffer pool + file-backed Metal buffers),
and why it shows green at 18 GB.

## File/Module Touchpoints

### 1. `crates/engine/src/weights.rs` — skip_copy decision (lines ~574-593)

**Current:** Only `embed_tokens` and `lm_head.trellis` set `skip_copy = true`.

**Change:** Expand `skip_copy = true` to all non-converted, non-CPU-read
packed EXL3 tensors.

```rust
// After:
let needs_cpu_read = is_cpu_read_tensor(name); // suh, svh, mcg, norms, scalars
let skip_copy = if force_resident {
    false  // KAPLUMBA_RESIDENT_WEIGHTS=1 forces old path
} else if offload_bf16_embedding {
    true
} else if converted || needs_cpu_read {
    false  // BF16 norms + CPU-read side-tensors stay in arena
} else {
    true   // ALL other EXL3 packed tensors → file-backed
};
```

CPU-read tensors (stay in arena, ~200 MB total):
- `*.suh`, `*.svh`, `*.mcg` — sign/codebook side-tensors read via `packed_u16()`
- Norm weights, `layer_scalar` — BF16, read via `read_f16()`
- (Exact list to be confirmed in T1 audit)

### 2. `crates/engine/src/weights.rs` — ShardReader: add mlock

**Change:** After all `no_copy_view()` calls complete (end of Phase 1 or start
of Phase 2), call `mlock()` on each ShardReader's full mmap range:

```rust
// After Phase 1:
for reader in shards.values() {
    let ret = unsafe { libc::mlock(reader.pointer.as_ptr().cast(), reader.length) };
    if ret != 0 {
        // Log warning but don't fail — mlock failure means pages may fault
        // but inference still works. Could hit vm.user_wire_limit.
        eprintln!(
            "warning: mlock({}) failed: {}",
            reader.path.display(),
            std::io::Error::last_os_error()
        );
    }
}
```

Also add `munlock()` in `ShardReader::drop()`:

```rust
impl Drop for ShardReader {
    fn drop(&mut self) {
        unsafe {
            libc::munlock(self.pointer.as_ptr().cast(), self.length);
            libc::munmap(self.pointer.as_ptr().cast(), self.length);
        }
    }
}
```

### 3. `crates/engine/src/model.rs` — alignment propagation

**Current:** `aligned` flag for trellis is set based on whether the tensor is
from the arena (aligned) or file-mapped (unaligned).

**Change:** Verify the flag correctly propagates for ALL file-backed layer
trellis (not just `lm_head.trellis`). The `no_copy_view()` delta `% 4 != 0`
→ `aligned = false`. Measured: 396/796 packed tensors are 2-byte misaligned.

The kernel already handles this via `ring_word<ALIGNED>` template (lines
266-276 of `linear.metal`). No kernel changes needed.

### 4. No kernel changes needed

`ring_word<false>` and `trellis_word()` already handle 2-byte-aligned data.
Implemented and battle-tested for `lm_head.trellis`.

## Sequencing and Dependencies

```
T1 (audit CPU-read tensors)
 └→ T2 (expand skip_copy)
     └→ T3 (alignment propagation)
         └→ T4 (mlock)
             └→ T5 (measurement)
                 └→ T6 (cleanup)
```

## Risks and Fallbacks

| Risk | Mitigation |
|---|---|
| `mlock` fails due to `vm.user_wire_limit` | Check limit in T4. Log warning, degrade gracefully (pages may fault but inference works). On macOS, default limit is typically generous. |
| `ALIGNED=false` trellis path slower | Measure in T5. Bandwidth-bound matvec; two 16-bit loads vs one 32-bit load is likely in noise. |
| `MAP_PRIVATE` + `PROT_READ` + `mlock` — do mlocked file pages stay external? | Yes: mlock pins pages but doesn't change their VM classification. File-backed clean pages remain external. This is the whole premise; verify in T5 with Activity Monitor. |
| `newBufferWithBytesNoCopy` + `HazardTrackingModeUntracked` for read-only weights | Safe: weights are never written by the GPU. Already proven for embed/lm_head. |
| ShardReader lifetime — buffers reference the mmap | ShardReader owned by `WeightStore`, lives for process lifetime. Already safe. |

## Requirement → Task Mapping

| Requirement | Task(s) |
|---|---|
| R1 (file-back EXL3 packed) | T1, T2, T3 |
| R2 (BF16 norms stay in arena) | T2 (skip_copy logic) |
| R3 (embed/lm_head unchanged) | Already done; T2 preserves |
| R4 (CPU-read tensors in arena) | T1 (audit), T2 (skip_copy logic) |
| R5 (mlock) | T4 |
| R6 (KAPLUMBA_RESIDENT_WEIGHTS) | T2 (force_resident path) |
| R7 (green pressure) | T5 (measure) |
| R8 (alignment) | T3 |
| R9 (tg/s no regression) | T5 (measure) |
