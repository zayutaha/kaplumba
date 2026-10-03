# File-Backed Weight Buffers — Implementation Notes

## Pre-implementation findings

### Tensor inventory (Gemma 4 31B EXL3 ~3.0 bpw checkpoint)

| Category | Size | Allocation today | Target |
|---|---|---|---|
| EXL3 packed (`.trellis`, `.su`, `.sv`) | ~11.23 GB | Arena (anonymous) | File-backed + mlock |
| EXL3 side-tensors (`.suh`, `.svh`, `.mcg`) | ~200 MB | Arena (anonymous) | Arena (CPU-read) |
| BF16 norms/scalars | ~2.63 GB | Arena (BF16→FP16 converted) | Arena (tiny, <10 KB each) |
| `embed_tokens` | ~2.82 GB | File-backed (already) | File-backed + mlock |
| `lm_head.trellis` | ~1.06 GB | File-backed (already) | File-backed + mlock |

### Alignment analysis

396 of 796 packed tensors have absolute file offsets that are 2-byte aligned
but NOT 4-byte aligned (`abs_offset % 4 == 2`). The `no_copy_view()` page-aligns
downward, so the sub-page delta inherits this misalignment (`delta % 4 == 2`).

The kernel handles this via the `ALIGNED` template parameter:
- `ring_word<true>`: one 32-bit load (`uint *` cast) — requires 4-byte alignment
- `ring_word<false>`: two 16-bit loads (`trellis_word()`) — works at 2-byte alignment

This was already implemented and tested for `lm_head.trellis`
(`crates/kernels/src/linear.metal` lines 266-276). The two-load path is
expected to be slightly slower but within noise for the decode matvec
(bandwidth-bound, not compute-bound).

### macOS memory pressure classification

- **Internal pages** (anonymous): `new_buffer()`, `malloc()` → counts toward pressure
- **External pages** (file-backed, clean): mmap + no writes → does NOT count
- `newBufferWithBytesNoCopy()` wrapping `PROT_READ/MAP_PRIVATE` mmap → external
- `mlock()` pins pages but does NOT reclassify them → still external
- KV cache (`new_buffer()` + GPU writes) → internal, must stay anonymous
- Scratch buffers → internal, but small (~100 MB)

### Why mlock + file-backed = green pressure

macOS pressure is driven by how much "internal" memory exists. `mlock()` pins
pages in physical RAM but does not change their classification from external to
internal. A file-backed page that is mlocked:
- Has a backing store (the file) → macOS classifies it as external
- Cannot be evicted → stays resident, no page faults
- Does NOT count toward memory pressure → green

This is the same mechanism MLX exploits: file-backed Metal buffers for weights.
MLX doesn't need mlock because its lazy evaluation means pages fault naturally
and stay resident in the file cache. We need mlock because we want guaranteed
zero faults from the first token.

### mlock limits on macOS

macOS has a per-process wired memory limit. Check with:
```bash
sysctl vm.user_wire_limit
```
On a 24 GB M4 Pro this is typically ~18 GB (75% of RAM). Our ~15 GB of shard
mmaps should fit. If not, mlock failure is non-fatal — pages will demand-fault
from the file cache instead.