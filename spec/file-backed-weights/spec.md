# File-Backed Weight Buffers

## Goal

Eliminate macOS yellow memory pressure during inference by converting the
~11.3 GB of EXL3 packed layer weights from anonymous Metal buffers
(`device.new_buffer()` + data copy) to file-backed Metal buffers
(`newBufferWithBytesNoCopy()` wrapping the mmap'd safetensors shard files),
pinned with `mlock()`.

macOS memory pressure is driven by **internal** (anonymous/dirty) pages.
File-backed pages are **external** — they don't count toward pressure even when
pinned via `mlock()`. Today the engine's ~11.3 GB arena is anonymous memory;
converting it to file-backed + mlocked memory should drop internal footprint
from ~12.5 GB to ~1.5 GB (KV cache + scratch + small side-tensors), eliminating
the yellow indicator while keeping all weight pages resident (no page faults,
no TTFT penalty).

## Non-goals / Restrictions

1. **No quality loss** — output must be bit-identical at `--temperature 0`.
2. **No KV cache changes** — KV buffers remain anonymous (`new_buffer()`). They
   are written by kernels and cannot be file-backed.
3. **No format changes** — safetensors files remain as-is. No on-disk
   conversion.
4. **No Python** — all work is Rust + Metal.
5. **No multi-user / batching concerns.**
6. **Do not break existing `KAPLUMBA_ZERO_COPY_WEIGHTS` or
   `KAPLUMBA_RESIDENT_WEIGHTS` env-var semantics** — but may deprecate/subsume
   them.
7. **Decode tg/s must not regress** — file-backing is not allowed to hurt
   throughput. The arena path remains available as a fallback.

## Requirements

1. **R1**: All EXL3 packed layer tensors (`.trellis`, `.mcg`, `.su`, `.suh`,
   `.sv`, `.svh`) that are not BF16-converted are loaded via
   `newBufferWithBytesNoCopy()` wrapping the existing `ShardReader` mmap,
   instead of `arena.allocate()` + `copy_tensor()`.

2. **R2**: The ~2.63 GB of BF16 language-model tensors (`input_layernorm`,
   `post_attention_layernorm`, `pre_feedforward_layernorm`,
   `post_feedforward_layernorm`, `k_norm`, `q_norm`, `layer_scalar`,
   `model.language_model.norm`) are small (< 10 KB each). They stay in the
   arena (need BF16 → FP16 conversion and CPU-side `read_f16()` access).

3. **R3**: The `embed_tokens` and `lm_head.trellis` remain file-backed as today
   (already done).

4. **R4**: Side-tensors read on the CPU via `read_f16()` / `packed_u16()`
   (`lm_head.suh`, `lm_head.svh`, `lm_head.mcg`, and per-layer `.suh`, `.svh`,
   `.mcg`) — only the ones that need CPU reads stay in the arena. Ones only read
   by the GPU can be file-backed.

5. **R5**: After all file-backed buffers are created, `mlock()` is called on
   each `ShardReader`'s mapped region to pin all weight pages in RAM. This
   ensures no page faults during inference and no TTFT penalty from demand
   paging. The pages remain classified as file-backed (external) by macOS's VM
   subsystem, so they do not contribute to memory pressure.

6. **R6**: `KAPLUMBA_RESIDENT_WEIGHTS=1` forces the old arena path for all
   weights (escape hatch / A/B comparison).

7. **R7**: Activity Monitor shows **green** memory pressure (not yellow) during
   inference on a 24 GB M4 Pro, with `phys_footprint` under 14 GB.

8. **R8**: Kernel alignment handling: the `ALIGNED` template parameter in
   `exl3_decode8`/`ring_word` must be set to `false` for file-backed trellis
   buffers (half the tensors are 2-byte misaligned in the safetensors). This
   already works for `lm_head.trellis`; extend to all file-backed trellis.

9. **R9**: TTFT should improve or stay the same (we skip the ~11 GB parallel
   copy; mlock pre-faults the pages). Decode tg/s should not regress.

## Acceptance Criteria

- **AC1**: `KAPLUMBA_RESIDENT_WEIGHTS=0` (default) produces bit-identical
  output vs `KAPLUMBA_RESIDENT_WEIGHTS=1` at `--temperature 0` for a fixed
  prompt.
- **AC2**: Activity Monitor shows green pressure during a 2500-token decode.
- **AC3**: `phys_footprint` (via `/usr/bin/time -l`) is under 14 GB.
  Target: ~4-5 GB (KV + scratch + small tensors), down from ~12.5 GB.
- **AC4**: Decode tg/s is within measurement spread of the baseline (A/B
  interleaved, medians with spread).
- **AC5**: TTFT improves (loading skips ~11 GB copy; mlock pre-faults pages
  faster than parallel `read_at`).
- **AC6**: `vm.swapusage` shows 0 swap during a 2500-token decode.
- **AC7**: Existing tests pass (`cargo test`).

## Open Questions

- **OQ1**: Which `.suh`/`.svh`/`.mcg` tensors are GPU-only vs CPU-read? Need to
  audit `read_f16()` / `packed_u16()` call sites. If some are GPU-only, they can
  be file-backed; if all are CPU-read, they must stay in the arena (~150 MB
  total, negligible).
- **OQ2**: Is the `ALIGNED=false` (two 16-bit loads) trellis path measurably
  slower than `ALIGNED=true` (one 32-bit load)? If yes, quantify. If it's
  within noise, file-back all trellis; if it's significant, the savings still
  vastly outweigh the cost (11 GB of anonymous → file-backed for a
  few-percent kernel slowdown).
- **OQ3**: `mlock()` on macOS has a per-process limit (`sysctl vm.user_wire_limit`).
  Verify this limit is large enough for ~15 GB of shard mmaps. If not,
  the process may need elevated privileges or the limit may need adjusting.
