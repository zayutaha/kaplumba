# Implementation Notes: Sampled Decode 15.8 tok/s

## 2026-10-14 — Task Initiation

- Initial state: Canonical decode benchmark on Gxma-repacked is ~12.44 tok/s.
- Root cause identified:
  1. Commit `eb9fd02` / `2080a86` introduced 4-trip unrolling and `decode_and_accum_4tile_2col` into `linear_repacked.metal`, causing register bloat and dropping memory bandwidth from >200 GB/s to 116–176 GB/s.
  2. `generate.rs` bypassed pipelined execution whenever `temperature > 0.0`, falling back to synchronous `forward_buffer_into_vec` and host CPU float conversions.
- Target: $\ge 15.8\text{ tok/s}$ at `temperature = 1.0, top_k = 64, top_p = 0.95`.
