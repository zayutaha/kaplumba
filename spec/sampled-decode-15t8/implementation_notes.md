# Implementation Notes: Sampled Decode 15.8 tok/s

## 2026-10-14 — Task Initiation

- Initial state: Canonical decode benchmark on Gxma-repacked is ~12.44 tok/s.
- Root cause identified:
  1. Commit `eb9fd02` / `2080a86` introduced 4-trip unrolling and `decode_and_accum_4tile_2col` into `linear_repacked.metal`, causing register bloat and dropping memory bandwidth from >200 GB/s to 116–176 GB/s.
  2. `generate.rs` bypassed pipelined execution whenever `temperature > 0.0`, falling back to synchronous `forward_buffer_into_vec` and host CPU float conversions.
- Target: $\ge 15.8\text{ tok/s}$ at `temperature = 1.0, top_k = 64, top_p = 0.95`.
- T1 Completed:
  - In `crates/kernels/src/linear_repacked.metal`, restored lean 2-trip burst loops in `linear_gemv_ishizuki_k3_colmajor_2col`, `linear_gemv_ishizuki_k3_colmajor_2col_sk2`, `linear_gemv_ishizuki_k3_fused2_2col`, and `linear_gemv_ishizuki_k3_fused3_2col`. Reused `float v[8]` array to maintain register count within ~28 registers. Removed unused bloated `decode_and_accum_4tile_2col`.
  - Sabotage verification: Injected `+ 1.0f` offset into accumulator of `linear_gemv_ishizuki_k3_colmajor_2col`. Confirmed `live_kernel_bit_exact` failed with 22708 mismatches. Reverted sabotage.
  - Regression verification: `live_kernel_bit_exact` (passed), `repacked_golden` (passed), and `fused_bit_exact` (passed, 3/3 tests).
