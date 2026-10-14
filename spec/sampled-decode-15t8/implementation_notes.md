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
- T2 Completed:
  - Enforced strict `top_k <= 1024` (`MAX_DEVICE_TOP_K`) bound across `GenerationConfig::validate`, `generate`, `generate_gpu`, `ChatSession::new`, and `ChatSession::turn`. Requests exceeding 1024 reject immediately with `SamplerError::DeviceTopKTooLarge { top_k, limit: 1024 }` with no silent clamp or CPU fallback.
  - Wired `HeadPick::Sample(params)` into `generate.rs`'s pipelined loop (`generate_gpu_greedy`). Prefill samples token 0 via host sampler and seeds the device sampler with `seed_device_sampler(state)`. Subsequent decode tokens are sampled on-device without 512 KB CPU copy bottleneck.
- T3 Completed:
  - Discovered root cause of 13.7 tok/s: commit 41f0171 added `&& gate_up_share_input_signs` and `&& qkv_share_input_signs` guards in `model.rs` because `Gxma-repacked` has distinct Hadamard sign vectors across projections. This caused all 60 layers to fall out of fused `gate_up_gelu` and fused QKV into 6 separate linear GEMVs + separate activation dispatches per layer (~12 ms overhead).
  - Added multi-input vector support (`stride_x`) across all fused kernels in `crates/kernels/src/linear_repacked.metal` (`fused2`, `fused2_2col`, `fused2_sk2`, `fused3`, `fused3_2col`, `fused3_sk2`) and `crates/kernels/src/linear.rs`.
  - Staged inputs at sequential offsets (0, `in_features * 2` bytes, etc.) inside existing `scratch.transformed` (which has 21,504 capacity, easily accommodating $3 \times 5,376 = 16,128$ and $2 \times 5,376 = 10,752$ with zero memory increase and zero allocations).
  - Restored fused `gate_up_gelu` and fused QKV paths in `crates/engine/src/model.rs` for all 60 layers.
  - Added unit test `fused2_matches_dual_independent_encodes_with_stride_x` to `crates/engine/tests/fused_bit_exact.rs`.
  - Sabotage verification: Mutated `stride_x` to 0 in `fused_bit_exact.rs`. Confirmed test failed loudly (`dual-input fused2 output B did not match reference`). Reverted sabotage.
  - Verified tests pass: `fused_bit_exact` (4/4 tests passed), `live_kernel_bit_exact` (passed), `probe_mlpdispatch_tree` (19/19 passed), `probe_dispatchlayer_census` (14/14 passed).

