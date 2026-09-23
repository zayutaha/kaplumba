# Tasks: Pure Decode 273 GB/s Bandwidth Saturation (≥ 20 t/s)

- [ ] **T1: Fused Post-Hadamard GEMV Kernel (`exl3_gemv_post`)**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`
  - **Intent**: Implement single-linear GEMV with fused in-threadgroup 128-column Hadamard butterfly + `svh` epilogue, outputting final `f16` directly.
  - **Testing**: Kernel unit test `gpu_gemv_post_matches_unfused` against isolated `had_out(gemv(xh))`. Verify exact Ulp agreement.

- [x] **T4: Batched Multi-Row Pre-Hadamard (`exl3_had_in_batched`)**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`
  - **Intent**: Batch multi-row `had_in` transforms into a single dispatch (`grid.y = 3` for QKV, `grid.y = 2` for Gate/Up), cutting separate `had_in` dispatches.
  - **Testing**: Unit test `gpu_had_in_batched_matches_separate_passes` verifying row 0, 1, 2 match isolated `had_in` passes. Passed.

- [x] **T2: Fused 2-Way Sibling GEMV (`exl3_gemv_fused2` for Gate + Up)**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`
  - **Intent**: Monolithic 2,688 out-tile kernel dispatch combining Gate (1344) and Up (1344), reading directly from their respective trellis spans via pointer table.
  - **Testing**: Kernel test `gpu_gemv_fused2_matches_separate_passes` verifying output matches separate gate and up GEMVs. Passed.

- [x] **T3: Fused 3-Way Sibling GEMV (`exl3_gemv_fused3` for Q + K + V)**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`
  - **Intent**: Monolithic 1,024 out-tile kernel dispatch combining Q (512), K (256), and V (256), expanding `k`/`v` grid size by 4x.
  - **Testing**: Kernel test `gpu_gemv_fused3_matches_separate_passes` verifying output matches separate Q, K, V GEMVs. Passed.

- [x] **T5: Engine Integration of Fused Handles (`Fused2Handles` & `Fused3Handles`)**
  - **Files**: `crates/engine/src/metal_linear.rs`, `crates/kernels/src/exl3.rs`
  - **Intent**: Construct zero-copy fused linear handles holding device spans to member trellises, `suh` stacks, and `svh` concatenations.
  - **Testing**: Engine unit test `gpu_encode_fused2_and_fused3_matches_separate_linears` verifying bit-for-bit equivalence. Passed.

- [x] **T6: Full Token Pipeline Integration in `gpu_decode.rs`**
  - **Files**: `crates/engine/src/gpu_decode.rs`, `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`
  - **Intent**: Replace sequential dispatches with monolithic fused calls in `encode_token`, reducing dispatches from ~36/layer to ~14/layer.
  - **Testing**: Fixed loop variable shadowing bug in `exl3.metal`. All 143 tests passed. `trace_chat_prompt_steps_slow` verified 20 steps bit-for-bit to token 818 ("The"). Passed.

- [ ] **T7: End-to-End Throughput Gate Verification (≥ 19.5 t/s, Target ≥ 20.0 t/s)**
  - **Files**: `crates/engine/src/gemma4.rs`
  - **Intent**: Benchmark forward-only decode throughput in `decode_throughput_slow`, verify effective bandwidth ≥ 236.7 GB/s, and flip assert to ≥ 19.5 t/s.
  - **Testing**: Gated test run with zero RSS/swap growth.
