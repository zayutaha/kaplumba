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

---

## P6 — decode bandwidth attack (2026-09-24)

Status of every lever investigated to lift decode above the 8.55 t/s baseline.

- [x] **P6-T1: LUT-decode A/B (`FC_LUT` + device codebook)**
  - Kernel-side f32-codebook LUT for `exl3_decode_f`/`_const` (FC-gated), `decode_table`/`codebook_buffer`/`set_decode_lut` plumbing, engine passthrough.
  - Fast gate `gpu_decode_lut_matches_alu` proved LUT bit-identical to ALU on GPU for all k/modes/alignments + end-to-end encode paths; gated A/B `bench_decode_lut_slow` on the real artifact (release, min-of-12) measured **LUT LOSES 1.6–3.1×**: down +59.7%, o +92.9%, lm +211.4%, gu2 +165.0%, qkv3 +129.0%.
  - **DEAD-END**: abandoned (L2 gather cost ≫ the ~5 ALU/decode it removes; M4 integer ALU is NOT the bottleneck). Reverted all three files; dead-end recorded in `implementation_notes.md`; commit `e0a93f9`.

- [x] **P6-T2: Fresh per-linear isolated bandwidth microbench (`bench_linear_bandwidth_slow`, engine, gated)**
  - Measured (release, warm, min-of-20; **real** trellis GB/s): gate 81.0 / up 80.3 / down 81.8 / o 57.3 / q 54.6 / k 37.9 / v 36.9 / lm **227.8** / gu2 94.7 / qkv3 72.2.
  - `bench_gemv_split_sweep_slow` re-run: policy argmaxes confirmed (down→8-16, o→8, q→2, k/v→8, lm→1); spec≈generic (≤5%).

- [x] **P6-T3: 4-way unroll in `exl3_gemv_spec` k≠7 loop**
  - Widened 2-way→4-way in-tile unroll (more LDG in flight/warp). Bit-exact (fast gates green). **Measured flat-to-worse**: down 81.7 (was 81.8), o 53.9 (was 57.3), k 35.9 (37.9), qkv3 69.3 (72.2). MDL already pipelines loads. **DEAD-END**: reverted.

- [ ] **P6-T4 (proposed): k=3 format-density change** — the one untested lever. lm (k=6) hits 228 GB/s; every k=3 linear plateaus at 37–95 GB/s. Per-byte decode-ALU for k=3 is ~2× k=6; geometry/nsplit/LUT/unroll all exhausted. Would require a repack of the on-disk trellis (converter + layout + memory), i.e. a new spec; not started (needs user decision).

- [ ] **T7**: still open — baseline 117.0 ms/token (8.55 t/s), target 50 ms/token requires ~239 GB/s in-context per layer (current 1.837 ms/layer ≈ 97.6 GB/s).
