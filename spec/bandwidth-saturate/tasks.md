# Tasks: Bandwidth Saturation (≥ 20.0 t/s Pure Decode)

- [x] **T1: Vectorized EXL3 Inner Loop in Metal (`exl3_decode_quad` + `dot`)**
  - **Files**: `crates/kernels/src/exl3.metal`
  - **Intent**: Replace scalar decode calls and serial FMA additions in `exl3_gemv_post`, `exl3_gemv_post_fused2`, and `exl3_gemv_post_fused3` with `uint4` vector arithmetic and `dot(float4, float4)`.
  - **Testing**: Run `cargo test -p kaplumba-kernels` to verify bit-exact outputs across all 32 specialization tests.

- [x] **T2: Single-Linear and Fused GEMV Loop Vectorization with Bit-Exact Parity**
  - **Files**: `crates/kernels/src/exl3.metal`
  - **Intent**: Apply `exl3_decode_quad` and `exl3_decode_quad_const` vectorization to `exl3_gemv_spec`, `exl3_gemv_post`, `exl3_gemv_post_fused2`, and `exl3_gemv_post_fused3` using pairwise summation `(x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3)` to achieve 4x ALU reduction while preserving 100% bit-exact parity.
  - **Testing**: Run `cargo test -p kaplumba-engine --lib` (all 131 tests pass, including `gpu_gemv_post_matches_unfused` and `trace_chat_prompt_steps_slow`).

- [x] **T3: Single-Dispatch GEMV for O and Down (`gemv_splits(336) = 1`)**
  - **Files**: `crates/kernels/src/exl3.rs`
  - **Intent**: Change `gemv_splits(336)` from `8` to `1` so `o_proj` and `down_proj` execute direct `exl3_gemv_post` in a single dispatch instead of K-splits with `gemv_reduce_post` and partials buffer.
  - **Testing**: `gpu_gemv_post_matches_unfused` and `gemma4::tests::trace_chat_prompt_steps_slow` to verify token 818 exact match. All 131 tests pass.

- [ ] **T4: Fused RMSNorm + Hadamard Prologue (`exl3_rms_had_in_batched`)**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`, `crates/engine/src/gpu_decode.rs`
  - **Intent**: Fuse `input_layernorm` and `pre_feedforward_layernorm` directly into `exl3_had_in_batched`, writing `xh_stack` directly without intermediate `xn` DRAM buffer.
  - **Testing**: Unit test verifying `rms_had_in_batched(x, ln, suh)` produces bit-identical result to `had_in_batched(rms(x, ln), suh)`.

- [x] **T5: Fused GELU + Hadamard Prologue (`exl3_gelu_had_in`)**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`, `crates/engine/src/gpu_decode.rs`
  - **Intent**: Fuse `gelu_mul(gate, up)` into `had_in` prologue for `down_proj`, eliminating standalone `gelu_mul` dispatch and `gu` buffer.
  - **Testing**: `gpu_gelu_had_in_matches_separate_passes` passes 100% bit-exact; `trace_chat_prompt_steps_slow` predicts golden token 818; decode throughput reaches 8.53 t/s (117.3 ms).

- [ ] **T6: Fused Attention Post-Processing (`exl3_qkv_prep`)**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`, `crates/engine/src/gpu_decode.rs`
  - **Intent**: Fuse Q/K RMS norm, RoPE rotation, and V cache write into unified `qkv_prep` kernel, cutting 5 dispatches per layer.
  - **Testing**: Compare KV cache and Q output against separate RMS/RoPE/Copy passes.

- [ ] **T7: End-to-End Throughput Verification (≥ 20.0 t/s)**
  - **Files**: `crates/engine/src/gemma4.rs`
  - **Intent**: Run `decode_throughput_slow` and assert forward decode throughput ≥ 20.0 t/s (≤ 50.0 ms/token).
  - **Testing**: Verify `trace_chat_prompt_steps_slow` predicts token 818 ("The") bit-identically.
