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

- [x] **T4: Fused RMSNorm + Hadamard Prologue (`exl3_rms_had_in_batched`)**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`, `crates/engine/src/gpu_decode.rs`
  - **Intent**: Fuse `input_layernorm` into `exl3_rms_had_in_batched`, writing `xh_stack` directly without intermediate `xn` DRAM buffer and eliminating 60 separate RMSNorm dispatches per token.
  - **Testing**: `gpu_rms_had_in_batched_matches_separate_passes` green bit-exact; `trace_chat_prompt_steps_slow` predicts golden token 818; decode throughput reaches 8.66 t/s (115.5 ms, 97.0 GB/s). All 135 unit tests pass.

- [x] **T5: Fused GELU + Hadamard Prologue (`exl3_gelu_had_in`)**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`, `crates/engine/src/gpu_decode.rs`
  - **Intent**: Fuse `gelu_mul(gate, up)` into `had_in` prologue for `down_proj`, eliminating standalone `gelu_mul` dispatch and `gu` buffer.
  - **Testing**: `gpu_gelu_had_in_matches_separate_passes` passes 100% bit-exact; `trace_chat_prompt_steps_slow` predicts golden token 818; decode throughput reaches 8.53 t/s (117.3 ms).

- [x] **T5.1: Fused Gate+Up GEMV with RMSNorm Prologue (`encode_rms_fused2_with_scratch`)**
  - **Files**: `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`, `crates/engine/src/gpu_decode.rs`
  - **Intent**: Fuse `pre_ff_ln` RMSNorm into Gate+Up GEMV via `encode_rms_fused2_with_scratch` and update `encode_gelu_had_in_down` to read `gate` and `up` slices directly from `gu_raw`, eliminating intermediate `xn` write and separate RMSNorm dispatch per layer (60 dispatches/token eliminated).
  - **Testing**: `gpu_gelu_had_in_matches_separate_passes` and all 135 unit tests pass; `trace_chat_prompt_steps_slow` predicts golden token 818; decode forward-only latency reaches 115.0 ms/token (8.70 t/s, 97.4 GB/s).

- [x] **T5.2: Coalesced 64-Byte SIMD Store for GEMV (`exl3_gemv`, `exl3_gemv_spec`, fused)**
  - **Files**: `crates/kernels/src/exl3.metal`
  - **Intent**: Replace the 8-lane strided 2-float stores in `exl3_gemv`, `exl3_gemv_spec`, `exl3_gemv_fused2`, and `exl3_gemv_fused3` with a 16-lane contiguous 64-byte burst write via `simd_shuffle`, eliminating uncoalesced memory transactions across all GEMV partial dispatches.
  - **Testing**: All 21 kernel tests pass; all 135 engine unit tests pass; `trace_chat_prompt_steps_slow` predicts golden token 818 bit-identically.

- [x] **T6: Fused Attention Post-Processing (`kr_rms_rope` + Direct Cache Write)**
  - **Files**: `crates/kernels/src/fuse.metal`, `crates/kernels/src/fuse.rs`, `crates/engine/src/metal_linear.rs`, `crates/engine/src/gpu_decode.rs`
  - **Intent**: Fuse per-head RMSNorm + Neox RoPE in registers via `kr_rms_rope` for Q and K, write K directly to KV cache, and normalize V directly into KV cache (eliminating `kr_copy`). Cuts 3 dispatches per layer (180 dispatches/token eliminated) and removes intermediate DRAM round-trips for Q/K/V.
  - **Testing**: `rms_rope_matches_separate_passes` bit-exact across all head dimensions, positions, and weighted/unweighted modes; all 22 kernel tests and 135 engine unit tests pass; `trace_chat_prompt_steps_slow` predicts golden token 818 (6.57s); forward-only decode latency drops to 114.1 ms/token (8.77 t/s, 98.2 GB/s).

- [x] **T6.1: Fused Residual Additions (`kr_rms_add` + `kr_rms_add_scale`)**
  - **Files**: `crates/kernels/src/fuse.metal`, `crates/kernels/src/fuse.rs`, `crates/engine/src/metal_linear.rs`, `crates/engine/src/gpu_decode.rs`
  - **Intent**: Fuse post-attention RMSNorm + residual addition (`kr_rms_add`) and post-feedforward RMSNorm + scaled residual addition (`kr_rms_add_scale`), eliminating 120 separate elementwise add launches per token and intermediate DRAM round-trips.
  - **Testing**: `rms_add_and_scale_matches_separate_passes` verified bit-exact; all 23 kernel tests and 135 engine unit tests pass; `trace_chat_prompt_steps_slow` drops to 6.00s predicting golden token 818; decode forward-only latency reaches 115.3 ms/token (8.67 t/s, 97.1 GB/s).

- [x] **T6.2: FC-Specialization and `exl3_decode_quad_const` for `exl3_gemv_fused2` and `exl3_gemv_fused3`**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`
  - **Intent**: Specialize `exl3_gemv_fused2` (Gate+Up) and `exl3_gemv_fused3` (QKV) with function constants `FC_ALIGNED`, `FC_MODE`, `FC_K` and vectorized `exl3_decode_quad_const`, eliminating 16 scalar decode function calls per iteration and runtime branch/bit-shift overhead across 60 layers (responsible for 61% of token decode latency).
  - **Testing**: `gpu_gemv_fused2_matches_separate_passes`, `gpu_gemv_fused3_matches_separate_passes`, and `gpu_encode_fused2_and_fused3_matches_separate_linears` bit-exact; all 23 kernel tests and 135 engine unit tests pass; `trace_chat_prompt_steps_slow` predicts golden token 818 in 5.76s.



- [ ] **T7: End-to-End Throughput Verification (≥ 20.0 t/s)** — **BAR KNOWN UNREACHABLE AT k=3 (2026-09-24)**
  - **Files**: `crates/engine/src/gemma4.rs`
  - **Was**: Run `decode_throughput_slow` and assert forward decode throughput ≥ 20.0 t/s (≤ 50.0 ms/token).
  - **Status**: 20.0 t/s needs 236.7 GB/s sustained and 19.5 t/s needs 242.6 GB/s, but a pure linear read on this M4 Pro measured **210 GB/s** (`bench_stream_bw_slow`). At 11.833 GB/token that caps k=3 decode near **17.8 t/s**. The assertion therefore stays **unflipped** and is expected to stay red. Bar kept by explicit user decision.
  - **Testing**: `trace_chat_prompt_steps_slow` must still predict token 818 ("The") bit-identically at every commit.
