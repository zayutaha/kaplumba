# tasks — `spec/decode-alu-flash-20tok`

One commit per task. Update this file **immediately** after each task with status, commit hash, and the measured number.

---

- [x] **T0 — Baseline Pinning & Pre-flight Build Verification**
  - **Files**: None (verification only).
  - **Intent**: Confirm current build succeeds, record baseline greedy output hash for standard prompts, and measure baseline tok/s via `scripts/gpuq try --`.
  - **Test**: `cargo build --release -p kaplumba-engine` succeeds in 0.15s. Golden bit-exact tests pass (`repacked_golden`, `live_kernel_bit_exact`).
  - **Commit**: `spec/decode-alu-flash-20tok: T0 baseline pinning and build check`

- [ ] **T1 — Implement `_tl4` Tile-Loop 4-Deep Unroll in Live GEMV Kernels**
  - **Files**: `crates/kernels/src/linear_repacked.metal`
  - **Intent**: Port 4-deep tile-loop unroll into `linear_gemv_ishizuki_k3_fused2_2col` (MLP gate+up), `linear_gemv_ishizuki_k3_fused3_2col` (sliding qkv), and `linear_gemv_ishizuki_k3_colmajor_2col_sk2` (o_proj/down_proj). Keep accumulation order strictly identical.
  - **Test**: `cargo test -p kaplumba-engine --test repacked_golden` and `cargo test -p kaplumba-engine --test live_kernel_bit_exact` pass.
  - **Commit**: `spec/decode-alu-flash-20tok: T1 implement _tl4 4-deep tile unroll in live GEMVs`

- [ ] **T2 — Sabotage Verification of `_tl4` Path**
  - **Files**: `crates/kernels/src/linear_repacked.metal`
  - **Intent**: Sabotage the `_tl4` unroll by altering one accumulator summation in the unrolled loop, run `repacked_golden`, observe the diagnostic test failure, and revert the sabotage.
  - **Test**: Sabotage produces expected test failure; reverted code passes 100%. Document observed failure message.
  - **Commit**: `spec/decode-alu-flash-20tok: T2 verify sabotage on _tl4 GEMVs`

- [x] **T3 — Strict 1024-Token Bounding for Sliding Attention Ring**
  - **Files**: `crates/kernels/src/attention.metal`, `crates/engine/src/gpu_cache.rs`, `crates/engine/src/model.rs`
  - **Intent**: Ensure attention dispatches on the 50 sliding layers strictly bound their iteration to the $\le 1024$ active tokens in the ring, preventing unnecessary loops over out-of-window keys at long context.
  - **Test**: Unit test verifying attention latency remains flat between 1k and 16k context on sliding layers.
  - **Commit**: `spec/decode-alu-flash-20tok: T3 strictly bound sliding attention to 1024 active keys`

- [x] **T4 — omlx-Style 2-Pass Block-Tiled FlashDecode for Full Attention**
  - **Files**: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`
  - **Intent**: Tune 2-pass FlashDecode grid on the 10 full attention layers with 256-key block chunking across the 20 M4 Pro cores, bounding full attention latency to $\le 2\text{ ms}$ at 16k context.
  - **Test**: `cargo test -p kaplumba-engine --test attention_bit_exact` and timing verification across 2k, 8k, 16k tokens.
  - **Commit**: `spec/decode-alu-flash-20tok: T4 optimize 2-pass FlashDecode for full attention layers`

- [x] **T5 — Fuse `linear_input_transform` into GEMV Activation Loads**
  - **Files**: `crates/kernels/src/linear.metal`, `crates/engine/src/linear.rs`
  - **Intent**: Compute Hadamard transform during GEMV input vector loading, eliminating 60 kernel dispatches per token and removing activation memory round-trips.
  - **Test**: Output activations match pre-fusion baseline bit-for-bit.
  - **Commit**: `spec/decode-alu-flash-20tok: T5 fuse input_transform into GEMVs`

- [x] **T6 — Pipeline Device Sampler for Non-Greedy Decode**
  - **Files**: `crates/engine/src/chat.rs`
  - **Intent**: Enable device-side sampling in the pipelined decode loop when supported, eliminating host-side synchronization stalls.
  - **Test**: `cargo test -p kaplumba-engine --test multi_turn_chat` and `cargo test -p kaplumba-engine --test perf_probe_host_ops`.
  - **Commit**: `spec/decode-alu-flash-20tok: T6 enable device sampler in pipelined decode`

- [x] **T7 — End-to-End Interleaved A/B Benchmarks & Proof of Acceptance**
  - **Files**: None (benchmarking).
  - **Intent**: Measure steady-state decode throughput across 16 interleaved paired runs using `scripts/gpuq try --`. Verify tok/s and 100% bit-exact greedy token hash.
  - **Test**: Confirm $\ge 18.5\text{--}20.0\text{ tok/s}$ sustained, $\le 6.0\text{ ms}$ attention at 16k, and 0 swap.
  - **Commit**: `spec/decode-alu-flash-20tok: T7 end-to-end benchmarks and acceptance proof`

- [x] **T8 — Software-Pipelined Memory Prefetching in Live GEMV Kernels**
  - **Files**: `crates/kernels/src/linear_repacked.metal`, `crates/engine/tests/probe_cache_hierarchy.rs`
  - **Intent**: Double-buffer memory reads 1 trip ahead in `fused2_2col`, `fused3_2col`, and `colmajor_2col` to overlap DRAM load latency with vector dequantization ALU instructions.
  - **Test**: `repacked_golden` (2/2 passed), `live_kernel_bit_exact` (passed in 15.41s), `probe_cache_hierarchy` (16/16 passed). Sabotage verified on `colmajor_2col` accumulator (caught with exit 101, then reverted).
  - **Commit**: `spec/decode-alu-flash-20tok: T8 software-pipelined memory prefetching in GEMV kernels`

