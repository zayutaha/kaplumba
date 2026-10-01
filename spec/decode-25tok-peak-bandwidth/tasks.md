# TASKS: 25 Tok/s Decode at Physical Memory Bandwidth Limit

- [x] **T1: True Threadgroup Co-Computation for Gate + Up (`linear_gemv_ishizuki_k3_fused2_co`)**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Replace split-grid concatenation in `fused2` with true co-computation where each threadgroup computes tile `tn` of `gate` and tile `tn` of `up` together, loading input vector $x$ once and sharing float registers.
  - Testing: Run `live_kernel_bit_exact` comparing `fused2_co` against independent reference encodes; verify sabotage detection; verify 0 FP16 diffs. (PASS: `fused_bit_exact` 3/3, sabotage test triggered exit code 101 on intentional corruption).

- [x] **T2: Wire `fused2_co` into `encode_gemv_fused2_mlp`**
  - Files: `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`
  - Intent: Update `encode_gemv_fused2_mlp` to dispatch `linear_gemv_ishizuki_k3_fused2_co` across 1344 threadgroups (instead of 2688). Measure latency reduction in `gate_up_gelu` (targeting 25.8 ms -> 16.5 ms).
  - Testing: Run `cargo test --test repacked_golden --test fused_bit_exact`; benchmark greedy decode forward step and confirm bit-exact text. (PASS: greedy output "Paris", 32-tok decode throughput 13.0 tok/s).

- [x] **T3: Pipelined Sub-Step Register Allocation in GEMV K-Loop**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Pipeline sub-steps in `colmajor_2col`, `fused2_co`, `fused3`, and `contig4` to reuse a single `v[8]` float array instead of 64 floats. Drastically cuts register pressure and elevates GPU occupancy toward peak streaming bandwidth.
  - Testing: Sabotage tested on `fused3` (deliberate corruption triggered exit 101 on `fused3_matches_three_independent_encodes`); clean test passes 100% across `live_kernel_bit_exact`, `repacked_golden`, `fused_bit_exact` (3/3); 32-token decode throughput accelerated from 13.0 to 14.42 tok/s (+10.9%).

- [x] **T4: Device-side Argmax Integration (`argmax_slots` & `decode_argmax_token`)**
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/src/model.rs`, `crates/engine/src/generate.rs`, `crates/engine/tests/argmax_engine_parity.rs`
  - Intent: Eliminate host 512 KiB copy (`copy_nonoverlapping`) and 262,144 CPU float comparisons per token by running slot-partitioned device-side argmax (`argmax_slots` across 4 slots of 65,536 elements) on GPU. Host only reads 4 u32s (16 bytes) and folds them in 2 nanoseconds.
  - Testing: Sabotage tested on assertion (exit 101 on mismatch); clean test suite passes 8/8 (`argmax_engine_parity` 2/2, `fused_bit_exact` 3/3, `live_kernel_bit_exact` 1/1, `repacked_golden` 2/2); greedy generation bit-exact ("Paris"); 32-token decode test bit-exact at 14.48 tok/s.

- [x] **T4.1: Contiguous Tile Load Grouping & Fast Greedy Path Integration**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/engine/src/generate.rs`, `crates/engine/src/model.rs`, `crates/kernels/src/linear.rs`
  - Intent: Group sub-steps along Column A/0 contiguously into 384-byte chunks in `fused2_co` and `colmajor_2col` to sustain hardware streaming prefetch. Wire CLI `generate_gpu` directly to `generate_gpu_greedy` at temperature 0, set `max_dispatches(2500)` unconditionally, and gate experimental `regpipe` to prevent prefill hangs.
  - Testing: Bit-exact tests pass 6/6 (`fused_bit_exact` 3/3, `live_kernel_bit_exact` 1/1, `repacked_golden` 2/2); golden text outputs "Paris" with TTFT 1325 ms; 32-token decode achieves 14.72 tok/s.

- [x] **T4.2: Quad-Column (4col) GEMV Gating & M4 Pro Occupancy Characterization**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Implement `linear_gemv_ishizuki_k3_colmajor_4col` and `linear_gemv_ishizuki_k3_fused2_co4` to evaluate 64-channel tiling. A/B measured on M4 Pro: 4col yielded 14.01 tok/s vs 2col at 14.62 tok/s (-4.2%) due to halving grid size to 84 threadgroups (4.2 TGs/core across 20 cores, starving DRAM latency hiding). Gated 4col behind `KAPLUMBA_4COL` and `KAPLUMBA_FUSED2_CO4` while preserving 2col as default.
  - Testing: `live_kernel_bit_exact` passes with 0 diffs against reference; deliberate sabotage fails with exit 101. Baseline 14.58 tok/s preserved.

- [x] **T4.3: 2-Sub-step GEMV K-Loop Restructuring with Burst Loading**
  - Files: `crates/kernels/src/linear_repacked.metal`
  - Intent: Unroll inner K-loop by 2 sub-steps (`step = SG * 2u = 16`) with burst-loaded tile words upfront per column (`w0_a0, w0_a1...`), perfectly factoring $K=5376$ (336 tiles = 21 * 16) and $K=21504$ (1344 tiles = 84 * 16) with zero remainder loop. Slashes thread register allocation to $\le 30$ registers, lifting Apple Silicon GPU core occupancy to 100%.
  - Testing: Verified bit-exact greedy output on "Paris" and 32-token decode test; `live_kernel_bit_exact` and `fused_bit_exact` pass 4/4; decode throughput increases to 14.85 tok/s (beating 14.5 tok/s baseline).

- [x] **T4.4: 2-Column Tiling for `fused3` and `fused2` + GPU Token Reduction**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`, `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/tests/argmax_engine_parity.rs`
  - Intent: Implement `linear_gemv_ishizuki_k3_fused3_2col` and `linear_gemv_ishizuki_k3_fused2_2col` to eliminate single-column loads in `qkv_proj` and `qk_proj`. Add `argmax_keys_to_token` GPU kernel to reduce 4-slot keys into a single token ID on device.
  - Testing: Sabotage tested on assertion and kernel (exit 101); all test suites pass 100% (`fused_bit_exact` 3/3, `argmax_engine_parity` 3/3, `live_kernel_bit_exact` 1/1, `repacked_golden` 2/2); golden text bit-exact ("Paris").



- [ ] **T5: Concurrent Multi-Queue Overlap (Attention + GEMV)**
  - Files: `crates/engine/src/model.rs`, `crates/kernels/src/encoder.rs`
  - Intent: Overlap Attention KV-cache streaming on a secondary Metal command queue with GEMV dequantization on the primary queue using `MTLEvent` synchronization to absorb the 3.2 ms attention latency.
  - Testing: Profile execution timeline with `KAPLUMBA_DECODE_PROFILE=kernel`; verify greedy output bit-exactness.

- [x] **T5.1: Zero-Bubble Pipelined Step Submission & GPU Token Hand-off**
  - Files: `crates/engine/src/model.rs`, `crates/engine/src/generate.rs`, `crates/engine/src/gpu_cache.rs`
  - Intent: Double-buffer decode staging (`query_positions_decode`, `token_ids_decode`, `host_tokens`), bypass host wait barrier `encoder.flush_and_wait()?` during decode (`tokens == 1`), dispatch GPU device-side token reduction (`encode_argmax_to_token`), and pipeline step $N+1$ submission ahead of waiting on step $N$. Add `retract` to `GpuKvCache` for clean early rollback on EOS.
  - Testing: Verified bit-exact output on golden prompt ("Paris", `stopped_on_eos=true`, 15.49 tok/s); verified multi-token generation on counting prompt (" 4, 5, 6, 7, 8, 9, 10, 11, 12, 13", 32 tokens, 15.33 tok/s); model resident memory strictly $\le 12$ GiB (measured 7.56 GiB); `embed_tokens` kept permanently resident without reloading.

- [x] **T5.2: MLP & Down-Proj Optimization (Split-K & Memory Coalescing)**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Optimize `down_proj` ($K=21504, N=5376$) and `o_proj` ($K=4096, N=5376$) using Split-K=2 to double threadgroup occupancy from 168 to 336 threadgroups across 20 M4 Pro GPU cores, and fuse reduction into post-FFN/pre-FFN norm scaling kernels. Use sequential 2-column streaming (`fused2_2col`) for `gate_up` to prevent cache thrashing.
  - Testing: Sabotage verified on `2col_sk2` (intentional corruption in accumulator fails with exit 101); full test suite passes (`live_kernel_bit_exact` 1/1, `fused_bit_exact` 3/3, `argmax_engine_parity` 3/3, `repacked_golden` 2/2); golden prompt bit-exact ("Paris"); 32-token decode throughput lifts to 15.75 tok/s (up from 15.33 tok/s), memory resident 7.86 GiB.

- [ ] **T5.3: RMSNorm Prologue Fusion & Dispatch Consolidation**
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/src/model.rs`
  - Intent: Fuse layer input RMSNorm with GEMV input transformation and fuse post-attention / post-FFN norm transformations to eliminate ~500 small elementwise dispatches per token, recovering ~3-4 ms of launch latency.
  - Testing: Verify numeric bit-exactness; benchmark dispatch reduction and step time.

- [x] **T5.4: Fused Q/K/V RMSNorm + RoPE + KV Cache Append (`norm_rope_qkv_cache`)**
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/src/model.rs`, `crates/engine/tests/norm_rope_qkv_bit_exact.rs`
  - Intent: Eliminate the separate `cache_append` dispatch in decode across all 60 layers (60 dispatches saved per token) by writing normalized and rotated Q/K/V elements directly into `key_cache`, `value_cache`, and `position_cache` in-place.
  - Testing: Sabotage tested on assertion in `norm_rope_qkv_cache_is_bit_identical_to_separate_append` (exit code 101 on intentional corruption); full test passes cleanly 4/4; golden prompt bit-exact ("Paris"); 32-token decode achieves 15.67 tok/s.

- [ ] **T6: End-to-End Benchmark & Acceptance Validation (20+ Tok/s)**
  - Files: `spec/decode-25tok-peak-bandwidth/implementation_notes.md`
  - Intent: Measure steady-state tok/s and latency across 16 interleaved A/B pairs using `scripts/gpuq try -- ./target/release/kaplumba`.
  - Testing: Prove AC1 ($\ge 20.0 - 25.0$ tok/s, $\le 40 - 50$ ms/step), AC2 (bit-exact greedy text), AC3 (0 test regressions), and AC5 ($\le 12.0$ GiB resident memory).
