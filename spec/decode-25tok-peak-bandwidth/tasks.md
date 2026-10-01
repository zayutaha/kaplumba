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


- [ ] **T5: Concurrent Multi-Queue Overlap (Attention + GEMV)**
  - Files: `crates/engine/src/model.rs`, `crates/kernels/src/encoder.rs`
  - Intent: Overlap Attention KV-cache streaming on a secondary Metal command queue with GEMV dequantization on the primary queue using `MTLEvent` synchronization to absorb the 3.2 ms attention latency.
  - Testing: Profile execution timeline with `KAPLUMBA_DECODE_PROFILE=kernel`; verify greedy output bit-exactness.

- [ ] **T5: Command Pipeline & Layer Dispatch Streamlining**
  - Files: `crates/engine/src/model.rs`, `crates/engine/src/generate.rs`
  - Intent: Consolidate remaining per-layer dispatch boundaries to eliminate ~5.0 ms of host CPU-GPU submission bubbles and threadgroup setup latency.
  - Testing: Count total dispatches per token with `KAPLUMBA_COUNT_BUFFERS=1`; verify reduction in end-to-end forward step latency.

- [ ] **T6: End-to-End Benchmark & Acceptance Validation**
  - Files: `spec/decode-25tok-peak-bandwidth/implementation_notes.md`
  - Intent: Measure steady-state tok/s and latency across 16 interleaved A/B pairs using `scripts/gpuq try -- ./target/release/kaplumba`.
  - Testing: Prove AC1 ($\ge 25.0$ tok/s, $\le 40.0$ ms/step), AC2 (bit-exact greedy text), AC3 (0 test regressions), and AC5 ($\le 16.5$ GiB resident memory).
