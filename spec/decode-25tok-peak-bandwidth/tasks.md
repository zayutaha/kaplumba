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
