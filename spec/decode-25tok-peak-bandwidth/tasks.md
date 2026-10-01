# TASKS: 25 Tok/s Decode at Physical Memory Bandwidth Limit

- [x] **T1: True Threadgroup Co-Computation for Gate + Up (`linear_gemv_ishizuki_k3_fused2_co`)**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Replace split-grid concatenation in `fused2` with true co-computation where each threadgroup computes tile `tn` of `gate` and tile `tn` of `up` together, loading input vector $x$ once and sharing float registers.
  - Testing: Run `live_kernel_bit_exact` comparing `fused2_co` against independent reference encodes; verify sabotage detection; verify 0 FP16 diffs. (PASS: `fused_bit_exact` 3/3, sabotage test triggered exit code 101 on intentional corruption).

- [x] **T2: Wire `fused2_co` into `encode_gemv_fused2_mlp`**
  - Files: `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`
  - Intent: Update `encode_gemv_fused2_mlp` to dispatch `linear_gemv_ishizuki_k3_fused2_co` across 1344 threadgroups (instead of 2688). Measure latency reduction in `gate_up_gelu` (targeting 25.8 ms -> 16.5 ms).
  - Testing: Run `cargo test --test repacked_golden --test fused_bit_exact`; benchmark greedy decode forward step and confirm bit-exact text. (PASS: greedy output "Paris", 32-tok decode throughput 13.0 tok/s).

- [ ] **T3: 128-Bit Aligned Vector Memory Transactions (`uint4` Loads)**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Implement 128-bit aligned vector memory loads (`uint4`) in GEMV K-loop to maximize DRAM bus utilization toward 250+ GB/s.
  - Testing: Run `live_kernel_bit_exact`; measure achieved bandwidth with `bench_colmajor`; verify sabotage proof.

- [ ] **T4: Concurrent Multi-Queue Overlap (Attention + GEMV)**
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
