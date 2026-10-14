# Tasks: Decode 17+ tok/s (up to 4k) and 15+ tok/s (up to 16k)

- [x] **T1 — Fix grid launch sizing for `linear_gemv_ishizuki_k3_fused2_2col` in `encode_gemv_fused2_mlp`.**
  - Files: `crates/kernels/src/linear.rs`, `crates/engine/tests/probe_mlpdispatch_tree.rs`.
  - Intent: `fused2_2col` calculates 2 columns of A and 2 columns of B per threadgroup, meaning `pairs_a = tiles_a / 2` threadgroups cover the entire computation. Launching `(tiles_a + tiles_b) / 2 = tiles_a` threadgroups causes exactly half the threadgroups to exit immediately on `tgid >= pairs_a`, polluting the Metal command queue. Launch exactly `tiles_a / 2` threadgroups.
  - Testing: Sabotaged threadgroup count to `tiles_a / 4` and verified `probe_mlpdispatch_tree` failed with `left: 336 right: 672`. Reverted sabotage. Ran `cargo test -p kaplumba-engine --test probe_mlpdispatch_tree` and confirmed 19/19 tests passed. Commit: fc51f24.

- [x] **T2 — Scalar 32-bit MCG extraction in `linear_gemv_ishizuki_k3_fused2_sk2` and `fused3_sk2`.**
  - Files: `crates/kernels/src/linear_repacked.metal`.
  - Intent: Replace 64-bit `ulong` shift windows (`((w0a << 32) | w0b) << sh0`) with 32-bit scalar MCG extraction (`extract_v_mcg_32`) in both split-K kernels.
  - Testing: Sabotaged `acc0 += p00 * v[0] * 2.0f + ...` in `fused2_sk2` and verified `fused2_matches_two_independent_encodes` failed with 0 passed, 1 failed. Reverted sabotage; confirmed 4/4 tests passed in `fused_bit_exact` and 19/19 passed in `probe_mlpdispatch_tree`. Commit.

- [ ] **T3 — Down projection Split-K and tile accumulation optimization.**
  - Files: `crates/engine/src/model.rs`, `crates/kernels/src/linear.rs`.
  - Intent: Profile `down_proj` (13.5 ms/step across 60 layers) under `colmajor_1col_sk2` vs `colmajor_2col` and select the fastest dispatch for `in_features = 21504, out_features = 5376`.
  - Testing: Sabotage `down_proj` output signs and confirm bit-exact test fails. Revert sabotage. Benchmark with `scripts/gpuq try --`. Commit.

- [ ] **T4 — Full-attention decode scalability at 4k, 8k, and 16k contexts.**
  - Files: `crates/kernels/src/attention.metal`, `crates/engine/src/model.rs`.
  - Intent: Benchmark decode attention time across the 10 full-attention layers at context lengths of 2,048, 4,096, 8,192, and 16,384 tokens. Ensure online-softmax key reduction scales efficiently without latency spikes.
  - Testing: Sabotage attention scaling parameter and confirm `attention_single_pass_bit_exact` fails. Revert sabotage. Verify 8/8 tests pass. Commit.

- [ ] **T5 — End-to-end benchmark demonstrating $\ge 17\text{ tok/s}$ at $\le 4\text{k}$ and $\ge 15\text{ tok/s}$ at 16k.**
  - Files: `spec/decode-17tps-16k/implementation_notes.md`.
  - Intent: Execute canonical 32-token sampled decode benchmarks at prompt lengths of 2k, 4k, 8k, and 16k tokens. Record throughput medians, verify zero swap growth, and verify output coherence.
  - Testing: Verify AC1, AC2, AC3, AC4, and AC5.
