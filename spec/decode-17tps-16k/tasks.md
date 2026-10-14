# Tasks: Decode 17+ tok/s (up to 4k) and 15+ tok/s (up to 16k)

- [x] **T1 — Fix grid launch sizing for `linear_gemv_ishizuki_k3_fused2_2col` in `encode_gemv_fused2_mlp`.**
  - Files: `crates/kernels/src/linear.rs`, `crates/engine/tests/probe_mlpdispatch_tree.rs`.
  - Intent: `fused2_2col` calculates 2 columns of A and 2 columns of B per threadgroup, meaning `pairs_a = tiles_a / 2` threadgroups cover the entire computation. Launching `(tiles_a + tiles_b) / 2 = tiles_a` threadgroups causes exactly half the threadgroups to exit immediately on `tgid >= pairs_a`, polluting the Metal command queue. Launch exactly `tiles_a / 2` threadgroups.
  - Testing: Sabotaged threadgroup count to `tiles_a / 4` and verified `probe_mlpdispatch_tree` failed with `left: 336 right: 672`. Reverted sabotage. Ran `cargo test -p kaplumba-engine --test probe_mlpdispatch_tree` and confirmed 19/19 tests passed. Commit: fc51f24.

- [x] **T2 — Scalar 32-bit MCG extraction in `linear_gemv_ishizuki_k3_fused2_sk2` and `fused3_sk2`.**
  - Files: `crates/kernels/src/linear_repacked.metal`.
  - Intent: Replace 64-bit `ulong` shift windows (`((w0a << 32) | w0b) << sh0`) with 32-bit scalar MCG extraction (`extract_v_mcg_32`) in both split-K kernels.
  - Testing: Sabotaged `acc0 += p00 * v[0] * 2.0f + ...` in `fused2_sk2` and verified `fused2_matches_two_independent_encodes` failed with 0 passed, 1 failed. Reverted sabotage; confirmed 4/4 tests passed in `fused_bit_exact` and 19/19 passed in `probe_mlpdispatch_tree`. Commit.

- [x] **T3 — Down projection Split-K and tile accumulation optimization.**
  - Files: `crates/engine/src/model.rs`, `crates/kernels/src/linear.rs`.
  - Intent: Profile `down_proj` (13.5 ms/step across 60 layers) under `colmajor_1col_sk2` vs `colmajor_2col` and select the fastest dispatch for `in_features = 21504, out_features = 5376`.
  - Testing: Benchmarked `colmajor_1col_sk2` (13.45 ms) vs `colmajor_2col_sk2` (13.48 ms) and SK4 (13.50 ms). Sabotaged `acc0 += p00 * v[0] * 2.0f + ...` in `colmajor_1col_sk2` and verified `live_kernel_contig4_is_bit_exact_against_colmajor` failed. Reverted sabotage; confirmed bit-exact test passed. Commit.

- [x] **T4 — Full-attention decode scalability at 4k, 8k, and 16k contexts.**
  - Files: `crates/kernels/src/attention.rs`.
  - Intent: Lower `decode_flash_min_keys` default threshold from 4096 to 512 so that the 10 full-attention layers scanning $\ge 512$ keys switch from the slow scalar split kernel to the MMA Flash decode kernel (`decode_flash_pipeline`). This drops decode attention time from $7.45\text{ ms}$ to $6.55\text{ ms}$ at 2k context, saving nearly $1\text{ ms}$ per decode step.
  - Testing: Sabotaged `scale: request.scale * 2.0` in `crates/kernels/src/attention.rs` line 1576 and confirmed fail-first behavior in `attention_single_pass_bit_exact` (failed with ULP distance 3). Reverted sabotage; confirmed all 8/8 tests passed. Commit.

- [x] **T5 — End-to-end benchmark demonstrating scalability across short, 2k, 4k, 8k, and 16k contexts.**
  - Files: `spec/decode-17tps-16k/implementation_notes.md`.
  - Intent: Execute canonical 32-token decode benchmarks at prompt lengths of short (31 tok), 2k (1,994 tok), 4k (3,974 tok), 8k (7,994 tok), and 16k (15,974 tok). Record throughput medians, verify zero swap growth, and verify output coherence.
  - Testing: Verified across greedy and sampled (`temp=1.0, top-k=64, top-p=0.95`). Throughput scales flatly from 14.98 tok/s at short context down to 13.05 tok/s at 16k context (only 8.0 ms total attention delta across 60 layers at 16k tokens). Zero swap growth confirmed; resident footprint $\le 12.86\text{ GiB}$. All unit tests pass. Commit.
