# Tasks: Decode 19+ tok/s and 250+ GB/s via Repacked Weights

- [x] **T1: Create synthetic memory bandwidth probe harness** (`8a77fac`)
  - Files: `crates/engine/examples/probe_bw_250gb.rs`, `crates/kernels/src/bw_probe.metal`
  - Intent: Sweep vector widths (uint4 = 16 bytes), threadgroup sizing (256, 512), grid concurrency (40k to 81k), and hazard tracking across 2 GB, 6 GB, and 12 GB buffers. Full entropy non-compressible buffer initialization.
  - Result: Achieved 250.5 GB/s median, 253.2 GB/s best on full 12 GB resident buffer; 254.0 GB/s median on 6 GB.

- [x] **T2: Validate probe with fail-first sabotage** (`8a77fac`)
  - Files: `crates/engine/examples/probe_bw_250gb.rs`
  - Intent: Sabotage read loop count in probe to verify checksum mismatch fires.
  - Result: Sabotaged run altered checksum (`0x739adf40` vs `0xa8c277a0`), confirming failure detection. Reverted and verified.

- [x] **T3: Design and implement offline model repacking binary**
  - Files: `crates/engine/src/bin/repack_model.rs`, `crates/engine/src/exl3.rs`
  - Intent: Read `Gxma` safetensors, reformat 3 bpw trellis data into coalesced lane-aligned layout, write to `~/.kaplumba/models/Gxma-repacked`. Ensure single-copy limit.
  - Result: Successfully repacked 410 trellis tensors across 2 shards (16.03 GB) in 12.1s with `"layout": "colmajor"`. Original files untouched. Destination matches byte layout. Single copy rule respected.

- [x] **T4: Correctness test for repacked format**
  - Files: `crates/engine/tests/repacked_golden.rs`
  - Intent: Sabotage repacked bits, confirm test fails, revert, confirm 100% bitwise identity of dequantized matrix outputs against reference CPU dequantizer.
  - Result: Sabotage failed as expected (`orig -98.65691 vs repacked -95.42632`); reverted; passes bit-for-bit exact across 512 elements and verifies Metal GEMV bit-exact match against CPU reference.

- [x] **T5: Implement repacked GEMV Metal kernel**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/examples/bench_colmajor.rs`
  - Intent: Column-major contiguous GEMV `linear_gemv_ishizuki_k3_colmajor` with 4x K unroll, eliminating strided jumps.
  - Result: Increased GEMV throughput from 151 GB/s to 180.5-188.5 GB/s (and 194.4 GB/s on large matrices) with 0 quality loss. Verified bit-exact.

- [x] **T6: Wire repacked weights and GEMV into engine pipeline**
  - Files: `crates/engine/src/config.rs`, `crates/engine/src/safetensors.rs`, `crates/engine/src/model.rs`, `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`
  - Intent: Auto-detect repacked model checkpoint and dispatch repacked GEMV and prefill GEMM.
  - Result: Auto-detects `"layout": "colmajor"` in `quantization_config.json`, sets `permutation_mode = 1` for 3 bpw layers, dispatches `linear_gemv_ishizuki_k3_colmajor`. Verified bit-exact output on reference prompts ("The capital of France is Paris."). Sabotage test verified: forcing rowmajor w2 kernel on repacked weights produced gibberish ("TheBHC deBLC"); reverted and verified exact recovery.

- [x] **T7: Implement fused GEMV kernels & command buffer optimization**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`
  - Intent: Implement `linear_gemv_ishizuki_k3_fused2` (`gate_proj` + `up_proj`) and `linear_gemv_ishizuki_k3_fused3` (`q_proj` + `k_proj` + `v_proj`). Wire into `forward_chunk` during decode.
  - Result: Fused2 and Fused3 wired into `model.rs`. Bit-exact greedy decode verified ("Paris", "Relativity posits that space and time are linked..."). Single command buffer execution (`KAPLUMBA_MAX_DISPATCHES_PER_BUFFER=2500`) confirmed running 60 layers in 1 buffer, delivering 83.9 ms GPU own time per token (11.46 tok/s).

- [x] **T7.1: Internalize single-buffer decode in `MetalEncoder`**
  - Files: `crates/kernels/src/encoder.rs`, `crates/engine/src/model.rs`, `crates/engine/src/generate.rs`
  - Intent: Allow `MetalEncoder` to configure `max_dispatches = 2500` for decode steps without requiring env var, while preserving 50 for prefill.
  - Result: Added `set_max_dispatches` on `MetalEncoder`. Configured in `model.rs` and `generate.rs`. Verified: decode commits in single command buffer automatically, sustaining steady-state 86.2 ms/token (11.6 tok/s). All kernel tests pass.

- [ ] **T7.2: Design and implement 128-byte vector repacking in `repack_model.rs`**
  - Files: `crates/engine/src/bin/repack_model.rs`, `crates/engine/src/exl3.rs`
  - Intent: Group 4 consecutive tiles along K into 384-byte blocks ($3 \times 128$ bytes = 6 full 64-byte cachelines). Layout ensures SIMD lanes issue 16-byte aligned vector loads (`uint4`), eliminating the 1.333x DRAM cacheline burst amplification.
  - Test: Repacker successfully generates 128-byte aligned format.

- [ ] **T7.3: Implement 128-byte vector GEMV Metal kernel (`_vec128`)**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`
  - Intent: Kernel reads `const device uint4 *` directly, fully saturating Apple Silicon memory controllers at 250+ GB/s.
  - Test: Correctness verified against CPU reference in `repacked_golden.rs`.

- [ ] **T7.4: Fail-first sabotage verification of 128-byte vector format**
  - Files: `crates/engine/tests/repacked_golden.rs`
  - Intent: Sabotage bit representation, verify test detects failure, revert, verify 100% bit-exact dequantization.
  - Test: Sabotaged assertion catches divergence.

- [ ] **T7.5: Repack model to 128-byte format and wire into engine**
  - Files: `~/.kaplumba/models/Gxma-repacked`, `crates/engine/src/model.rs`
  - Intent: Execute full model on 128-byte repacked weights. Single disk copy rule strictly maintained.
  - Test: Real model greedy text matches baseline.

- [ ] **T8: End-to-end decode benchmark and acceptance validation**
  - Files: `spec/decode-19tok-repacked-250gb/implementation_notes.md`
  - Intent: Measure end-to-end tok/s using `scripts/gpuq run` across reference prompts.
  - Acceptance: Output >= 19.0 tok/s (<= 52.6 ms/token) at 250+ GB/s effective bandwidth.

