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

- [x] **T7.2: Vector coalescing & Split-K investigation**
  - Files: `crates/kernels/src/linear.metal`, `crates/engine/examples/bench_colmajor.rs`
  - Intent: Test 128-byte coalesced SIMD loads and Split-K 4 to address memory controller gather and grid under-subscription.
  - Result: `linear_gemv_ishizuki_k3_colmajor_simd` achieved 128.1 GB/s on 5376x5376 (+18% over baseline 108.5 GB/s). `linear_gemv_ishizuki_k3_colmajor_sk4` (Split-K 4) increased throughput to 142.6 GB/s (+31% over baseline) by expanding grid from 336 to 1344 threadgroups. On wide shapes (`gate_up 5376x43008`), baseline colmajor is optimal at 189.3 GB/s.

- [x] **T7.3: Implement and verify SIMD coalesced GEMV kernel**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/repacked_golden.rs`
  - Intent: Register `linear_gemv_ishizuki_k3_colmajor_simd`, verify against CPU reference.
  - Result: Verified 100% bit-exact across all 512 elements against original Ishizuki reference dequantizer.

- [x] **T7.4: Measure full model decode breakdown and identify remaining gap**
  - Files: `spec/decode-19tok-repacked-250gb/implementation_notes.md`
  - Intent: Dissect 85.7 ms steady-state decode step via `KAPLUMBA_DECODE_PROFILE=kernel`.
  - Result: Verified per-step breakdown: `gate_up_proj` 35.58 ms, `down_proj` 18.37 ms, `qkv_proj` 19.94 ms, `o_proj` 9.74 ms, `head/gemv` 4.17 ms, non-linear ops 7.04 ms. Total linear projections: 87.8 ms. At 188 GB/s full-model bandwidth, theoretical floor is 52.5 ms (19.0 tok/s).

- [x] **T7.5: Implement vector coalescing, finalize wiring, and Split-K investigation**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/repacked_golden.rs`, `crates/engine/examples/bench_colmajor.rs`
  - Intent: Wire finalize pipeline into `encode_ishizuki_gemv_named`, test `linear_gemv_ishizuki_k3_colmajor_u4` with 128-byte aligned loads across 384-byte chunks (4 tiles).
  - Result: Finalize pipeline wired and tested in `repacked_golden.rs`. `u4` vector loading achieved 95.8 GB/s on `q_proj` (+41% over baseline 67.9 GB/s) and 153.5 GB/s on `up_proj` (+13.5%). Split-K 4 evaluated on `o_proj`: adds 60 reduction dispatches per token, creating command buffer dispatch overhead. Inner-loop threadgroup barriers identified as throughput killer; barrier-free K-loop restored.

- [x] **T7.6.1: Direct MCG specialization and full-attn Q+K fusion** (`4b7c528`)
  - Files: `crates/kernels/src/linear.metal`, `crates/engine/src/model.rs`
  - Intent: Eliminate runtime `decode_state` branch in inner K loops by calling `decode_state_mcg` directly. Fuse Q + K in full-attention layers using `encode_gemv_fused2`.
  - Result: Zero branches in unrolled inner loops across all colmajor kernels. Q and K fused in full-attention decode layers. Verified bit-exact ("The capital of France is Paris.").

- [x] **T7.6.2: Fused MLP activation pipeline (`fused2_mlp`)** (`0288b49`)
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`
  - Intent: Merge `gate_up` GEMV, Hadamard output transform, and `gated_gelu` into `encode_gemv_fused2_mlp`. Output directly to `scratch.activated`.
  - Result: Eliminated `scratch.gate` and `scratch.up` buffers from global DRAM (20.6 MB roundtrips per token eliminated). Eliminated 120 kernel dispatches per token. Measured steady-state forward step dropped from 85.5 ms to 81.0 ms (saving 4.5 ms per token). 100% bit-exact greedy match verified.

- [x] **T7.6.3: Fast zero-copy argmax in `generate_gpu`** (`9848a2b`)
  - Files: `crates/engine/src/generate.rs`
  - Intent: Eliminate the 262,144 element `f16 -> f32` conversion and heap `Vec` allocations on every token step. Use direct integer/half comparisons on `scratch.capped` or `logits` slice.
  - Result: Eliminated 262,144 f32 allocation and conversion on CPU per token. Forward step dropped to 80.8 ms. Bit-exact greedy text verified.

- [x] **T7.6.4: Contiguous 4-tile unrolled GEMV (`contig4`) evaluation** (`4182ef6`, `9844142`)
  - Files: `crates/kernels/src/linear.rs`, `crates/engine/examples/bench_colmajor.rs`
  - Intent: Wire `linear_gemv_ishizuki_k3_colmajor_contig4` into `linear.rs` and benchmark across all layer projection shapes.
  - Result: Achieved 129 GB/s on 5376x5376 and 188 GB/s on gate_up. Replaced inner loops of fused2/fused3 with contiguous 4-tile loading.

- [x] **T7.6.5: Switch `encode_ishizuki_gemv_named` to SIMD butterfly finalize** (`6faaf7f`)
  - Files: `crates/kernels/src/linear.rs`
  - Intent: Replace slow vector `finalize_pipeline` with 32-thread SIMD butterfly `finalize_simd_pipeline`.
  - Result: `down_proj` dropped from 18.95 ms to 14.47 ms; `o_proj` dropped from 10.72 ms to 6.80 ms.

- [x] **T7.6.6: Switch `encode_gemv_fused3` to SIMD butterfly finalize** (`aea8af5`)
  - Files: `crates/kernels/src/linear.rs`
  - Intent: Replace vector finalize in `encode_gemv_fused3` with SIMD butterfly `finalize_simd_pipeline`.
  - Result: `qkv_proj` dropped from 14.97 ms down to 9.48 ms per step. Total kernel time dropped from 76.5 ms to 71.0 ms.

- [x] **T7.6.7: Fuse `down_proj` input transform and fuse QKV finalize** (`6327c84`)
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`
  - Intent: Merge `down_proj` input transform into `gate_up_gelu` output, eliminating `scratch.activated` DRAM writes and 60 dispatches. Fuse Q, K, V finalize into single-dispatch `linear_finalize_simd3`.
  - Result: 100% bit-exact greedy decode verified. Total kernel time down to 70.7 ms/step.

- [ ] **T8: End-to-end decode benchmark and acceptance validation**
  - Files: `spec/decode-19tok-repacked-250gb/implementation_notes.md`
  - Intent: Measure end-to-end tok/s using `scripts/gpuq run` across reference prompts.
  - Acceptance: Pure autoregressive decode throughput characterization against physical 250+ GB/s ceiling.

## T14 — 32 codes per lane (halve the 2.667× request amplification)

Sole route to 15 tok/s. Each lever in this group is worth **+3.13 tok/s alone**; the
`+8` previously attributed to the non-GEMV group was a double count (see spec §5).
Sequenced probe-first on purpose: the 1.333× prediction extrapolates a law fitted at two
points, and plan §4.3 explains why that is not good enough to build against.

- [ ] **T14.1: Microbenchmark probe for the `D = 32` load pattern** (AC7)
  - Files: `crates/kernels/src/linear_repacked.metal` (new `*_d32_probe` kernels, garbage output), `crates/kernels/src/linear.rs` (dispatch arm), `crates/engine/examples/probe_d32.rs` (harness)
  - Intent: Issue 8 lanes × 4 words per 16×16 tile with the real `decode_state_mcg` body, on the 11 real shapes, cold via `MADV_DONTNEED`, interleaved A/B against `_contig4`. Report the per-shape request-rate ratio.
  - Test: the harness must compute the ratio from measured ns, and a unit test must assert the *predicted* ratio is 2.667/1.333 = 2.0 from first principles. **Sabotage-verified**: break the probe's word index and confirm the ratio moves; a probe that cannot fail is not evidence (AGENTS.md §3.4).
  - **Gate:** median ratio within 1.6× of 2.0. Otherwise T14.2 is **cancelled** and the amplification law is recorded as a 2-point coincidence.

- [ ] **T14.2: Implement `D = 32` GEMV, 4 tiles per simdgroup** (AC8 prerequisite)
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`
  - Intent: The real kernel. 8 lanes/tile, `uint4` + `uint2` for the straddling lane, 32 decodes/lane into 2 accumulators, `simd_shuffle_xor` 1/2/4 within an 8-lane group, threadgroup reduce across 8 simdgroups, 64 outputs per threadgroup.
  - Test: `cargo test -p kaplumba-engine --test live_kernel_bit_exact` extended to all 9 real decode shapes; `fused_bit_exact.rs` (fused vs **independent** `encode_gemv`) re-run for the fused2/fused3 paths, since those now cover 4 tiles per simdgroup.
  - Must be **sabotage-verified** in both directions: an off-by-one in the tile-to-lane map must fail the test, and the f32 reference must be allowed to differ.

- [ ] **T14.3: Assert fp16 bit-identity on all 9 real shapes** (AC8)
  - Files: `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: `to_bits()` equality of the **fp16** output of `D = 32` vs `_contig4` on every real decode shape. f32 divergence is expected (up to ~51 ULP is the `_contig4` precedent) and recorded, not gated.
  - **All 9 or nothing.** No env var, no partial rollout, no "ship it and measure quality later" — this repo has no perplexity/KKL instrument, so greedy-output identity on 5 prompts is the only downstream gate and it cannot detect a small argmax-flip rate.
  - Sabotage-verified by perturbing one accumulation order and confirming the fp16 assertion fires.

- [ ] **T14.4: Per-shape A/B and env-gated wiring** (AC9, AC10)
  - Files: `crates/kernels/src/linear.rs` (`ishizuki_gemv_pipeline` arm, env var **default off**), `crates/engine/examples/bench_residency.rs` (all 11 `bench.rs:SHAPES` shapes for both kernels)
  - Intent: Report every shape for both kernels, interleaved, medians and spread. **Narrow-shape regressions are expected and reported, not hidden**: 4 tiles/simdgroup means `down_proj` 336 → 84 threadgroups, 4.2 waves on 20 cores instead of 16.8. Then decide go/no-go on the whole table.
  - Test: `KAPLUMBA_TRACE_GEMV=1` re-run in both configurations to prove which kernel dispatches (AC10) — the historical failure here was a golden test asserting kernels that did not run. AC6 re-verified: generator ×2, `git diff` clean.

- [ ] **T15: Attribute and reduce the ~8.9 ms host/dispatch gap**
  - Files: `crates/engine/src/decode_profile.rs`, `crates/engine/src/generate.rs`, `crates/engine/src/model.rs`
  - Intent: First *measure* it — step mode's `dp_span_over_wall` has never been recorded in any file in this repo. Then attack what it shows: the `model.rs:3192` empty commit+wait, the terminal argmax barrier, ~1407 dispatches/step, and the unconditional `argmax_with_margin` at `generate.rs:867` (legacy path only; the default path guards it correctly).
  - Worth **+3.21 tok/s** and, unlike T14, does not depend on the amplification law at all. This is the fallback if AC7 fails.


