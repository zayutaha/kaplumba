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

- [x] **T7.6.8: Host pipeline & decode warmup optimization (eliminate 1.27s JIT stall and empty buffer flush)**
  - Files: `crates/kernels/src/encoder.rs`, `crates/engine/src/model.rs`, `crates/engine/src/generate.rs`, `crates/engine/src/exl3.rs`
  - Intent: Eliminate the 1.27s Metal JIT shader compilation penalty on the first decode step via `warmup_decode` at model load. Skip empty command buffer commits in `flush_and_wait`. Optimize `argmax_logits` comparison loop and reuse `logits` vector in-place via `forward_buffer_into_vec`. Fix shift overflow in `pack_integer_windows`.
  - Result: Step 0 dropped from 1,277 ms to 69.9 ms. Decode throughput boosted from 12.39 tok/s to 14.04 tok/s. TTFT dropped from 3.23s to 1.23s. 100% bit-exact golden output verified.

- [x] **T7.6.9: Split-K gating and MCG ALU half2 vector optimization**
  - Files: `crates/kernels/src/linear.rs`, `crates/kernels/src/linear.metal`, `crates/kernels/src/linear_repacked.metal`
  - Intent: Benchmark Split-K vs contiguous 4-tile (`contig4`) layout. Gate Split-K behind `KAPLUMBA_USE_SK=1` because wide/medium matrices already saturate 20 GPU cores (134 SIMD groups/core) and Split-K incurs DRAM roundtrips and smaller reduction loops. Optimize `decode_state_mcg` with direct `half2` vector bitcast.
  - Result: Eliminating Split-K reduction roundtrips dropped forward step latency from ~80.9 ms to ~69.8-70.4 ms (decode speed jumped from 12.35 tok/s to 14.02 tok/s). `half2` bitcast replaced scalar shifts/masks across 32 dequant calls per loop step. 0 FP16 diffs in `live_kernel_bit_exact` and `fused_bit_exact`.

- [x] **T7.6.10a: Count dispatches, not just command buffers** (`021067c`)
  - Files: `crates/kernels/src/encoder.rs`, `crates/engine/src/model.rs`, `crates/engine/src/generate.rs`
  - Intent: A fusion is measured on dispatches removed, and the only counter was `command_buffers_created()`. Under `KAPLUMBA_DECODE_PROFILE` the stamped boundaries dominate it (a 48-step profiled run reported 730 buffers/step against 1407 real dispatches), so it cannot answer the question.
  - Result: `MetalEncoder::dispatches_total`, folded per decode step and printed by the existing `KAPLUMBA_COUNT_BUFFERS` diagnostic. The tally for the real decode loop must be taken in `generate.rs` — `forward_buffer_into_step` drives the CALLER's encoder, so `model.rs`'s own hook never fires for it — and before `commit_terminal`, which consumes the encoder. **Baseline: 1407 dispatches/token, 13.26-13.32 tok/s.**

- [x] **T7.6.10b: `norm_rope_qkv` kernel, rewritten and proven bit-exact** (`479d720`)
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/tests/norm_rope_qkv_bit_exact.rs`
  - Intent: Finish and verify the kernel a previous pass wrote but never wired.
  - Result: **The kernel as written was wrong in three ways**, each of which the test was confirmed to catch (see the failure modes below):
    1. `const float val = target[base + lid]` with no strided loop, so `head_dim 512` (the 10 full-attention layers) summed only the first 256 columns. The 50 sliding layers (256) happened to be right.
    2. It staged the normed value in an fp32 register across the rotation, skipping the fp16 rounding the separate norm writes. `rope` reads that half back, so the fused kernel was computing a *different function*. **This is the fusion's whole difficulty and the reason the obvious version is not bit-exact.**
    3. The unrotated tail copied only the low half, leaving the high half of every tail pair unwritten — the same bug `rope` itself once had (`elements.metal:363`).
  - Rewritten to mirror the reference exactly: strided sum of squares in `rms_norm_weighted_tg`'s order with the 256-lane tree; `normed_shared[i] = float(half(normed))` (load-bearing); `threadgroup float normed_shared[512]` so both head dims work unrolled; one thread per PAIR, strided, for both the rotation and the tail; V unweighted with no RoPE.
  - **Sabotage evidence** (AGENTS.md §3.4 — each applied, run, and reverted; the kernel is byte-identical to its pre-sabotage state):
    | sabotage | result |
    |---|---|
    | drop `float(half(normed))` | 328/1024 differ, worst 19 ULP, first at head 0 column 2 |
    | `partial[lid] = (lid==0) ? sum : 0` | 8192/8192 differ, worst 9266 ULP |
    | drop the tail's high half | exactly 256 elements at column 192; poison test independently reports 256/1024 unwritten, `rotated_pairs 64` |
  - **The poison test was itself wrong on first write and is worth recording.** It uploaded real input OVER a poison sentinel, then asserted the sentinel did not survive — which can never happen, so it passed *while the tail was being dropped*. It now makes the INPUT the poison (all `0x7bff`, weights 1.0), so a written column comes out near 1.0 and an unwritten one is still `0x7bff`. Verified it fails under sabotage before being trusted.
  - 7 shape cases: both head dims, both rope modes, offsets 0/5/11/37/1023, proportional 0.5 and 0.75 (both leave an unrotated tail). 3/3 tests pass.

- [x] **T7.6.10c: Wire `norm_rope_qkv` into the decode path** (`9d4c44c`)
  - Files: `crates/engine/src/model.rs`
  - Intent: Five dispatches per layer (3 head-width norms + 2 ropes) → one.
  - Result: Gated on `decode`. Prefill keeps the five-dispatch path for two load-bearing reasons: `norm_rope_qkv` has no token axis so `offset` IS the position, and the reference norms switch to the per-row kernel above `THREADGROUP_NORM_MAX_ROWS` (8192), which a 1024-row × 32-head prefill chunk crosses — a different reduction order is a different `rsqrt`.
  - `KAPLUMBA_NO_NORM_ROPE_QKV` restores the old path so the A/B runs from ONE binary; two rebuilt binaries differ by build layout alone and this box drifts 10%+ run to run.
  - **MEASURED** (same binary, flag A/B, 96 greedy tokens, 4 interleaved pairs): unfused 1407 dispatches/token, median 13.215 tok/s (75.67 ms/step), spread 1.1%; fused 1167 dispatches/token, median 13.299 tok/s (75.19 ms/step), spread 0.9%. **Delta -0.48 ms/step, +0.64%** — 240 dispatches/token removed, exactly 4 × 60 layers as predicted. The time delta is *inside* the spread, so see the negative result note below.
  - **BIT-EXACTNESS, end to end**: full stdout over 128 greedy tokens byte-identical between arms, sha256 `2fff2c970b611f64c4a485d8a3a3dc719a0f03293a3f48c167f4dff5466795b2` for both, including the `--show-ids` token list. A single fp16 ULP anywhere in the 60-layer forward would change that list.

- [ ] **T7.6.11: Fused Down Finalize + Post-FFN Residual & Layer Scale**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`
  - Intent: Combine `linear_finalize_simd` and `encode_rms_norm_residual_layer_scale` into a single kernel dispatch (`linear_finalize_simd_with_residual_scale`), eliminating 60 dispatches and `scratch.proj` DRAM buffer traffic.
  - Estimated impact: ~2.0 ms reduction in step latency. **See the T7.6.10 result before starting: the measured marginal value of a dispatch on this box is ~0.002 ms, so 60 dispatches is ~0.12 ms, not 2.0 ms.**

- [ ] **T7.6.12: Fused Post-Attention Norm, Residual, and Gate Input Transform**
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/src/model.rs`
  - Intent: Merge post-attention residual add, pre-FFN RMS norm, and Hadamard gate input transform into a single dispatch per layer, eliminating 120 dispatches per token.
  - Estimated impact: ~2.0 ms reduction in step latency. **Revise against the measured marginal dispatch cost, as above.**

- [ ] **T7.6.13: Attention Fast-Path Single Dispatch for Short Decode Context**
  - Files: `crates/kernels/src/attention.rs`, `crates/engine/src/model.rs`
  - Intent: When decode KV context length is < 64 (splits == 1), bypass split pipelines (`decode_split_pipelines` + `decode_combine`), dispatching single-pass attention directly.
  - Estimated impact: ~1.5 ms reduction in step latency.

- [ ] **T8: End-to-end decode benchmark and acceptance validation**
  - Files: `spec/decode-19tok-repacked-250gb/implementation_notes.md`
  - Intent: Measure end-to-end tok/s using `scripts/gpuq run` across reference prompts.
  - Acceptance: Pure autoregressive decode throughput >= 19.0 tok/s (<= 52.6 ms/step) against physical 250+ GB/s ceiling with 100% bit-exact greedy generation.


