# Tasks: Decode 15+ tok/s and 250+ GB/s via Repacked Weights

> **Title corrected from 19 to 15** (the target was lowered in `spec.md`'s header; this
> file's title and T8's acceptance line were never updated and would send a future
> agent back to 19). **T8's acceptance line is corrected below.** The target *number*
> itself is unchanged and remains the user's call — see `spec.md` R6's pending-decision
> note.
>
> **T7.6.11–13 are budgeted at 3.5 us per dispatch removed** (`spec.md` §9), not at
> kernel time. T7.6.13 carried no revision note and is fixed here.

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

- [x] **T7.2: Vector coalescing & Split-K investigation** — **CANCELLED, results superseded**
  - Files: `crates/kernels/src/linear.metal`, `crates/engine/examples/bench_colmajor.rs`
  - Intent: Test 128-byte coalesced SIMD loads and Split-K 4 to address memory controller gather and grid under-subscription.
  - Result: `linear_gemv_ishizuki_k3_colmajor_simd` achieved 128.1 GB/s on 5376x5376 (+18% over baseline 108.5 GB/s). `linear_gemv_ishizuki_k3_colmajor_sk4` (Split-K 4) increased throughput to 142.6 GB/s (+31% over baseline) by expanding grid from 336 to 1344 threadgroups. On wide shapes (`gate_up 5376x43008`), baseline colmajor is optimal at 189.3 GB/s.
  - **Superseded on both halves, measured.** (a) *Split-K is unavailable under the no-quality-loss rule* — it measured 3–5 ULP from the reference on every shape (`spec.md` §5), and in-engine it was **slower** (T7.6.9: contig4 188.2 vs sk4 168.1 on `gate_up`). The "+31%" came from a partly-cached working set, the same artifact that inflated T12's "+24%". (b) *The `_simd` coalescing arm is the `simd_shuffle` family*, which measured **180.5 → 146.5 GB/s on `up_proj`** when A/B'd properly — the win is on the 5376×5376 shape only and does not survive on a wide one. Both kernels are dead on the decode path.

- [x] **T7.3: Implement and verify SIMD coalesced GEMV kernel** — **CANCELLED**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/repacked_golden.rs`
  - Intent: Register `linear_gemv_ishizuki_k3_colmajor_simd`, verify against CPU reference.
  - Result: Verified 100% bit-exact across all 512 elements against original Ishizuki reference dequantizer.
  - **Bit-exactness is real and is not the problem; performance is.** The kernel is not on the decode path: `simd_shuffle` cooperative loads measured **180.5 → 146.5 GB/s** on `up_proj` because dynamic per-lane shuffles serialize through the SIMD crossbar (`implementation_notes.md` finding 1). Cancel the task; the verification method is worth reusing, the kernel is not worth shipping.

- [x] **T7.4: Measure full model decode breakdown and identify remaining gap**
  - Files: `spec/decode-19tok-repacked-250gb/implementation_notes.md`
  - Intent: Dissect 85.7 ms steady-state decode step via `KAPLUMBA_DECODE_PROFILE=kernel`.
  - Result: Verified per-step breakdown: `gate_up_proj` 35.58 ms, `down_proj` 18.37 ms, `qkv_proj` 19.94 ms, `o_proj` 9.74 ms, `head/gemv` 4.17 ms, non-linear ops 7.04 ms. Total linear projections: 87.8 ms.
  - **Corrected:** the original line ended "At 188 GB/s full-model bandwidth, theoretical floor is 52.5 ms (19.0 tok/s)." **Both halves are withdrawn.** The weight traffic is **12.039 GB/token**, not the 9.88 GB that number assumed, so 12.039/188 = 64.0 ms, not 52.5 — and 19 tok/s is excluded outright (`spec.md` §6.2; AC1's 188 GB/s "ceiling" was itself a fit to land on 92% of 273). The `87.8 ms` vs `85.7 ms` inconsistency above is a per-kernel-sum vs whole-step difference; the current profile is in `spec.md` §6.3.

- [x] **T7.5: Implement vector coalescing, finalize wiring, and Split-K investigation**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/repacked_golden.rs`, `crates/engine/examples/bench_colmajor.rs`
  - Intent: Wire finalize pipeline into `encode_ishizuki_gemv_named`, test `linear_gemv_ishizuki_k3_colmajor_u4` with 128-byte aligned loads across 384-byte chunks (4 tiles).
  - Result: Finalize pipeline wired and tested in `repacked_golden.rs`. `u4` vector loading achieved 95.8 GB/s on `q_proj` (+41% over baseline 67.9 GB/s) and 153.5 GB/s on `up_proj` (+13.5%). Split-K 4 evaluated on `o_proj`: adds 60 reduction dispatches per token, creating command buffer dispatch overhead. Inner-loop threadgroup barriers identified as throughput killer; barrier-free K-loop restored.
  - **CANCELLED, and the "+41%" does not survive.** `_u4` is the 384-byte `uint4` staging attack on the amplification thesis, and A/B on the real shape gives **180.5 → 173.7 GB/s on `gate_up` — a loss** (`spec.md` §6.6). The `q_proj` +41% is a narrow-shape, partly-cached-arm number, the same artifact as T12's withdrawn +24%. **This is the load-width axis that the withdrawn 128-byte-burst thesis was built on**, and it is now closed three ways: `_u4` (uint4) loses, `_u8` (deeper unroll) is 1.01–1.03×, and `D=32` (1.333× amplification) is 0.288–0.918× on all 11 shapes. **The kernels are not on the decode path.**

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
  - **Superseded — the 14.04 tok/s was a single-arm reading on a drifting box, not a measurement.** AGENTS.md 10.5 records 3.7–6.4 tok/s from *one* binary on this box, and §20.7 of the notes records 4–12% spread across 16 *unpaired* samples. A single before/after pair cannot resolve a 13% effect. The real current figure, from a 16-pair interleaved A/B with sign test p = 0.0005, is **13.511 tok/s** (T7.6.10c). The 14.04 is inside the historical range (12.14 median, 12.76 median, 12.50 median across other sessions) and should not be quoted as a result.

- [x] **T7.6.9: Split-K gating and MCG ALU half2 vector optimization**
  - Files: `crates/kernels/src/linear.rs`, `crates/kernels/src/linear.metal`, `crates/kernels/src/linear_repacked.metal`
  - Intent: Benchmark Split-K vs contiguous 4-tile (`contig4`) layout. Gate Split-K behind `KAPLUMBA_USE_SK=1` because wide/medium matrices already saturate 20 GPU cores (134 SIMD groups/core) and Split-K incurs DRAM roundtrips and smaller reduction loops. Optimize `decode_state_mcg` with direct `half2` vector bitcast.
  - Result: Eliminating Split-K reduction roundtrips dropped forward step latency from ~80.9 ms to ~69.8-70.4 ms (decode speed jumped from 12.35 tok/s to 14.02 tok/s). `half2` bitcast replaced scalar shifts/masks across 32 dequant calls per loop step. 0 FP16 diffs in `live_kernel_bit_exact` and `fused_bit_exact`.
  - **The 12.35 → 14.02 tok/s figure is unpaired and does not survive the evidence standard.** Same failure mode as T7.6.8: two arm readings on a box that drifts 4–12% across unpaired samples. What *is* solid here is the **isolated A/B behind it** (`implementation_notes.md` §19.1, one binary, interleaved): contig4 **188.2** vs sk4 **168.1** GB/s on `gate_up`, 177.4 vs 176.7 on `down_proj`, 150.6 vs 140.2 on `q_proj` — contig4 wins every shape, which is why Split-K is gated off. The `half2` bitcast is separately verified bit-exact (0 fp16 diffs). Quote §19.1, not the tok/s delta.

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
  - **MEASURED, final** (same binary, flag A/B, 96 greedy tokens, **16 interleaved pairs**): unfused 1407 dispatches/token, median 13.320 tok/s (75.08 ms/step), spread 4.5%; fused 1167 dispatches/token, median 13.511 tok/s (74.01 ms/step), spread 4.1%. **Median -0.84 ms/step, +1.14%.** B faster in **15/16 pairs**, sign test **p = 0.0005**, paired t = 6.80 on 14 df. 240 dispatches/token removed, exactly 4 × 60 layers as predicted. **The win is ~1%, not the 3.5-4.5 ms the task originally estimated — see the marginal-cost note below.**
  - **MARGINAL COST PER DISPATCH: 3.5 us** (0.84 ms / 240 dispatches). This is the number to plan against, and it revises the remaining fusion estimates by an order of magnitude.
  - **Kernel own time is UNCHANGED** (`KAPLUMBA_DECODE_PROFILE=kernel`, 96 tokens): `kv_norm_rope` 0.8598 ms/token unfused vs 0.8622 ms/token fused, and total `dpk_own_ms_per_token` 71.649 vs 71.644. **The 0.84 ms is NOT saved kernel work — it is removed launch/bubble latency.** The five kernels and the one kernel do the same arithmetic over the same data; what changed is that four boundaries per layer are gone. This is direct confirmation of the bubble hypothesis in the task brief, and it also means the win cannot be scaled by making the kernel itself faster.
  - **BIT-EXACTNESS, end to end**: full stdout over 128 greedy tokens byte-identical between arms, sha256 `2fff2c970b611f64c4a485d8a3a3dc719a0f03293a3f48c167f4dff5466795b2` for both, including the `--show-ids` token list. A single fp16 ULP anywhere in the 60-layer forward would change that list.

- [ ] **T7.6.11: Fused Down Finalize + Post-FFN Residual & Layer Scale**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`
  - Intent: Combine `linear_finalize_simd` and `encode_rms_norm_residual_layer_scale` into a single kernel dispatch (`linear_finalize_simd_with_residual_scale`), eliminating 60 dispatches and `scratch.proj` DRAM buffer traffic.
  - Estimated impact: ~2.0 ms reduction in step latency. **REVISED BY MEASUREMENT: at the measured 3.5 us marginal cost per dispatch, 60 dispatches is ~0.21 ms, not 2.0 ms — an order of magnitude less than estimated.** The original estimate appears to have assumed the dispatch's *own* kernel time, which T7.6.10 measured as unchanged. Build only with that expectation set.

- [ ] **T7.6.12: Fused Post-Attention Norm, Residual, and Gate Input Transform**
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/src/model.rs`
  - Intent: Merge post-attention residual add, pre-FFN RMS norm, and Hadamard gate input transform into a single dispatch per layer, eliminating 120 dispatches per token.
  - Estimated impact: ~2.0 ms. **REVISED BY MEASUREMENT: 120 dispatches x 3.5 us = ~0.42 ms, not 2.0 ms.** Note this one also removes a `scratch.proj`-width DRAM round trip (5376 fp16 x 2 = 10.5 KB per layer, 0.63 MB/token), which at 250 GB/s is ~2.5 us of *bandwidth*, so the dispatch saving dominates.

- [x] **T7.6.13: Attention Fast-Path Single Dispatch for Short Decode Context** — **built, measured, bit-exact, NOT enabled by default**
  - Files: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`, `crates/engine/tests/attention_single_pass_bit_exact.rs`. `model.rs`/`generate.rs` untouched: the arm lives inside the existing `encode_online_attention_split`.
  - New kernel `attention_decode_single_{256_2,512_8}` (the two model shapes only), gated by `KAPLUMBA_DECODE_SINGLE_PASS`, **default off**.
  - Intent was as written: bypass split+combine when the decode KV context is short enough that one threadgroup can hold the reduction.
  - **MEASURED, and the roadmap's 1.5 ms estimate was wrong by ~15x.** Per-layer GPU-clocked own_ms, median of 9 interleaved samples after 3 warm-ups:
    - Crossover is **64 keys**, exactly `ATTENTION_MIN_KEYS_PER_SPLIT` and therefore exactly where `decode_splits` already returns 1. Below it the two paths launch the *identical* grid and single-pass wins by only the combine dispatch: **1.059x** (sliding) / **1.034x** (full) at 64 keys, 1.14x / 1.10x at 16 keys.
    - At 65 keys `decode_splits` returns 2 and single-pass is already **0.68x** (sliding) / **0.63x** (full) — it loses immediately and by a widening margin: 0.31x at 256, 0.19x at 512, **0.127x at the sliding layers' 1024-key cap**; 0.28x / 0.16x / **0.141x** for the full layers at 256 / 512 / 8192.
    - End-to-end value: 60 dispatches x ~0.4 us = **~0.025 ms/token**, not 0.21 ms and nowhere near 1.5 ms. It applies only to the first 64 tokens of a conversation.
  - **Bit-exact at `splits == 1`, and only there.** `to_bits()` identical on **294912/294912** fp16 elements across both shapes x 12 key counts. At `splits == 8` it is *not* bit-exact (6/49152 elements, worst 1 fp16 ULP) — the combine's cross-split rescale is a different association — so the gate is `key_count <= 64` and never anything wider.
  - Left **off by default** because a 0.025 ms/token window that only covers the first 64 tokens is not worth changing the production decode path for. `KAPLUMBA_DECODE_SINGLE_PASS=1` enables it.
  - **Negative result worth keeping:** the sliding window capping context at 1024 for 50 of 60 layers does *not* make single-pass viable at all times, which was the hypothesis to check. A bounded KV context bounds the *work*, not the *parallelism need* — and the parallelism need is set by `kv_heads` (16 and 4 workgroups), which is exactly the number of cores-scale units the split path multiplies.

- [ ] **T8: End-to-end decode benchmark and acceptance validation**
  - Files: `spec/decode-19tok-repacked-250gb/implementation_notes.md`
  - Intent: Measure end-to-end tok/s using `scripts/gpuq run` across reference prompts.
  - Acceptance: Pure autoregressive decode throughput **>= 15.0 tok/s (<= 66.7 ms/step)** against the physical 250+ GB/s ceiling with 100% bit-exact greedy generation, by the paired method in AC5 (>= 16 interleaved pairs, median + spread + paired test).
  - **Corrected from ">= 19.0 tok/s (<= 52.6 ms/step)".** 19 was withdrawn in the spec header; 19 is additionally excluded by arithmetic (`spec.md` §6.2: 19 tok/s needs 264–283 GB/s of *dequantizing* GEMV, i.e. 105–113% of a pure copy). **Current: 13.511 tok/s (74.01 ms/step, 1167 dispatches/token, T7.6.10c, 16 pairs, spread 4.1%).**
  - The target number is under review and is the user's call — see `spec.md` R6. The remaining fusion budget (T7.6.11–13) is ~0.8 ms, which reaches 13.67.



- [x] **T7.6.14: Prove every decode-reachable pipeline is compiled before the timed loop**
  - Files: `crates/kernels/src/pipeline_audit.rs` (new), `crates/kernels/src/{lib,linear,elements,attention,exl3}.rs`, `crates/engine/src/{model,generate}.rs`, `crates/engine/tests/pipeline_warmup_coverage.rs` (new)
  - Intent: A cold kernel inside a timed A/B measures Metal shader compilation, not the kernel, and the error is not small and not signed. Make "the families compile once, at load" a measurement instead of a reading of the code, and make a gap fail the run.
  - Result: **No gap.** 173 pipeline builds, 157 distinct kernels, 10.3 ms, all at load; `late_builds=0` across a 614-token prefill and 48 greedy decode steps on the real model. Every construction site in the four kernel modules now goes through `pipeline_audit::build`, and a test fails the build if one bypasses it.
  - Enumeration: `LinearKernels::new` builds 127 (1 transform + 31 ishizuki + 48 EXL3 slots + 24 NR0 + 2 `gemv_reduce` + 8 `linear_gemv_fast` + 13 fixed) over a library of 111 kernels; `ElementKernels` 21; `AttentionKernels` 25. All behind a `OnceLock`.
  - **Six of them are never dispatched**: `gemv_reduce_sk2`, `gemv_reduce_sk4`, `linear_finalize_simd3_sk2`, `linear_zero_raw`, `attention_dim_256`, `attention_dim_512`. Load cost, zero decode relevance. Do not read a build count as a decode-kernel count.
  - `repacked_gemv_by_name` is **always empty**: `repacked_kernel_names` matches `linear_gemv_repacked_`, and the only occurrence of that string is a comment. Every repacked kernel is named `linear_gemv_ishizuki_k3_*` and is built by `ishizuki_kernel_names` instead. Documented; left in place because `trace_gemv` chains it.
  - Live decode kernels, for the record: `linear_gemv_ishizuki_k3_w2` (420 decoder linears), `linear_gemv_exl3_k6_u_mcg` (`lm_head` -- its trellis starts at `packed_offset % 4 == 2`, so the unaligned arm, not ishizuki), `linear_finalize_simd`, `linear_input_transform`, and two attention specialisations `attention_decode_split_256_2` / `attention_decode_split_512_8` with `attention_decode_combine_256` / `_512`.
  - **Sabotage evidence** (AGENTS.md 3.4, each applied, run, reverted):
    | sabotage | result |
    |---|---|
    | raw `new_compute_pipeline_state_with_function` restored at `attention.rs:323` | `every_pipeline_construction_goes_through_the_audit` FAILED: "attention.rs:323 constructs a pipeline outside pipeline_audit::build" |
    | `linear_finalize_simd` built in a `OnceLock` on first dispatch instead of in `new()` | `decode_shaped_dispatches_compile_nothing` FAILED: "decode-shaped dispatches compiled 1 pipeline(s) after load: [(\"linear_finalize_simd\", 56)]" |
    | (unit) build a pipeline after a mark, call the real `check_no_builds_since` | `#[should_panic]` observed: "KAPLUMBA_ASSERT_NO_LATE_PIPELINES: 1 pipeline(s) were compiled after load (56 us total, ...)" |
  - **The prefill warmup step was added, measured, and REVERTED.** `warmup_decode` could not reach the multi-query kernels (`tokens == 1` is the gate), so a two-query forward was put ahead of the decode step. Interleaved A/B, 614-token prompt: with = load 6010 ms / ttft 6274 ms / 97.90 pp/s; without = load 5687 ms / ttft 6249 ms / 98.27 pp/s. It is **not additive** (the two-query forward takes ~3.0 s but the warmup totals ~3.1 s either way -- the *decode* step is what then costs ~3.0 s; the prefill step only changes which forward pays the weight-path warm), and it **buys nothing on TTFT**. Reverted; the audit arming survived. Consequence recorded in the notes: the ~3 s first-forward cost is not prefill-kernel first-dispatch, and warming the prefill path does not remove it.
  - **`--metrics` adds no per-token readback or synchronisation.** It is read in `crates/cli/src/main.rs:109` *after* `generate_gpu` returns; its only work is one `getrusage` and string formatting. Every tok/s quoted with `--metrics` equals the uninstrumented one. The adjacent real cost is not `--metrics`: `generate_gpu_greedy` (the `--legacy-path` route) evaluates `argmax_with_margin(&logits)` unconditionally -- `MarginLog::record` checks `enabled` *inside* -- at a measured **0.152 ms/token** of waste (0.21% of a 74 ms step), and `last_logits()` clones 512 KiB every step for 0.0066 ms. Neither is inside `step_ms`. `generate_gpu` already gates its margin correctly behind `track_margin`.
  - **A test in this file failed for the wrong reason first.** The three tests in `pipeline_warmup_coverage.rs` share one process-global audit window, and `cargo test` runs them on many threads: `the_attention_family_...` failed listing 140 `linear_*` kernels it had nothing to do with, because another test was inside `LinearKernels::shared()` at the same moment. Fixed with a serialising lock. A red test blaming the wrong code is worse than no test, and it was silent about the property under test.
  - **Two process facts, recorded because they cost hours:** HEAD moved `e9c2785` -> `9cd57a7` -> `379ae9e` under this work, and another session's `git add -A` swept this task's `generate.rs` change into `9cd57a7`. The shared tree's `linear_repacked.metal` also carries an uncommitted `_bench` kernel with a syntax error (`[[buffer(0)],`, missing `]`) that makes every library compile fail. All measurement was done in a worktree at a pinned HEAD with its own `CARGO_TARGET_DIR`.
