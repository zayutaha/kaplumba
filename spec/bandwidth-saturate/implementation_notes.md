# Implementation Notes: Bandwidth Saturation (≥ 20.0 t/s Pure Decode)

## Initial State & Baseline (2026-09-24)

- **Hardware**: Apple M4 Pro (20 GPU cores, 273 GB/s peak memory bandwidth).
- **Model**: Gemma 4 31B EXL3 (~11.20 GB weights per forward pass).
- **Baseline Measured State**:
  - `decode forward-only`: **116.0 ms/token** (8.62 t/s, 96.6 GB/s memory bandwidth = 35.4% bus utilization).
  - `host between tokens`: **0.34 ms/token** (host overhead is negligible).
  - `lm_head` alone (k=6): **4.67 ms** (226.6 GB/s = 83.0% bus utilization).
  - 60 layers take 111.3 ms (1.855 ms per layer).
  - Dispatches per layer: 24 (total 1,440 dispatches per token).

## Architectural Discoveries & Dead-Ends Checked

1. **`had_in` in-kernel prologue is a proven NO-GO (Commit d9e6618)**:
   - Fusing `had_in` prologue into every GEMV threadgroup causes the 5376-point Hadamard butterfly to be recomputed redundantly 336 times per layer, adding 3-8% extra ALU time to an already compute-heavy loop.
   - `had_in_batched` as a separate dispatch takes only ~20 μs, so prologue fusion into every threadgroup is a net loss.
   - Correct approach: Fuse producer operations (RMSNorm, GELU) *into* `had_in_batched`, NOT `had_in` into GEMV.

2. **The 3-bit vs 6-bit Bandwidth Paradox**:
   - `lm_head` (6-bit) streams at **226.6 GB/s** (302 Giga-weights/second).
   - `gate+up` (3-bit) streams at **72.4 GB/s** (193 Giga-weights/second).
   - Because 3-bit weights are half the byte-width of 6-bit weights, reaching 225 GB/s on 3-bit weights requires 600 Giga-weights/second.
   - The current scalar decode loop requires ~7 ALU instructions per weight, capping ALU issue rate at ~300 Gw/s.
   - Solution: Vectorize decode to `uint4` and `dot(float4, float4)` to drop ALU instructions per weight to ~1.75 ops/weight, allowing the ALUs to sustain 600+ Gw/s.
3. **Hardware `dot(float4, float4)` vs Pairwise Float Summation**:
   - `dot(float4, float4)` in Metal evaluates pairwise products with slightly different internal summation rounding than scalar adds, causing 1-8 ULP differences in f32 tests (`gpu_gemv_post_matches_unfused` and `gpu_gemv_fused2_matches_separate_passes`).
   - Using pairwise addition `acc += (x0 * w.x + x1 * w.y) + (x2 * w.z + x3 * w.w)` retains 100% exact bit-equivalence with the reference math while still reaping the 4x ALU instruction reduction from `exl3_decode_quad` (vectorized `uint4` integer multiplies and XORs).
4. **Single-Dispatch GEMV for O and Down (`gemv_splits(336) = 1`) vs K-splits**:
   - Setting `gemv_splits(336) = 1` directs `o_proj` and `down_proj` into `exl3_gemv_post` directly without K-splits, launching 42 threadgroups total (2.1 warps/core), which causes DRAM stalls on M4 Pro (1.18 ms for down).
   - Restoring `gemv_splits(336) = 8` launches 2,688 threadgroups (134 warps/core), rocketing `down` to 0.538 ms (80.6 GB/s, 2.2x faster).
5. **T5 Fused GELU + Hadamard Prologue (`exl3_gelu_had_in`)**:
   - Evaluates `(half)(gelu_tanh(gate) * up)` inside the threadgroup register file and directly multiplies by `suh` before running Sylvester FWT.
   - Eliminates 60 standalone `gelu_mul` dispatches and completely deletes the 21,504-element intermediate DRAM buffer `gu` from `GpuDecoder`.
   - Verified 100% bit-exact in `gpu_gelu_had_in_matches_separate_passes` and `trace_chat_prompt_steps_slow` (golden token 818).
   - Latency dropped to **117.3 ms/token (8.53 t/s, 95.5 GB/s)**.

## 2026-09-24 — RETRACTION: the memory-pattern premise is falsified

The control experiment `bench_stream_bw_slow`
(`crates/engine/src/metal_linear.rs`) had been written but never run. Run on the M4 Pro
over a 512 MB synthetic u32 buffer:

```
P0 contig 512MB: min 2.558 med 2.625 ms -> 210 GB/s
P0 strided nsplit=1: min 0.949 med 1.006 ms (192 MB) -> 212 GB/s
P0 strided nsplit=2: min 0.624 med 0.664 ms (192 MB) -> 323 GB/s
P0 strided nsplit=4: min 0.544 med 0.561 ms (192 MB) -> 370 GB/s
P0 strided nsplit=8: min 0.548 med 0.560 ms (192 MB) -> 368 GB/s
```

`stream_sum_strided` replicates the `exl3_gemv` grid (32 lanes/TG, tile stride
`out_tiles * twords` words) and walks the trellis with **no arithmetic at all**.

**Two earlier conclusions in this file are wrong and are withdrawn:**

- **WITHDRAWN — "the ~21 KB tile stride caps the bus at ~68 GB/s"** (the premise inherited
  from `spec/trellis-dense-repack`, restated as the "336 straws" model in the ASCII
  explanation). The strided walk is *not slower than a fully linear read*: 212 vs 210 GB/s.
  The native `[it][ot]` layout is fine as-is. This retrospectively explains why both dense
  repack candidates (W1, W4) measured **slower** than the plain word layout — they were
  built to remove a stride the hardware handles natively.
- **WITHDRAWN — "per-symbol decode ALU is effectively free"** (from `exl3_gemv_probe`,
  m0 ≈ m2). That probe kept the production per-lane gather structure and the production
  grid, so mode 0 and mode 2 were both pinned at the same non-DRAM ceiling. It measured the
  kernel, not the memory system. The related "14 ops/weight = 113 ms of pure ALU" model is
  likewise unsupported.

**What survives:** the real gap is *inside* `exl3_gemv_spec` — 30–95 GB/s achieved against
212 GB/s available for the identical walk with no math. That ~3x is decomposed
stage-by-stage in `spec/exl3-load-latency`.

**Caveats (do not quote these numbers without them):**
- 370 GB/s exceeds the 273 GB/s nominal bus figure, so it is almost certainly SLC residency
  and/or boost-clock variance on a 192 MB working set. The defensible comparison point is
  the nsplit=1 figure (212 GB/s), which is same-shape against the contiguous control.
- 210 GB/s is a single 512 MB launch. Real linears are 16–87 MB and may sit partly in the
  SLC, so a single "ceiling" number is misleading.
- Consequently **20.0 t/s (236.7 GB/s) and the ≥ 19.5 t/s gate (242.6 GB/s) both sit above
  the 210 GB/s measured linear-read figure**, implying ~17.8 t/s at 11.833 GB/token. The
  bar is kept by explicit user decision; the assertion stays unflipped and is expected to
  stay red. Reaching 20 t/s at k=3 would require fewer bytes per token, not better kernels.

## 2026-09-25 — T5.1: Fused Gate+Up GEMV with RMSNorm Prologue (`encode_rms_fused2_with_scratch`)
- Fused `pre_ff_ln` RMSNorm prologue directly into `gate` and `up` GEMV passes using `encode_rms_fused2_with_scratch`.
- RMSNorm and Hadamard transform run directly on `x_mid` without intermediate DRAM write to `xn`.
- `gate` and `up` outputs written directly to contiguous slices of `gu_raw`.
- `encode_gelu_had_in_down` updated to consume `DeviceSpan` directly from `gu_raw` slices.
- Eliminates 60 separate RMSNorm launches per token (and avoids DRAM round-trip for 60 * 5376 floats).
- Forward-only decode latency drops to **115.0 ms/token (8.70 t/s, 97.4 GB/s)**. Bit-exact parity verified with golden token 818.

## 2026-09-25 — T5.2: Coalesced 64-Byte SIMD Store for GEMV (`exl3_gemv`, `exl3_gemv_spec`, fused)
- Replaced the 8-lane strided 2-float stores `(g & 3u) == 0u` in `exl3_gemv`, `exl3_gemv_spec`, `exl3_gemv_fused2`, and `exl3_gemv_fused3` with a 16-lane contiguous 64-byte burst store via `simd_shuffle`.
- Eliminates uncoalesced memory transactions across 320,000+ threadgroup stores per token.
- Bit-exact parity maintained across all test suites and `trace_chat_prompt_steps_slow` (token 818).

## 2026-09-25 — T6: Fused Attention Post-Processing (`kr_rms_rope` + Direct Cache Write)
- Implemented `kr_rms_rope` in `fuse.metal` to evaluate per-head RMSNorm and Neox RoPE in threadgroup registers without DRAM round-trips.
- Applied `kr_rms_rope` for Q (in-place in `q_span`) and K (directly targeting the KV cache row buffer).
- Eliminated `kr_copy` by encoding weightless RMSNorm on V directly into the KV cache row buffer.
- Cuts 3 kernel launches per layer (180 dispatches/token eliminated).
- Bit-exact parity verified with new `rms_rope_matches_separate_passes` test across all configurations.
- Forward-only decode latency drops to **114.1 ms/token (8.77 t/s, 98.2 GB/s)**. Bit-exact golden token 818 verified (6.57s).

## 2026-09-25 — T6.1: Fused Residual Additions (`kr_rms_add` + `kr_rms_add_scale`)
- Implemented `kr_rms_add` and `kr_rms_add_scale` in `fuse.metal` and `fuse.rs`.
- Fused post-attention RMSNorm and residual addition (`x = x + post_attn_norm(attn_out)`), writing directly to `x_mid`.
- Fused post-feedforward RMSNorm and residual addition with layer scaling (`x = (x + post_ff_norm(ff_out)) * scalar`), writing directly to `x`.
- Eliminates 120 separate elementwise add kernel dispatches per token (cutting total decode dispatches by another 120).
- Bit-exact parity verified in `rms_add_and_scale_matches_separate_passes`.
- `trace_chat_prompt_steps_slow` latency drops to **6.00s** (predicting golden token 818 bit-identically). Forward decode throughput reaches 8.67 t/s (115.3 ms/token, 97.1 GB/s).

## 2026-09-25 — T6.2: FC-Specialization and Quad Vectorization for Fused2 & Fused3
- Vectorized and specialized `exl3_gemv_fused2` (Gate+Up) and `exl3_gemv_fused3` (QKV) in `exl3.metal` using function constants `FC_ALIGNED`, `FC_MODE`, and `FC_K` alongside `exl3_decode_quad_const`.
- Replaced per-quad scalar function calls with 4-element SIMD vector decoding directly into registers, eliminating branch conditions and bit-shifts on the inner loop.
- Wired function constant specialization into `exl3.rs` via `spec_fn_pipeline` for both fused kernels.
- Benchmark breakdown (`bench_decode_breakdown_slow`) measured across all 60 layers:
  - Gate + Up (`exl3_gemv_fused2`): 47.47 ms (72.8 GB/s)
  - Down Proj (`encode_gelu_had_in_down`): 24.00 ms (72.7 GB/s)
  - QKV (`exl3_gemv_fused3`): 21.97 ms (52.4 GB/s)
  - O Proj (`encode_linear_with_scratch`): 11.33 ms (43.4 GB/s)
  - LM Head: 5.14 ms (221.6 GB/s)
  - All non-GEMV stages (attention, rope, rms, adds): 4.17 ms total (< 4% of decode).
- `trace_chat_prompt_steps_slow` execution time drops from 6.00s to **5.76s** with bit-exact prediction of golden token 818.
## 2026-09-25 — T6.3: FC-Specialization and Quad Vectorization for Post-Hadamard Kernels
- Propagated compile-time function constants `FC_ALIGNED`, `FC_MODE`, and `FC_K` alongside `exl3_decode_quad_const` to `exl3_gemv_post` (LM head & single linears), `exl3_gemv_post_fused2` (Gate+Up production path), and `exl3_gemv_post_fused3` (QKV production path).
- Replaced runtime dynamic branch conditions, bit-shift calculations, and scalar decode function calls in the production single-dispatch paths with compile-time constants folded by the Metal compiler.
- Verified bit-exactness: all 23 kernel unit tests and all 135 engine unit tests pass.
- Verified end-to-end golden token parity: `trace_chat_prompt_steps_slow` predicts golden token 818 ("The") bit-identically.

## 2026-09-25 — T6.4: Unified Attention Prologue (`kr_rms_rope_qkv`)
- Implemented `kr_rms_rope_qkv` in `fuse.metal` combining per-head RMSNorm + Neox RoPE for Q, per-head RMSNorm + Neox RoPE for K, and unweighted RMSNorm for V into a single unified kernel dispatch of `n_heads + 2 * n_kv` (64) threadgroups.
- Q roped heads written directly in-place into `q_span`, K roped heads written directly into `k_cache_span`, and V normalized heads written directly into `v_cache_span`.
- Eliminates 2 kernel launches per layer (120 launches/token eliminated).
- Unit tested bit-exactness in `rms_rope_qkv_matches_separate_passes`. All 24 kernel tests and 135 engine unit tests pass.
- Verified golden token 818 ("The") in `trace_chat_prompt_steps_slow`.
- Pass 2 latency (`RoPE Q,K + Norm V`) dropped from 0.60 ms to **0.34 ms** across 60 layers. Total decode latency drops to **113.14 ms**.

## 2026-09-25 — Coherence gates + decode bottleneck campaign (8.63 t/s, still 112.4 ms/token)

### Gates landed and proven effective
- `9ba195b` added `trace_chat_prompt_steps_slow` (asserts all 20 prompt-step argmaxes
  `[236773,236773,236773,616,236749,7243,1331,3207,7001,1852,108,669,100,100,100,100,45518,107,101,818]`)
  and `decode_coherence_slow` (asserts the 24-token `EXPECTED` vector +
  `text.contains("capital of France is Paris")`). Both caught the first fused-dispatch bug
  on their first run.
- **Run them one at a time.** They are *env-gated*, not `#[ignore]` — `KAPLUMBA_RUN_SLOW=1 cargo test
  -p kaplumba-engine --release --lib trace_chat_prompt_steps_slow -- --nocapture` (drop `--ignored`).
  Running two gates in one process loads the 11.8 GB artifact twice: 157 s vs 8 s each.
  Model load is one `load_exl3` into a single shared device buffer (64-byte aligned offsets →
  `aligned=1` always); peak RSS 8.14 GB single test / 8.93 GB both — no double residency.

### Isolated stage table (`bench_decode_breakdown_slow`, 60 layers) — sums to 112.55 ≈ 112.40 measured
| stage | ms | MB | GB/s |
|---|---|---|---|
| QKV | 21.90 | 2023 | 92 |
| RoPE Q,K + Norm V | 0.35 | — | — |
| Attention (s=1) | 0.40 | — | — |
| O proj | 11.16 | 1156 | 104 |
| Post-Attn RMSAdd | 1.65 | — | — |
| Gate+Up | 47.07 | 5202 | 111 |
| Down (GELU+Had) | 23.75 | 2601 | 110 |
| Post-FF RMSAddScale | 1.58 | — | — |
| FinalNorm+LM | 4.69 | 1057 | **225** |
No hidden cross-stage overhead. `MetalLinear::calls()` does not count `encode_attention`/`encode_rms`.

### Dispatch count is NOT the bottleneck (probe in `encode_token`, `KAPLUMBA_PROBE_SKIP_EW`/`_ATTN`)
baseline 113.7 ms · skip elementwise (120 dispatches) 110.1 · skip attn+rope 112.5 · both 110.1.
**Marginal ≈ 0.015 ms/dispatch.** 591 dispatches/token (`bench_dispatch_count_slow`, 9.85/layer).

### Per-in-tile-step cost is fixed at ~300 ns — and it is NOT latency/MLP-bound
Derived: bandwidth = resident_warps(~336) × bytes_per_step / step_time. Everything that changes
concurrency is flat: `FC_WARPS` (1/2/4/8 → 114.1/113.5/113.3/113.3 ms), `nsplit` 1/2/4/8 flat,
`exl3_gemv_mg` no better. **Deeper MLP is also flat**, which kills the DRAM-latency theory:
- unroll-2 load hoisting (issue all 8 loads before any decode): Gate+Up 46.05 → 46.73 ms.
- unroll-4 hoisted loads: Gate+Up → 46.32 ms, QKV 21.90 → 21.83 ms.
Conclusion: the step is **issue/ALU-bound**, not outstanding-bytes-bound. Do not retry MLP tuning.

### Codeword decode = 35% of GEMV time — and it cannot be looked up
- Probe: stub `exl3_decode_quad_const` to `return float4(1.0f)` (loads + FMAs intact) →
  **Gate+Up 46.76 → 30.05 ms (−35%)**.
- Attempted fix (REVERTED): host-built 65536-entry f32 codebook + `exl3_decode_quad_lut(cb, cw)`
  bound at `buffer(7)` on `exl3_gemv_spec/fused2/fused3/post`. **4–7× regression**:
  O 11.06→46.44, Down 23.75→105.06, LM 4.74→24.75, `encode_gemv_fused2` 46.78→221.92 ms.
  Control (keep `cb` bound, restore inline decode) → 11.21/46.49/4.69 = baseline, so the binding
  was correct and the **LUT itself** is the cost.
- Cause: `cw` is data-dependent, so lookups are random 4-B reads into 256 KB — each fetches a
  64-B line (16× amplification, no reuse within a step) and sits on the critical path *after* the
  trellis DRAM load. Any `cw → value` precompute needs a dependent load, which costs more than the
  ~50 ALU ops it replaces.
- **Do not retry a codebook/weight LUT** unless keys are lane-adjacent (coalesced) or the table
  fits in registers. `git checkout -- crates/kernels/src/exl3.metal crates/kernels/src/exl3.rs` reverts.

### Structural reason lm_head is 2× faster per byte
`words_per_tile` = 96 (lm) vs 48 (all layer tensors, from `quantization_config.json` trellis shapes
`[336,512,48] q, [336,256,48] k, [512,336,48] o, [336,1344,48] gate, [1344,336,48] down`).
Each lane issues 4 `ld_word`s per in-tile step = 512 B of load traffic regardless of cell size, so
redundancy = 512 / cell_bytes = **5.33× for layers (96 B cell) vs 2.67× for lm (192 B cell)** →
105 vs 225 GB/s. Larger cells amortize the fixed load count; the fix would be a trellis re-slice
with a matching output mapping, not a kernel-local change.

### Read-ceiling model caps a perfect GEMV at ~10.3 t/s
`t(ms) = 0.216 + MB/269` fitted to `bench_read_ceiling_slow`. Applied per stage:
QKV 20.5, O 17.3, Gate+Up 32.3, Down 22.6, LM 4.14 + non-GEMV 4.7 ≈ **97 ms → 10.3 t/s**.
Measured 112.4, so the whole gap to the model is **Gate+Up (+14.5 ms)**; O actually *beats* its
model by 6.2 ms. Hitting 12 t/s (83.3 ms, 144.5 GB/s) requires beating the read ceiling, i.e.
raising bytes-per-step (larger `words_per_tile`) — not tuning dispatch geometry.
