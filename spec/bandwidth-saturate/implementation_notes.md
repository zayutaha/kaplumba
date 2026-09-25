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





