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
4. **Single-Dispatch GEMV for O and Down (`gemv_splits(336) = 1`)**:
   - Setting `gemv_splits(336) = 1` directs `o_proj` and `down_proj` into `exl3_gemv_post` directly instead of K-splits (nsplit=8).
   - Eliminates 120 dispatches per token (2 dispatches eliminated per layer across 60 layers) and cuts all DRAM roundtrips to `scratch.partial`.
   - All 131 tests pass with 100% bit-exact parity including `trace_chat_prompt_steps_slow` (predicting token 818).


