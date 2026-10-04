# Prefill ≥120 tok/s at 16k–24k context

## Goal

Push prefill throughput from **~78 tok/s** (measured at 614 tokens, `PREFILL_CHUNK=1024`)
to **≥120 tok/s** at short context, and **≥80 tok/s** at 24k context, by closing the
two dominant gaps: GEMM MMA utilization (35% → 60%+ of tensor peak) and attention
throughput at long context (scalar QK → MMA-based).

The aspirational target is 200+ tok/s. The physics ceiling is ~160 tok/s at short
context and ~90 tok/s at 24k (see §3). We aim for the highest achievable
throughput and report honestly where the wall is.

## Non-goals / restrictions

Carried from `AGENTS.md` §8, **non-negotiable**:

- **No quality loss.** Zero degradation in output quality. Greedy output on
  reference prompts must remain correct.
- **Numerical accuracy:** worst normalized GEMM error ≤ **2.16e-3** against
  the CPU reference (the current approved bound from `spec/engine-performance`
  §2.1). Must not worsen.
- **No model memory increase.** Weight footprint must stay ≤ **11.51 GiB**.
  Repacking into a different bit layout is allowed; enlarging is not. Not a
  single GB.
- **No new large scratch buffers.** Existing 372 MB scratch at
  `PREFILL_CHUNK=1024` is acceptable. New fixed-size buffers for attention
  partials (already exist at 1.05 MB) are fine.
- **No Python** in engine/product code.
- Single-user chat only. No batching for its own sake.
- **No KV cache quantization.** No lossy requantization.
- **No materialized score matrix / no O(L²) prefill scratch.** Prefill attention
  must stay online-softmax (flash-style) with O(L) working set.
- Scratch buffers explicitly owned, fixed-size, independent of context length.
- Peak RSS is a measured acceptance number, not a prediction.

### Additional constraints from this interview

- **Context length: 16k–24k.** The system must handle prompts up to 24k tokens.
  `max_sequence_len` is already set from `max_position_embeddings` (262144),
  but `HOST_TOKEN_SLOTS` is 8192 and KV sizing / scratch have not been validated
  at 24k.
- **Decode improvements welcome.** Current decode is ~15 tok/s. If prefill
  optimizations also help decode, include them. Do not regress decode.
- **Nondeterminism is a separate concern** (the chunked-prefill hazard documented
  in `spec/engine-performance` §14.3). Not addressed here.

## Requirements

- **R1 — Baseline at 16k and 24k.** Measure current prefill tok/s, GEMM TFLOPS,
  attention ms/token, and peak `phys_footprint` at 1k / 8k / 16k / 24k. This is
  the "before" for every optimization.

- **R2 — Attention MMA.** Replace the scalar QK dot product in
  `attention_gqa_body` with `simdgroup_multiply_accumulate` for the QK score
  computation. At 24k context, attention is 40–60% of prefill and the scalar
  QK loop is instruction-bound (57% `cvt.f32.f16` / 29% FMA / 14% loads,
  per `spec/engine-performance` §16). An MMA-based QK batches 8 keys per step
  and eliminates the per-key float conversion.

- **R3 — GEMM MMA utilization.** Investigate and implement at least one approach
  to close the gap between 5.31 TFLOPS (35% of peak) and the ~7.8 TFLOPS
  staging-stripped ceiling (76.7 tok/s). Candidates:
  - Tile shape variants that improve the load:MMA ratio (e.g., 4 simdgroups ×
    4 row blocks × 4 column blocks = 0.5 loads/MMA vs current 0.75)
  - Weight layout optimized for the MMA tile access pattern
  - Better occupancy through reduced threadgroup memory
  - Any approach NOT listed in `spec/engine-performance` §14 (rejected experiments)

- **R4 — Context extension validation.** `HOST_TOKEN_SLOTS` (8192) must be
  raised to support 24k prompts. KV cache sizing must be validated:
  - Full-attention KV at 24k: 10 layers × 4 heads × 512 × 2 × 2B × 24k = **1.88 GiB**
  - Sliding-window KV: 50 layers × 16 heads × 256 × 2 × 2B × 1024 = **0.78 GiB**
  - Total KV at 24k: ~**2.66 GiB**
  - Weights: 11.51 GiB
  - Total without scratch: ~14.17 GiB against 24 GiB RAM
  - Scratch at PREFILL_CHUNK=1024: 0.37 GiB
  - Must fit with OS overhead.

- **R5 — PV MMA.** If R2 lands and attention is still significant at 24k,
  investigate MMA for the PV (weighted value accumulation) step. Currently
  the PV loop reads one value vector per key with scalar `wt * float4(...)`.

- **R6 — Measured tok/s at 1k / 8k / 16k / 24k** with achieved TFLOPS and
  peak `phys_footprint`, reported against the roofline.

- **R7 — No regression.** All existing tests pass. Five reference prompts
  produce correct greedy output. Decode ≥ 15 tok/s at short context.

## Physics ceiling (reference, not a target)

| metric | value | source |
|---|---|---|
| tensor peak (measured) | 15.18 TFLOPS | `mma_peak.rs`, 16 chains |
| SIMT fp32 peak | 8.76 TFLOPS | `alu_peak.rs` |
| GEMM FLOPs/token | 5.75e10 | 2 × 28.74e9 params |
| GEMM-only ceiling (tensor, attention=0) | 264 tok/s | 15.18e12 / 5.75e10 |
| current GEMM TFLOPS | 5.31 | 35% of peak |
| staging-stripped diagnostic ceiling | 76.7 tok/s | current tile shape |
| current prefill @ 614 | ~78 tok/s | PREFILL_CHUNK=1024 |
| current prefill @ 8k | ~75 tok/s | measured |

### Projected ceilings by context length

At long context, attention's O(L²) term dominates. Assuming:
- GEMM at 60% tensor peak (9.1 TFLOPS) = 6.3 ms/token
- Attention with MMA (2.4× speedup from measured scalar)
- Other overhead: ~1 ms/token

| context | GEMM ms | attn ms (MMA) | other | total ms | tok/s |
|---|---|---|---|---|---|
| 614 | 6.3 | 0.4 | 1.0 | 7.7 | **130** |
| 8k | 6.3 | 2.5 | 1.0 | 9.8 | **102** |
| 16k | 6.3 | 5.5 | 1.0 | 12.8 | **78** |
| 24k | 6.3 | 9.5 | 1.0 | 16.8 | **60** |

These are aggressive estimates. The GEMM column assumes 9.1 TFLOPS which is a 
1.7× improvement over current. Attention MMA has not been measured, so the 2.4×
factor is an estimate from `spec/engine-performance` §16. The "other" column is
layernorm + RoPE + dispatch overhead, currently ~1 ms at short context.

**For 120 tok/s at 24k, we would need GEMM at ~12 TFLOPS (79% of peak) AND
attention at 4× the current speed. This is at the edge of physical possibility.**

## Acceptance criteria

1. Prefill ≥ **120 tok/s** at 614 tokens, or a measured attribution of every
   remaining ms/token to a named hardware wall.
2. Prefill ≥ **80 tok/s** at 8k tokens.
3. Prefill measured and reported at **16k and 24k** tokens with honest numbers.
4. Achieved GEMM TFLOPS ≥ **7.5** (50% of tensor peak), or a measured
   demonstration that the tile shape ceiling has been reached.
5. Attention throughput improved by ≥ **1.5×** at 8k context.
6. Peak `phys_footprint` reported at 1k / 8k / 16k / 24k, within the 24 GiB
   system budget.
7. All existing tests pass. Five reference prompts correct. Decode ≥ 15 tok/s.
8. Worst normalized GEMM error ≤ 2.16e-3.
9. `HOST_TOKEN_SLOTS` raised to ≥ 24k, and a 24k prefill demonstrated.

## Open questions

- **OQ1 — Attention MMA register pressure.** An 8×8 MMA fragment for QK needs
  8 query rows = 128 accumulator float registers per lane. The notes say
  128 registers per lane spills. The actual spill threshold on M4 Pro is
  unknown and must be measured, not assumed. If it spills, a 4-key MMA tile
  or a mixed scalar/MMA approach may be needed.

- **OQ2 — PV MMA viability.** The PV step multiplies a single scalar weight
  by a full value vector. MMA would batch 8 keys × 8 dims, but the weight
  is one value per key (not a matrix). A `simdgroup_multiply` where one
  operand is a diagonal may not be efficient. Measure before building.

- **OQ3 — PREFILL_CHUNK at 24k.** At 24k tokens with PREFILL_CHUNK=1024,
  there are 24 chunks. Larger chunks reduce dispatch fragmentation but increase
  scratch. The sweet spot may differ from the current 1024.

- **OQ4 — Host-side token buffer.** `HOST_TOKEN_SLOTS = 8192` is a fixed buffer
  for staged host tokens. Raising it to 24k+ is straightforward but adds 64 KB
  of buffer. Verify no other code path assumes 8192.
