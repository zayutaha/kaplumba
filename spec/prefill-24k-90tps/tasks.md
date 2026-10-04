# Tasks: 90 tok/s prefill and 14 tok/s decode at 24k

Small, independently verifiable steps. Updated immediately after completion.

---

## T1 — Diagnose the 24k decode collapse

**Status:** ✅ done (diagnosis in `implementation_notes.md` §2b/§2c; no code).

**Result — the collapse was a profiler artifact.** Clean (unprofiled) 24k
decode is **5.44 tok/s**, not 0.30–0.43; the `KAPLUMBA_GPU_PROFILE=1` run
reported 0.082 tok/s on the same prompt. `swaps 0`, 18.35 GiB peak footprint,
`sys` 20 s of 434 s → **no memory cliff** (OQ1 answered). Decode is
latency-bound in the attention kernels: the 24k decode grid is only
`16 splits × 4 kv_heads = 64` workgroups for the 10 full layers, each serially
scanning its split.

**Prefill budget at 24k (the number that matters):** GEMM 213.7 s, attention
192.9 s (full ≈ 176.7 s, sliding ≈ 16.3 s), other 3.2 s. Target 90 tok/s =
266 s, so **AC2 needs attention 193 s → ≤ 52 s with GEMM untouched**. Full
attention currently runs at **1.06 TFLOPS** (12% of SIMT peak, 7% of tensor
peak) → at 6 TFLOPS the run lands at 91.8 tok/s. Reachable only via MMA.

---

## T2 — Decode at 24k: 5.44 → 14 tok/s (2.6×)

**Status:** ☐ not started. **Depends on:** T1.

**Files:** `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`.

**Intent:** The decode split kernel (`attention_gqa_body` with `PARTIALS=true`)
shares the scalar QK/PV structure with prefill. It is latency-bound: 64
workgroups for the full layers, each 8 simdgroups × 47 serial 32-key tiles.

**Work:** raise in-flight parallelism per workgroup (more resident splits, or a
flash-style MMA split body), *not* more splits (already clamped at 16).

**Test:** decode tok/s at 24k ≥ 14, no short/8k decode regression, five prompts
correct.

---

## T3 — Prefill attention: flash-style MMA, 64-key tiles

**Status:** ☐ not started. **This is the only path to AC2.**

**Intent:** Fix all three structural costs read from `attention_gqa_mma_body`:
(a) only 4 of 8 simdgroups engage QK; (b) PV is scalar with 8× redundant V
read+convert across the GQA group; (c) the Q fragment reloads from `qhstage`
on every `d8` step although it is key-tile-invariant.

**Files:** `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`.

**Work:**
1. Widen the tile to 64 keys: eight 8-key m-tiles, one owned per simdgroup,
   so all 8 engage QK.
2. Replace scalar PV with `simdgroup_multiply_accumulate`:
   `P[8 heads × 64 keys] × V[64 keys × 512 dims]`, V loaded from device with
   `simdgroup_load(.., kv_stride)` (no threadgroup V staging; 64×512 halves is
   over the 32 KB cliff).
3. Hoist the Q fragment load out of the key loop where register pressure allows.

**Test:** `attention_qk_mma_matches_scalar` + `attention_single_pass_bit_exact`
green (pin `AttentionQk::Scalar` where required); `attention_qk_mma_bench`
full-shape ratio; then end-to-end 8k A/B before spending a 24k run.

---

## T4 — Prefill attention: verify the full-attention win end to end

**Status:** ☐ not started. **Depends on:** T3.

**Test:** interleaved 8k and 24k A/B, medians with spread; 24k ≥ 90 tok/s or a
measured explanation of the residual.

---

## T5 — Verify GEMM at 24k

**Status:** ☐ not started

**Intent:** Confirm GEMM per token is context-independent at 24k (R4).

**Test:** profiled GEMM `own_ms`/token at 8k vs 24k within spread.

---

## T6 — Final measurement and acceptance

**Status:** ☐ not started. **Depends on:** T1–T5.

**Work:** interleaved A/B prefill (8k/16k/24k) and decode (short/8k/16k/24k),
TFLOPS, peak `phys_footprint`, five reference prompts, worst GEMM error, full
suite. Evaluate AC1–AC5.

**Test:** scorecard with medians and spread.
