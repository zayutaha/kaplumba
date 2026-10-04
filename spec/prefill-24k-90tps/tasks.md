# Tasks: 90 tok/s prefill and 14 tok/s decode at 24k

Small, independently verifiable steps. Updated immediately after completion.

---

## T1 — Diagnose the 24k decode collapse

**Status:** 🔄 in progress

**Intent:** Attribute the 2.3–3.3 s/token at 24k to a named cause (R1). Every
decode change is gated on this.

**Files:** none (measurement).

**Work:**
1. `KAPLUMBA_GPU_PROFILE=1` at 24k, 4 decode tokens, `/usr/bin/time -l`,
   `sysctl vm.swapusage` before/after.
2. Compare decode per-layer kernel `own_ms` at 24k vs the 8k profile: full
   attention should scale ~2.9×, sliding stay flat, everything else flat.
3. Read `getrusage` `swaps` and the system swap delta.
4. Note the inter-boundary gap for each decode token.

**Test:** a written attribution — every second of the per-token cost assigned to
a kernel, a host step, or paging — or an explicit "cannot attribute" with the
profile attached as the reason.

---

## T2 — Fix the decode cause found in T1

**Status:** ☐ not started. **Depends on:** T1.

**Files:** TBD by T1.

**Work / Test:** defined once T1 lands. Must show decode ≥ 14 tok/s at 24k with
no 8k/short-context decode regression, five prompts still correct.

---

## T3 — Prefill attention: query blocking

**Status:** ☐ not started

**Intent:** Amortise K/V load + `half→float` conversion across `Q_BLOCK` queries
per threadgroup (R3.1).

**Files:** `crates/kernels/src/attention.metal` (`attention_gqa_body`),
`crates/kernels/src/attention.rs` (grid).

**Work:**
1. Add a `Q_BLOCK` template parameter; threadgroup handles `Q_BLOCK` queries of
   one KV head; per-query accumulators and softmax state.
2. Tail query handling when `queries % Q_BLOCK != 0`.
3. Keep the scalar arm's bit-exactness with the single-pass kernel
   (`attention_single_pass_bit_exact` pins `AttentionQk::Scalar`).

**Test:** `attention_qk_mma_matches_scalar` and the bit-exact test green;
`Q_BLOCK=2` vs `1` A/B at 8k with medians; sabotage-check the Q_BLOCK indexing.

---

## T4 — Prefill attention: 64-key MMA publish

**Status:** ☐ not started. **Depends on:** T3 result.

**Intent:** Engage all 8 simdgroups and halve the publish barrier rate on the
full-attention shape (R3.2).

**Files:** `crates/kernels/src/attention.metal` (`attention_gqa_mma_body`).

**Test:** `attention_qk_mma_bench` full-shape ratio; end-to-end 8k/24k A/B.

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
