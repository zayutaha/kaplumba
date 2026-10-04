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

**Status:** ✅ done. `attention_prefill_flash_body` + `KAPLUMBA_FA_FLASH` in
`attention.metal`; `AttentionQk::Flash` + `attention_flash_pipeline` in
`attention.rs`; `attention_qk_for(groups, head_dim)` now selects it for
`head_dim == 512 && groups == 8`.

**Measured, interleaved A/B, 3 rounds (`measure-flash.tsv`):**

| | 614 | 8k |
|---|---|---|
| flash (new default) | 98.84 / 99.78 | 84.96 / 85.01 / 84.97 |
| scalar (`KAPLUMBA_SCALAR_ATTENTION=1`) | 99.84 / 99.97 | 78.59 / 78.75 / 78.14 |
| delta | −0.6% (noise) | **+8.1%** |

Also +5.5% over the MMA arm it replaces (80.52 @8k on the shipped default).
Correctness: `attention_flash_matches_scalar_on_the_full_shape`, worst fp16
abs diff **4.88e-4** (one ulp) over 5 full-shape cases incl. tails, multi-query
and non-unit scale. **Sabotage check (AGENTS.md 3.4):** forcing `kt0 = 2*(lane & 15)`
raises the diff to **0.374756** (77x the 5e-3 tolerance) and fails the test with
the diagnostic message; reverted. Full workspace suite: same **4 pre-existing**
failures as at base, no new ones.

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

**Status:** 🔄 in progress. 8k done (above, +8.1%). Query blocking built,
measured, **rejected**; the kernel deficit is now explained and an instrument
exists. **Next:** raise MMA/fragment-load in the QK loop.

### T4a — query blocking (`BQ`): built, correct, **does not pay**

`attention_prefill_flash_body<HEAD_DIM, NSG, BQ>` with two instantiations
(`online_attention_flash_512_8` BQ=2, `online_attention_flash_bq1_512_8` BQ=1),
shared grid `ceil(queries/BQ) * kv_heads`, block chosen by
`flash_query_block_for(key_count)`, `KAPLUMBA_FLASH_BQ=1|2` to override, and
`encode_online_attention_qk_block` so tests cover both. `KAPLUMBA_FULL_QK`
adds a **full-shape-only** arm override, because the global flags also move the
50 sliding layers and cannot express a full-layer A/B.

Correctness: both blocks vs scalar, worst fp16 abs diff **4.88e-4** over 6
full-shape cases including two new 5-query cases (BQ=2 grid tail). **Sabotage
check:** dropping `b` from `msb[b*NSG+sg]` fails only BQ=2 on the multi-query
cases (0.575–0.745 vs tol 5e-3) and leaves BQ=1 at one ulp.

**Measured, interleaved (`measure-bq.tsv`):**

| prompt | BQ=1 | BQ=2 |
|---|---|---|
| 614 | 99.671 | 99.472 |
| 8k | 83.912 | 82.436 |
| 24k | **63.154** | **59.250** |

-1.8% at 8k, **-6.2% at 24k**. The §5 K/V byte-traffic model does not describe
this kernel. Recorded as a negative result; BQ=1 is the shipped rule.

### T4b — the instrument, and the real limiter

`attention_arm_throughput_at_prefill_geometry` (new, in
`attention_qk_mma_bench.rs`): no model load, real prefill geometry
(`queries = 1024`, one threadgroup per (query, KV head)), TFLOP/s per arm.
~25 s for all arms at two key counts.

| arm | 8192 keys | 24576 keys |
|---|---|---|
| scalar | 397.34 ms (1.38) | 1243.66 ms (1.33) |
| **mma** | **253.21 ms (2.17)** | **880.21 ms (1.87)** |
| flash bq1 | 360.50 ms (1.52) | 1184.23 ms (1.39) |
| flash bq2 | 384.45 ms (1.43) | 1216.92 ms (1.36) |

**MMAs per fragment load is the limiter.** A `simdgroup_load` is a real
instruction; the GEMM runs at 1.60 MMA/load (6.75 TFLOP/s), the flash QK at
0.50 (1.39 TFLOP/s), and 1.60/0.50 = 3.2 predicts the 3.6x measured gap. Heads
(N) is fixed at 8, so the only axis that can grow is keys per simdgroup.

**Bug found by this instrument:** the first BQ cut loaded the P fragment inside
the n-tile loop, dropping PV from 0.89 to 0.67 MMA/load. Fixed by preloading
the `BQ` P fragments once per k-step (0.89 at BQ=1, 1.60 at BQ=2).

**Remaining:** raise the QK's M-tiles from 1 to 2, and decide the shipped
full-shape arm (E2E `KAPLUMBA_FULL_QK` A/B in flight).

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
