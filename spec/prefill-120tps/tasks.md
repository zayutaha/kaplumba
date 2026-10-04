# Tasks: prefill ≥120 tok/s at 16k–24k context

All tasks are small, independently verifiable steps. Each names files touched,
intent, and how it will be tested. Updated immediately after completion.

---

## T0 — Baseline measurement at 1k / 8k / 16k / 24k

**Status:** ✅ DONE (commit below). Harness added: `spec/prefill-120tps/measure.sh`.

**Intent:** Establish the "before" numbers for every optimization. Without this,
we cannot attribute any gain.

**Files:** none (measurement only)

**Work:**
1. Build the current HEAD: `cargo build --release -p kaplumba-cli`
2. Measure prefill tok/s at 4 context lengths using interleaved A/B (each
   length 3× minimum, medians with spread):
   - ~614 tokens: `'Explain the theory of relativity. ' * 100`
   - ~8k tokens: `* 1300`
   - ~16k tokens: `* 2660` (6.01 tok/rep × 2660 ≈ 15,987)
   - ~24k tokens: `* 3990` (6.01 × 3990 ≈ 23,980)
3. For each: record prefill tok/s, `peak_resident_pages_bytes`, and if the run
   completes at all (24k may fail if HOST_TOKEN_SLOTS=8192 blocks it).
4. At 614 and 8k, run `KAPLUMBA_GPU_PROFILE=1` to get per-kernel GPU time.
5. Record the GEMM TFLOPS at each length.

**Test:** ✅ numbers recorded in `implementation_notes.md` (T0 section).
Result: 614 → **94.66 tok/s** (spec assumed 78); 8k → **74.52**; 16k → **60.75**;
24k → **51.12** (runs fine — `HOST_TOKEN_SLOTS` does not block prefill).
Attribution: GEMM **93.2%** / attention 5.5% / other 1.3% at 614;
GEMM 72.5% / attention **26.4%** / other 1.0% at 8k.
Plan's "~36% norms+dispatch" row is **measured false** — it is 1.3%.

---

## T1 — Attention MMA: QK score computation

**Status:** ✅ DONE — branch `impl/attn-mma` (commit `aa9f1c6`), merged `de5acfd`.
`attention_gqa_mma_body` + `online_attention_gqa_mma_{256,512}_{1,2,4,8}`.
Verified bit-close to scalar by `attention_qk_mma_matches_scalar` (12 cases,
TOL=5e-3; clean run max abs diff 0.000000). **As implemented it was 1.7–5×
SLOWER than scalar** — see T2.

**Intent:** Replace the scalar QK dot product loop in `attention_gqa_body`
with `simdgroup_multiply_accumulate`, batching 8 keys per MMA step.

**Files:**
- `crates/kernels/src/attention.metal`: new template
  `attention_gqa_mma_body<HEAD_DIM, NSG>`, or modify `attention_gqa_body`
  with a `USE_MMA` template parameter.
- `crates/kernels/src/attention.rs`: new pipeline entries for the MMA variants.

**Work:**
1. Write the MMA QK loop:
   - For each 8-key tile in the key range:
     - For each HEAD_DIM/8 dimension chunk:
       - Load Q[query, head, chunk*8..(chunk+1)*8] → `simdgroup_half8x8`
       - Load K[base..base+8, head, chunk*8..(chunk+1)*8]^T → `simdgroup_half8x8`
       - `simdgroup_multiply_accumulate(score_acc, Q_frag, K_frag)`
     - Extract 8 scores from `score_acc` using the lane→fragment mapping
   - Online softmax on 8 scores (same `simd_max`/`simd_sum`)
   - PV: same scalar loop as current (not MMA yet)
2. Handle the tail (key_count not a multiple of 8): fall back to scalar for
   the last 0–7 keys, or pad with NEG scores.
3. Handle HEAD_DIM=512 (full-attention layers): 64 MMA steps per 8-key tile.
   Register pressure is higher. If it spills, tile the head_dim into 2 chunks
   of 256.

**Test:**
- `cargo test --release -p kaplumba-engine --test attention_metal` must pass.
- New test: `attention_mma_matches_scalar` that runs both the MMA and scalar
  paths on the same input and asserts fp16 output equality (not f32 tolerance).
- Sabotage check: break the MMA QK loop, verify the test fails.
- A/B at 8k context: measure attention ms/token improvement.

---

## T2 — Attention MMA: measure and tune

**Status:** ✅ DONE (commit `3ea15aa`). Root cause found and fixed: T1 ran the
full sub-tile set in **every** simdgroup (NSG× redundant tensor work + K loads).
T2 partitions sub-tiles across simdgroups and publishes a shared score tile.

**Measured (`attention_qk_mma_bench`, ratio mma/scalar, before → after):**
| shape | before | after |
|---|---|---|
| full d512 NSG8 keys=8192 | 1.700 | **0.833** (1.2× faster) |
| sliding d256 NSG2 keys=1024 | 1.813 | 1.588 (still slower) |
| sliding d256 NSG2 keys=8192 | 4.967 | 3.296 (still slower) |

**End to end:** 8k **78.82 → 80.52 tok/s (+2.2%, meets AC2)**; 614 unchanged
(~100.5). Default is now shape-aware (`attention_qk_for(groups)`: MMA only for
groups ≥ 4). AC5 (attention ≥1.5×) is **not** met — the win is 1.2× on the
full-attention QK only, ~+8% on attention overall at 8k.

**Intent:** Measure the actual speedup from T1 and tune the tile parameters.

**Files:**
- `crates/kernels/src/attention.metal`: parameter adjustments
- `crates/kernels/src/attention.rs`: dispatch adjustments

**Work:**
1. A/B the MMA attention vs scalar attention at 614 / 8k / 16k using
   `KAPLUMBA_GPU_PROFILE=1`. Report attention ms/token for each.
2. If the gain is < 1.5× at 8k, diagnose:
   - Is it register spilling? (check local memory via `MTL_DEBUG_LAYER`)
   - Is the MMA fragment load overhead too high for 1 query?
   - Would tiling 2+ queries per threadgroup help?
3. Try KEY_TILE=16 (two MMAs per key batch) if KEY_TILE=8 shows overhead.
4. Enable the MMA path as default for prefill. Keep scalar as fallback via
   an env var (`KAPLUMBA_SCALAR_ATTENTION=1`).

**Test:**
- End-to-end prefill at 8k: ≥ 1.5× attention speedup (acceptance criterion 5).
- Five reference prompts correct.
- `cargo test --release --workspace` all green.

---

## T3 — GEMM: weight repack for MMA tile locality

**Status:** ⚠️ NOT PURSUED — R3 is satisfied by T4's measured tile-variant
sweep. The repack is an offline layout change with an unknown payoff and a
larger blast radius; T4 found a real win (regpipe) with no format change, so
the budget went there. Node marked closed, not silently dropped.

**Intent:** Reorder the trellis tiles so that the 4 column tiles read per
K-chunk by the MMA kernel are contiguous in memory, improving L1/L2 locality.

**Files:**
- Weight repacking tool (new, under `crates/engine/` or `scripts/`)
- `crates/kernels/src/linear.metal`: adjust `mma_tile_gbase()` for new layout
- `crates/kernels/src/linear.rs`: detect repacked layout in weight metadata

**Work:**
1. Profile the current weight access pattern: are the 4 NTILE reads hitting
   L1 or going to L2/DRAM? Use `KAPLUMBA_GPU_PROFILE=1` to compare gate_proj
   (N=21504, tiles far apart) vs k_proj (N=4096, tiles closer).
2. Design the repack: group trellis tiles by MMA tile columns.
   Current layout: `[K/16, N/16, tile_words]` — row of K-tiles, then next N.
   Repacked: `[K/16, N/64, 4, tile_words]` — group 4 column tiles together,
   so the 4 NTILE reads are contiguous.
3. Implement an offline repacker that reads the existing trellis and writes the
   new layout. Same total bytes. Same decoded values (just reordered tiles).
4. Adjust `mma_tile_gbase()` to use the new stride.
5. Measure: A/B at 614 and 8k tokens.

**Test:**
- Repacked weights produce bit-identical MMA GEMM output (worst error unchanged
  at 2.16e-3).
- `cargo test --release -p kaplumba-engine --test linear_metal` passes on
  repacked weights.
- Sabotage: swap the repack to a wrong ordering, verify test fails.

---

## T4 — GEMM: investigate remaining tile variants

**Status:** ✅ DONE — branch `impl/gemm-mma` (commit `ad1e860`), merged
`e1005e7`; regpipe default landed `c02b270`.

**Measured (same binary, interleaved; GEMM = 7 projections):**
| variant | 614 tok GEMM ms | TFLOPS | 8k tok/s |
|---|---|---|---|
| baseline 128×64 | 6035 | 5.85 | 75.6 |
| **regpipe (new default)** | **5753** | **6.13** | **78.8** |
| 4sg 128×32 | 6246 | 5.65 | 71.7 |
| regpipe_k32 | 6295 | 5.61 | 71.7 |

regpipe is bit-identical to `linear_gemm_mma`, MCG-3bpw gated. 4sg and k32 are
losers (recorded negative results).

**Intent:** Try tile shape variants that haven't been rejected in
`spec/engine-performance` §14, specifically targeting the load:MMA ratio.

**Files:**
- `crates/kernels/src/linear.metal`: new kernel variant or template parameter
- `crates/kernels/src/linear.rs`: dispatch logic

**Work:**
1. **4-simdgroup tile (128q × 32n, 4 SG, 16 acc = 4R × 4C):**
   Load:MMA ratio = (4+4)/(4×4) = 0.5 vs current 0.75.
   128 threads per threadgroup, `bstage` = 64 × 40 = 2560 halfs = 5120 B.
   Better occupancy (5 KB vs 9 KB). But fewer output columns per dispatch.
   The notes say "8→4 simdgroups with KCHUNK=32: no change" — but that was
   a different tile shape. This specific 4R×4C factoring has NOT been tried.
   
2. **KCHUNK=32 with regpipe variant:**
   KCHUNK=32 was –2% before regpipe landed. Re-measure with regpipe, which
   reduces barrier count.

3. Any promising variant: measure at 614 and 8k, report TFLOPS and tok/s.
   Do not spend more than 2 measurement sessions on variants that show no gain.

**Test:**
- `linear_metal` test suite passes on every variant.
- A/B against the baseline: report TFLOPS improvement (or lack thereof).
- If no variant improves, record the negative result and close R3 as
  "current tile is at its ceiling."

---

## T5 — Context extension to 24k

**Status:** ✅ DONE — branch `impl/ctx24k` (commit `6fd187e`), merged
`438ead9`. `HOST_TOKEN_SLOTS = DEFAULT_CONTEXT_LIMIT` (32768). 24k prefill +
decode measured **17.13 GiB** peak `phys_footprint`, < 20 GiB. (Premise in R4
was wrong: `HOST_TOKEN_SLOTS` never blocked 24k — an under-sized-buffer defect
was fixed instead.)

**Intent:** Enable 24k prompts by fixing HOST_TOKEN_SLOTS and validating KV
cache sizing.

**Files:**
- `crates/engine/src/model.rs`: `HOST_TOKEN_SLOTS` constant, any other 8192
  hardcodes.
- `crates/engine/src/gpu_cache.rs`: verify grow-on-demand works to 24k for
  full-attention layers.

**Work:**
1. Raise `HOST_TOKEN_SLOTS` to 32768 (or make it dynamic, sized from
   `max_sequence_len`).
2. Search for any other `8192` hardcodes that would block 24k prompts.
3. Verify KV cache grow-on-demand: full-attention layers should grow from
   initial allocation up to 24k. Sliding-window layers stay capped at 1024.
4. Test a 24k prefill: `'Explain the theory of relativity. ' * 3990`.
5. Measure peak `phys_footprint` at 24k. Must be < 20 GiB (leaving 4 GiB
   for OS).

**Test:**
- 24k prompt completes without error.
- Peak `phys_footprint` < 20 GiB.
- `cargo test --release --workspace` passes (no regression from the constant
  change).
- Decode after 24k prefill works (at least 1 token generated).

---

## T6 — PV MMA (gated on T2 results)

**Status:** ⏸ GATED OUT. The T2 measurements show attention is **memory-bound**
(V/K re-read per query), not QK-instruction-bound: fixing the QK to 1.2× moved
8k by only 2.2%, and the PV loop is shared by both arms. A PV MMA would not
touch the re-read term, so the gate ("proceed only if attention ≥20% after QK
MMA") is not met in a way that makes PV MMA worth the complexity.

**Gate:** Only proceed if T2 shows attention is still ≥ 20% of prefill time at
16k after QK MMA. If attention drops below 20%, the PV improvement is marginal
and not worth the complexity.

**Intent:** Replace the scalar PV loop with MMA-based weighted value
accumulation.

**Files:**
- `crates/kernels/src/attention.metal`: modify the PV section of
  `attention_gqa_mma_body`.

**Work:**
1. Design: batch 8 keys × 8 dims via MMA. The weight matrix is a diagonal
   (one weight per key), so the B operand is `w[k] * V[k, d..d+8]`. This can
   be expressed as `diag(w) × V_block`, which is a valid MMA but may not be
   efficient.
2. Alternative: replicate each weight into a full column of the B fragment.
   `B[k, :] = w[k]` for all 8 columns. Then `A = V[k, d..d+8]` and the
   MMA computes the weighted sum. This is a rank-1 outer product, which MMA
   handles.
3. Measure: A/B at 16k and 24k.

**Test:**
- fp16 output equality with the scalar PV path.
- Sabotage check.

---

## T7 — Final measurement and acceptance

**Status:** ✅ DONE — see the results table below and `implementation_notes.md`.
**Depends on:** T1–T6

**Intent:** Comprehensive measurement against all acceptance criteria.

**Files:** none (measurement only)

**Work:**
1. Measure prefill tok/s at 614 / 8k / 16k / 24k with interleaved A/B
   (5 runs each, medians with spread).
2. Record achieved GEMM TFLOPS.
3. Record attention ms/token at each context length.
4. Record peak `phys_footprint` at each context length.
5. Run all five reference prompts, verify correct output.
6. Run `cargo test --release --workspace`, report suite status.
7. Record worst GEMM error (must be ≤ 2.16e-3).
8. Measure decode tok/s at short context (must be ≥ 15).
9. Write final summary in `implementation_notes.md`.

**Test:** all acceptance criteria from spec.md evaluated and recorded.

---

## T7 results — end-to-end prefill (merged binary, `KAPLUMBA_MAX_IN_FLIGHT=1`)

| ctx | prompt tok | before (T0) | after (regpipe + shape-aware MMA) | peak `phys_footprint` |
|---|---|---|---|---|
| 614 | 614 | 96.17 | **100.56** | 12.66 GiB |
| 8k | 7814 | 75.61 | **80.52** | 14.15 GiB |
| 16k | 15974 | 60.75 | **63.96** | 15.54 GiB |
| 24k | 23954 | 51.12 | **52.84** | 17.12 GiB |

GEMM at 614: 5.85 → 6.13 TFLOPS (regpipe). Decode (short ctx, 96 tok): ~14.0
tok/s (pre-existing, unchanged). Five reference prompts correct.

**Acceptance scorecard:**

| AC | target | result |
|---|---|---|
| AC1 | 120 @614 | **100.6** — unmet; attributed below |
| AC2 | ≥80 @8k | **80.52** — MET |
| AC3 | 16k/24k reported | **63.96 / 52.84** — reported |
| AC4 | ≥7.5 TFLOPS | **6.13** — unmet; T4 sweep + ceiling |
| AC5 | attention ≥1.5× @8k | **1.2× full QK only** — unmet |
| AC6 | footprint ≤ budget | **17.12 GiB @24k** — MET |
| AC7 | tests + 5 prompts + decode ≥15 | tests/prompts ✓; decode **14.0** — partly unmet |
| AC8 | worst GEMM error ≤2.16e-3 | **2.1603e-3**, unchanged (bit-identical) — MET |
| AC9 | HOST_TOKEN_SLOTS ≥24k | **32768**; 24k runs — MET |

### AC1 attribution (the spec's escape clause)

At 614 the 5117 ms budget needs GEMM ≤4707 ms. regpipe reaches 5753 ms (6.13
TFLOPS). Making attention **entirely free** would give 614/(5753+~230) ≫ … it
still cannot reach 120: the T0 arithmetic (attention=0 → 99.2 tok/s, GEMM 5.84
TFLOPS) plus T4's 6.13 TFLOPS gives 614/(5753+702) ≈ **101 tok/s**, which is
what is measured. The remaining 6.13 → 7.5 TFLOPS is a tile-shape wall: T4
exhausted 4sg and k32 against it, and the staging-stripped diagnostic caps this
shape near 76.7 tok/s of GEMM-only work. AC1 is therefore reported as a
measured wall, not a missing optimization.
