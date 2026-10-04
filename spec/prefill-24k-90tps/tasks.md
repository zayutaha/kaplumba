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

**Status:** 🔄 in progress. 8k +8.1% (T3). Query blocking **rejected in its
grid-halving form**; cost split measured (T4e); arm decision settled (T4c);
**T4f shipped +15-21% on the bench, +1.6% E2E at 8k** (`1254780`). The key-split
plan was refuted before building it. **T4g done (negative, §10):** the §9 ablation
does not reproduce and the microbench cannot attribute cost. **Next:** GEMM (T5)
vs decode (T2), E2E only.

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

### T4c — full-shape arm decided (E2E), and the bench's ranking is discredited

`KAPLUMBA_FULL_QK=flash|mma` E2E, interleaved: **tied** at 614 (flash 101.46 /
101.49 / 101.34 vs mma 101.60 / 98.29 / 101.49) and at 8k (flash 80.09 / 81.80 vs
mma 79.89 / 79.55). The TFLOP/s bench's mma-over-flash ranking (2.14 vs 1.58 at
8k) does not reproduce; **`flash` stays the shipped full-shape arm.** Recorded as
§8.7: the bench is a ceiling instrument, not a ranking instrument.

### T4d — QK M-tiles (KSUB): tried at 4, **negative**, reverted

A 256-key pass (4 m-tiles per simdgroup) halves the QK's `simdgroup_load`s per
MMA but needs 28 KB of threadgroup memory; **13% slower** in the engine at 8k
(attention 20.6 s -> 23.4 s) and slower on the bench (1118 -> 1318 ms at 24 576
keys). Recorded as §8.1; `FLASH_KEY_SUBTILES = 1` documents the shipped value.

### T4e — ablate the flash body; the cost split is now measured ✅

Ablations in `attention.metal` (24 576 keys, BQ=1): baseline 1148 ms, PV off 433,
QK off 689, **all MMAs and their loads off 36.7**. So the softmax/barriers/staging
are 3%; the strided device loads are 44% (K -171 ms, V -338 ms) and MMA issue 52%,
**and the two do not overlap** (they sum instead of maxing). Attention runs at
175 GB/s = 64% DRAM and 1.44 TFLOP/s = 9% tensor peak, simultaneously.

| tried | result |
|---|---|
| bank-conflict padding on `qstage`/`wq` | 1111 ms, neutral |
| 2 independent QK accumulators | 1126 ms, ~-2% |
| K fragment software-pipelined 1 `d8` ahead | 1166 ms, slower |
| remove `qstage`, load Q transposed from device | 1151 ms, neutral (but it does work: transposed 8x8 spans the 8 heads) |

So occupancy, operand reuse and bank conflicts are all **out**; the two levers
left are (a) **BQ=2 with the grid restored** by splitting the key range
(§9.1b -- halves the 44% load term without halving the threadgroups, needs a
partial/combine pass like the decode split path) and (b) overlapping a tile's
MMAs with the next tile's loads (the 48% idle).

**Arm decision (third confirmation):** `flash` beats `mma` end to end at 16k
(69.36 vs 66.95) and 24k (62.87 flash), even though the bench says `mma` is 28%
faster. The bench is valid for *work* (1.44 TFLOP/s predicts the engine's 131 s
of full attention against 132 s measured) and invalid for *ranking*.

| prompt | prefill tok/s |
|---|---|
| 626 | 101.5 |
| 8 204 | 80.1 (flash) |
| 16 394 | 69.36 (flash) / 66.95 (mma) |
| 24 590 | **62.87** (target 90) |

### T4e-b — the key-split-with-combine plan is **refuted, do not build it**

The §9.1b plan was to halve the K/V bytes with BQ=2 and restore the threadgroup
count by splitting the key range, which needs a partial/combine pass. Measured
first: the bench now sweeps `queries` 256 / 1024 / 4096 (grid 1024 / 4096 / 16384)
and flash runs at **1.78 / 1.80 / 1.80 TFLOP/s** -- flat. The kernel is **not
grid-starved**, so halving the grid was never what cost BQ=2 its win, and a
combine pass would have bought nothing. Recorded so nobody spends the effort.

### T4f — contiguous V dim window per simdgroup ✅ `1254780`

PV assigned its n-tiles `nt = sg + j*NSG`, so each simdgroup read **8 scattered
16-byte windows at every 64-dim step** of each V row, with the 8 rows 4 KB apart
-- structurally the opposite of the K load, which walks `d8` contiguously over its
own 8 rows and never over-fetches. That asymmetry is exactly why the ablation
put the V load at 338 ms against the K load at 171 ms for byte-identical work.

Fix, one line in two places (PV loop + epilogue write): `nt = sg * NPER + j`.
The n-tiles are still a partition of `[0, 64)`, so the output rows are unchanged.

**Measured, no-model bench, warm, `mma` as an untouched control (871 ms before
and after):**

| case | flash bq1 before -> after | flash bq2 (what the engine runs at 24k) |
|---|---|---|
| 8192 keys / 1024 q | 334.06 -> **286.68** (+17%) | 318.08 -> **292.44** (+9%) |
| 24576 keys / 1024 q | 1099.79 -> **916.24** (+20%) | 1062.91 -> **902.17** (+15%) |
| 24576 keys / 256 q | 275.52 -> **232.02** (+19%) | 223.42 |
| 24576 keys / 4096 q | 4456.60 -> **3670.43** (+21%) | 3588.84 |

**Correctness:** `attention_qk_mma_matches_scalar` worst |flash - scalar| still
**4.88e-4** (one ulp, tol 5e-3) over all 6 full-shape cases incl. every straddle;
`attention_single_pass_bit_exact` 8/8. **E2E 8k:** 80.09 / 81.80 -> **82.42 /
82.24**, with mma unmoved at 79.80 / 79.80 (it never touched this code, so it is a
real control).

**Caveat:** a ~6% effect at 24k cannot be resolved on this box -- two back-to-back
clean runs of the *same* binary gave 62.72 and 49.43 tok/s (21% spread) while the
box was compressing. See OQ5.

**Remaining (T4g):** re-ablate on the updated body to find the next dominant term,
then choose between the GEMM (~225 s of the 24k wall, 59%, untouched by this whole
spec) and T2 (decode, untouched). Per §5, 90 tok/s needs **both** the GEMM and
attention to improve.

### T4g — re-ablation of the post-T4f body: **the T4e split does not reproduce** ✅

**Status:** ✅ done, negative. `implementation_notes.md` §10. No product code
changed; the working tree is byte-identical to `1254780` apart from these two spec
files.

Re-ran the §9 ablation against the T4f body, same bench, 24 576 keys / 1024
queries / `flash bq1`, median of 3, tree restored after every arm:

| variant | T4e | T4g |
|---|---|---|
| baseline | 1148 | 958.39 |
| `ksteps = 0` | 433 | **966.88** |
| QK off | 689 | **959.90** |
| both off | 36.7 | **941.88** |
| eighth of the tiles | -- | 947.33 |
| `return` at the top of the body | -- | **964.79** |
| 3 M-iteration threadgroup spin | -- | **4 828.96** |
| `mma` control | 869-871 | 867-894 |

Deleting the PV work, the QK work, both, or the entire body changes nothing; an
added spin loop costs 3.9 s. Ruled out: build failure (`cargo` rc 0, marker
present in the binary via `strings`), wrong kernel (`eprintln!` in the dispatcher
shows `qk=Flash block=1 flash_pipeline=true`), host cost (no-commit arm 0.05 ms),
bandwidth (`KV_ELEMS` 64x shrink buys 6%), and work-proportionality (time *is*
linear in keys x queries: 2.28 / 32.24 / 144.36 / 947.35 ms at 64 / 1024 / 4096 /
24576 keys).

**Consequences:** §9's cost split and everything derived from it (§9.1, §9.1b,
§9.2) is withdrawn as a basis for design. §9.1b stays refuted. The microbench is a
throughput instrument, not an attribution instrument and not a ranking instrument
(third independent confirmation, §8.7/§9.4/§10.2).

**Rule adopted:** an ablation reading "the change did nothing" is only believable
after a positive control has been shown to move the same instrument in the same
build.

---

## T5 — Verify GEMM at 24k

**Status:** ☐ not started

**Intent:** Confirm GEMM per token is context-independent at 24k (R4).

**Test:** profiled GEMM `own_ms`/token at 8k vs 24k within spread.

**Priority after T4g:** this and T2 are the only untouched levers. GEMM is ~225 s
of the ~381 s 24k wall (59%) at 6.13-6.44 TFLOPS against a ~7.8 ceiling, and
attention cannot alone reach 90 tok/s (381 s - 132 s attention = 249 s = 96.6
tok/s *if attention were free*, which it cannot be).

---

## T6 — Final measurement and acceptance

**Status:** ☐ not started. **Depends on:** T1–T5.

**Work:** interleaved A/B prefill (8k/16k/24k) and decode (short/8k/16k/24k),
TFLOPS, peak `phys_footprint`, five reference prompts, worst GEMM error, full
suite. Evaluate AC1–AC5.

**Test:** scorecard with medians and spread.

---

## T2b — MMA decode attention kernel ✅ `55cd5df`

**Status:** ✅ done and measured. **Files:** `crates/kernels/src/attention.metal`,
`crates/kernels/src/attention.rs`,
`crates/engine/tests/attention_decode_flash_matches_split.rs` (new),
`crates/engine/tests/attention_single_pass_bit_exact.rs`.

**Intent:** replace decode's per-key 32-lane `simd_sum` + online-softmax update
with 8x8 simdgroup MMA. §6 diagnosed decode as latency-bound; oMLX v0.7.0
(`07d88dbd`, `omlx/patches/mimo_v2/sdpa_flash.py`) diagnosed the same kernel
shape as ALU-bound at ~1/3 of DRAM bandwidth on different hardware and a
different GQA ratio, and fixed it the same way.

**Work:** `attention_decode_flash_512_8` = the prefill flash body with `BQ=1` and
a new `PARTIALS` template parameter. Emits the same `(m, s, O)` layout the
scalar split kernel emits, so `attention_decode_combine_*` and its exactness
argument are reused untouched. Carried over from oMLX: **split count stays a
runtime value** (a template split count cost them ~115 ms JIT/step and -22%
decode @8k), fp32 MMA and fp32 partials, 4096-key threshold.

**Correctness (AGENTS.md 3.4):** `attention_decode_flash_matches_split`, 15
cases -- worst |flash - scalar| fp16 abs diff **0.000488** (one ulp), tol 5e-3,
0 NaNs, across splits 1/2/3/5/13/16, a slice running off the end, and EMPTY
splits (`key_count` < `splits * 64`). **Sabotage check:** dropping `split` from
the partials index fails at `keys=64 splits=2: worst |flash - scalar| =
0.5576477 exceeds tol 0.005` (111x) and only on `splits >= 2`, correct because
splits=1 makes the paths identical by construction.

**Regression found and fixed:** `the_one_wave_split_rule_costs_one_fp16_ulp_and_is_not_bit_exact`
pins EXACT differing-element counts for the scalar rule; following the
production default silently re-pointed it at the new kernel (8 passed -> 7
passed, 1 failed). It now pins the scalar arm through `encode_decode_split_explicit`
-- preserves its original subject rather than weakening the assertion.

**Measured (interleaved A/B, 3 rounds, 8204 tokens, `--max-new-tokens 24`):**

| arm | r1 | r2 | r3 | median |
|---|---|---|---|---|
| scalar | 12.000 | 12.059 | 11.990 | 12.000 (+-0.3%) |
| **MMA** | 12.978 | 12.979 | 12.972 | **12.978 (+-0.03%)** |

**+8.2% decode @8k**, and **+4.4% @4k** (12.787 -> 13.346, one round). Effect arm
is an order of magnitude tighter than the control, so not a spread artefact.
**AC3 (decode >= 12 @8k) was being met by 0.0% and is now met with 8% margin.**
Gain grows with context (oMLX: +2.8% @8k -> +71.6% @256k), so 12.98 @8k is a
**floor** for 24k, not an estimate of it.

**Gates:** `KAPLUMBA_DECODE_FLASH=0` off, `KAPLUMBA_DECODE_FLASH_MIN_KEYS=<n>`
threshold (default 4096), shape enforced in `encode_decode_split_explicit`
(head_dim 512, 8 q-heads/kv-head only -- a test caught that this was initially
unchecked, which would have dispatched a 256-thread kernel onto the 64-thread
sliding shape).

**Verified:** attention_single_pass_bit_exact 8 passed, attention_decode_flash_matches_split
3 passed, attention_qk_mma_matches_scalar 3 passed, all rc 0.

---

## N1 — Prefill: slice the key range into successive dispatches (OQ9)

**Status:** ☐ not started. **Priority: highest remaining.**

**Files:** `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`,
`crates/engine/src/model.rs` (prefill chunk loop).

**Intent:** port oMLX's prefill key-range slicing. From
`omlx/utils/nax_attention.py`: "several dispatches over consecutive key ranges
(`OMLX_NAX_ATTN_PASS_KEYS` keys each, default 8192) that hand the fp32
online-softmax row state (O accumulator, running max and sum) from one to the
next through device memory, so each row computes exactly what one dispatch
computes (bit-identical output)", because "one dispatch over a long key range
lets its threadgroups drift apart until each streams its KV head from DRAM;
per-slice dispatches keep the K/V they share in the on-chip caches."

**Why:** it attacks the cache-residency loss §10 measured directly (a 64x smaller
K/V buffer bought only 6%, so not DRAM-bound -- but T4f's *access-pattern* change
was worth +6.7% E2E, so layout is what pays).

**Constraints:** scratch fixed-size and context-independent (AGENTS.md §8). Row
state = 1024 queries x 32 heads x 512 dims x 4 B = **67 MB fixed per chunk**,
affordable inside the 20 GiB budget (currently 18.35 GiB). **No combine pass
needed** -- unlike §12.2's parallel key-split, which stays an alternative.

**Test:** bit-exact vs the current one-dispatch body; then E2E at 16k **first**
(a ~6% effect is resolvable there; 24k carries ±20% per OQ5). Slice size is a
sweep knob.

**Risk:** medium. 67 MB written + read per slice may cost more than the cache
residency it buys. Measure before believing.

---

## N2 — Unblock 24k measurement (OQ8)

**Status:** ☐ blocked on user direction. **Blocks:** AC1 and validation of N1 at
the context that matters.

24k decode is **structurally unmeasurable** on this box: the run's own
18.35 GiB footprint evicts the desktop and the compressor drives decode to
0.055 tok/s (99x pessimisation). Three attempts have produced invalid samples.
A gate cannot fix this. Options: (a) quieter machine, (b) lower-footprint 24k
configuration, (c) re-baseline AC1 onto 8k. See `NEXT.md` §3.

---

## N3 — Cheap micro-wins from oMLX's decode kernel

**Status:** ☐ not started. Low effort, unquantified.

- Guard the O rescale on `m_new != m_run` ("a row whose max did not move has
  factor 1: skipping its rescale is exact").
- Head-dim split over simdgroup pairs (their `HS == 2`). We spend all 8
  simdgroups on 8 query heads; splitting head dim trades simdgroups for a
  shorter dependent chain. Our head dim is 512 vs their 192/128, so more
  relevant here than there.
- Also recorded from `07d88dbd`: their fused decode layer stack (one
  row-concatenated q/k/v mat-vec, fused residual+RMSNorm) cut ~2.8k kernels/step
  for a MoE model. Not portable as written, but it says **look at our
  element-kernel count per decode step**.

---

## N4 — Re-examine `decode_splits` with the MMA kernel

**Status:** ☐ not started. Supersedes the T2a sweep, whose arms were all invalid.

The scalar kernel was latency-bound on a 64-workgroup grid; the MMA kernel's
parallelism trade is different and may want **more** splits.
`ATTENTION_SPLITS_MAX = 16` is a hard cap never probed above. oMLX keeps the
split count a **runtime** value for a measured reason (template split count =
~115 ms JIT/step, -22% decode @8k); ours already is.

**Prerequisite:** N5 (a trustworthy harness).

---

## N5 — Fix the red OQ5 gate in the sweep harness

**Status:** ✅ done. `scripts/gpuq_mem.sh` (new, sourceable), `scripts/test_gpuq_mem.sh`
(new, the fixed and committed successor of `/tmp/test_parse.sh`),
`scripts/bench_decode_splits.sh` (new, the committed successor of
`/tmp/decode_splits_sweep.sh`).

**Root cause of the red test (it was not a regex problem).** `parse()` was
invoked in **pipeline position** — `avail=$(printf '%s\n' "$1" | parse)` — and
bash does not pass a function's positional parameters to a command that runs as
the *last* element of a pipeline (no `lastpipe`). So `$1` inside `parse` was
always unbound, the helper returned empty every time, and the synthetic
`healthy=1` cases read unhealthy. §13.2's hypothesis was right. The fix reads
stdin instead of `$1`, and no parser sits in pipeline position.

**Bigger problem found while fixing it:** the test tested a *reimplementation*
of the parser pasted into the test file, so it could pass while the harness's
real parser was broken — a coverage illusion of exactly the kind AGENTS.md §3.4
warns about. The parser now lives once, in `scripts/gpuq_mem.sh`, and the test
sources it.

**Files.** `scripts/gpuq_mem.sh` provides `gpuq_mem_available_gib`,
`gpuq_mem_healthy` (fail-closed), `box_healthy`, `top_physmem_fields`
(named-field, unit-aware, pure — reads stdin so it is testable),
`top_physmem_live`, `mem_poll_record`, `mem_sample_verdict`.
`scripts/bench_decode_splits.sh` is now generic over the swept knob (`VAR=`, so
N4 and N1 use the same harness), **interleaves** arms by round, polls memory
**during** each arm, logs `phys_footprint` from `/usr/bin/time -l`, and records
**no number at all** for a sample its own poller judged invalid.

**Test:** `bash scripts/test_gpuq_mem.sh` → **ALL PASS, exit 0**. 19 assertions,
including two fail-closed shapes added here (a garbage `memory:` line, and a
line with no digits at all), the historical `554M` trap asserted as a *number*
(`554M compressor -> 0.54`, `126M unused -> 0.12`, both GiB), the no-poll case,
and two end-to-end checks: a stubbed `gpuq` reporting a garbage memory line must
make the harness SKIP the arm and never invoke the binary, and a stubbed healthy
`gpuq` must let the arm through.

**Sabotage check (AGENTS.md 3.4), observed failure modes.** Replacing the
harness's pre-arm gate with `if false; then` fails the suite with
`FAIL: harness ran the model on a box it could not parse` **and**
`FAIL: no SKIPPED line in the log`; restoring it returns exit 0 and the harness
is byte-identical to the backup. An earlier version of the sentinel check grepped
the harness's stdout for a marker the harness redirects into its own temp file,
so it passed under sabotage — replaced by a sentinel *file*, which cannot be
hidden by stream redirection.

---

## N6 — final scorecard
