# What we are supposed to do next

**Spec:** `spec/prefill-24k-90tps/` · **Branch:** `mtp-speculation` · **HEAD:** `b73090d`
**Read this first, then `implementation_notes.md` §10–§14 for the evidence.**

---

## 1. Where the numbers stand

| | before | now | target | verdict |
|---|---|---|---|---|
| prefill @24k | 52.84 | **67.10** tok/s | 90 | ❌ **25% short** |
| prefill @8k | 80.52 | **82.42** tok/s | ≥80 | ✅ |
| prefill @614 | — | **101.5** tok/s | ≥100 | ✅ |
| decode @8k | 12.27 | **12.98** tok/s | ≥12 | ✅ (was passing by 0.0%) |
| decode @24k | 5.44 | **unmeasurable** | 14 | ❌ **blocked, see §3** |
| peak `phys_footprint` @24k | 18.35 GiB | 18.35 GiB | < 20 GiB | ✅ |
| quality | — | **unchanged** (one fp16 ulp) | no loss | ✅ |

**Neither headline target is met.** Prefill is 25% short; decode cannot be
measured at 24k at all. Everything below is aimed at those two.

---

## 2. What has been established (do not re-derive these)

These cost real time to find. Each is committed with its evidence.

1. **§12 — the "86 s attention bandwidth floor" was wrong by ~350×.** It counted
   2048 B of *DRAM* per (query,key) pair; `PREFILL_CHUNK = 1024` means a whole
   chunk shares every key. Real 24k KV traffic is **67.1 GB ≈ 0.25 s**. The
   kernel moves **0.45 GB/s ≈ 0.16% of DRAM bandwidth**.
   **AC2 is an attention ask (149.7 s → ≤ 56 s, ~1.74 → ~3.4 TFLOP/s = 22% of
   tensor peak), not a GEMM ask.** OQ4 and OQ6 withdrawn. Do not re-open the GEMM.
2. **§10 — the T4e cost split does not reproduce.** `ksteps = 0`, QK-off,
   both-off and an immediate `return` all land within 1.8% of baseline, while an
   added spin loop costs +3.9 s. §9.1/§9.1b/§9.2 are **withdrawn as a design
   basis**. The microbench is a throughput instrument, **not** an attribution or
   ranking instrument (four independent confirmations it ranks arms wrongly).
   **F1 CORRECTION (2026-10-05) — this bullet's premise is now known to be false,
   and so is its conclusion.** The §10 arms never reached the kernel:
   `attention.metal` is `include_str!`'d into the crate, and the flash path
   dispatches 256 threads per threadgroup, so §10's `if (tid < 256u) { return; }`
   was an **empty kernel** and read 964.79 ms while an empty body at that grid
   costs **0.19 ms** — a 6 162x signal. The microbench **is** an attribution
   instrument; what it cannot do is **rank arms end to end**, and that half
   stands. §22 replaces §9.1/§9.1b/§9.2 and §10.2 #3, and closes OQ7.
3. **§9.1b's "BQ=2 + key-split + combine" is re-opened, not refuted** — its
   refutation used the withdrawn §9 split. But naive BQ=2 still measures
   −2.0% @8k / −6.2% @24k, so the *grid* has to be restored too.
   **F1 CLOSES IT (2026-10-05).** The loads that BQ halves are worth **8%**
   (every fragment load L1-resident, MMA count untouched, is 1.09x — §22 arm 8),
   and on the bench `BQ=1` is **4-19% faster** than `BQ=2` at both 8 192 and
   24 576 keys. There is nothing for it to recover. Do not build the combine pass.
4. **§14 — oMLX v0.7.0 independently reached our §6 decode diagnosis** (per-key
   `simd_sum` decode is ALU-bound at ~⅓ of DRAM). Ported; see §4.

---

## 3. The blocker: 24k decode cannot be measured (OQ8)

The 24k run's own footprint is **18.35 GiB of a 24 GiB machine**. It evicts the
desktop, and every later allocation becomes a compression event:

```
PhysMem: 23G used (9165M wired, 5954M compressor), 110M unused
decode_tok_s = 0.055     <- 99x pessimisation, this is the compressor
```

**Three attempts have produced invalid 24k decode samples.** A measurement gate
cannot fix this — it can refuse to *report* a compressed sample, not produce a
valid one.

> **Decision needed from the user.** Either (a) measure on a quieter machine
> (desktop closed / second box), (b) find a lower-footprint 24k configuration to
> measure, or (c) re-baseline AC1 onto 8k, where we *can* measure and where the
> kernel is verified. This is the one thing blocking AC1 that is not a kernel
> problem.

---

## 4. Shipped: T2b, the MMA decode kernel (`55cd5df`)

`attention_decode_flash_512_8` — the prefill flash body with `BQ=1` and
`PARTIALS`, so decode QK/PV are 8×8 simdgroup MMA instead of a 32-lane
`simd_sum` per (key, head). Writes the same `(m, s, O)` partials, so the existing
combine pass and its exactness argument are reused unchanged. Ported from
`omlx/patches/mimo_v2/sdpa_flash.py` (oMLX v0.7.0, commit `07d88dbd`).

**Measured, interleaved A/B, 3 rounds @8k:** 12.000/12.059/11.990 →
**12.978/12.979/12.972** = **+8.2%**, effect arm ±0.03% vs control ±0.3%.
**@4k: +4.4%.** Gain grows with context, as oMLX measured (+2.8% @8k → +71.6%
@256k), so 12.98 @8k is a **floor** for 24k, not an estimate.

**Gates:** `KAPLUMBA_DECODE_FLASH=0` off · `KAPLUMBA_DECODE_FLASH_MIN_KEYS=<n>`
threshold (default 4096) · head_dim 512 with 8 q-heads/kv-head only, enforced in
`encode_decode_split_explicit`.

**Quality:** new `attention_decode_flash_matches_split`, 15 cases, worst
|flash − scalar| = **0.000488 (one ulp)**, tol 5e-3, no NaNs, across ragged
splits, off-the-end slices and empty splits. Sabotage check: removing `split`
from the partials index fails at **0.5576477 (111× tol)** and only on `splits≥2`.

**Not taken from oMLX, deliberately:** all NAX/M5 code (wrong GPU), all MTP (per
brief), TurboQuant KV (forbidden by AGENTS.md §8), and the MoE-specific fused
decode layer stack (wrong model shape — Gxma is dense). The last is a lead, not
a dead end: oMLX reports "a decode step dispatched ~2.8k kernels, most of them
small element-wise ops", which may apply to our element-kernel count too.

---

## 5. Next, in priority order

### N1 — Prefill: slice the key range into successive dispatches (OQ9)
**The strongest remaining prefill idea, and it is a direct port.**

oMLX's prefill path runs several dispatches over consecutive key ranges
(`OMLX_NAX_ATTN_PASS_KEYS`, default 8192), handing the fp32 online-softmax row
state (O accumulator, running max and sum) from one to the next through device
memory — **bit-identical output**, no combine pass — because *"one dispatch over
a long key range lets its threadgroups drift apart until each streams its KV head
from DRAM; per-slice dispatches keep the K/V they share in the on-chip caches."*

- **Why here:** it targets the same cache-residency loss §10 measured (a 64×
  smaller K/V buffer bought only 6%, so the kernel is not DRAM-bound — but its
  *access pattern* is what T4f improved for +6.7% E2E).
- **Shape:** not §12.2's parallel key-split + combine (which needs a partials
  buffer and a combine pass). Sequential slices carrying row state.
- **Constraints:** scratch must stay **fixed-size and context-independent**
  (AGENTS.md §8). Row state = 1024 queries × 32 heads × 512 dims × 4 B = 67 MB,
  fixed per chunk — affordable inside the 20 GiB budget (currently 18.35).
- **Test:** bit-exact vs the current one-dispatch body (§4's standard), then E2E
  at 16k and 24k. Slice size becomes a sweep knob (8192 is oMLX's default).
- **Risk:** medium. The O accumulator round-trip through device memory is
  67 MB written + read per slice, which may cost more than the cache residency
  it buys. **Measure at 16k first, where a ~6% effect is resolvable** — 24k
  carries ±20% (OQ5).

### N2 — Unblock 24k measurement
Until OQ8 is decided, **no 24k decode or prefill number from this box is
trustworthy**, and N1 cannot be validated at the context that matters. Everything
else is downstream of this. See §3.

### N3 — Apply oMLX's cheap micro-wins to the decode kernel
From `sdpa_flash.py`, not yet tried here:
- **"A row whose max did not move has factor 1: skipping its rescale is exact"** —
  guard the O rescale on `m_new != m_run`. Free, and exact.
- **Head-dim split over simdgroup pairs** (their `HS == 2` scheme): we currently
  spend all 8 simdgroups on 8 query heads; splitting the head dim across pairs
  trades simdgroups for a shorter dependent chain.
- Head dim is **512** here vs 192/128 there, so both are more relevant to us.

### N4 — Re-examine `decode_splits` now that the kernel changed
`§13`'s sweep never produced a usable curve (all arms invalid). With the MMA
kernel the parallelism trade is completely different: the scalar kernel was
latency-bound on a 64-workgroup grid, the MMA one may want **more** splits, and
`ATTENTION_SPLITS_MAX = 16` is a hard cap that has never been probed above.
oMLX makes the split count a **runtime** value deliberately (a template split
count cost them ~115 ms of JIT/step and −22% decode @8k) — ours already is.

### N5 — Fix the red OQ5 gate in the sweep harness
`/tmp/test_parse.sh` is **written and failing** (`$1: unbound variable` under
`set -u`); it caught two real parser bugs already. N1/N4 both need a trustworthy
sweep harness. See `implementation_notes.md` §13.2.

### N6 — T6 / final scorecard
Only after N1–N4. Interleaved A/B at 8k/16k/24k, peak `phys_footprint` via
`/usr/bin/time -l`, five reference prompts, full suite (4 failures are
**pre-existing at base** — do not attribute them to this work).

---

## 6. Ground rules for whoever continues

- **Interleave A/B** (AGENTS.md §10.2 #5). Single-round pairs carry **3.7%**
  spread at 4k and up to ±20% at 24k — measured, §14.5.
- **Never pipe a long measurement run into `tail`.** It buffers until EOF, so an
  interrupted run loses *everything*. Append to a file per arm. This cost a
  ~50-minute sweep.
- **A reading of "the change did nothing" is only believable after a positive
  control has moved the same instrument in the same build.**
- **Sabotage every new claim.** A test-coverage claim requires breaking the code
  and seeing it fail.
- **Check `top`'s `unused` and the process's CPU time before trusting a 24k
  sample.** >2% CPU in compression means the sample is invalid — discard it, do
  not report it.
- **One model process at a time**, via `scripts/gpuq try --` (exit 75 = busy).
---

## 7. Session update (`48df478`) — what changed and what it cost

This section supersedes §1-§5 above. Read it instead of them.

### 7.1 Scorecard

| | 614 | 8k | 16k | 24k |
|---|---|---|---|---|
| prefill, shipped | 100.90 | 85.01 | 74.72 | 67.10 (older, 1 round) |
| prefill, `workgroups` | 101.04 | 85.52 | 75.00 | — |
| decode, shipped | 14.862 | 12.754 | 12.067 | **unmeasurable** |
| decode, `workgroups` | 14.860 | **12.911** | **12.329** | — |
| `phys_footprint` | 12.65 | 14.16 | 15.54 | 18.35 (older) |

3 interleaved rounds. `phys_footprint` 12.65-15.54 GiB against the 20 GiB budget.
**Zero swapouts** across all sixteen runs. Five reference prompts byte-identical
between arms. Full suite: 116 binaries ok, 4 failed, all four verified at base
`b92234f`. Detail: `implementation_notes.md` §20.

### 7.2 What landed

1. **N3(a), the `msv == 1` rescale guard** (`4bd162c`). Exact, verified against the
   *scalar* arm — because comparing against another flash run passed 8/8 with the
   guard sabotaged to always fire. Third self-comparison trap in this spec.
2. **N1, prefill key-range passes** (`386c9c2`), implemented and **bit-exact**,
   **measured neutral** (`c61dcba`). The durable output is the tile-alignment
   invariant (§16.2): unaligned pass edges change the output by one fp16 ulp,
   10x inside the tolerance every other attention test here uses.
3. **N4, the decode split sweep** (`d956e68`). The optimum is **13, not the shipped
   16**, and `decode_one_wave_workgroups(256)/kv_heads = 54/4 = 13` predicts it.
   Worth **+2.17% decode @16k**, and the gain grows with context. Available as
   `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`; the default flip is one line and is not
   made here.

   **F6 correction: this is a *reproduction*, not a third *independent*
   reproduction.** `spec/decode-19tok` §2 read the budget of 54 off one sweep of
   three shapes, and its `kv_heads = 4 → 13` row **is** the number this session
   re-measured. The shapes that would make `budget / kv_heads` a rule rather than a
   fit (`kv_heads` 5 and 6 → 10 and 9, built by holding `q_heads / kv_heads = 8`
   and varying `kv_heads`) **are not in Gxma and have never been re-measured on
   this machine**. The claim that survives, and is what the decision should rest
   on, is that 13 is a **real wave boundary**: an interleaved 8-arm curve is
   monotone rising to 13 and **cliffs −1.5% at 14** (13 → 52 workgroups, 14 → 56,
   and 54 is one wave / 55 is two), recovering only to −0.9% at 16.
4. **N5, the OQ5 gate** (`d6683a6`, `086073b`). Green, and the honest finding is
   negative: **a good run and a stalled run print the same memory numbers on this
   box** (a 626-token run at 100.9 tok/s reads 9.47 GiB compressor / 0.09 GiB
   unused; the documented stall reads 5.90 / 0.11). No threshold separates them.
5. Five harnesses in `scripts/`, all tested: `gpuq_mem.sh`, `test_gpuq_mem.sh`,
   `bench_prefill_passes.sh`, `bench_decode_splits.sh`, `scorecard.sh`,
   `refprompts.sh`.

### 7.3 What was refuted, with the evidence

* **N1 (prefill key-range passes) does not pay at any context this box can
  measure.** +0.05% / +0.27% at 16k against a 0.84% control spread, with a
  positive control (scalar attention) moving the same instrument 20% on the same
  binary and prompt. oMLX measured it at ~85 vs ~103 TFLOPS over **1M** keys; ours
  is 134 MB of KV per full layer at 16k, and §10 already showed a 64x smaller K/V
  allocation buys 6%. **No residency loss exists here for the slices to recover.**
* **N3(b), the `HS == 2` head-dim split, is not applicable.** `QB = (8 * 1) / 8 = 1`
  band at our decode shape, so `HS = 2` would idle six of eight simdgroups, and PV
  is already split 8 ways over the value dims by the pre-existing `nt = sg*NPER + j`
  partition. Arithmetic in §19.
* **`ATTENTION_SPLITS_MAX = 16` is not the constraint.** 24 and 32 clamp (measured:
  12.777 / 12.780 against 16's 12.776) and the curve peaked at 13, *below* the cap.
  Raising it could only help the 64-thread sliding shape, which does not use the
  MMA kernel at all.
* **§10's microbench cannot attribute or rank attention arms.** Re-confirmed and
  left in force: every conclusion in this session came from E2E interleaved A/B.

### 7.4 The findings worth more than the numbers

* **The measurement, not the kernel, is the binding constraint on this box** (OQ8,
  now with numbers). The memory signals cannot separate a good run from a stalled
  one; `sys` CPU is 3.5-32% of wall on *good* runs because it is dominated by the
  11.5 GiB mmap'd weight read, so OQ5's ">2% CPU" rule read that way rejects good
  runs; and 24k decode's own 18.35 GiB footprint evicts the desktop. **A quieter
  box, not a better kernel, is what unblocks AC1.**
* **A self-comparison cannot detect a uniform corruption.** Three instances in this
  spec now (two shipped historically). When adding an optimisation, the control
  must be an *independent implementation of the same maths*, not another run of the
  optimised code.
* **"Bit-exact" claims need a tolerance an order of magnitude tighter than the
  house one.** A one-fp16-ulp regression is 10x inside the 5e-3 that every
  attention test here uses; it would have shipped as "exact".
* **Pass boundaries must land on tile boundaries** — a correctness invariant of the
  online softmax, not a performance detail (§16.2).
* **WITHDRAWN (F7): the "PRE-EXISTING crash" this file used to list.** A short
  prompt was filed as failing with `Metal command buffer failed (unlabelled)
  status=Error` at `--max-new-tokens >= 32`, "reproducible every time", verified
  on base `b92234f`. It **does not reproduce**: 29 attempts across nine token
  counts all returned rc=0 with the right answer, and `refprompts.sh` returned
  rc=0 twice with all five answers correct. The base-build leg was never logged.
  **No defect is filed and no time should be spent on the prompt.** The plausible
  reading — offered as a reading, not a finding — is that it was a symptom of the
  OQ8 stall (`unused` at 0.03–0.08 GiB on *good* runs), not a code path. See
  implementation_notes.md §20.1.

### 7.5 What is still open

> **SUPERSEDED IN PART BY THE FIXER PASS (2026-10-05, `fixer_report.md`).** Read
> this list together with §7.6 below; three of the five items have answers.

* **AC2 (prefill ≥ 90 @24k): unmet, 20% short, and unmoved.** Four levers tried
  (T3, T4f, T2b, N1) against one target. §12.1's arithmetic still holds: attention
  must go 149.7 s → ≤ 56 s, a 2.7× kernel speedup, at 1.74 TFLOP/s against a ~3.4
  target. N1 was supposed to be the mechanism for it and does not apply here.
  **F5: the arithmetic in that sentence is wrong by ~1.4x on the ask — see §23.**
  88% of the budget is spent before attention starts, attention gets 33.5 s and
  not 56 s, and the ask is **3.8x at 46% of tensor peak** against a measured 1.9.
* **AC1 (decode ≥ 14 @24k): unmeasurable.** Needs a quieter box (OQ8) or a
  lower-footprint 24k configuration. The splits=13 result is the best decode lever
  this spec found and its effect is still rising at 16k.
* **OQ7 is still open** and now more pointed: removing the QK and PV *instructions*
  from the flash kernel does not speed it up, and §16 adds that its bytes are not
  the constraint either. Whatever it is, it is not DRAM traffic and not instruction
  count as currently modelled.
  **F1: CLOSED — and it was a stale build, not a mystery.** §10's arms never
  reached the shader (§22). With the ablation carried in the uniform block, an
  empty body reads **0.19 ms** against 1 143 ms, removing QK is **2.26x**, removing
  PV is **1.68x**, the softmax is **0.4%** and every fragment load together is
  **8-11%**. The kernel is **MMA-issue-bound**: ~88% `simdgroup_multiply_accumulate`,
  with the accumulator-chain, register-pressure, memory and occupancy explanations
  each eliminated by a measurement.
* **The `workgroups` default flip.** One line, +2.17% @16k, provably a no-op for
  83% of layers. Needs an owner.
  **F4: measured, and the reason it is a real decision.** The "no-op for 83% of
  layers" is the **sliding ring** (`key_count = min(position, 1024)`), not the wave
  budget, and the first context at which the two rules can differ is **896**, not
  120 — so N6's "five reference prompts byte-identical" was vacuous. The greedy
  token delta is in §24. **Not flipped here**: the quality decision is the owner's.
* **The pre-existing Metal crash** (§20.1). Needs per-dispatch decode attribution.
  **F7: WITHDRAWN — 29 attempts across nine token counts, rc=0 every time. No
  defect is filed and no time should be spent on the prompt.**

### 7.6 What the fixer pass added (2026-10-05)

* **An attribution instrument that works** (§22), and the first real cost split of
  the flash kernel. This is the durable output: not the numbers, the *method* — an
  ablation selector in the uniform block, so one build times every arm and a stale
  shader is impossible.
* **The o_proj lead is dead for prefill** (§23.1). `spec/decode-20tps`'s
  "o_proj at half the bandwidth" is a **decode GEMV (`M = 1`)** measurement. In
  prefill all seven projections run at 6.4-8.6 TFLOP/s, o_proj at **98% of
  down_proj's efficiency**, and the whole GEMM at **43% of tensor peak**. There is
  no anomalous projection to fix and 273 GB/s is irrelevant (the chunked GEMM needs
  1.3 GB/s of it).
* **The sliding-ring reason** (`key_count = min(position, 1024)`) written down for
  the first time, which is what actually makes the `workgroups` flip a 10-layer
  change rather than a 60-layer one.
* **The rescale guard is worth 2.5-4.9%**, measured in the same sweep across six shapes (F8).
  It is a real few percent, not an unmeasured optimisation. §17.1 argued exactness
  only; it now has a number too.
* **What is still missing**: a mechanism for the MMA issue rate. Every lever that
  follows from the measured split is negative (§22.3, §23.5).
